# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optional display of material fragments and the mechanical DOFs they share.

Material component labels must come from positive-area face adjacency, independently
of solver node indices. A node shared across those labels is a mechanical bridge;
sharing an original parent tetrahedron alone is not. Topology is inspected only
when explicitly rebuilt. Animated positions remain on the device.

Opposite incision faces which retain shared basis nodes are highlighted even
within one material component, exposing unresolved motion across a finite crack.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from .fragment_device import FragmentArrays
from .knife import _polygon_overlap_2d

_EdgeMasks = wp.types.vector(length=64, dtype=wp.uint64)


@dataclass(frozen=True)
class ConnectivityLinks:
    """Representative fragment links, excluding inactive reserved storage."""

    bridges: np.ndarray
    connections: np.ndarray
    shared_nodes: np.ndarray
    component_count: int


def connectivity_links(nodes: np.ndarray, components: np.ndarray, count: int) -> ConnectivityLinks:
    """Find shared solver nodes, with compact spanning links rather than cliques.

    A bridge connects different material components; a connection stays within
    one component. Duplicate node slots and duplicate fragment pairs are removed.
    Component IDs need not be dense. Negative node indices are inactive slots.
    """
    nodes, components = np.asarray(nodes), np.asarray(components)
    if nodes.ndim != 2 or components.ndim != 1 or not 0 <= count <= min(len(nodes), len(components)):
        raise ValueError("Expected node rows and component labels for every active fragment")
    if np.any(components[:count] < 0):
        raise ValueError("Active fragments need nonnegative material component labels")
    owners = {}
    for fragment, indices in enumerate(nodes[:count]):
        for node in np.unique(indices):
            if node >= 0:
                owners.setdefault(int(node), {}).setdefault(int(components[fragment]), []).append(fragment)
    bridges, connections, shared_nodes = set(), set(), []
    for node, groups in owners.items():
        representatives = [indices[0] for indices in groups.values()]
        if len(groups) > 1:
            shared_nodes.append(node)
            for other in representatives[1:]:
                bridges.add(tuple(sorted((representatives[0], other))))
        for indices in groups.values():
            for other in indices[1:]:
                connections.add((indices[0], other))
    return ConnectivityLinks(
        np.asarray(sorted(bridges), dtype=np.int32).reshape(-1, 2),
        np.asarray(sorted(connections), dtype=np.int32).reshape(-1, 2),
        np.asarray(sorted(shared_nodes), dtype=np.int32),
        len(np.unique(components[:count])),
    )


def incision_links(nodes: np.ndarray, faces: np.ndarray, vertices: np.ndarray) -> np.ndarray:
    """Match overlapping opposite crack lips which retain a common solver node.

    Face rows hold owner, original parent, interface ID and polygon size. Vertex
    coordinates use the parent's three independent barycentric coordinates.
    Grouping by parent and interface avoids comparing unrelated cut surfaces.
    """
    groups = {}
    node_sets = [set(row[row >= 0]) for row in nodes]
    for index, (owner, parent, interface, size) in enumerate(faces):
        polygon = vertices[index, :size]
        relative = polygon - polygon[0]
        normal = np.cross(relative, np.roll(relative, -1, axis=0)).sum(axis=0)
        length = np.linalg.norm(normal)
        if length > 0:
            groups.setdefault((int(parent), int(interface)), []).append(
                (int(owner), polygon, normal / length, polygon.min(axis=0), polygon.max(axis=0))
            )
    pairs = set()
    for group in groups.values():
        if len(group) < 2:
            continue
        lower = np.min([item[3] for item in group], axis=0)
        upper = np.max([item[4] for item in group], axis=0)
        span = max(np.linalg.norm(upper - lower), 1e-12)
        tolerance = 256 * np.finfo(float).eps * span
        axis = int(np.argmax(upper - lower))
        ordered = sorted(group, key=lambda item: item[3][axis])
        for index, (owner_a, polygon_a, normal_a, lower_a, upper_a) in enumerate(ordered):
            for other in range(index + 1, len(ordered)):
                owner_b, polygon_b, normal_b, lower_b, upper_b = ordered[other]
                if lower_b[axis] > upper_a[axis] + tolerance:
                    break
                if owner_a == owner_b or not node_sets[owner_a].intersection(node_sets[owner_b]):
                    continue
                if np.any(np.minimum(upper_a, upper_b) < np.maximum(lower_a, lower_b) - tolerance):
                    continue
                if normal_a @ normal_b > -1.0 + 1e-10:
                    continue
                if np.max(np.abs((polygon_b - polygon_a[0]) @ normal_a)) > tolerance:
                    continue
                projection = np.delete(np.arange(3), np.argmax(np.abs(normal_a)))
                area = _polygon_overlap_2d(polygon_a[:, projection], polygon_b[:, projection], tolerance)
                if area > tolerance * span:
                    pairs.add(tuple(sorted((owner_a, owner_b))))
    return np.asarray(sorted(pairs), dtype=np.int32).reshape(-1, 2)


@wp.kernel(enable_backward=False)
def _count_cut_faces(fragments: FragmentArrays, counts: wp.array[int]):
    fragment = wp.tid()
    count = int(0)
    for face in range(fragments.face_count[fragment]):
        if fragments.face_kind[fragment, face] == 2 and fragments.face_size[fragment, face] >= 3:
            count += 1
    counts[fragment] = count


@wp.kernel(enable_backward=False)
def _pack_cut_faces(
    fragments: FragmentArrays,
    offsets: wp.array[int],
    metadata: wp.array2d[wp.int64],
    vertices: wp.array2d[wp.vec3d],
):
    fragment = wp.tid()
    index = offsets[fragment]
    for face in range(fragments.face_count[fragment]):
        size = fragments.face_size[fragment, face]
        if fragments.face_kind[fragment, face] == 2 and size >= 3:
            metadata[index, 0] = wp.int64(fragment)
            metadata[index, 1] = wp.int64(fragments.parent[fragment])
            metadata[index, 2] = fragments.face_interface[fragment, face]
            metadata[index, 3] = wp.int64(size)
            for corner in range(size):
                value = fragments.vertices[fragment, fragments.faces[fragment, face, corner]]
                vertices[index, corner] = wp.vec3d(value[1], value[2], value[3])
            index += 1


def _device_incision_links(fragments, nodes, count, device):
    counts = wp.empty(count, dtype=int, device=device)
    wp.launch(_count_cut_faces, dim=count, inputs=[fragments, counts], device=device, record_tape=False)
    offsets = np.concatenate(([0], np.cumsum(counts.numpy(), dtype=np.int32)))
    face_count = int(offsets[-1])
    if face_count == 0:
        return np.empty((0, 2), dtype=np.int32)
    metadata = wp.empty((face_count, 4), dtype=wp.int64, device=device)
    vertices = wp.empty((face_count, fragments.vertex_capacity), dtype=wp.vec3d, device=device)
    wp.launch(
        _pack_cut_faces,
        dim=count,
        inputs=[fragments, wp.array(offsets[:-1], dtype=int, device=device), metadata, vertices],
        device=device,
        record_tape=False,
    )
    return incision_links(nodes, metadata.numpy(), vertices.numpy())


@wp.kernel(enable_backward=False)
def _count_edges(fragments: FragmentArrays, counts: wp.array[int]):
    fragment = wp.tid()
    masks = _EdgeMasks(wp.uint64(0))
    count = int(0)
    for face in range(fragments.face_count[fragment]):
        size = fragments.face_size[fragment, face]
        for corner in range(size):
            first = fragments.faces[fragment, face, corner]
            second = fragments.faces[fragment, face, (corner + 1) % size]
            a, b = wp.min(first, second), wp.max(first, second)
            bit = wp.uint64(1) << wp.uint64(b)
            if a != b and masks[a] & bit == wp.uint64(0):
                masks[a] = masks[a] | bit
                count += 1
    counts[fragment] = count


@wp.kernel(enable_backward=False)
def _pack_edges(
    fragments: FragmentArrays,
    sample_offsets: wp.array[int],
    edge_offsets: wp.array[int],
    components: wp.array[int],
    edges: wp.array2d[int],
    colors: wp.array[wp.vec3],
):
    fragment = wp.tid()
    masks = _EdgeMasks(wp.uint64(0))
    index = edge_offsets[fragment]
    # A repeating palette makes separate material components visible without
    # assigning meaning to the non-dense root labels used by connectivity.
    color_id = components[fragment] % 6
    color = wp.vec3(0.35, 0.85, 1.0)
    if color_id == 1:
        color = wp.vec3(0.85, 0.5, 1.0)
    elif color_id == 2:
        color = wp.vec3(0.3, 1.0, 0.55)
    elif color_id == 3:
        color = wp.vec3(1.0, 0.6, 0.3)
    elif color_id == 4:
        color = wp.vec3(0.95, 0.9, 0.5)
    elif color_id == 5:
        color = wp.vec3(0.95, 0.55, 0.7)
    for face in range(fragments.face_count[fragment]):
        size = fragments.face_size[fragment, face]
        for corner in range(size):
            first = fragments.faces[fragment, face, corner]
            second = fragments.faces[fragment, face, (corner + 1) % size]
            a, b = wp.min(first, second), wp.max(first, second)
            bit = wp.uint64(1) << wp.uint64(b)
            if a != b and masks[a] & bit == wp.uint64(0):
                masks[a] = masks[a] | bit
                edges[index, 0] = sample_offsets[fragment] + a
                edges[index, 1] = sample_offsets[fragment] + b
                colors[index] = color
                index += 1


@wp.kernel(enable_backward=False)
def _pack_samples(
    fragments: FragmentArrays,
    offsets: wp.array[int],
    coordinates: wp.array[wp.vec4d],
    owners: wp.array[int],
):
    fragment = wp.tid()
    center = wp.vec4d(0.0)
    count = fragments.vertex_count[fragment]
    for vertex in range(count):
        value = fragments.vertices[fragment, vertex]
        center += value
        coordinates[offsets[fragment] + vertex] = value
        owners[offsets[fragment] + vertex] = fragment
    coordinates[fragment] = center / wp.float64(count)
    owners[fragment] = fragment


@wp.kernel(enable_backward=False)
def _sample_linear(
    q: wp.array[wp.vec3],
    owners: wp.array[int],
    nodes: wp.array2d[int],
    coordinates: wp.array[wp.vec4d],
    points: wp.array[wp.vec3],
):
    sample = wp.tid()
    position = wp.vec3d(0.0)
    fragment = owners[sample]
    for corner in range(4):
        position += coordinates[sample][corner] * wp.vec3d(q[nodes[fragment, corner]])
    points[sample] = wp.vec3(position)


@wp.kernel(enable_backward=False)
def _line_positions(
    points: wp.array[wp.vec3],
    pairs: wp.array2d[int],
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    line = wp.tid()
    starts[line] = points[pairs[line, 0]]
    ends[line] = points[pairs[line, 1]]


class CutConnectivityDebug:
    """Opt-in physical fragment edges and shared-DOF connections [m].

    Call :meth:`rebuild` after enabling or accepting a topology change. It reads
    compact active topology once, never the reserved face buffers. :meth:`render`
    evaluates positions on the device without host readback. The disabled path
    allocates nothing, reads nothing, and launches no kernels.
    """

    class _Lines:
        def __init__(self, pairs, colors, device):
            self.pairs = pairs if isinstance(pairs, wp.array) else wp.array(pairs, dtype=int, device=device)
            if isinstance(colors, wp.array):
                self.colors = colors
            else:
                self.colors = wp.empty(len(self.pairs), dtype=wp.vec3, device=device)
                self.colors.fill_(wp.vec3(*colors))
            self.starts = wp.empty(len(self.pairs), dtype=wp.vec3, device=device)
            self.ends = wp.empty_like(self.starts)

    def __init__(self, *, device):
        self.device = wp.get_device(device)
        self.enabled = False
        self.show_connections = False
        self.show_wireframe = True
        self.component_count = self.shared_node_count = self.bridge_count = self.connection_count = 0
        self.incision_connection_count = 0
        self.fragment_count = 0
        self.cross_component_nodes = np.empty(0, dtype=np.int32)
        self._lines = {}
        self._visible = set()

    def rebuild(self, fragments, components, *, basis_nodes=None):
        """Read active topology and prepare compact, reusable display buffers.

        Args:
            fragments: Accepted FragmentArrays or FragmentDevice geometry.
            components: Independent material component labels on the device.
            basis_nodes: Actual solver node indices, overriding geometry bindings.
        """
        if not self.enabled:
            return
        fragments = getattr(fragments, "arrays", fragments)
        if fragments.vertex_capacity > 64:
            raise ValueError("Connectivity debug supports at most 64 vertices per convex fragment")
        count = int(fragments.count.numpy()[0])
        if not 0 <= count <= fragments.capacity:
            raise ValueError("Invalid active fragment count")
        nodes = fragments.nodeindices if basis_nodes is None else basis_nodes
        if nodes.ndim != 2 or nodes.shape[1] != 4 or nodes.shape[0] < count:
            raise ValueError("Expected four solver node indices for every active fragment")
        if components.ndim != 1 or components.shape[0] < count:
            raise ValueError("Expected a material component label for every active fragment")
        node_values = nodes[:count].numpy()
        labels = components[:count].numpy()
        links = connectivity_links(node_values, labels, count)
        self.fragment_count = count
        self.component_count = links.component_count
        self.cross_component_nodes = links.shared_nodes
        self.shared_node_count = len(links.shared_nodes)
        self.bridge_count, self.connection_count = len(links.bridges), len(links.connections)
        incisions = (
            _device_incision_links(fragments, node_values, count, self.device)
            if count
            else np.empty((0, 2), dtype=np.int32)
        )
        # Cross-component shared nodes already have red bridges. Orange instead
        # explains why crack lips within one remaining material piece interact.
        incisions = incisions[labels[incisions[:, 0]] == labels[incisions[:, 1]]]
        self.incision_connection_count = len(incisions)
        self._nodes = wp.array(node_values, dtype=int, device=self.device)
        self._lines = {
            "bridges": self._Lines(links.bridges, (1.0, 0.12, 0.12), self.device),
            "connections": self._Lines(links.connections, (1.0, 0.85, 0.15), self.device),
            "incisions": self._Lines(incisions, (1.0, 0.4, 0.05), self.device),
        }
        if count == 0:
            self._lines["fragments"] = self._Lines(np.empty((0, 2), dtype=np.int32), (1.0, 1.0, 1.0), self.device)
            return
        vertex_counts = fragments.vertex_count[:count].numpy()
        sample_offsets = count + np.concatenate(([0], np.cumsum(vertex_counts, dtype=np.int32)))
        sample_count = int(sample_offsets[-1])
        offsets = wp.array(sample_offsets[:-1], dtype=int, device=self.device)
        self._coordinates = wp.empty(sample_count, dtype=wp.vec4d, device=self.device)
        self._owners = wp.empty(sample_count, dtype=int, device=self.device)
        self._points = wp.empty(sample_count, dtype=wp.vec3, device=self.device)
        wp.launch(
            _pack_samples,
            dim=count,
            inputs=[fragments, offsets, self._coordinates, self._owners],
            device=self.device,
            record_tape=False,
        )
        edge_counts = wp.empty(count, dtype=int, device=self.device)
        wp.launch(_count_edges, dim=count, inputs=[fragments, edge_counts], device=self.device, record_tape=False)
        edge_offsets = np.concatenate(([0], np.cumsum(edge_counts.numpy(), dtype=np.int32)))
        edge_count = int(edge_offsets[-1])
        edges = wp.empty((edge_count, 2), dtype=int, device=self.device)
        colors = wp.empty(edge_count, dtype=wp.vec3, device=self.device)
        # Dense palette indices prevent component root IDs separated by six
        # from accidentally receiving the same first color.
        palette = np.unique(labels, return_inverse=True)[1].astype(np.int32)
        wp.launch(
            _pack_edges,
            dim=count,
            inputs=[
                fragments,
                offsets,
                wp.array(edge_offsets[:-1], dtype=int, device=self.device),
                wp.array(palette, dtype=int, device=self.device),
                edges,
                colors,
            ],
            device=self.device,
            record_tape=False,
        )
        self._lines["fragments"] = self._Lines(edges, colors, self.device)

    def render(self, viewer, q, *, kind="linear", hide=False):
        """Draw deformed links; disabled calls only hide previously visible lines."""
        active = self.enabled and not hide and self.fragment_count > 0
        if active:
            if kind != "linear":
                raise ValueError("State kind must match the basis used when rebuilding debug topology")
            sample_count = len(self._points) if self.show_wireframe else self.fragment_count
            wp.launch(
                _sample_linear,
                dim=sample_count,
                inputs=[q, self._owners, self._nodes, self._coordinates, self._points],
                device=self.device,
                record_tape=False,
            )
        for name, lines in self._lines.items():
            visible = active and len(lines.pairs) > 0
            if name == "connections":
                visible = visible and self.show_connections
            elif name == "fragments":
                visible = visible and self.show_wireframe
            if visible:
                wp.launch(
                    _line_positions,
                    dim=len(lines.pairs),
                    inputs=[self._points, lines.pairs, lines.starts, lines.ends],
                    device=self.device,
                    record_tape=False,
                )
                viewer.log_lines(f"cut_debug/{name}", lines.starts, lines.ends, lines.colors)
                self._visible.add(name)
            elif name in self._visible:
                viewer.log_lines(f"cut_debug/{name}", lines.starts, lines.ends, lines.colors, hidden=True)
                self._visible.discard(name)

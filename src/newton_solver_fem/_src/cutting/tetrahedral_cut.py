# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU incisions and material removal for piecewise-linear virtual tetrahedra.

Clipping changes material domains, not their affine interpolation. Each fragment
keeps four original basis positions; disconnected regions receive exact copies.
The FEM solver integrates its constant strain over each fragment's volume.
Boundary contact must use the clipped surface, never the enclosing basis nodes.
"""

from dataclasses import dataclass

import numpy as np
import warp as wp

from . import fragment_connectivity_gpu as connectivity_kernels
from .adjacency import CutAdjacency
from .fragment_adjacency import FragmentAdjacency
from .fragment_connectivity_gpu import FragmentConnectivityGPU
from .fragment_device import (
    FragmentArrays,
    FragmentDevice,
    _clip_patch,
    _compact_fragments,
    _copy_geometry,
    _fragment_center,
    _reset_lineage,
    _reset_stage,
    _validate_patch,
)
from .fragment_merge import FragmentMerger
from .knife import _knife_polygons, make_tetrahedral_fragments


@wp.kernel(enable_backward=False)
def _pack_positions(positions: wp.array[wp.vec3], coordinates: wp.array[wp.float64]):
    node = wp.tid()
    point = positions[node]
    for component in range(3):
        coordinates[12 * node + component] = wp.float64(point[component])


@wp.kernel(enable_backward=False)
def _publish_nodes(geometry: FragmentArrays, nodes: wp.array2d[int]):
    fragment, corner = wp.tid()
    if fragment < geometry.count[0]:
        geometry.nodeindices[fragment, corner] = nodes[fragment, corner]


@wp.kernel(enable_backward=False)
def _integrate_fragments(
    geometry: FragmentArrays,
    parent_volumes: wp.array[wp.float64],
    fractions: wp.array[float],
    mass_integrals: wp.array[wp.vec4],
    second_moments: wp.array3d[wp.float64],
    error: wp.array[int],
):
    fragment = wp.tid()
    fractions[fragment] = 0.0
    mass_integrals[fragment] = wp.vec4(0.0)
    if second_moments.shape[0] > 0:
        for i in range(4):
            for j in range(4):
                second_moments[fragment, i, j] = wp.float64(0.0)
    if fragment >= geometry.count[0] or geometry.error[0] != 0:
        return
    volume = wp.float64(0.0)
    moment = wp.vec4d(0.0)
    origin = _fragment_center(geometry, fragment)
    for face in range(geometry.face_count[fragment]):
        a = geometry.vertices[fragment, geometry.faces[fragment, face, 0]]
        for corner in range(1, geometry.face_size[fragment, face] - 1):
            b = geometry.vertices[fragment, geometry.faces[fragment, face, corner]]
            c = geometry.vertices[fragment, geometry.faces[fragment, face, corner + 1]]
            da, db, dc = a - origin, b - origin, c - origin
            v = wp.dot(
                wp.vec3d(da[1], da[2], da[3]),
                wp.cross(wp.vec3d(db[1], db[2], db[3]), wp.vec3d(dc[1], dc[2], dc[3])),
            ) / wp.float64(6.0)
            volume += v
            moment += v * (origin + a + b + c) / wp.float64(4.0)
            if second_moments.shape[0] > 0:
                total = origin + a + b + c
                for i in range(4):
                    for j in range(4):
                        product = origin[i] * origin[j] + a[i] * a[j] + b[i] * b[j] + c[i] * c[j]
                        second_moments[fragment, i, j] += v * (total[i] * total[j] + product) / wp.float64(20.0)
    if not wp.isfinite(volume) or volume <= wp.float64(0.0):
        wp.atomic_max(error, 0, 1)
        return
    scale = wp.float64(6.0) * parent_volumes[geometry.parent[fragment]]
    fractions[fragment] = float(wp.float64(6.0) * volume)
    result = wp.vec4(0.0)
    for corner in range(4):
        value = moment[corner] * scale
        if value < wp.float64(-1.0e-12) * scale or not wp.isfinite(value):
            wp.atomic_max(error, 0, 1)
        result[corner] = float(wp.max(value, wp.float64(0.0)))
    mass_integrals[fragment] = result
    if second_moments.shape[0] > 0:
        for i in range(4):
            for j in range(4):
                second_moments[fragment, i, j] *= scale


@wp.kernel(enable_backward=False)
def _build_boundary(
    geometry: FragmentArrays,
    nodes: wp.array2d[int],
    components: wp.array[int],
    exterior: wp.array[int],
    rest: wp.array[wp.vec3d],
    elements: wp.array2d[int],
    vertex_nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    triangles: wp.array[int],
    triangle_components: wp.array[int],
    triangle_kinds: wp.array[int],
    reference_area: wp.array[float],
    triangle_fragments: wp.array[int],
    triangle_interfaces: wp.array[wp.int64],
    reference_weights: wp.array[wp.vec4d],
    count: wp.array[int],
    error: wp.array[int],
):
    fragment = wp.tid()
    if fragment >= geometry.count[0] or geometry.error[0] != 0:
        return
    for face in range(geometry.face_count[fragment]):
        kind = geometry.face_kind[fragment, face]
        visible = kind == 2
        if kind == 0:
            interface = int(-geometry.face_interface[fragment, face] - wp.int64(1))
            if interface >= 0 and interface < exterior.shape[0]:
                visible = exterior[interface] != 0
        if not visible:
            continue
        a = geometry.vertices[fragment, geometry.faces[fragment, face, 0]]
        for corner in range(1, geometry.face_size[fragment, face] - 1):
            b = geometry.vertices[fragment, geometry.faces[fragment, face, corner]]
            c = geometry.vertices[fragment, geometry.faces[fragment, face, corner + 1]]
            triangle = wp.atomic_add(count, 0, 1)
            if triangle >= triangle_components.shape[0]:
                wp.atomic_max(error, 0, 2)
                return
            first, second, third = wp.vec3d(0.0), wp.vec3d(0.0), wp.vec3d(0.0)
            parent = geometry.parent[fragment]
            for k in range(4):
                position = rest[elements[parent, k]]
                first += a[k] * position
                second += b[k] * position
                third += c[k] * position
                for vertex in range(3):
                    vertex_nodes[3 * triangle + vertex, k] = nodes[fragment, k]
            weights[3 * triangle] = wp.vec4(a)
            weights[3 * triangle + 1] = wp.vec4(b)
            weights[3 * triangle + 2] = wp.vec4(c)
            for vertex in range(3):
                triangles[3 * triangle + vertex] = 3 * triangle + vertex
            triangle_components[triangle] = components[fragment]
            triangle_kinds[triangle] = kind
            triangle_fragments[triangle] = fragment
            triangle_interfaces[triangle] = geometry.face_interface[fragment, face]
            if reference_weights.shape[0] > 0:
                reference_weights[3 * triangle] = a
                reference_weights[3 * triangle + 1] = b
                reference_weights[3 * triangle + 2] = c
            reference_area[triangle] = float(wp.length(wp.cross(second - first, third - first)) * wp.float64(0.5))


class _BoundedConnectivity(FragmentConnectivityGPU):
    """Use bounded incident groups instead of a dense node-by-fragment table."""

    def __init__(self, elements, node_count, *, fragment_capacity, edge_capacity, max_copies, group_capacity, device):
        self.device = wp.get_device(device)
        self.node_count, self.max_copies = int(node_count), int(max_copies)
        self.fragment_capacity, self.edge_capacity = int(fragment_capacity), int(edge_capacity)
        pool_count = node_count * max_copies
        self.elements = wp.array(elements, dtype=int, device=self.device)

        def empty(shape):
            return wp.empty(shape, dtype=int, device=self.device)

        self.basis, self.group_source, self.group_size, self.assigned = (empty(4 * fragment_capacity) for _ in range(4))
        self.parent_regions, self.material_components = empty(fragment_capacity), empty(fragment_capacity)
        self.inherited, self.nodes = empty((fragment_capacity, 4)), empty((fragment_capacity, 4))
        self.used, self.group_count = empty(pool_count), empty(pool_count)
        self.groups = empty((pool_count, group_capacity))
        self.clone_sources = empty((pool_count, 2))
        self.clone_count, self.error = empty(1), empty(1)
        self._no_overflow = wp.zeros(1, dtype=int, device=self.device)
        self._inputs = None
        wp.load_module(module=connectivity_kernels, device=self.device, block_dim=256)


@dataclass
class LinearCutSurface:
    """Fixed-capacity surface triangles, with affine coordinates per vertex."""

    vertex_nodes: wp.array
    weights: wp.array
    indices: wp.array
    components: wp.array
    kind: wp.array
    reference_area: wp.array
    count: wp.array
    error: wp.array
    fragments: wp.array | None = None
    interfaces: wp.array | None = None
    reference_weights: wp.array | None = None

    @classmethod
    def allocate(cls, triangle_capacity, device, *, consistent_mass=False):
        return cls(
            wp.zeros((3 * triangle_capacity, 4), dtype=int, device=device),
            wp.zeros(3 * triangle_capacity, dtype=wp.vec4, device=device),
            wp.zeros(3 * triangle_capacity, dtype=int, device=device),
            wp.zeros(triangle_capacity, dtype=int, device=device),
            wp.zeros(triangle_capacity, dtype=int, device=device),
            wp.zeros(triangle_capacity, dtype=float, device=device),
            wp.zeros(1, dtype=int, device=device),
            wp.zeros(1, dtype=int, device=device),
            fragments=wp.zeros(triangle_capacity, dtype=int, device=device),
            interfaces=wp.zeros(triangle_capacity, dtype=wp.int64, device=device),
            reference_weights=wp.zeros(3 * triangle_capacity if consistent_mass else 0, dtype=wp.vec4d, device=device),
        )


class LinearTetCut:
    """Captured local geometry edits with transactional publication.

    ``propose`` leaves published geometry unchanged. After checking metadata,
    the caller copies coordinates/velocities using ``candidate.clone_sources``,
    updates its sparse topology, and calls ``commit``. Geometry edits and
    mechanical integration remain separate operations.
    """

    def __init__(
        self,
        rest_positions,
        elements,
        *,
        fragment_capacity=None,
        node_copies=4,
        vertex_capacity=12,
        face_capacity=12,
        group_capacity=128,
        triangle_capacity=None,
        max_patches=2,
        consistent_mass=False,
        device=None,
    ):
        points = np.asarray(rest_positions, dtype=np.float64)
        topology = np.asarray(elements, dtype=np.int32)
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("rest_positions must contain finite three-dimensional coordinates")
        if topology.ndim != 2 or topology.shape[1] != 4 or np.any(topology < 0) or np.any(topology >= len(points)):
            raise ValueError("elements must contain four valid indices per tetrahedron")
        tetrahedra = points[topology]
        volumes = np.linalg.det((tetrahedra[:, 1:] - tetrahedra[:, :1]).transpose(0, 2, 1)) / 6.0
        if np.any(volumes <= 0) or not np.isfinite(volumes).all():
            raise ValueError("reference tetrahedra must have positive finite volume")
        fragment_capacity = max(16, 2 * len(topology)) if fragment_capacity is None else int(fragment_capacity)
        if fragment_capacity < len(topology) or min(node_copies, group_capacity, max_patches) < 1:
            raise ValueError("capacities must be positive and contain the input mesh")
        self.device = wp.get_device(device)
        self.node_count, self.node_copies = len(points), int(node_copies)
        self.capacity, self.max_patches = fragment_capacity, int(max_patches)
        self.rest = wp.array(points, dtype=wp.vec3d, device=self.device)
        self.elements = wp.array(topology, dtype=int, device=self.device)
        self.parent_volumes = wp.array(volumes, dtype=wp.float64, device=self.device)
        self.coordinates = wp.zeros(12 * self.node_count * node_copies, dtype=wp.float64, device=self.device)
        self._input_positions = wp.array(np.tile(points, (node_copies, 1)), dtype=wp.vec3, device=self.device)
        self.tolerance = float(128 * np.finfo(float).eps * max(np.linalg.norm(np.ptp(points, axis=0)), 1.0))

        def storage(capacity):
            return FragmentDevice(
                capacity, device=self.device, vertex_capacity=vertex_capacity, face_capacity=face_capacity
            )

        self.geometry = storage(fragment_capacity)
        self.buffers = [storage(fragment_capacity) for _ in range(2)]
        self.scratch = [storage(fragment_capacity) for _ in range(2)]
        self.clipped = storage(6 * fragment_capacity)
        fragments = make_tetrahedral_fragments(topology)
        self.geometry.upload(fragments, topology, points, topology)
        interfaces = self.geometry.original_interfaces
        multiplicities = {}
        interface_owners = {}
        for fragment in fragments:
            for face in fragment.faces:
                multiplicities[face.interface] = multiplicities.get(face.interface, 0) + 1
                interface_owners.setdefault(face.interface, []).append(fragment.parent)
        interface_parents = np.full((len(interfaces), 2), -1, dtype=np.int32)
        for interface, face in interfaces.items():
            owners = interface_owners[face]
            if len(owners) > 2:
                raise ValueError("Original tetrahedral faces must have at most two incident elements")
            interface_parents[-interface - 1, : len(owners)] = owners
        self.exterior = wp.array(
            [int(multiplicities[interfaces[-index - 1]] == 1) for index in range(len(interfaces))],
            dtype=int,
            device=self.device,
        )
        self.polygons = wp.zeros((max_patches, 4), dtype=wp.vec3d, device=self.device)
        self.polygon_sizes = wp.zeros(max_patches, dtype=int, device=self.device)
        self.normals = wp.zeros(max_patches, dtype=wp.vec3d, device=self.device)
        self.origins = wp.zeros(max_patches, dtype=wp.vec3d, device=self.device)
        self.interface_start = wp.array([1], dtype=wp.int64, device=self.device)
        self.changed = wp.zeros(len(topology), dtype=int, device=self.device)
        self.merger = FragmentMerger(len(topology), fragment_capacity, device=self.device)
        edge_capacity = 16 * fragment_capacity
        self.adjacency = FragmentAdjacency(
            face_capacity=face_capacity * fragment_capacity,
            edge_capacity=edge_capacity,
            vertex_capacity=vertex_capacity,
            minimum_area=1.0e-14 * np.linalg.norm(np.ptp(points, axis=0)) ** 2,
            device=self.device,
        )
        self._cut_adjacency = CutAdjacency(self.adjacency, interface_parents)
        self._cut_adjacency.initialize(self.geometry, self.rest, self.elements)
        self.candidate = _BoundedConnectivity(
            topology,
            len(points),
            fragment_capacity=fragment_capacity,
            edge_capacity=edge_capacity,
            max_copies=node_copies,
            group_capacity=group_capacity,
            device=self.device,
        )
        self.volume_fractions = wp.zeros(fragment_capacity, dtype=float, device=self.device)
        self.mass_integrals = wp.zeros(fragment_capacity, dtype=wp.vec4, device=self.device)
        # Exact P1 mass requires second moments of the clipped domain. The
        # lumped-mass example skips this work; monolithic shell/solid snapshots
        # also retain double-precision boundary weights for reference mapping.
        self.second_moments = wp.zeros(
            (fragment_capacity if consistent_mass else 0, 4, 4), dtype=wp.float64, device=self.device
        )
        self.integration_error = wp.zeros(1, dtype=int, device=self.device)
        triangle_capacity = 12 * fragment_capacity if triangle_capacity is None else int(triangle_capacity)
        self.surface = LinearCutSurface.allocate(triangle_capacity, self.device, consistent_mass=consistent_mass)
        self._published_components = wp.zeros(fragment_capacity, dtype=int, device=self.device)
        self._graphs = {}
        self._sphere = None
        self._sphere_graph = None
        self._pending = None
        self._interface_value = 1
        self._pending_interface_stride = 0
        self._integrate(self.geometry)
        self._surface(self.geometry, self.geometry.nodeindices, self._published_components)

    @property
    def material_components(self):
        """Accepted material-component labels, independent of shared solver nodes."""
        return self._published_components

    def propose_triangles(self, positions, triangles):
        """Stage a finite knife incision described by world-space triangles."""
        return self.propose(positions, _knife_polygons(triangles, self.tolerance))

    def _run(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _integrate(self, geometry):
        self.integration_error.zero_()
        self._run(
            _integrate_fragments,
            self.capacity,
            [
                geometry.arrays,
                self.parent_volumes,
                self.volume_fractions,
                self.mass_integrals,
                self.second_moments,
                self.integration_error,
            ],
        )

    def _surface(self, geometry, nodes, components):
        surface = self.surface
        surface.count.zero_()
        surface.error.zero_()
        self._run(
            _build_boundary,
            self.capacity,
            [
                geometry.arrays,
                nodes,
                components,
                self.exterior,
                self.rest,
                self.elements,
                surface.vertex_nodes,
                surface.weights,
                surface.indices,
                surface.components,
                surface.kind,
                surface.reference_area,
                surface.fragments,
                surface.interfaces,
                surface.reference_weights,
                surface.count,
                surface.error,
            ],
        )

    def launch(self, positions, patch_count):
        """Enqueue fixed-buffer geometry, connectivity and surface updates."""
        if not 1 <= patch_count <= self.max_patches:
            raise ValueError("knife patch count exceeds reserved capacity")
        if positions.shape[0] != self.node_count * self.node_copies:
            raise ValueError("positions must include the reserved node-copy pool")
        self._run(_pack_positions, len(positions), [positions, self.coordinates])
        self.changed.zero_()
        self._run(_reset_lineage, self.capacity, [self.geometry.arrays])
        current = self.geometry
        for patch in range(patch_count):
            target = self.buffers[patch % 2]
            self._run(_reset_stage, 1, [self.clipped.arrays, target.arrays, *(item.arrays for item in self.scratch)])
            self._run(
                _validate_patch,
                1,
                [
                    self.polygons,
                    self.polygon_sizes,
                    patch,
                    self.tolerance,
                    self.normals,
                    self.origins,
                    self.clipped.error,
                ],
            )
            self._run(
                _clip_patch,
                self.capacity,
                [
                    current.arrays,
                    self.clipped.arrays,
                    *(item.arrays for item in self.scratch),
                    self.coordinates,
                    self.polygons,
                    self.polygon_sizes,
                    self.normals,
                    self.origins,
                    patch,
                    self.interface_start,
                    self.rest,
                    self.elements,
                    self.tolerance,
                    self.changed,
                ],
            )
            self._run(
                _compact_fragments,
                self.clipped.capacity,
                [
                    current.arrays,
                    self.clipped.arrays,
                    target.arrays,
                    *(item.arrays for item in self.scratch),
                ],
            )
            current = target
        self.merger.launch(current, self.buffers[patch_count % 2], self.scratch[0])
        return self._complete_geometry(current)

    def _complete_geometry(self, current):
        """Rebuild the material boundary and connectivity for either tool."""
        self._cut_adjacency.propose(current, self.changed, self.rest, self.elements)
        self.candidate.launch(
            current.parent,
            current.nodeindices,
            current.count,
            self.adjacency.edges,
            self.adjacency.edge_count,
            self.adjacency.overflow,
        )
        self._integrate(current)
        self._surface(current, self.candidate.nodes, self.candidate.material_components)
        return current

    def propose(self, positions, polygons):
        """Propose one or more finite blade polygons; return compact diagnostics.

        A metadata readback occurs after the captured geometry transaction.
        The caller owns sparse solver rebuilding and state transfer timing.
        """
        if positions.dtype != wp.vec3 or positions.shape != self._input_positions.shape:
            raise ValueError("positions must be vec3 coordinates for the full reserved node-copy pool")
        patches = tuple(np.asarray(polygon, dtype=np.float64) for polygon in polygons)
        if not 1 <= len(patches) <= self.max_patches:
            raise ValueError("knife patch count exceeds reserved capacity")
        values = np.zeros((self.max_patches, 4, 3))
        sizes = np.zeros(self.max_patches, dtype=np.int32)
        for index, patch in enumerate(patches):
            if patch.shape not in ((3, 3), (4, 3)) or not np.isfinite(patch).all():
                raise ValueError("each knife patch must contain three or four finite points")
            values[index, : len(patch)] = patch
            sizes[index] = len(patch)
        self.polygons.assign(values)
        self.polygon_sizes.assign(sizes)
        wp.copy(self._input_positions, positions)
        key = len(patches)
        while True:
            if self.device.is_cuda:
                if key not in self._graphs:
                    self._capture(key)
                graph, pending = self._graphs[key]
                wp.capture_launch(graph)
            else:
                pending = self.launch(self._input_positions, len(patches))
            error = int(pending.error.numpy()[0])
            if error not in (1, 2) or not self._grow_geometry(error):
                break
        self._pending_interface_stride = 5 * self.max_patches * self.capacity
        return self._diagnostics(pending)

    def _diagnostics(self, pending):
        self._pending = pending
        diagnostics = {
            "geometry_error": int(pending.error.numpy()[0]),
            "connectivity_error": int(self.candidate.error.numpy()[0]),
            "integration_error": int(self.integration_error.numpy()[0]),
            "surface_error": int(self.surface.error.numpy()[0]),
            "fragment_count": int(pending.count.numpy()[0]),
            "clone_count": int(self.candidate.clone_count.numpy()[0]),
            "triangle_count": int(self.surface.count.numpy()[0]),
            "changed_parents": int(np.count_nonzero(self.changed.numpy())),
        }
        if any(
            diagnostics[name] for name in ("geometry_error", "connectivity_error", "integration_error", "surface_error")
        ):
            self.discard()
            raise ValueError(f"linear tetrahedron edit rejected: {diagnostics}")
        return diagnostics

    def _launch_sphere(self):
        self.changed.zero_()
        self._run(_reset_lineage, self.capacity, [self.geometry.arrays])
        target = self.buffers[0]
        self._sphere.launch(
            self.geometry.arrays,
            target.arrays,
            self._input_positions,
            self.rest,
            self.elements,
            self.interface_start,
            self.changed,
            self.tolerance,
        )
        return self._complete_geometry(target)

    def prepare_sphere(self, work_capacity=256):
        """Reserve local clipping scratch and capture an empty spherical edit."""
        if self._sphere is not None:
            return
        self._prepare_subtractor(work_capacity)

    def _prepare_subtractor(self, work_capacity, *, scratch_vertex_capacity=None, scratch_face_capacity=None):
        """Replace only local clipping storage, keeping accepted material intact."""
        from .sphere import SphereSubtractor  # noqa: PLC0415

        previous = self._sphere, self._sphere_graph
        if previous[0] is not None:
            scratch = previous[0].scratch[0]
            if scratch_vertex_capacity is None:
                scratch_vertex_capacity = scratch.vertex_capacity
            if scratch_face_capacity is None:
                scratch_face_capacity = scratch.face_capacity
        try:
            self._sphere = SphereSubtractor(
                vertex_capacity=self.geometry.vertices.shape[1],
                face_capacity=self.geometry.faces.shape[1],
                work_capacity=work_capacity,
                scratch_vertex_capacity=scratch_vertex_capacity,
                scratch_face_capacity=scratch_face_capacity,
                validate_volume=previous[0].validate_volume if previous[0] is not None else False,
                device=self.device,
            )
            self._sphere_graph = None
            self._sphere.assign((1.0e10, 1.0e10, 1.0e10), 1.0)
            self._launch_sphere()
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
                with wp.ScopedCapture(device=self.device) as capture:
                    self._launch_sphere()
                self._sphere_graph = capture.graph
            self.discard()
        except (MemoryError, RuntimeError, ValueError):
            self._sphere, self._sphere_graph = previous
            self.discard()
            raise

    def _grow_geometry(self, error):
        """Grow bounded polygon storage without changing material or FEM arrays."""
        vertices, faces = self.geometry.vertex_capacity, self.geometry.face_capacity
        if error == 1 and vertices < 64:
            vertices = min(64, max(48, 2 * vertices))
        elif error == 2 and faces < 48:
            faces = min(48, max(32, 2 * faces))
        else:
            return False
        self.discard()
        names = ("geometry", "buffers", "scratch", "clipped", "adjacency", "_graphs", "_sphere", "_sphere_graph")
        previous = {name: getattr(self, name) for name in names}

        def storage(capacity):
            return FragmentDevice(capacity, vertex_capacity=vertices, face_capacity=faces, device=self.device)

        try:
            geometry = storage(self.capacity)
            buffers = [storage(self.capacity) for _ in range(2)]
            scratch = [storage(self.capacity) for _ in range(2)]
            clipped = storage(6 * self.capacity)
            old_adjacency = self.adjacency
            adjacency = FragmentAdjacency(
                face_capacity=faces * self.capacity,
                edge_capacity=old_adjacency.edge_capacity,
                vertex_capacity=vertices,
                minimum_area=old_adjacency.minimum_area,
                relative_area=old_adjacency.relative_area,
                candidate_capacity=max(old_adjacency.candidate_capacity, 4 * faces * self.capacity),
                device=self.device,
            )
            geometry.original_interfaces = self.geometry.original_interfaces
            self._run(_copy_geometry, self.capacity, [self.geometry.arrays, geometry.arrays])
            self.geometry, self.buffers, self.scratch, self.clipped = geometry, buffers, scratch, clipped
            self.adjacency, self._graphs = adjacency, {}
            # Accepted fragment IDs and edges are unchanged; only the projected
            # face workspace needs replacing. Subsequent graphs use the new
            # buffers while the existing FEM graph keeps its original arrays.
            self._cut_adjacency.adjacency = adjacency
            if self._sphere is not None:
                self._prepare_subtractor(len(self._sphere.work))
            self.discard()
        except (MemoryError, RuntimeError, ValueError):
            for name, value in previous.items():
                setattr(self, name, value)
            self._cut_adjacency.adjacency = self.adjacency
            self.discard()
            raise
        return True

    def _grow_convex_scratch(self):
        """Keep transient cap partitions local instead of padding the mesh."""
        error = max(int(scratch.error.numpy()[0]) for scratch in self._sphere.scratch)
        scratch = self._sphere.scratch[0]
        vertices, faces = scratch.vertex_capacity, scratch.face_capacity
        if error == 1 and vertices < 64:
            vertices = 64
        elif error == 2 and faces < 64:
            faces = min(64, 2 * faces)
        else:
            return False
        self.discard()
        self._prepare_subtractor(len(self._sphere.work), scratch_vertex_capacity=vertices, scratch_face_capacity=faces)
        return True

    def propose_sphere(self, positions, start, end, radius):
        """Subtract a swept polyhedral sphere of radius [m] from current material.

        The inscribed icosahedron follows the complete start-to-end segment;
        tissue geometry and connectivity remain on the GPU. Publication uses
        the same transaction as a finite line incision.
        """
        from .sphere import spherical_sweep_planes  # noqa: PLC0415

        if positions.dtype != wp.vec3 or positions.shape != self._input_positions.shape:
            raise ValueError("positions must be vec3 coordinates for the full reserved node-copy pool")
        self.prepare_sphere()
        planes = spherical_sweep_planes(start, end, radius)
        return self._propose_convex(positions, start, end, radius, planes, grow_work=False)

    def prepare_cylinder(self, work_capacity=256):
        """Reserve a cylindrical edit using the shared convex subtraction graph."""
        self.prepare_sphere(work_capacity=work_capacity)

    def propose_cylinder(self, positions, start, end, radius):
        """Subtract an inscribed 16-sided cylinder with flat endpoints [m].

        A long through-hole may touch more tetrahedra than a local sphere edit.
        If necessary, grow only clipping scratch and retry before publication.
        Further edits within that reserve reuse the same captured graph.
        """
        from .cylinder import cylindrical_cut_planes  # noqa: PLC0415

        if positions.dtype != wp.vec3 or positions.shape != self._input_positions.shape:
            raise ValueError("positions must be vec3 coordinates for the full reserved node-copy pool")
        planes = cylindrical_cut_planes(start, end, radius)
        self.prepare_cylinder()
        return self._propose_convex(positions, start, end, radius, planes, grow_work=True)

    def _propose_convex(self, positions, start, end, radius, planes, *, grow_work):
        while True:
            self._sphere.assign_planes(start, end, radius, planes)
            self._launch_convex(positions)
            error = int(self.buffers[0].error.numpy()[0])
            if error in (1, 2) and (self._grow_convex_scratch() or self._grow_geometry(error)):
                continue
            if grow_work and error == 7:
                required = int(self._sphere.count.numpy()[0])
                maximum = int(self.geometry.count.numpy()[0])
                if len(self._sphere.work) < required <= maximum:
                    capacity = len(self._sphere.work)
                    while capacity < required:
                        capacity *= 2
                    self.discard()
                    self._prepare_subtractor(min(capacity, maximum))
                    continue
            break
        return self._convex_diagnostics()

    def _launch_convex(self, positions):
        wp.copy(self._input_positions, positions)
        if self.device.is_cuda:
            wp.capture_launch(self._sphere_graph)
        else:
            self._launch_sphere()

    def _convex_diagnostics(self):
        self._pending_interface_stride = 50 * self.capacity
        return self._diagnostics(self.buffers[0])

    def _capture(self, patch_count):
        self.launch(self._input_positions, patch_count)
        wp.synchronize_device(self.device)
        with wp.ScopedCapture(device=self.device) as capture:
            pending = self.launch(self._input_positions, patch_count)
        self._graphs[patch_count] = capture.graph, pending

    def prepare(self, patch_counts=(1, 2)):
        """Compile and capture empty edits during setup, avoiding first-cut JIT."""
        self.polygon_sizes.zero_()
        for count in patch_counts:
            if not 1 <= count <= self.max_patches:
                raise ValueError("knife patch count exceeds reserved capacity")
            if self.device.is_cuda:
                if count not in self._graphs:
                    self._capture(count)
            else:
                self.launch(self._input_positions, count)
        self.discard()

    @property
    def pending_geometry(self):
        """Validated unpublished fragment geometry after ``propose``."""
        if self._pending is None:
            raise RuntimeError("no validated cut is pending")
        return self._pending

    def commit(self):
        """Publish after the caller has transferred state and rebuilt mechanics."""
        pending = self.pending_geometry
        self._run(_publish_nodes, (self.capacity, 4), [pending.arrays, self.candidate.nodes])
        self.geometry.copy_from(pending)
        self._cut_adjacency.commit()
        wp.copy(self._published_components, self.candidate.material_components)
        self._interface_value += self._pending_interface_stride
        self.interface_start.assign([self._interface_value])
        self._pending = None
        self._pending_interface_stride = 0

    def discard(self):
        """Restore surface and integrals after a rejected mechanics update."""
        self._pending = None
        self._pending_interface_stride = 0
        self._integrate(self.geometry)
        self._surface(self.geometry, self.geometry.nodeindices, self._published_components)

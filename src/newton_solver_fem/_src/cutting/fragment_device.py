# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded device storage and finite-patch clipping for the manual knife.

All geometric coordinates are barycentric in the original parent tetrahedron.
Published and candidate buffers are independent. Overflow rejects an edit; no
fragment or face may be silently truncated to fit a reserved allocation.
"""

import numpy as np
import warp as wp

from .knife import (
    FragmentFace,
    TetrahedralFragment,
)


@wp.struct
class FragmentArrays:
    parent: wp.array[int]
    source: wp.array[int]
    created: wp.array[int]
    vertex_count: wp.array[int]
    face_count: wp.array[int]
    vertices: wp.array2d[wp.vec4d]
    faces: wp.array3d[int]
    face_size: wp.array2d[int]
    face_kind: wp.array2d[int]
    face_axis: wp.array2d[int]
    face_interface: wp.array2d[wp.int64]
    nodeindices: wp.array2d[int]
    count: wp.array[int]
    error: wp.array[int]
    capacity: int
    vertex_capacity: int
    face_capacity: int


class FragmentDevice:
    """Persistent fragment buffers shared by geometry, connectivity and rendering."""

    def __init__(self, capacity, *, device, vertex_capacity=32, face_capacity=32):
        if min(capacity, vertex_capacity, face_capacity) < 4:
            raise ValueError("fragment, vertex and face capacities must be at least four")
        self.device = wp.get_device(device)
        self.capacity, self.vertex_capacity, self.face_capacity = capacity, vertex_capacity, face_capacity
        arrays = FragmentArrays()
        for name in ("parent", "source", "created", "vertex_count", "face_count"):
            setattr(arrays, name, wp.zeros(capacity, dtype=int, device=device))
        arrays.vertices = wp.zeros((capacity, vertex_capacity), dtype=wp.vec4d, device=device)
        arrays.faces = wp.zeros((capacity, face_capacity, vertex_capacity), dtype=int, device=device)
        for name in ("face_size", "face_kind", "face_axis"):
            setattr(arrays, name, wp.zeros((capacity, face_capacity), dtype=int, device=device))
        arrays.face_interface = wp.zeros((capacity, face_capacity), dtype=wp.int64, device=device)
        arrays.nodeindices = wp.zeros((capacity, 4), dtype=int, device=device)
        arrays.count, arrays.error = wp.zeros(1, dtype=int, device=device), wp.zeros(1, dtype=int, device=device)
        arrays.capacity, arrays.vertex_capacity, arrays.face_capacity = capacity, vertex_capacity, face_capacity
        self.arrays = arrays
        for name in (
            "parent",
            "source",
            "created",
            "vertex_count",
            "face_count",
            "vertices",
            "faces",
            "face_size",
            "face_kind",
            "face_axis",
            "face_interface",
            "nodeindices",
            "count",
            "error",
        ):
            setattr(self, name, getattr(arrays, name))

    def upload(self, fragments, nodes, positions, elements):
        """Initialize or restore geometry outside a graph; never called per cut."""
        if len(fragments) > self.capacity:
            raise ValueError("fragment capacity exceeded")
        values = {
            name: getattr(self, name).numpy()
            for name in (
                "parent",
                "source",
                "created",
                "vertex_count",
                "face_count",
                "vertices",
                "faces",
                "face_size",
                "face_kind",
                "face_axis",
                "face_interface",
                "nodeindices",
            )
        }
        interfaces = sorted({face.interface for f in fragments for face in f.faces if face.kind == "original"})
        self.original_interfaces = {-(i + 1): face for i, face in enumerate(interfaces)}
        original_ids = {face: index for index, face in self.original_interfaces.items()}
        projections = {}
        for index, fragment in enumerate(fragments):
            if len(fragment.vertices) > self.vertex_capacity or len(fragment.faces) > self.face_capacity:
                raise ValueError("fragment geometry exceeds reserved vertex or face capacity")
            values["parent"][index] = fragment.parent
            values["source"][index] = index
            values["created"][index] = 1
            values["nodeindices"][index] = nodes[index]
            values["vertex_count"][index], values["face_count"][index] = len(fragment.vertices), len(fragment.faces)
            values["vertices"][index, : len(fragment.vertices)] = fragment.vertices
            for face_id, face in enumerate(fragment.faces):
                if len(face.indices) > self.vertex_capacity:
                    raise ValueError("fragment polygon exceeds reserved vertex capacity")
                values["faces"][index, face_id, : len(face.indices)] = face.indices
                values["face_size"][index, face_id] = len(face.indices)
                values["face_kind"][index, face_id] = {"original": 0, "auxiliary": 1, "cut": 2}[face.kind]
                values["face_interface"][index, face_id] = (
                    original_ids[face.interface] if face.kind == "original" else face.interface[0]
                )
                points = (
                    positions[list(face.interface[1:])]
                    if face.kind == "original"
                    else fragment.vertices[list(face.indices)] @ positions[elements[fragment.parent]]
                )
                normal = np.cross(points - points[0], np.roll(points, -1, axis=0) - points[0]).sum(axis=0)
                # Tiny clipped pieces can round an equal pair of normal
                # components differently. Every shared interface must use
                # the same projection; its largest polygon is best resolved.
                key = (face.kind, face.interface)
                area = float(normal @ normal)
                if key not in projections or area > projections[key][0]:
                    projections[key] = (area, int(np.argmax(np.abs(normal))))
        for index, fragment in enumerate(fragments):
            for face_id, face in enumerate(fragment.faces):
                values["face_axis"][index, face_id] = projections[face.kind, face.interface][1]
        for name, value in values.items():
            getattr(self, name).assign(value)
        self.count.assign([len(fragments)])
        self.error.zero_()

    def download(self, original_interfaces=None):
        """Diagnostic CPU snapshot; the captured cutting path does not use this."""
        count = int(self.count.numpy()[0])
        if int(self.error.numpy()[0]):
            raise ValueError("GPU fragment edit exceeded its geometric capacity or failed validation")
        original_interfaces = self.original_interfaces if original_interfaces is None else original_interfaces
        values = {
            name: getattr(self, name).numpy()[:count]
            for name in (
                "parent",
                "source",
                "vertex_count",
                "face_count",
                "vertices",
                "faces",
                "face_size",
                "face_kind",
                "face_interface",
                "nodeindices",
            )
        }
        fragments = []
        for index in range(count):
            faces = []
            for face_id in range(values["face_count"][index]):
                kind = ("original", "auxiliary", "cut")[values["face_kind"][index, face_id]]
                interface = int(values["face_interface"][index, face_id])
                faces.append(
                    FragmentFace(
                        tuple(map(int, values["faces"][index, face_id, : values["face_size"][index, face_id]])),
                        original_interfaces[interface] if kind == "original" else (interface,),
                        kind,
                    )
                )
            fragments.append(
                TetrahedralFragment(
                    int(values["parent"][index]),
                    values["vertices"][index, : values["vertex_count"][index]],
                    tuple(faces),
                )
            )
        return tuple(fragments), values["source"], values["nodeindices"]

    def copy_from(self, candidate):
        """Copy an accepted candidate into stable published storage on device."""
        if (self.capacity, self.vertex_capacity, self.face_capacity) != (
            candidate.capacity,
            candidate.vertex_capacity,
            candidate.face_capacity,
        ):
            raise ValueError("device geometry capacities must match")
        wp.launch(_copy_geometry, dim=self.capacity, inputs=[candidate.arrays, self.arrays], device=self.device)


@wp.func
def _copy_fragment(source: FragmentArrays, i: int, target: FragmentArrays, j: int):
    target.parent[j] = source.parent[i]
    target.source[j] = source.source[i]
    target.created[j] = source.created[i]
    target.vertex_count[j] = source.vertex_count[i]
    target.face_count[j] = source.face_count[i]
    for corner in range(4):
        target.nodeindices[j, corner] = source.nodeindices[i, corner]
    for vertex in range(source.vertex_count[i]):
        target.vertices[j, vertex] = source.vertices[i, vertex]
    for face in range(source.face_count[i]):
        target.face_size[j, face] = source.face_size[i, face]
        target.face_kind[j, face] = source.face_kind[i, face]
        target.face_axis[j, face] = source.face_axis[i, face]
        target.face_interface[j, face] = source.face_interface[i, face]
        for vertex in range(source.face_size[i, face]):
            target.faces[j, face, vertex] = source.faces[i, face, vertex]


@wp.func
def _add_vertex(target: FragmentArrays, fragment: int, point: wp.vec4d):
    found = int(-1)
    for index in range(target.vertex_count[fragment]):
        if wp.max(wp.abs(target.vertices[fragment, index] - point)) <= wp.float64(2.0e-14):
            found = index
            break
    if found < 0:
        found = target.vertex_count[fragment]
        if found >= target.vertex_capacity:
            wp.atomic_max(target.error, 0, 1)
            return int(0)
        target.vertices[fragment, found] = point
        target.vertex_count[fragment] = found + 1
    return found


@wp.func
def _simplify_fragment(geometry: FragmentArrays, fragment: int):
    """Drop roundoff-collinear corners and vertices unused by any face."""
    face = int(0)
    while face < geometry.face_count[fragment]:
        size = geometry.face_size[fragment, face]
        index = int(0)
        while size >= 3 and index < size:
            a = geometry.vertices[fragment, geometry.faces[fragment, face, (index + size - 1) % size]]
            b = geometry.vertices[fragment, geometry.faces[fragment, face, index]]
            c = geometry.vertices[fragment, geometry.faces[fragment, face, (index + 1) % size]]
            ab, ac = b - a, c - a
            cross = wp.cross(wp.vec3d(ab[1], ab[2], ab[3]), wp.vec3d(ac[1], ac[2], ac[3]))
            if wp.length_sq(cross) <= wp.float64(4.0e-28) * wp.length_sq(ac) and wp.dot(b - a, b - c) <= wp.float64(
                4.0e-28
            ):
                for corner in range(index, size - 1):
                    geometry.faces[fragment, face, corner] = geometry.faces[fragment, face, corner + 1]
                size -= 1
                index = wp.max(index - 1, 0)
            else:
                index += 1
        geometry.face_size[fragment, face] = size
        if size < 3:
            last = geometry.face_count[fragment] - 1
            geometry.face_size[fragment, face] = geometry.face_size[fragment, last]
            geometry.face_kind[fragment, face] = geometry.face_kind[fragment, last]
            geometry.face_axis[fragment, face] = geometry.face_axis[fragment, last]
            geometry.face_interface[fragment, face] = geometry.face_interface[fragment, last]
            for corner in range(geometry.face_size[fragment, last]):
                geometry.faces[fragment, face, corner] = geometry.faces[fragment, last, corner]
            geometry.face_count[fragment] = last
        else:
            face += 1
    # Most fragments already use every vertex. Marking references once makes
    # that common case linear in polygon size instead of scanning every face
    # for every vertex. Larger custom capacities retain the general fallback.
    used_mask = wp.uint64(0)
    if geometry.vertex_count[fragment] <= 64:
        for face in range(geometry.face_count[fragment]):
            for corner in range(geometry.face_size[fragment, face]):
                used_mask = used_mask | (wp.uint64(1) << wp.uint64(geometry.faces[fragment, face, corner]))
    destination = int(0)
    for vertex in range(geometry.vertex_count[fragment]):
        used = bool(False)
        if geometry.vertex_count[fragment] <= 64:
            used = (used_mask & (wp.uint64(1) << wp.uint64(vertex))) != wp.uint64(0)
        else:
            for face in range(geometry.face_count[fragment]):
                for corner in range(geometry.face_size[fragment, face]):
                    if geometry.faces[fragment, face, corner] == vertex:
                        used = True
        if used:
            if destination != vertex:
                geometry.vertices[fragment, destination] = geometry.vertices[fragment, vertex]
                for face in range(geometry.face_count[fragment]):
                    for corner in range(geometry.face_size[fragment, face]):
                        if geometry.faces[fragment, face, corner] == vertex:
                            geometry.faces[fragment, face, corner] = destination
            destination += 1
    geometry.vertex_count[fragment] = destination


@wp.func
def _polar_precedes(a: wp.vec2d, b: wp.vec2d):
    """Compare counterclockwise directions without evaluating angles."""
    upper_a = a[1] > wp.float64(0.0) or (a[1] == wp.float64(0.0) and a[0] >= wp.float64(0.0))
    upper_b = b[1] > wp.float64(0.0) or (b[1] == wp.float64(0.0) and b[0] >= wp.float64(0.0))
    if upper_a != upper_b:
        return upper_a
    return a[0] * b[1] - a[1] * b[0] >= wp.float64(0.0)


@wp.func
def _clip_fragment(
    source: FragmentArrays,
    i: int,
    target: FragmentArrays,
    j: int,
    plane: wp.vec4d,
    sign: wp.float64,
    interface: wp.int64,
    kind: int,
    tolerance: wp.float64,
    positions: wp.array[wp.vec3d],
    elements: wp.array2d[int],
):
    minimum, maximum = wp.float64(1.0e30), wp.float64(-1.0e30)
    for vertex in range(source.vertex_count[i]):
        value = sign * wp.dot(source.vertices[i, vertex], plane)
        minimum, maximum = wp.min(minimum, value), wp.max(maximum, value)
    target.vertex_count[j] = 0
    target.face_count[j] = 0
    if minimum >= -tolerance:
        _copy_fragment(source, i, target, j)
        return int(1)
    if maximum <= tolerance:
        return int(0)
    target.parent[j] = source.parent[i]
    target.source[j] = source.source[i]
    target.created[j] = 1
    for corner in range(4):
        target.nodeindices[j, corner] = source.nodeindices[i, corner]
    for face in range(source.face_count[i]):
        out_face = target.face_count[j]
        if out_face >= target.face_capacity - 1:
            wp.atomic_max(target.error, 0, 2)
            return int(0)
        size, emitted = source.face_size[i, face], int(0)
        previous = source.vertices[i, source.faces[i, face, size - 1]]
        a = sign * wp.dot(previous, plane)
        if wp.abs(a) <= tolerance:
            a = wp.float64(0.0)
        for edge in range(size):
            current = source.vertices[i, source.faces[i, face, edge]]
            b = sign * wp.dot(current, plane)
            if wp.abs(b) <= tolerance:
                b = wp.float64(0.0)
            if (a >= wp.float64(0.0)) != (b >= wp.float64(0.0)):
                intersection = previous + (a / (a - b)) * (current - previous)
                vertex = _add_vertex(target, j, intersection)
                if emitted == 0 or target.faces[j, out_face, emitted - 1] != vertex:
                    if emitted >= target.vertex_capacity:
                        wp.atomic_max(target.error, 0, 1)
                        return int(0)
                    target.faces[j, out_face, emitted] = vertex
                    emitted += 1
            if b >= wp.float64(0.0):
                vertex = _add_vertex(target, j, current)
                if emitted == 0 or target.faces[j, out_face, emitted - 1] != vertex:
                    if emitted >= target.vertex_capacity:
                        wp.atomic_max(target.error, 0, 1)
                        return int(0)
                    target.faces[j, out_face, emitted] = vertex
                    emitted += 1
            previous, a = current, b
        if emitted > 1 and target.faces[j, out_face, 0] == target.faces[j, out_face, emitted - 1]:
            emitted -= 1
        if emitted >= 3:
            target.face_size[j, out_face] = emitted
            target.face_kind[j, out_face] = source.face_kind[i, face]
            target.face_interface[j, out_face] = source.face_interface[i, face]
            target.face_axis[j, out_face] = source.face_axis[i, face]
            target.face_count[j] = out_face + 1
    # A face clipped to an edge may have emitted vertices but no valid face.
    # Remove those before collecting the cap or repeatedly clipping this hull.
    _simplify_fragment(target, j)
    cap, cap_count, center = target.face_count[j], int(0), wp.vec3d(0.0)
    for vertex in range(target.vertex_count[j]):
        point = target.vertices[j, vertex]
        if wp.abs(wp.dot(point, plane)) <= wp.max(tolerance, wp.float64(1.0e-13)):
            target.faces[j, cap, cap_count] = vertex
            cap_count += 1
            center += wp.vec3d(point[1], point[2], point[3])
    if cap_count < 3 or target.error[0] != 0:
        target.vertex_count[j] = 0
        target.face_count[j] = 0
        return int(0)
    center /= wp.float64(cap_count)
    normal = -sign * wp.normalize(wp.vec3d(plane[1] - plane[0], plane[2] - plane[0], plane[3] - plane[0]))
    axis = wp.vec3d(1.0, 0.0, 0.0)
    if wp.abs(normal[1]) < wp.abs(normal[0]) and wp.abs(normal[1]) <= wp.abs(normal[2]):
        axis = wp.vec3d(0.0, 1.0, 0.0)
    elif wp.abs(normal[2]) < wp.abs(normal[0]) and wp.abs(normal[2]) < wp.abs(normal[1]):
        axis = wp.vec3d(0.0, 0.0, 1.0)
    first = wp.normalize(wp.cross(normal, axis))
    second = wp.cross(normal, first)
    for vertex in range(1, cap_count):
        index = target.faces[j, cap, vertex]
        point = target.vertices[j, index]
        relative = wp.vec3d(point[1], point[2], point[3]) - center
        direction = wp.vec2d(wp.dot(relative, first), wp.dot(relative, second))
        place = vertex
        while place > 0:
            before = target.vertices[j, target.faces[j, cap, place - 1]]
            relative_before = wp.vec3d(before[1], before[2], before[3]) - center
            before_direction = wp.vec2d(wp.dot(relative_before, first), wp.dot(relative_before, second))
            if _polar_precedes(before_direction, direction):
                break
            target.faces[j, cap, place] = target.faces[j, cap, place - 1]
            place -= 1
        target.faces[j, cap, place] = index
    target.face_size[j, cap] = cap_count
    target.face_kind[j, cap] = kind
    target.face_interface[j, cap] = interface
    # Derive the shared projection axis from the plane itself. Recomputing it
    # from different cap polygons can choose different axes at a near tie.
    parent = target.parent[j]
    origin = positions[elements[parent, 0]]
    side_b = positions[elements[parent, 1]] - origin
    side_c = positions[elements[parent, 2]] - origin
    side_d = positions[elements[parent, 3]] - origin
    world_normal = (
        (plane[1] - plane[0]) * wp.cross(side_c, side_d)
        + (plane[2] - plane[0]) * wp.cross(side_d, side_b)
        + (plane[3] - plane[0]) * wp.cross(side_b, side_c)
    )
    projection = int(0)
    if wp.abs(world_normal[1]) > wp.abs(world_normal[projection]):
        projection = 1
    if wp.abs(world_normal[2]) > wp.abs(world_normal[projection]):
        projection = 2
    target.face_axis[j, cap] = projection
    target.face_count[j] = cap + 1
    _simplify_fragment(target, j)
    return int(1)


@wp.func
def _fragment_center(geometry: FragmentArrays, fragment: int):
    center = wp.vec4d(0.0)
    count = geometry.vertex_count[fragment]
    for vertex in range(count):
        center += geometry.vertices[fragment, vertex]
    if count > 0:
        center /= wp.float64(count)
    return center


@wp.func
def _volume_device(geometry: FragmentArrays, fragment: int):
    volume = wp.float64(0.0)
    # A convex fragment's interior origin gives nonnegative pyramids. Using
    # the parent origin instead cancels large terms for small cut remnants.
    origin = _fragment_center(geometry, fragment)
    for face in range(geometry.face_count[fragment]):
        a = geometry.vertices[fragment, geometry.faces[fragment, face, 0]] - origin
        for corner in range(1, geometry.face_size[fragment, face] - 1):
            b = geometry.vertices[fragment, geometry.faces[fragment, face, corner]] - origin
            c = geometry.vertices[fragment, geometry.faces[fragment, face, corner + 1]] - origin
            volume += wp.dot(
                wp.vec3d(a[1], a[2], a[3]), wp.cross(wp.vec3d(b[1], b[2], b[3]), wp.vec3d(c[1], c[2], c[3]))
            ) / wp.float64(6.0)
    return volume


@wp.kernel(enable_backward=False)
def _reset_stage(
    candidate: FragmentArrays, output: FragmentArrays, scratch_a: FragmentArrays, scratch_b: FragmentArrays
):
    output.count[0] = 0
    output.error[0] = 0
    candidate.error[0] = 0
    scratch_a.error[0] = 0
    scratch_b.error[0] = 0


@wp.kernel(enable_backward=False)
def _reset_lineage(geometry: FragmentArrays):
    index = wp.tid()
    if index < geometry.count[0]:
        geometry.source[index] = index
        geometry.created[index] = 0


@wp.kernel(enable_backward=False)
def _copy_geometry(source: FragmentArrays, target: FragmentArrays):
    index = wp.tid()
    if index == 0:
        target.count[0] = source.count[0]
        target.error[0] = source.error[0]
    if index < source.count[0] and source.error[0] == 0:
        _copy_fragment(source, index, target, index)


@wp.kernel(enable_backward=False)
def _validate_patch(
    polygons: wp.array2d[wp.vec3d],
    sizes: wp.array[int],
    patch: int,
    tolerance: wp.float64,
    normals: wp.array[wp.vec3d],
    origins: wp.array[wp.vec3d],
    error: wp.array[int],
):
    size = sizes[patch]
    if size == 0:
        return
    if size < 3 or size > 4:
        error[0] = 7
        return
    coordinate_scale = wp.float64(0.0)
    origin = polygons[patch, 0]
    for index in range(size):
        point = polygons[patch, index]
        for axis in range(3):
            if not wp.isfinite(point[axis]):
                error[0] = 7
        coordinate_scale = wp.max(coordinate_scale, wp.max(wp.abs(point)))
        if wp.length(point) < wp.length(origin):
            origin = point
    normal, length = wp.vec3d(0.0), wp.float64(0.0)
    # The first triangle can contain the screen projection's tiny near edge.
    # A wide triangle determines the same plane without magnifying its noise.
    for first_index in range(size - 2):
        for second_index in range(first_index + 1, size - 1):
            for third_index in range(second_index + 1, size):
                candidate = wp.cross(
                    polygons[patch, second_index] - polygons[patch, first_index],
                    polygons[patch, third_index] - polygons[patch, second_index],
                )
                area = wp.length(candidate)
                if area > length:
                    normal, length = candidate, area
    if length <= tolerance * tolerance:
        error[0] = 7
        return
    normal /= length
    roundoff = wp.float64(1.4210854715202004e-14)
    distance_tolerance = wp.max(tolerance, roundoff * coordinate_scale)
    for index in range(size):
        a = polygons[patch, index]
        b = polygons[patch, (index + 1) % size]
        c = polygons[patch, (index + 2) % size]
        first_edge, second_edge = b - a, c - b
        first_length, second_length = wp.length(first_edge), wp.length(second_edge)
        if wp.abs(wp.dot(a - origin, normal)) > distance_tolerance:
            error[0] = 7
        if first_length <= distance_tolerance or wp.dot(wp.cross(first_edge, second_edge), normal) <= wp.max(
            tolerance * tolerance, roundoff * first_length * second_length
        ):
            error[0] = 7
    normals[patch] = normal
    origins[patch] = origin


@wp.kernel(enable_backward=False)
def _clip_patch(
    source: FragmentArrays,
    candidates: FragmentArrays,
    scratch_a: FragmentArrays,
    scratch_b: FragmentArrays,
    q: wp.array[wp.float64],
    polygons: wp.array2d[wp.vec3d],
    polygon_sizes: wp.array[int],
    normals: wp.array[wp.vec3d],
    origins: wp.array[wp.vec3d],
    patch: int,
    interface_start: wp.array[wp.int64],
    positions: wp.array[wp.vec3d],
    elements: wp.array2d[int],
    tolerance: wp.float64,
    changed: wp.array[int],
):
    fragment = wp.tid()
    for child in range(6):
        candidates.face_count[6 * fragment + child] = 0
        candidates.vertex_count[6 * fragment + child] = 0
    if fragment >= source.count[0] or source.error[0] != 0 or candidates.error[0] != 0:
        return
    if polygon_sizes[patch] == 0:
        _copy_fragment(source, fragment, candidates, 6 * fragment)
        return
    # These basis nodes can lie outside the remaining material after a cut.
    # Their affine orientation does not test the fragment's physical Jacobian.
    for corner in range(4):
        dof = 12 * source.nodeindices[fragment, corner]
        for coordinate in range(12):
            if not wp.isfinite(q[dof + coordinate]):
                wp.atomic_max(candidates.error, 0, 6)
                return
    normal = normals[patch]
    plane = wp.vec4d(0.0)
    for corner in range(4):
        dof = 12 * source.nodeindices[fragment, corner]
        point = wp.vec3d(q[dof], q[dof + 1], q[dof + 2])
        plane[corner] = wp.dot(point - origins[patch], normal)
    minimum, maximum = wp.float64(1.0e30), wp.float64(-1.0e30)
    for vertex in range(source.vertex_count[fragment]):
        value = wp.dot(source.vertices[fragment, vertex], plane)
        minimum, maximum = wp.min(minimum, value), wp.max(maximum, value)
    if minimum >= -tolerance or maximum <= tolerance:
        _copy_fragment(source, fragment, candidates, 6 * fragment)
        return
    _copy_fragment(source, fragment, scratch_a, fragment)
    inside, failed = int(1), int(0)
    for edge in range(polygon_sizes[patch]):
        first = polygons[patch, edge]
        next_point = polygons[patch, (edge + 1) % polygon_sizes[patch]]
        inward = wp.normalize(wp.cross(normal, next_point - first))
        # Either endpoint defines the same side plane. A near endpoint avoids
        # subtracting kilometre-scale camera depths from metre-scale objects.
        origin = first
        if wp.length(next_point) < wp.length(first):
            origin = next_point
        boundary = wp.vec4d(0.0)
        for corner in range(4):
            dof = 12 * source.nodeindices[fragment, corner]
            point = wp.vec3d(q[dof], q[dof + 1], q[dof + 2])
            boundary[corner] = wp.dot(point - origin, inward)
        tag = interface_start[0] + wp.int64(5 * (patch * source.capacity + fragment) + edge)
        if edge % 2 == 0:
            _clip_fragment(
                scratch_a,
                fragment,
                candidates,
                6 * fragment + edge,
                boundary,
                wp.float64(-1.0),
                tag,
                1,
                tolerance,
                positions,
                elements,
            )
            inside = _clip_fragment(
                scratch_a,
                fragment,
                scratch_b,
                fragment,
                boundary,
                wp.float64(1.0),
                tag,
                1,
                tolerance,
                positions,
                elements,
            )
        else:
            _clip_fragment(
                scratch_b,
                fragment,
                candidates,
                6 * fragment + edge,
                boundary,
                wp.float64(-1.0),
                tag,
                1,
                tolerance,
                positions,
                elements,
            )
            inside = _clip_fragment(
                scratch_b,
                fragment,
                scratch_a,
                fragment,
                boundary,
                wp.float64(1.0),
                tag,
                1,
                tolerance,
                positions,
                elements,
            )
        if inside == 0:
            failed = 1
            break
    if failed == 0:
        tag = interface_start[0] + wp.int64(5 * (patch * source.capacity + fragment) + 4)
        if polygon_sizes[patch] % 2 == 0:
            negative = _clip_fragment(
                scratch_a,
                fragment,
                candidates,
                6 * fragment + 4,
                plane,
                wp.float64(-1.0),
                tag,
                2,
                tolerance,
                positions,
                elements,
            )
            positive = _clip_fragment(
                scratch_a,
                fragment,
                candidates,
                6 * fragment + 5,
                plane,
                wp.float64(1.0),
                tag,
                2,
                tolerance,
                positions,
                elements,
            )
        else:
            negative = _clip_fragment(
                scratch_b,
                fragment,
                candidates,
                6 * fragment + 4,
                plane,
                wp.float64(-1.0),
                tag,
                2,
                tolerance,
                positions,
                elements,
            )
            positive = _clip_fragment(
                scratch_b,
                fragment,
                candidates,
                6 * fragment + 5,
                plane,
                wp.float64(1.0),
                tag,
                2,
                tolerance,
                positions,
                elements,
            )
        if negative == 0 or positive == 0:
            failed = 1
    if failed != 0:
        for child in range(6):
            candidates.face_count[6 * fragment + child] = 0
            candidates.vertex_count[6 * fragment + child] = 0
        _copy_fragment(source, fragment, candidates, 6 * fragment)
    else:
        volume = wp.float64(0.0)
        for child in range(6):
            volume += _volume_device(candidates, 6 * fragment + child)
        if wp.abs(volume - _volume_device(source, fragment)) > wp.float64(1.0e-11):
            wp.atomic_max(candidates.error, 0, 3)
        wp.atomic_max(changed, source.parent[fragment], 1)


@wp.kernel(enable_backward=False)
def _compact_fragments(
    source: FragmentArrays,
    candidates: FragmentArrays,
    target: FragmentArrays,
    scratch_a: FragmentArrays,
    scratch_b: FragmentArrays,
):
    index = wp.tid()
    error = wp.max(source.error[0], wp.max(candidates.error[0], wp.max(scratch_a.error[0], scratch_b.error[0])))
    if error != 0:
        wp.atomic_max(target.error, 0, error)
        return
    if candidates.face_count[index] == 0:
        return
    destination = wp.atomic_add(target.count, 0, 1)
    if destination >= target.capacity:
        wp.atomic_max(target.error, 0, 4)
        return
    _copy_fragment(candidates, index, target, destination)


_vec10d = wp.types.vector(length=10, dtype=wp.float64)

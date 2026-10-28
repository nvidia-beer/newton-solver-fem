# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Captured convex subtraction for a spherical material-removal tool.

The cutter uses an inscribed icosahedron, so removal stays inside the displayed
sphere. Only fragments overlapping the sphere's bounds enter the clipping
workspace. All retained material, moments, and connectivity remain represented
by the same affine parent tetrahedra as the finite knife.
"""

from functools import lru_cache
from itertools import combinations

import numpy as np
import warp as wp

from .fragment_device import (
    FragmentArrays,
    FragmentDevice,
    _add_vertex,
    _clip_fragment,
    _copy_fragment,
    _simplify_fragment,
    _volume_device,
)

# Volumes are measured in the unit parent simplex (volume 1/6). Regions
# below its geometric roundoff scale cannot support independent material DOFs.
_MIN_FRAGMENT_VOLUME = wp.constant(float(64.0 * np.finfo(np.float64).eps))


@lru_cache(maxsize=1)
def _icosahedron():
    """Return the vertices and outward planes of a unit inscribed icosahedron."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [(0, a, b * golden) for a in (-1, 1) for b in (-1, 1)]
        + [(a, b * golden, 0) for a in (-1, 1) for b in (-1, 1)]
        + [(a * golden, 0, b) for a in (-1, 1) for b in (-1, 1)],
        dtype=np.float64,
    )
    vertices /= np.linalg.norm(vertices[0])
    planes = []
    for i, j, k in combinations(range(len(vertices)), 3):
        normal = np.cross(vertices[j] - vertices[i], vertices[k] - vertices[i])
        length = np.linalg.norm(normal)
        if length < 1.0e-12:
            continue
        normal /= length
        offset = float(normal @ vertices[i])
        if offset < 0:
            normal, offset = -normal, -offset
        if np.max(vertices @ normal - offset) <= 1.0e-12:
            planes.append((*normal, offset))
    if len(planes) != 20:
        raise RuntimeError("Failed to construct the spherical cutter's convex surface")
    return vertices, np.asarray(planes, dtype=np.float64)


@lru_cache(maxsize=1)
def _icosahedron_edges():
    vertices, _ = _icosahedron()
    pairs = np.asarray(tuple(combinations(range(len(vertices)), 2)))
    vectors = vertices[pairs[:, 1]] - vertices[pairs[:, 0]]
    lengths = np.linalg.norm(vectors, axis=1)
    edge_mask = np.abs(lengths - lengths.min()) < 1.0e-10
    return vertices[pairs[edge_mask, 0]], vectors[edge_mask]


def spherical_sweep_planes(start, end, radius):
    """Return convex-hull planes for the swept inscribed spherical tool.

    The hull is the Minkowski sum of the icosahedron and the motion segment.
    Its original faces and silhouette-edge faces bound one continuous sweep,
    including motion longer than the tool diameter. Offsets are relative to
    ``start`` and normalized by ``radius``.
    """
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("Sphere endpoints must be finite and radius must be positive")
    vertices, base_planes = _icosahedron()
    with np.errstate(over="ignore", invalid="ignore"):
        delta = (end - start) / radius
    if not np.isfinite(delta).all():
        raise ValueError("Sphere sweep is too large relative to its radius")
    normals = base_planes[:, :3]
    travel = float(np.hypot.reduce(delta))
    if not np.isfinite(travel):
        raise ValueError("Sphere sweep is too large relative to its radius")
    if travel > 1.0e-12:
        origins, edges = _icosahedron_edges()
        side_normals = np.cross(edges, delta)
        lengths = np.linalg.norm(side_normals, axis=1)
        usable = lengths > 1.0e-12 * travel
        side_normals = side_normals[usable] / lengths[usable, None]
        projections = (vertices @ side_normals.T).T - np.sum(origins[usable] * side_normals, axis=1)[:, None]
        lower, upper = projections.min(axis=1), projections.max(axis=1)
        side_normals[lower >= -1.0e-12] *= -1
        side_normals = side_normals[(lower >= -1.0e-12) | (upper <= 1.0e-12)]
        normals = np.concatenate((normals, side_normals), axis=0)
        # A repeated plane would partition a coplanar cap into two identical
        # polygons because both closed halfspaces contain that cap.
        distances = np.sum((normals[:, None, :] - normals[None, :, :]) ** 2, axis=2)
        normals = normals[~np.any(np.tril(distances < 1.0e-20, k=-1), axis=1)]
    supports = (vertices @ normals.T).max(axis=0) + np.maximum(normals @ delta, 0)
    return np.column_stack((normals, supports))


@wp.kernel(enable_backward=False)
def _reset(source: FragmentArrays, target: FragmentArrays, count: wp.array[int], scratch_error: wp.array[int]):
    target.count[0] = 0
    target.error[0] = source.error[0]
    count[0] = 0
    scratch_error[0] = 0


@wp.func
def _append(source: FragmentArrays, index: int, target: FragmentArrays):
    # Local clipping may temporarily need more storage than the compact
    # published geometry. Reject a genuinely oversized result before copying.
    if source.vertex_count[index] > target.vertex_capacity:
        wp.atomic_max(target.error, 0, 1)
        return
    if source.face_count[index] > target.face_capacity:
        wp.atomic_max(target.error, 0, 2)
        return
    destination = wp.atomic_add(target.count, 0, 1)
    if destination >= target.capacity:
        wp.atomic_max(target.error, 0, 4)
    else:
        _copy_fragment(source, index, target, destination)


@wp.kernel(enable_backward=False)
def _select(
    source: FragmentArrays,
    target: FragmentArrays,
    positions: wp.array[wp.vec3],
    center: wp.array[wp.vec3d],
    end: wp.array[wp.vec3d],
    radius: wp.array[wp.float64],
    planes: wp.array[wp.vec4d],
    plane_count: wp.array[int],
    tolerance: wp.float64,
    work: wp.array[int],
    count: wp.array[int],
):
    fragment = wp.tid()
    if fragment >= source.count[0] or source.error[0] != 0:
        return
    lower, upper = wp.vec3d(1.0e30), wp.vec3d(-1.0e30)
    for vertex in range(source.vertex_count[fragment]):
        point = wp.vec3d(0.0)
        weights = source.vertices[fragment, vertex]
        for corner in range(4):
            point += weights[corner] * wp.vec3d(positions[source.nodeindices[fragment, corner]])
        if not wp.isfinite(point[0]) or not wp.isfinite(point[1]) or not wp.isfinite(point[2]):
            wp.atomic_max(target.error, 0, 6)
            return
        lower, upper = wp.min(lower, point), wp.max(upper, point)
    tool_lower, tool_upper = (
        wp.min(center[0], end[0]) - wp.vec3d(radius[0]),
        wp.max(center[0], end[0]) + wp.vec3d(radius[0]),
    )
    separated = bool(False)
    for axis in range(3):
        if upper[axis] <= tool_lower[axis] or lower[axis] >= tool_upper[axis]:
            separated = True
    if not separated:
        for plane_index in range(plane_count[0]):
            direction = planes[plane_index]
            normal = wp.vec3d(direction[0], direction[1], direction[2])
            plane = wp.vec4d(0.0)
            position_scale = wp.float64(0.0)
            for corner in range(4):
                point = wp.vec3d(positions[source.nodeindices[fragment, corner]])
                plane[corner] = wp.dot(point - center[0], normal) - radius[0] * direction[3]
                position_scale = wp.max(position_scale, wp.dot(wp.abs(point), wp.abs(normal)))
            minimum = wp.float64(1.0e30)
            for vertex in range(source.vertex_count[fragment]):
                minimum = wp.min(minimum, wp.dot(source.vertices[fragment, vertex], plane))
            # Current mechanical positions are float32. Recutting a sidewall
            # separated only by their rounding error creates near-coincident
            # cap planes and unresolvable slivers. Reject that grazing case
            # before topology work; actual clipping retains double precision.
            coordinate_error = wp.float64(2.384185791015625e-7) * position_scale
            if minimum >= -wp.max(tolerance, coordinate_error):
                separated = True
                break
    if separated:
        _append(source, fragment, target)
    else:
        slot = wp.atomic_add(count, 0, 1)
        if slot >= work.shape[0]:
            wp.atomic_max(target.error, 0, 7)
        else:
            work[slot] = fragment


@wp.func
def _plane(
    geometry: FragmentArrays,
    fragment: int,
    positions: wp.array[wp.vec3],
    center: wp.vec3d,
    radius: wp.float64,
    direction: wp.vec4d,
):
    normal = wp.vec3d(direction[0], direction[1], direction[2])
    plane = wp.vec4d(0.0)
    for corner in range(4):
        point = wp.vec3d(positions[geometry.nodeindices[fragment, corner]])
        plane[corner] = wp.dot(point - center, normal) - radius * direction[3]
    return plane


@wp.kernel(enable_backward=False)
def _order_planes(
    source: FragmentArrays,
    positions: wp.array[wp.vec3],
    center: wp.array[wp.vec3d],
    radius: wp.array[wp.float64],
    planes: wp.array[wp.vec4d],
    plane_count: wp.array[int],
    work: wp.array[int],
    count: wp.array[int],
    ordered: wp.array2d[wp.vec4d],
):
    slot = wp.tid()
    if slot >= count[0]:
        return
    fragment = work[slot]
    weights = wp.vec4d(0.0)
    for vertex in range(source.vertex_count[fragment]):
        weights += source.vertices[fragment, vertex]
    weights /= wp.float64(source.vertex_count[fragment])
    midpoint = wp.vec3d(0.0)
    for corner in range(4):
        midpoint += weights[corner] * wp.vec3d(positions[source.nodeindices[fragment, corner]])
    relative = (midpoint - center[0]) / radius[0]
    # Retain the bulk of each fragment first, before partitioning the small
    # material region around the tool. This avoids artificial distant slabs.
    for index in range(plane_count[0]):
        value = planes[index]
        score = wp.dot(relative, wp.vec3d(value[0], value[1], value[2])) - value[3]
        insertion = index
        while insertion > 0:
            previous = ordered[slot, insertion - 1]
            previous_score = wp.dot(relative, wp.vec3d(previous[0], previous[1], previous[2])) - previous[3]
            if previous_score >= score:
                break
            ordered[slot, insertion] = previous
            insertion -= 1
        ordered[slot, insertion] = value


@wp.func
def _polygon_clip(
    source: wp.array2d[wp.vec4d],
    target: wp.array2d[wp.vec4d],
    slot: int,
    count: int,
    plane: wp.vec4d,
    sign: wp.float64,
    tolerance: wp.float64,
    error: wp.array[int],
):
    emitted = int(0)
    if count < 3:
        return emitted
    previous = source[slot, count - 1]
    a = sign * wp.dot(previous, plane)
    if wp.abs(a) <= tolerance:
        a = wp.float64(0.0)
    for index in range(count):
        current = source[slot, index]
        b = sign * wp.dot(current, plane)
        if wp.abs(b) <= tolerance:
            b = wp.float64(0.0)
        if (a >= wp.float64(0.0)) != (b >= wp.float64(0.0)):
            point = previous + a / (a - b) * (current - previous)
            if emitted == 0 or wp.max(wp.abs(target[slot, emitted - 1] - point)) > wp.float64(2.0e-14):
                if emitted >= target.shape[1]:
                    wp.atomic_max(error, 0, 1)
                    return int(0)
                target[slot, emitted] = point
                emitted += 1
        if b >= wp.float64(0.0):
            if emitted == 0 or wp.max(wp.abs(target[slot, emitted - 1] - current)) > wp.float64(2.0e-14):
                if emitted >= target.shape[1]:
                    wp.atomic_max(error, 0, 1)
                    return int(0)
                target[slot, emitted] = current
                emitted += 1
        previous, a = current, b
    if emitted > 1 and wp.max(wp.abs(target[slot, 0] - target[slot, emitted - 1])) <= wp.float64(2.0e-14):
        emitted -= 1
    return emitted


@wp.func
def _polygon_face(
    geometry: FragmentArrays,
    fragment: int,
    polygon: wp.array2d[wp.vec4d],
    slot: int,
    count: int,
    kind: int,
    axis: int,
    interface: wp.int64,
):
    if count < 3 or geometry.error[0] != 0:
        return
    origin = polygon[slot, 0]
    normal = wp.vec3d(0.0)
    for index in range(1, count - 1):
        a, b = polygon[slot, index] - origin, polygon[slot, index + 1] - origin
        normal += wp.cross(wp.vec3d(a[1], a[2], a[3]), wp.vec3d(b[1], b[2], b[3]))
    if wp.length_sq(normal) <= wp.float64(1.0e-52):
        return
    face = geometry.face_count[fragment]
    if face >= geometry.face_capacity:
        wp.atomic_max(geometry.error, 0, 2)
        return
    for index in range(count):
        vertex = _add_vertex(geometry, fragment, polygon[slot, index])
        if geometry.error[0] != 0:
            return
        geometry.faces[fragment, face, index] = vertex
    geometry.face_size[fragment, face] = count
    geometry.face_kind[fragment, face] = kind
    geometry.face_axis[fragment, face] = axis
    geometry.face_interface[fragment, face] = interface
    geometry.face_count[fragment] = face + 1


@wp.func
def _partition_cap(
    geometry: FragmentArrays,
    fragment: int,
    positions: wp.array[wp.vec3],
    center: wp.vec3d,
    radius: wp.float64,
    planes: wp.array2d[wp.vec4d],
    plane_count: int,
    first_plane: int,
    order_slot: int,
    polygon_a: wp.array2d[wp.vec4d],
    polygon_b: wp.array2d[wp.vec4d],
    polygon_out: wp.array2d[wp.vec4d],
    tolerance: wp.float64,
):
    # Only the part inside every remaining cutter plane is a cavity wall.
    # The complementary coplanar polygons keep neighboring pieces connected.
    face = geometry.face_count[fragment] - 1
    count = geometry.face_size[fragment, face]
    axis, interface = geometry.face_axis[fragment, face], geometry.face_interface[fragment, face]
    for index in range(count):
        polygon_a[fragment, index] = geometry.vertices[fragment, geometry.faces[fragment, face, index]]
    geometry.face_count[fragment] = face
    _simplify_fragment(geometry, fragment)
    for plane_index in range(first_plane + 1, plane_count):
        plane = _plane(geometry, fragment, positions, center, radius, planes[order_slot, plane_index])
        outside = _polygon_clip(
            polygon_a, polygon_out, fragment, count, plane, wp.float64(1.0), tolerance, geometry.error
        )
        _polygon_face(geometry, fragment, polygon_out, fragment, outside, 1, axis, interface)
        count = _polygon_clip(polygon_a, polygon_b, fragment, count, plane, wp.float64(-1.0), tolerance, geometry.error)
        for index in range(count):
            polygon_a[fragment, index] = polygon_b[fragment, index]
        if count < 3 or geometry.error[0] != 0:
            break
    _polygon_face(geometry, fragment, polygon_a, fragment, count, 2, axis, interface)
    _simplify_fragment(geometry, fragment)


@wp.kernel(enable_backward=False)
def _intersect(
    source: FragmentArrays,
    target: FragmentArrays,
    scratch_a: FragmentArrays,
    scratch_b: FragmentArrays,
    positions: wp.array[wp.vec3],
    rest: wp.array[wp.vec3d],
    elements: wp.array2d[int],
    center: wp.array[wp.vec3d],
    radius: wp.array[wp.float64],
    planes: wp.array2d[wp.vec4d],
    plane_count: wp.array[int],
    work: wp.array[int],
    count: wp.array[int],
    removed: wp.array[wp.float64],
    tolerance: wp.float64,
):
    slot = wp.tid()
    removed[slot] = wp.float64(0.0)
    if slot >= count[0] or target.error[0] != 0:
        return
    fragment = work[slot]
    _copy_fragment(source, fragment, scratch_a, slot)
    intersects = int(1)
    # AABB overlap alone cannot establish positive-volume removal. Complete
    # the intersection before emitting any changed material or cut faces.
    for plane_index in range(plane_count[0]):
        plane = _plane(source, fragment, positions, center[0], radius[0], planes[slot, plane_index])
        intersects = _clip_fragment(
            scratch_a,
            slot,
            scratch_b,
            slot,
            plane,
            wp.float64(-1.0),
            wp.int64(0),
            1,
            tolerance,
            rest,
            elements,
        )
        if intersects == 0:
            break
        _copy_fragment(scratch_b, slot, scratch_a, slot)
    intersection_error = wp.max(scratch_a.error[0], scratch_b.error[0])
    if intersection_error != 0:
        wp.atomic_max(target.error, 0, intersection_error)
        return
    if intersects == 0 or _volume_device(scratch_a, slot) <= wp.float64(1.0e-18):
        return
    removed[slot] = _volume_device(scratch_a, slot)


@wp.kernel(enable_backward=False)
def _emit_regions(
    source: FragmentArrays,
    target: FragmentArrays,
    scratch_a: FragmentArrays,
    scratch_b: FragmentArrays,
    outside: FragmentArrays,
    positions: wp.array[wp.vec3],
    rest: wp.array[wp.vec3d],
    elements: wp.array2d[int],
    center: wp.array[wp.vec3d],
    radius: wp.array[wp.float64],
    planes: wp.array2d[wp.vec4d],
    plane_count: wp.array[int],
    work: wp.array[int],
    count: wp.array[int],
    kept: wp.array[wp.float64],
    discarded: wp.array[wp.float64],
    interface_start: wp.array[wp.int64],
    polygon_a: wp.array2d[wp.vec4d],
    polygon_b: wp.array2d[wp.vec4d],
    polygon_out: wp.array2d[wp.vec4d],
    tolerance: wp.float64,
):
    slot = wp.tid()
    outside.face_count[slot] = 0
    outside.vertex_count[slot] = 0
    work_slot, plane_index = slot // planes.shape[1], slot % planes.shape[1]
    if work_slot >= count[0] or plane_index >= plane_count[0] or target.error[0] != 0:
        return
    fragment = work[work_slot]
    _copy_fragment(source, fragment, scratch_a, slot)
    # Each cutter face owns one disjoint outside region. Independent prefix
    # clipping exposes parallel work even when the tool touches few elements.
    for previous in range(plane_index):
        plane = _plane(source, fragment, positions, center[0], radius[0], planes[work_slot, previous])
        tag = interface_start[0] + wp.int64(planes.shape[1] * fragment + previous)
        present = _clip_fragment(
            scratch_a,
            slot,
            scratch_b,
            slot,
            plane,
            wp.float64(-1.0),
            tag,
            1,
            tolerance,
            rest,
            elements,
        )
        if present == 0:
            return
        _copy_fragment(scratch_b, slot, scratch_a, slot)
    plane = _plane(source, fragment, positions, center[0], radius[0], planes[work_slot, plane_index])
    tag = interface_start[0] + wp.int64(planes.shape[1] * fragment + plane_index)
    present = _clip_fragment(scratch_a, slot, outside, slot, plane, wp.float64(1.0), tag, 1, tolerance, rest, elements)
    if present != 0:
        cap = outside.face_count[slot] - 1
        if outside.face_interface[slot, cap] == tag:
            _partition_cap(
                outside,
                slot,
                positions,
                center[0],
                radius[0],
                planes,
                plane_count[0],
                plane_index,
                work_slot,
                polygon_a,
                polygon_b,
                polygon_out,
                tolerance,
            )
        volume = _volume_device(outside, slot)
        if volume > wp.float64(_MIN_FRAGMENT_VOLUME):
            wp.atomic_add(kept, work_slot, volume)
        else:
            # Repeated cuts of a float32 deformed field can leave numerical
            # sheets that would otherwise exhaust the basis-node copy pool.
            wp.atomic_add(discarded, work_slot, wp.max(volume, wp.float64(0.0)))
            outside.face_count[slot] = 0
    error = wp.max(scratch_a.error[0], wp.max(scratch_b.error[0], outside.error[0]))
    if error != 0:
        wp.atomic_max(target.error, 0, error)


@wp.kernel(enable_backward=False)
def _validate_subtraction(
    source: FragmentArrays,
    target: FragmentArrays,
    work: wp.array[int],
    count: wp.array[int],
    removed: wp.array[wp.float64],
    kept: wp.array[wp.float64],
    discarded: wp.array[wp.float64],
    expected_removed: wp.array[wp.float64],
    validate_volume: bool,
    changed: wp.array[int],
    scratch_error_a: wp.array[int],
    scratch_error_b: wp.array[int],
    scratch_error_out: wp.array[int],
):
    slot = wp.tid()
    removed[slot] = wp.float64(0.0)
    error = wp.max(scratch_error_a[0], wp.max(scratch_error_b[0], scratch_error_out[0]))
    if error != 0:
        wp.atomic_max(target.error, 0, error)
    if slot >= count[0] or target.error[0] != 0:
        return
    fragment = work[slot]
    volume = _volume_device(source, fragment)
    # Keep cleanup volume separate from the true tool intersection so the
    # independent interior-volume oracle retains its original tolerance.
    difference = volume - kept[slot] - discarded[slot]
    tolerance = wp.float64(2.0e-11) * volume + wp.float64(1.0e-15)
    if not wp.isfinite(difference) or difference < -tolerance or difference > volume + tolerance:
        wp.atomic_max(target.error, 0, 3)
    elif validate_volume and wp.abs(difference - expected_removed[slot]) > tolerance:
        wp.atomic_max(target.error, 0, 3)
    elif difference <= tolerance:
        # Preserve the original partition when the broadphase only grazes
        # the tool; publishing auxiliary pieces would grow invisible seams.
        _append(source, fragment, target)
    else:
        removed[slot] = difference + discarded[slot]
        wp.atomic_max(changed, source.parent[fragment], 1)


@wp.kernel(enable_backward=False)
def _publish_regions(
    regions: FragmentArrays,
    target: FragmentArrays,
    count: wp.array[int],
    removed: wp.array[wp.float64],
    planes_per_fragment: int,
):
    slot = wp.tid()
    work_slot = slot // planes_per_fragment
    if (
        target.error[0] == 0
        and work_slot < count[0]
        and removed[work_slot] > wp.float64(0.0)
        and regions.face_count[slot] != 0
    ):
        _append(regions, slot, target)


class SphereSubtractor:
    """Fixed-workspace GPU subtraction; overflow rejects the complete edit.

    Production derives removed volume from the disjoint retained regions.
    ``validate_volume=True`` additionally constructs the interior intersection
    and checks its volume independently, for diagnostics and regression tests.
    """

    def __init__(
        self,
        *,
        vertex_capacity=32,
        face_capacity=24,
        work_capacity=256,
        scratch_vertex_capacity=None,
        scratch_face_capacity=None,
        validate_volume=False,
        device=None,
    ):
        if min(vertex_capacity, face_capacity, work_capacity) < 4:
            raise ValueError("Spherical clipping capacities must be at least four")
        self.device = wp.get_device(device)
        self.validate_volume = bool(validate_volume)
        self.center = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.end = wp.zeros(1, dtype=wp.vec3d, device=self.device)
        self.radius = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.planes = wp.zeros(50, dtype=wp.vec4d, device=self.device)
        self.plane_count = wp.zeros(1, dtype=int, device=self.device)
        self.work = wp.empty(work_capacity, dtype=int, device=self.device)
        self.count = wp.zeros(1, dtype=int, device=self.device)
        self.ordered_planes = wp.empty((work_capacity, 50), dtype=wp.vec4d, device=self.device)
        self.removed = wp.zeros(work_capacity, dtype=wp.float64, device=self.device)
        self.expected_removed = wp.zeros(work_capacity, dtype=wp.float64, device=self.device)
        self.kept = wp.zeros(work_capacity, dtype=wp.float64, device=self.device)
        self.discarded = wp.zeros(work_capacity, dtype=wp.float64, device=self.device)
        # Cap partitioning can briefly exceed the published vertex count.
        # Keep that headroom local instead of padding every mesh fragment.
        scratch_vertex_capacity = max(
            vertex_capacity, 48 if scratch_vertex_capacity is None else scratch_vertex_capacity
        )
        scratch_face_capacity = max(
            face_capacity, face_capacity if scratch_face_capacity is None else scratch_face_capacity
        )
        self.scratch = [
            FragmentDevice(
                work_capacity * 50,
                vertex_capacity=scratch_vertex_capacity,
                face_capacity=scratch_face_capacity,
                device=self.device,
            )
            for _ in range(3)
        ]
        self.polygons = [
            wp.empty((work_capacity * 50, scratch_vertex_capacity), dtype=wp.vec4d, device=self.device)
            for _ in range(3)
        ]

    def assign(self, center, radius, end=None):
        """Set the sphere sweep endpoints [m] and positive radius [m]."""
        center = np.asarray(center, dtype=np.float64)
        end = center if end is None else np.asarray(end, dtype=np.float64)
        planes = spherical_sweep_planes(center, end, radius)
        self.assign_planes(center, end, radius, planes)

    def assign_planes(self, start, end, radius, planes):
        """Set a convex tool bounded by its endpoint segment plus ``radius``.

        Unit-normal halfspaces use coordinates ``(point - start) / radius``.
        The endpoint/radius envelope is a conservative broadphase bound. Sphere
        sweeps and flat-ended cylinders share these persistent GPU buffers and
        their captured clipping graph.
        """
        start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
        planes = np.asarray(planes, dtype=np.float64)
        if (
            start.shape != (3,)
            or end.shape != (3,)
            or not np.isfinite(start).all()
            or not np.isfinite(end).all()
            or not np.isscalar(radius)
            or not np.isfinite(radius)
            or radius <= 0
        ):
            raise ValueError("Convex tool endpoints must be finite and radius must be positive")
        if (
            planes.ndim != 2
            or planes.shape[1] != 4
            or not 4 <= len(planes) <= len(self.planes)
            or not np.isfinite(planes).all()
            or not np.allclose(np.linalg.norm(planes[:, :3], axis=1), 1.0, rtol=0, atol=1.0e-12)
        ):
            raise ValueError("Convex tool requires 4 to 50 finite, unit-normal planes")
        values = np.zeros((len(self.planes), 4), dtype=np.float64)
        values[: len(planes)] = planes
        self.planes.assign(values)
        self.plane_count.assign([len(planes)])
        self.center.assign(start.reshape(1, 3))
        self.end.assign(end.reshape(1, 3))
        self.radius.assign([float(radius)])

    def launch(self, source, target, positions, rest, elements, interface_start, changed, tolerance):
        """Enqueue subtraction into unpublished ``target`` without host reads."""
        scratch = self.scratch[0]
        if (
            source.vertex_capacity > scratch.vertex_capacity
            or source.face_capacity > scratch.face_capacity
            or target.vertex_capacity < source.vertex_capacity
            or target.face_capacity < source.face_capacity
        ):
            raise ValueError("Source geometry must fit both the spherical scratch and target capacities")

        def run(kernel, dim, inputs, *, block_dim=256):
            wp.launch(kernel, dim=dim, inputs=inputs, device=self.device, block_dim=block_dim)

        for scratch in self.scratch:
            scratch.error.zero_()
        self.kept.zero_()
        self.discarded.zero_()
        run(_reset, 1, [source, target, self.count, self.scratch[0].error])
        run(
            _select,
            source.capacity,
            [
                source,
                target,
                positions,
                self.center,
                self.end,
                self.radius,
                self.planes,
                self.plane_count,
                tolerance,
                self.work,
                self.count,
            ],
        )
        run(
            _order_planes,
            len(self.work),
            [
                source,
                positions,
                self.center,
                self.radius,
                self.planes,
                self.plane_count,
                self.work,
                self.count,
                self.ordered_planes,
            ],
        )
        if self.validate_volume:
            run(
                _intersect,
                len(self.work),
                [
                    source,
                    target,
                    self.scratch[0].arrays,
                    self.scratch[1].arrays,
                    positions,
                    rest,
                    elements,
                    self.center,
                    self.radius,
                    self.ordered_planes,
                    self.plane_count,
                    self.work,
                    self.count,
                    self.expected_removed,
                    tolerance,
                ],
                block_dim=32,
            )
        run(
            _emit_regions,
            len(self.work) * len(self.planes),
            [
                source,
                target,
                *(item.arrays for item in self.scratch),
                positions,
                rest,
                elements,
                self.center,
                self.radius,
                self.ordered_planes,
                self.plane_count,
                self.work,
                self.count,
                self.kept,
                self.discarded,
                interface_start,
                *self.polygons,
                tolerance,
            ],
            block_dim=32,
        )
        run(
            _validate_subtraction,
            len(self.work),
            [
                source,
                target,
                self.work,
                self.count,
                self.removed,
                self.kept,
                self.discarded,
                self.expected_removed,
                self.validate_volume,
                changed,
                *(item.error for item in self.scratch),
            ],
        )
        run(
            _publish_regions,
            len(self.work) * len(self.planes),
            [self.scratch[2].arrays, target, self.count, self.removed, len(self.planes)],
        )

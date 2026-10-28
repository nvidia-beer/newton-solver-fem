# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Captured GPU adjacency for convex material fragments of a knife incision.

Faces use a common two-dimensional reference projection on each interface.
Spatial hash buckets generate candidates on the device; exact convex polygon
overlap then emits fragment pairs for the device connectivity solver. All buffers have
fixed capacities, and overflow rejects an edit instead of dropping connections.
"""

import numpy as np
import warp as wp


@wp.func
def _point(points: wp.array2d[wp.vec2d], counts: wp.array[int], orientation: wp.array[int], face: int, i: int):
    index = i
    if orientation[face] < 0:
        index = counts[face] - 1 - i
    return points[face, index]


@wp.func
def _cross(a: wp.vec2d, b: wp.vec2d):
    return a[0] * b[1] - a[1] * b[0]


@wp.func
def _boundary_integral(
    points: wp.array2d[wp.vec2d],
    counts: wp.array[int],
    orientation: wp.array[int],
    first: int,
    second: int,
    origin: wp.vec2d,
    tolerance: wp.float64,
    exclude_shared: bool,
):
    """Integrate clipped polygon edges without allocating a temporary polygon."""
    integral = wp.float64(0.0)
    for i in range(counts[first]):
        a = _point(points, counts, orientation, first, i) - origin
        b = _point(points, counts, orientation, first, (i + 1) % counts[first]) - origin
        low, high = wp.float64(0.0), wp.float64(1.0)
        shared = bool(False)
        for j in range(counts[second]):
            c = _point(points, counts, orientation, second, j) - origin
            d = _point(points, counts, orientation, second, (j + 1) % counts[second]) - origin
            edge = d - c
            da, db = _cross(edge, a - c), _cross(edge, b - c)
            # Both the tested point and the clipping edge inherit coordinate
            # roundoff. A short edge's directional error grows with distance
            # to the tested endpoints; omitting it makes common-edge integrals
            # disagree and can spuriously reconnect separated material.
            edge_tolerance = tolerance * (wp.length(edge) + wp.max(wp.length(a - c), wp.length(b - c)))
            if wp.abs(da) <= edge_tolerance:
                da = wp.float64(0.0)
            if wp.abs(db) <= edge_tolerance:
                db = wp.float64(0.0)
            if exclude_shared and wp.abs(da) <= edge_tolerance and wp.abs(db) <= edge_tolerance:
                if wp.dot(b - a, edge) > wp.float64(0.0):
                    shared = True
            delta = db - da
            if delta > wp.float64(0.0):
                low = wp.max(low, -da / delta)
            elif delta < wp.float64(0.0):
                high = wp.min(high, -da / delta)
            elif da < -edge_tolerance:
                high = wp.float64(-1.0)
        if high > low and not shared:
            integral += _cross(a + low * (b - a), a + high * (b - a))
    return integral


@wp.kernel
def _pack_geometry(
    reference_positions: wp.array[wp.vec3d],
    elements: wp.array2d[int],
    parent: wp.array[int],
    vertices: wp.array2d[wp.vec4d],
    faces: wp.array3d[int],
    sizes: wp.array2d[int],
    face_kinds: wp.array2d[int],
    axes: wp.array2d[int],
    face_interfaces: wp.array2d[wp.int64],
    fragment_face_count: wp.array[int],
    fragment_count: wp.array[int],
    faces_per_fragment: int,
    active_parents: wp.array[int],
    interface_parents: wp.array2d[int],
    points: wp.array2d[wp.vec2d],
    counts: wp.array[int],
    kinds: wp.array[int],
    interfaces: wp.array[wp.int64],
    fragments: wp.array[int],
    parents: wp.array[int],
    area_scale: wp.array[wp.float64],
    tolerance: wp.array[wp.float64],
    face_count: wp.array[int],
):
    index = wp.tid()
    fragment, face = index // faces_per_fragment, index % faces_per_fragment
    counts[index] = 0
    if index == 0:
        face_count[0] = fragment_count[0] * faces_per_fragment
    if fragment < fragment_count[0] and face < fragment_face_count[fragment]:
        count = sizes[fragment, face]
        if count >= 3 and face_kinds[fragment, face] != 2:
            element = parent[fragment]
            if active_parents.shape[0] > 0:
                active = active_parents[element] != 0
                if not active and face_kinds[fragment, face] == 0:
                    interface = int(-face_interfaces[fragment, face] - wp.int64(1))
                    if interface >= 0 and interface < interface_parents.shape[0]:
                        for side in range(2):
                            adjacent = interface_parents[interface, side]
                            if adjacent >= 0 and active_parents[adjacent] != 0:
                                active = True
                if not active:
                    return
            axis = axes[fragment, face]
            first_axis, second_axis = int(0), int(1)
            if axis == 0:
                first_axis, second_axis = 1, 2
            elif axis == 1:
                first_axis, second_axis = 0, 2
            origin, previous = wp.vec3d(0.0), wp.vec3d(0.0)
            normal = wp.vec3d(0.0)
            span = wp.float64(0.0)
            coordinate_scale = wp.float64(0.0)
            for i in range(count):
                barycentric = vertices[fragment, faces[fragment, face, i]]
                point = wp.vec3d(0.0)
                for corner in range(4):
                    point += barycentric[corner] * reference_positions[elements[element, corner]]
                coordinate_scale = wp.max(coordinate_scale, wp.length(point))
                points[index, i] = wp.vec2d(point[first_axis], point[second_axis])
                if i == 0:
                    origin = point
                else:
                    normal += wp.cross(previous - origin, point - origin)
                    span = wp.max(span, wp.length(point - origin))
                previous = point
            if wp.abs(normal[axis]) > wp.float64(0.0):
                counts[index] = count
                kinds[index] = face_kinds[fragment, face]
                interfaces[index] = face_interfaces[fragment, face]
                fragments[index] = fragment
                parents[index] = element
                area_scale[index] = wp.length(normal) / wp.abs(normal[axis])
                # Tiny source/cage intersections still inherit the roundoff of
                # world-space coordinates, not just their much smaller extent.
                tolerance[index] = wp.max(
                    wp.float64(5.684341886080802e-14) * span,
                    wp.float64(3.552713678800501e-15) * coordinate_scale,
                )


@wp.func
def _interface_bucket(interface: wp.int64, mask: int):
    return int((wp.uint64(interface) * wp.uint64(2654435761)) & wp.uint64(mask))


@wp.func
def _spatial_bucket(interface: wp.int64, level: int, x: int, y: int, mask: int):
    key = wp.uint64(interface) * wp.uint64(2654435761)
    key = key ^ (wp.uint64(wp.int64(x)) * wp.uint64(11400714819323198485))
    key = key ^ (wp.uint64(wp.int64(y)) * wp.uint64(14029467366897019727))
    key = key ^ (wp.uint64(wp.int64(level)) * wp.uint64(1609587929392839161))
    key = (key ^ (key >> wp.uint64(30))) * wp.uint64(13787848793156543929)
    key = key ^ (key >> wp.uint64(27))
    return int(key & wp.uint64(mask))


@wp.kernel
def _clear(heads: wp.array[int], max_levels: wp.array[int], edge_count: wp.array[int], overflow: wp.array[int]):
    i = wp.tid()
    heads[i] = -1
    max_levels[i] = -1024
    if i == 0:
        edge_count[0] = 0
        overflow[0] = 0


@wp.kernel
def _hash_faces(
    points: wp.array2d[wp.vec2d],
    counts: wp.array[int],
    kinds: wp.array[int],
    interfaces: wp.array[wp.int64],
    face_count: wp.array[int],
    bucket_mask: int,
    heads: wp.array[int],
    following: wp.array[int],
    orientation: wp.array[int],
    lower: wp.array[wp.vec2d],
    upper: wp.array[wp.vec2d],
    levels: wp.array[int],
    cells: wp.array[wp.vec2i],
    max_levels: wp.array[int],
):
    face = wp.tid()
    if face < face_count[0] and counts[face] >= 3 and kinds[face] != 2:
        minimum, maximum = wp.vec2d(points[face, 0]), wp.vec2d(points[face, 0])
        area = wp.float64(0.0)
        origin = points[face, 0]
        for i in range(counts[face]):
            value = points[face, i]
            minimum = wp.min(minimum, value)
            maximum = wp.max(maximum, value)
            area += _cross(value - origin, points[face, (i + 1) % counts[face]] - origin)
        lower[face] = minimum
        upper[face] = maximum
        orientation[face] = 1
        if area < wp.float64(0.0):
            orientation[face] = -1
        center = minimum * wp.float64(0.5) + maximum * wp.float64(0.5)
        span = maximum - minimum
        coordinate_scale = wp.max(wp.abs(center[0]), wp.abs(center[1]))
        # One cell per face. Bounding its integer coordinates also handles tiny
        # polygons far from the origin without integer overflow.
        extent = wp.max(wp.max(span[0], span[1]), coordinate_scale * wp.float64(9.313225746154785e-10))
        extent = wp.max(extent, wp.float64(9.094947017729282e-13))
        level = int(wp.ceil(wp.log2(extent)))
        cell_size = wp.pow(wp.float64(2.0), wp.float64(level))
        if cell_size < extent:
            level += 1
            cell_size *= wp.float64(2.0)
        x, y = int(wp.floor(center[0] / cell_size)), int(wp.floor(center[1] / cell_size))
        levels[face] = level
        cells[face] = wp.vec2i(x, y)
        wp.atomic_max(max_levels, _interface_bucket(interfaces[face], bucket_mask), level)
        bucket = _spatial_bucket(interfaces[face], level, x, y, bucket_mask)
        following[face] = wp.atomic_exch(heads, bucket, face)


@wp.kernel(enable_backward=False)
def _candidate_pairs(
    counts: wp.array[int],
    kinds: wp.array[int],
    interfaces: wp.array[wp.int64],
    fragments: wp.array[int],
    parents: wp.array[int],
    face_count: wp.array[int],
    bucket_mask: int,
    heads: wp.array[int],
    following: wp.array[int],
    lower: wp.array[wp.vec2d],
    upper: wp.array[wp.vec2d],
    levels: wp.array[int],
    cells: wp.array[wp.vec2i],
    max_levels: wp.array[int],
    pair_capacity: int,
    pairs: wp.array2d[int],
    pair_count: wp.array[int],
    overflow: wp.array[int],
):
    first = wp.tid()
    if first < face_count[0] and counts[first] >= 3 and kinds[first] != 2:
        maximum_level = max_levels[_interface_bucket(interfaces[first], bucket_mask)]
        cell_size = wp.pow(wp.float64(2.0), wp.float64(levels[first]))
        coordinate_scale = wp.max(wp.length(lower[first]), wp.length(upper[first]))
        for level in range(levels[first], maximum_level + 1):
            # A candidate at this level extends at most half a cell from its
            # center. Query the expanded bounds, with outward rounding, so
            # small faces also find larger faces across grid boundaries.
            scale = wp.max(coordinate_scale, cell_size)
            padding = wp.float64(0.5) * cell_size + wp.float64(8.881784197001252e-16) * scale
            low = (lower[first] - wp.vec2d(padding)) / cell_size
            high = (upper[first] + wp.vec2d(padding)) / cell_size
            for x in range(int(wp.floor(low[0])), int(wp.floor(high[0])) + 1):
                for y in range(int(wp.floor(low[1])), int(wp.floor(high[1])) + 1):
                    bucket = _spatial_bucket(interfaces[first], level, x, y, bucket_mask)
                    second = heads[bucket]
                    while second >= 0:
                        compatible = levels[second] == level and cells[second] == wp.vec2i(x, y)
                        compatible = compatible and interfaces[first] == interfaces[second]
                        compatible = compatible and (level > levels[first] or second > first)
                        a, b = wp.min(first, second), wp.max(first, second)
                        compatible = compatible and fragments[a] != fragments[b]
                        if kinds[a] == 0 and parents[a] == parents[b]:
                            compatible = False
                        if compatible:
                            overlap = wp.min(upper[a], upper[b]) - wp.max(lower[a], lower[b])
                            if overlap[0] > wp.float64(0.0) and overlap[1] > wp.float64(0.0):
                                index = wp.atomic_add(pair_count, 0, 1)
                                if index < pair_capacity:
                                    pairs[index, 0] = a
                                    pairs[index, 1] = b
                                else:
                                    wp.atomic_max(overflow, 0, 2)
                        second = following[second]
            cell_size *= wp.float64(2.0)


@wp.kernel(enable_backward=False)
def _adjacency(
    points: wp.array2d[wp.vec2d],
    counts: wp.array[int],
    fragments: wp.array[int],
    area_scale: wp.array[wp.float64],
    tolerance: wp.array[wp.float64],
    orientation: wp.array[int],
    lower: wp.array[wp.vec2d],
    upper: wp.array[wp.vec2d],
    pairs: wp.array2d[int],
    pair_count: wp.array[int],
    minimum_area: wp.float64,
    relative_area: wp.float64,
    edge_capacity: int,
    edges: wp.array2d[int],
    edge_count: wp.array[int],
    overflow: wp.array[int],
):
    index = wp.tid()
    if index < wp.min(pair_count[0], pairs.shape[0]):
        a, b = pairs[index, 0], pairs[index, 1]
        origin = points[a, 0]
        epsilon = wp.max(tolerance[a], tolerance[b])
        integral = _boundary_integral(points, counts, orientation, a, b, origin, epsilon, False)
        integral += _boundary_integral(points, counts, orientation, b, a, origin, epsilon, True)
        area = wp.abs(integral) * wp.float64(0.5) * area_scale[a]
        first_span = upper[a] - lower[a]
        second_span = upper[b] - lower[b]
        scale = wp.min(first_span[0] * first_span[1], second_span[0] * second_span[1])
        threshold = wp.max(minimum_area, relative_area * scale * area_scale[a])
        if relative_area > wp.float64(0.0):
            # Preserve the zero-area common-edge guard
            # independently of broad-phase grid size.
            roundoff = wp.float64(8.0) * epsilon * (wp.length(first_span) + wp.length(second_span))
            threshold = wp.max(threshold, roundoff * area_scale[a])
        if area > threshold:
            index = wp.atomic_add(edge_count, 0, 1)
            if index < edge_capacity:
                edges[index, 0] = fragments[a]
                edges[index, 1] = fragments[b]
            else:
                wp.atomic_max(overflow, 0, 1)


class FragmentAdjacency:
    """Reusable face buffers and graph-compatible positive-area adjacency.

    ``kinds`` uses 0=original, 1=auxiliary, 2=cut. Original faces only connect
    different parents. ``area_scale`` converts projected area to square meters.
    All faces sharing an interface ID must use exactly the same projection.
    ``launch`` performs no allocation, host readback, or synchronization.
    """

    def __init__(
        self,
        *,
        face_capacity,
        edge_capacity,
        vertex_capacity=32,
        minimum_area=1e-14,
        relative_area=0.0,
        candidate_capacity=None,
        device=None,
    ):
        if min(face_capacity, edge_capacity) < 1 or vertex_capacity < 3:
            raise ValueError("adjacency capacities must be positive, with at least three vertices per face")
        if candidate_capacity is None:
            candidate_capacity = max(edge_capacity, 4 * face_capacity)
        if candidate_capacity < 1:
            raise ValueError("candidate capacity must be positive")
        self.candidate_capacity = candidate_capacity
        self.device = wp.get_device(device)
        self.face_capacity, self.edge_capacity, self.vertex_capacity = face_capacity, edge_capacity, vertex_capacity
        self.minimum_area = float(minimum_area)
        self.relative_area = float(relative_area)
        if not np.isfinite([minimum_area, relative_area]).all() or min(minimum_area, relative_area) < 0.0:
            raise ValueError("adjacency area tolerances must be finite and nonnegative")
        self.bucket_count = 1 << (2 * face_capacity - 1).bit_length()
        self.points = wp.zeros((face_capacity, vertex_capacity), dtype=wp.vec2d, device=self.device)
        for name in ("counts", "kinds", "fragments", "parents", "following", "orientation", "levels"):
            setattr(self, name, wp.zeros(face_capacity, dtype=int, device=self.device))
        self.interfaces = wp.zeros(face_capacity, dtype=wp.int64, device=self.device)
        self.area_scale = wp.ones(face_capacity, dtype=wp.float64, device=self.device)
        self.tolerance = wp.full(face_capacity, 2e-14, dtype=wp.float64, device=self.device)
        self.face_count = wp.zeros(1, dtype=int, device=self.device)
        self.heads = wp.empty(self.bucket_count, dtype=int, device=self.device)
        self.max_levels = wp.empty(self.bucket_count, dtype=int, device=self.device)
        self.cells = wp.empty(face_capacity, dtype=wp.vec2i, device=self.device)
        self.lower = wp.empty(face_capacity, dtype=wp.vec2d, device=self.device)
        self.upper = wp.empty(face_capacity, dtype=wp.vec2d, device=self.device)
        self.pairs = wp.empty((candidate_capacity, 2), dtype=int, device=self.device)
        self.pair_count = wp.zeros(1, dtype=int, device=self.device)
        self.edges = wp.empty((edge_capacity, 2), dtype=int, device=self.device)
        self.edge_count = wp.zeros(1, dtype=int, device=self.device)
        self.overflow = wp.zeros(1, dtype=int, device=self.device)
        self._all_parents = wp.empty(0, dtype=int, device=self.device)
        self._no_interfaces = wp.empty((0, 2), dtype=int, device=self.device)

    def pack_device_geometry(
        self, geometry, reference_positions, elements, *, active_parents=None, interface_parents=None
    ):
        """Project a persistent device fragment store without host geometry reads.

        ``geometry`` exposes the fixed arrays of ``FragmentDevice``. Interface
        projection axes must be inherited identically on both sides of a face.
        With ``active_parents``, project only those parents and original faces
        bordering them. ``interface_parents[-interface_id - 1]`` gives the two
        original parents, using -1 for an exterior side. Omit both arrays to
        retain the full-geometry path.
        """
        faces_per_fragment = geometry.face_size.shape[1]
        required = geometry.parent.shape[0] * faces_per_fragment
        if required > self.face_capacity or geometry.faces.shape[2] > self.vertex_capacity:
            raise ValueError("device fragment store exceeds adjacency capacity")
        if (active_parents is None) != (interface_parents is None):
            raise ValueError("Active parents and their original interface map must be supplied together")
        if active_parents is not None:
            if active_parents.shape != (elements.shape[0],) or active_parents.dtype != wp.int32:
                raise ValueError("Active parents must contain one integer flag per original element")
            if interface_parents.ndim != 2 or interface_parents.shape[1] != 2 or interface_parents.dtype != wp.int32:
                raise ValueError("Original interface parents must have two integer columns")
        wp.launch(
            _pack_geometry,
            required,
            inputs=[
                reference_positions,
                elements,
                geometry.parent,
                geometry.vertices,
                geometry.faces,
                geometry.face_size,
                geometry.face_kind,
                geometry.face_axis,
                geometry.face_interface,
                geometry.face_count,
                geometry.count,
                faces_per_fragment,
                self._all_parents if active_parents is None else active_parents,
                self._no_interfaces if interface_parents is None else interface_parents,
                self.points,
                self.counts,
                self.kinds,
                self.interfaces,
                self.fragments,
                self.parents,
                self.area_scale,
                self.tolerance,
                self.face_count,
            ],
            device=self.device,
        )

    def launch(self):
        """Append adjacent fragment pairs, or set ``overflow`` if capacity is exceeded."""
        wp.launch(
            _clear,
            self.bucket_count,
            inputs=[self.heads, self.max_levels, self.edge_count, self.overflow],
            device=self.device,
        )
        wp.launch(
            _hash_faces,
            self.face_capacity,
            inputs=[
                self.points,
                self.counts,
                self.kinds,
                self.interfaces,
                self.face_count,
                self.bucket_count - 1,
                self.heads,
                self.following,
                self.orientation,
                self.lower,
                self.upper,
                self.levels,
                self.cells,
                self.max_levels,
            ],
            device=self.device,
        )
        self.pair_count.zero_()
        wp.launch(
            _candidate_pairs,
            self.face_capacity,
            inputs=[
                self.counts,
                self.kinds,
                self.interfaces,
                self.fragments,
                self.parents,
                self.face_count,
                self.bucket_count - 1,
                self.heads,
                self.following,
                self.lower,
                self.upper,
                self.levels,
                self.cells,
                self.max_levels,
                self.candidate_capacity,
                self.pairs,
                self.pair_count,
                self.overflow,
            ],
            device=self.device,
        )
        wp.launch(
            _adjacency,
            self.candidate_capacity,
            inputs=[
                self.points,
                self.counts,
                self.fragments,
                self.area_scale,
                self.tolerance,
                self.orientation,
                self.lower,
                self.upper,
                self.pairs,
                self.pair_count,
                self.minimum_area,
                self.relative_area,
                self.edge_capacity,
                self.edges,
                self.edge_count,
                self.overflow,
            ],
            device=self.device,
        )

    def upload(self, polygons, *, fragments, parents, interfaces, kinds=None, area_scale=None):
        """Populate buffers from a reference fixture; production device packing bypasses this method."""
        count = len(polygons)
        if count > self.face_capacity or any(len(polygon) > self.vertex_capacity for polygon in polygons):
            raise ValueError("face geometry exceeds adjacency capacity")
        lengths = np.asarray([len(polygon) for polygon in polygons], dtype=np.int32)
        if np.any(lengths < 3):
            raise ValueError("face polygons require at least three vertices")
        points = np.zeros((self.face_capacity, self.vertex_capacity, 2))
        for i, polygon in enumerate(polygons):
            points[i, : len(polygon)] = polygon
        if not np.isfinite(points).all():
            raise ValueError("face polygons must be finite")
        values = {"counts": lengths, "fragments": fragments, "parents": parents, "interfaces": interfaces}
        values["kinds"] = np.zeros(count, dtype=np.int32) if kinds is None else kinds
        values["area_scale"] = np.ones(count) if area_scale is None else area_scale
        staged = {}
        for name, value in values.items():
            dtype = np.float64 if name == "area_scale" else np.int64 if name == "interfaces" else np.int32
            array = np.asarray(value, dtype=dtype)
            if array.shape != (count,):
                raise ValueError(f"{name} must contain one value per face")
            padded = np.zeros(self.face_capacity, dtype=dtype)
            padded[:count] = array
            staged[name] = padded
        self.points.assign(points)
        for name, value in staged.items():
            getattr(self, name).assign(value)
        self.face_count.assign(np.array([count], dtype=np.int32))

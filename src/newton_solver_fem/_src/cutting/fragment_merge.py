# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded exact convex unions across disposable auxiliary material faces."""

import warp as wp

from .fragment_device import FragmentArrays, _copy_fragment, _volume_device

_SCAN_GROUP = 256


@wp.func
def _xyz(point: wp.vec4d):
    return wp.vec3d(point[1], point[2], point[3])


@wp.func
def _normal(geometry: FragmentArrays, fragment: int, face: int):
    origin = _xyz(geometry.vertices[fragment, geometry.faces[fragment, face, 0]])
    normal = wp.vec3d(0.0)
    for index in range(1, geometry.face_size[fragment, face] - 1):
        a = _xyz(geometry.vertices[fragment, geometry.faces[fragment, face, index]])
        b = _xyz(geometry.vertices[fragment, geometry.faces[fragment, face, index + 1]])
        normal += wp.cross(a - origin, b - origin)
    return wp.normalize(normal)


@wp.func
def _contains(geometry: FragmentArrays, first: int, face: int, second: int, other_face: int):
    normal = _normal(geometry, first, face)
    if wp.length(normal) < wp.float64(0.5):
        return False
    origin = _xyz(geometry.vertices[first, geometry.faces[first, face, 0]])
    for index in range(geometry.face_size[second, other_face]):
        point = _xyz(geometry.vertices[second, geometry.faces[second, other_face, index]])
        if wp.abs(wp.dot(point - origin, normal)) > wp.float64(1e-12):
            return False
        for edge in range(geometry.face_size[first, face]):
            a = _xyz(geometry.vertices[first, geometry.faces[first, face, edge]])
            b = _xyz(
                geometry.vertices[first, geometry.faces[first, face, (edge + 1) % geometry.face_size[first, face]]]
            )
            if wp.dot(wp.cross(b - a, point - a), normal) < -wp.float64(1e-12) * wp.length(b - a):
                return False
    return True


@wp.func
def _convex_union(geometry: FragmentArrays, first: int, first_face: int, second: int, second_face: int):
    for side in range(2):
        owner, other, excluded = first, second, first_face
        if side == 1:
            owner, other, excluded = second, first, second_face
        for face in range(geometry.face_count[owner]):
            if face != excluded:
                normal = _normal(geometry, owner, face)
                origin = _xyz(geometry.vertices[owner, geometry.faces[owner, face, 0]])
                for vertex in range(geometry.vertex_count[other]):
                    if wp.dot(_xyz(geometry.vertices[other, vertex]) - origin, normal) > wp.float64(1e-12):
                        return False
    return True


@wp.func
def _owner(index: int, first: int, second: int, first_count: int):
    if index < first_count:
        return first, index
    return second, index - first_count


@wp.func
def _matching_plane(geometry: FragmentArrays, first: int, face: int, second: int, other_face: int):
    if geometry.face_kind[first, face] != geometry.face_kind[second, other_face]:
        return False
    if geometry.face_kind[first, face] != 2:
        if geometry.face_interface[first, face] != geometry.face_interface[second, other_face]:
            return False
    normal, other_normal = _normal(geometry, first, face), _normal(geometry, second, other_face)
    if wp.length(normal - other_normal) > wp.float64(1e-12):
        return False
    a = _xyz(geometry.vertices[first, geometry.faces[first, face, 0]])
    b = _xyz(geometry.vertices[second, geometry.faces[second, other_face, 0]])
    return wp.abs(wp.dot(b - a, normal)) <= wp.float64(1e-12)


@wp.func
def _add_bounded(target: FragmentArrays, fragment: int, point: wp.vec4d):
    for index in range(target.vertex_count[fragment]):
        if wp.max(wp.abs(target.vertices[fragment, index] - point)) <= wp.float64(2e-14):
            return index
    index = target.vertex_count[fragment]
    if index == target.vertex_capacity:
        return int(-1)
    target.vertices[fragment, index] = point
    target.vertex_count[fragment] = index + 1
    return index


@wp.func
def _candidate(
    source: FragmentArrays,
    first: int,
    first_face: int,
    second: int,
    second_face: int,
    target: FragmentArrays,
    points: wp.array2d[wp.vec4d],
    processed: wp.array2d[int],
    hull: wp.array2d[int],
):
    work = source.parent[first]
    target.vertex_count[first] = 0
    target.face_count[first] = 0
    target.parent[first] = source.parent[first]
    target.source[first] = source.source[first]
    target.created[first] = 1
    for corner in range(4):
        target.nodeindices[first, corner] = source.nodeindices[first, corner]
    first_count = source.face_count[first]
    count = first_count + source.face_count[second]
    for index in range(count):
        processed[work, index] = 0
    processed[work, first_face] = 1
    processed[work, first_count + second_face] = 1
    for index in range(count):
        if processed[work, index] == 0:
            owner, face = _owner(index, first, second, first_count)
            normal = _normal(source, owner, face)
            point_count = int(0)
            original_area = wp.float64(0.0)
            for other_index in range(index, count):
                other, other_face = _owner(other_index, first, second, first_count)
                if processed[work, other_index] == 0 and _matching_plane(source, owner, face, other, other_face):
                    processed[work, other_index] = 1
                    polygon_normal = wp.vec3d(0.0)
                    origin = _xyz(source.vertices[other, source.faces[other, other_face, 0]])
                    for vertex in range(1, source.face_size[other, other_face] - 1):
                        a = _xyz(source.vertices[other, source.faces[other, other_face, vertex]])
                        b = _xyz(source.vertices[other, source.faces[other, other_face, vertex + 1]])
                        polygon_normal += wp.cross(a - origin, b - origin)
                    original_area += wp.length(polygon_normal)
                    for vertex in range(source.face_size[other, other_face]):
                        point = source.vertices[other, source.faces[other, other_face, vertex]]
                        found = bool(False)
                        for k in range(point_count):
                            old = points[work, k]
                            if wp.max(wp.abs(point - old)) <= wp.float64(2e-14):
                                found = True
                        if not found:
                            if point_count == 64:
                                return False
                            points[work, point_count] = point
                            point_count += 1
            if point_count < 3:
                return False
            axis = int(0)
            if wp.abs(normal[1]) > wp.abs(normal[axis]):
                axis = 1
            if wp.abs(normal[2]) > wp.abs(normal[axis]):
                axis = 2
            x, y = int(1), int(2)
            if axis == 0:
                x, y = 2, 3
            elif axis == 1:
                x, y = 1, 3
            start = int(0)
            for vertex in range(1, point_count):
                dx = points[work, vertex][x] - points[work, start][x]
                if dx < -wp.float64(2e-14) or (
                    wp.abs(dx) <= wp.float64(2e-14) and points[work, vertex][y] < points[work, start][y]
                ):
                    start = vertex
            hull_count = int(0)
            current = start + int(0)
            for _boundary in range(64):
                hull[work, hull_count] = current
                hull_count += 1
                following = (current + 1) % point_count
                for vertex in range(point_count):
                    if vertex != current:
                        ax = points[work, following][x] - points[work, current][x]
                        ay = points[work, following][y] - points[work, current][y]
                        bx = points[work, vertex][x] - points[work, current][x]
                        by = points[work, vertex][y] - points[work, current][y]
                        turn = ax * by - ay * bx
                        if turn < -wp.float64(1e-14) or (
                            wp.abs(turn) <= wp.float64(1e-14) and bx * bx + by * by > ax * ax + ay * ay
                        ):
                            following = vertex
                current = following
                if current == start:
                    break
            if current != start or hull_count > target.vertex_capacity or hull_count < 3:
                return False
            hull_normal = wp.vec3d(0.0)
            p = int(hull[work, 0])
            origin = _xyz(points[work, p])
            for vertex in range(1, hull_count - 1):
                p, r = int(hull[work, vertex]), int(hull[work, vertex + 1])
                a = _xyz(points[work, p])
                b = _xyz(points[work, r])
                hull_normal += wp.cross(a - origin, b - origin)
            if wp.abs(wp.length(hull_normal) - original_area) > wp.float64(2e-11) * original_area + wp.float64(1e-15):
                return False
            out_face = target.face_count[first]
            if out_face == target.face_capacity:
                return False
            orientation = normal[axis]
            if axis == 1:
                orientation = -orientation
            for vertex in range(hull_count):
                h = vertex
                if orientation < wp.float64(0.0):
                    h = hull_count - vertex - 1
                p = int(hull[work, h])
                point = points[work, p]
                stored = _add_bounded(target, first, point)
                if stored < 0:
                    return False
                target.faces[first, out_face, vertex] = stored
            target.face_size[first, out_face] = hull_count
            target.face_kind[first, out_face] = source.face_kind[owner, face]
            target.face_axis[first, out_face] = source.face_axis[owner, face]
            target.face_interface[first, out_face] = source.face_interface[owner, face]
            target.face_count[first] = out_face + 1
    before = _volume_device(source, first) + _volume_device(source, second)
    after = _volume_device(target, first)
    return after > wp.float64(0.0) and wp.abs(after - before) <= wp.float64(2e-11) * before + wp.float64(1e-15)


@wp.kernel(enable_backward=False)
def _count_parent_members(
    source: FragmentArrays,
    counts: wp.array[int],
    changed: wp.array[int],
):
    fragment = wp.tid()
    if source.error[0] != 0 or fragment >= wp.min(source.count[0], source.capacity):
        return
    parent = source.parent[fragment]
    if parent < 0 or parent >= counts.shape[0]:
        wp.atomic_max(source.error, 0, 4)
        return
    wp.atomic_add(counts, parent, 1)
    if source.created[fragment] != 0:
        wp.atomic_max(changed, parent, 1)


@wp.kernel(enable_backward=False)
def _scan_parent_counts(counts: wp.array[int], offsets: wp.array[int], totals: wp.array[int]):
    group = wp.tid()
    total = int(0)
    for parent in range(group * _SCAN_GROUP, wp.min((group + 1) * _SCAN_GROUP, counts.shape[0])):
        offsets[parent] = total
        total += counts[parent]
    totals[group] = total


@wp.kernel(enable_backward=False)
def _scan_group_totals(totals: wp.array[int], offsets: wp.array[int]):
    total = int(0)
    for group in range(totals.shape[0]):
        offsets[group] = total
        total += totals[group]


@wp.kernel(enable_backward=False)
def _finish_parent_offsets(
    counts: wp.array[int],
    changed: wp.array[int],
    group_offsets: wp.array[int],
    offsets: wp.array[int],
    sizes: wp.array[int],
):
    parent = wp.tid()
    offset = offsets[parent] + group_offsets[parent // _SCAN_GROUP]
    offsets[parent] = offset
    if parent == counts.shape[0] - 1:
        offsets[parent + 1] = offset + counts[parent]
    sizes[parent] = counts[parent] if changed[parent] != 0 else 0


@wp.kernel(enable_backward=False)
def _scatter_parent_members(
    source: FragmentArrays,
    offsets: wp.array[int],
    cursors: wp.array[int],
    members: wp.array[int],
):
    fragment = wp.tid()
    if source.error[0] == 0 and fragment < wp.min(source.count[0], source.capacity):
        parent = source.parent[fragment]
        destination = offsets[parent] + wp.atomic_add(cursors, parent, 1)
        if destination < members.shape[0]:
            members[destination] = fragment
        else:
            wp.atomic_max(source.error, 0, 4)


@wp.kernel(enable_backward=False)
def _sort_parent_members(members: wp.array[int], offsets: wp.array[int], sizes: wp.array[int], error: wp.array[int]):
    parent = wp.tid()
    if error[0] != 0:
        return
    start = offsets[parent]
    # Atomic scatter order is unspecified. Restore the previous dense scan's
    # ascending fragment order so convex-union decisions remain identical.
    for index in range(1, sizes[parent]):
        value = members[start + index]
        insertion = index
        while insertion > 0:
            previous = members[start + insertion - 1]
            if previous <= value:
                break
            members[start + insertion] = previous
            insertion -= 1
        members[start + insertion] = value


@wp.kernel(enable_backward=False)
def _initialize(
    source: FragmentArrays,
    output: FragmentArrays,
    alive: wp.array[int],
):
    index = wp.tid()
    alive[index] = int(index < source.count[0])
    if index == 0:
        output.count[0] = 0
        output.error[0] = source.error[0]
        if source.count[0] < 0 or source.count[0] > source.capacity:
            source.error[0] = 4
            output.error[0] = 4


@wp.kernel(enable_backward=False)
def _merge_parents(
    source: FragmentArrays,
    scratch: FragmentArrays,
    members: wp.array[int],
    offsets: wp.array[int],
    sizes: wp.array[int],
    alive: wp.array[int],
    points: wp.array2d[wp.vec4d],
    processed: wp.array2d[int],
    hull: wp.array2d[int],
):
    parent = wp.tid()
    count = sizes[parent]
    if source.error[0] != 0 or count < 2:
        return
    start = offsets[parent]
    for _iteration in range(count):
        changed = bool(False)
        for i in range(count):
            first = members[start + i]
            if alive[first] != 0:
                for j in range(i + 1, count):
                    second = members[start + j]
                    compatible = alive[second] != 0
                    for corner in range(4):
                        if source.nodeindices[first, corner] != source.nodeindices[second, corner]:
                            compatible = False
                    if compatible:
                        merged = bool(False)
                        for fa in range(source.face_count[first]):
                            if source.face_kind[first, fa] == 1:
                                for fb in range(source.face_count[second]):
                                    if (
                                        source.face_kind[second, fb] == 1
                                        and source.face_interface[first, fa] == source.face_interface[second, fb]
                                    ):
                                        if _contains(source, first, fa, second, fb) and _contains(
                                            source, second, fb, first, fa
                                        ):
                                            if _convex_union(source, first, fa, second, fb) and _candidate(
                                                source, first, fa, second, fb, scratch, points, processed, hull
                                            ):
                                                _copy_fragment(scratch, first, source, first)
                                                alive[second] = 0
                                                changed, merged = True, True
                                                break
                                if merged:
                                    break
        if not changed:
            break


@wp.kernel(enable_backward=False)
def _compact(source: FragmentArrays, output: FragmentArrays, alive: wp.array[int]):
    index = wp.tid()
    if index == 0:
        output.error[0] = source.error[0]
    if index < source.count[0] and alive[index] != 0 and source.error[0] == 0:
        destination = wp.atomic_add(output.count, 0, 1)
        _copy_fragment(source, index, output, destination)


class FragmentMerger:
    """Simplify changed parents; failed convex unions leave their inputs intact."""

    def __init__(self, parent_count, capacity, *, device):
        if parent_count < 1 or capacity < 1:
            raise ValueError("parent count and fragment capacity must be positive")
        self.device = wp.get_device(device)
        self.members = wp.empty(capacity, dtype=int, device=device)
        self.counts = wp.empty(parent_count, dtype=int, device=device)
        self.changed = wp.empty(parent_count, dtype=int, device=device)
        self.cursors = wp.empty(parent_count, dtype=int, device=device)
        self.offsets = wp.empty(parent_count + 1, dtype=int, device=device)
        groups = (parent_count + _SCAN_GROUP - 1) // _SCAN_GROUP
        self._group_totals = wp.empty(groups, dtype=int, device=device)
        self._group_offsets = wp.empty(groups, dtype=int, device=device)
        self.sizes = wp.empty(parent_count, dtype=int, device=device)
        self.alive = wp.empty(capacity, dtype=int, device=device)
        self.points = wp.empty((parent_count, 64), dtype=wp.vec4d, device=device)
        self.processed = wp.empty((parent_count, 64), dtype=int, device=device)
        self.hull = wp.empty((parent_count, 64), dtype=int, device=device)

    def launch(self, source, output, scratch):
        if source.capacity > self.members.size or source.capacity != output.capacity:
            raise ValueError("fragment storage exceeds the merger capacity or output layout")
        if source.capacity > scratch.capacity:
            raise ValueError("merge scratch must contain every source fragment")
        self.counts.zero_()
        self.changed.zero_()
        self.cursors.zero_()
        wp.launch(
            _initialize,
            self.alive.size,
            inputs=[source.arrays, output.arrays, self.alive],
            device=self.device,
        )
        wp.launch(
            _count_parent_members,
            source.capacity,
            inputs=[source.arrays, self.counts, self.changed],
            device=self.device,
        )
        wp.launch(
            _scan_parent_counts,
            self._group_totals.size,
            inputs=[self.counts, self.offsets, self._group_totals],
            device=self.device,
        )
        wp.launch(
            _scan_group_totals,
            1,
            inputs=[self._group_totals, self._group_offsets],
            device=self.device,
        )
        wp.launch(
            _finish_parent_offsets,
            self.sizes.size,
            inputs=[self.counts, self.changed, self._group_offsets, self.offsets, self.sizes],
            device=self.device,
        )
        wp.launch(
            _scatter_parent_members,
            source.capacity,
            inputs=[source.arrays, self.offsets, self.cursors, self.members],
            device=self.device,
        )
        wp.launch(
            _sort_parent_members,
            self.sizes.size,
            inputs=[self.members, self.offsets, self.sizes, source.error],
            device=self.device,
        )
        wp.launch(
            _merge_parents,
            self.sizes.size,
            inputs=[
                source.arrays,
                scratch.arrays,
                self.members,
                self.offsets,
                self.sizes,
                self.alive,
                self.points,
                self.processed,
                self.hull,
            ],
            device=self.device,
        )
        wp.launch(_compact, self.alive.size, inputs=[source.arrays, output.arrays, self.alive], device=self.device)
        source.copy_from(output)

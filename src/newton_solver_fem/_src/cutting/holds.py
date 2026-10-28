# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Implicit world attachments to physical material points in clipped P1 tets."""

import numpy as np
import warp as wp

from .fragment_device import FragmentArrays

_GRID = 8
_SAMPLES = _GRID * _GRID


@wp.func
def _surface_point(vertex: int, nodes: wp.array2d[int], weights: wp.array[wp.vec4], q: wp.array[wp.vec3]):
    point = wp.vec3(0.0)
    for corner in range(4):
        point += weights[vertex][corner] * q[nodes[vertex, corner]]
    return point


@wp.kernel(enable_backward=False)
def _patch_frame(
    triangle: int, nodes: wp.array2d[int], weights: wp.array[wp.vec4], q: wp.array[wp.vec3], frame: wp.array[wp.vec3]
):
    a = _surface_point(3 * triangle, nodes, weights, q)
    b = _surface_point(3 * triangle + 1, nodes, weights, q)
    c = _surface_point(3 * triangle + 2, nodes, weights, q)
    normal = wp.normalize(wp.cross(b - a, c - a))
    axis = wp.vec3(1.0, 0.0, 0.0)
    if wp.abs(normal[0]) > 0.8:
        axis = wp.vec3(0.0, 1.0, 0.0)
    frame[0] = wp.normalize(wp.cross(normal, axis))
    frame[1] = wp.cross(normal, frame[0])
    frame[2] = normal


@wp.func
def _patch_cell(
    triangle: int,
    picked: int,
    center: wp.vec3,
    radius: float,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    components: wp.array[int],
    frame: wp.array[wp.vec3],
):
    a = _surface_point(3 * triangle, nodes, weights, q)
    b = _surface_point(3 * triangle + 1, nodes, weights, q)
    c = _surface_point(3 * triangle + 2, nodes, weights, q)
    normal = wp.cross(b - a, c - a)
    point = (a + b + c) / 3.0
    if triangle == picked:
        point = center
    displacement = point - center
    accepted = (
        components[triangle] == components[picked]
        and wp.length(displacement) <= radius
        and wp.dot(wp.normalize(normal), frame[2]) > 0.2
    )
    cell, score = int(-1), float(0.0)
    if accepted or triangle == picked:
        u, v = wp.dot(displacement, frame[0]), wp.dot(displacement, frame[1])
        x = wp.clamp(int(wp.floor((u / radius + 1.0) * float(_GRID) * 0.5)), 0, _GRID - 1)
        y = wp.clamp(int(wp.floor((v / radius + 1.0) * float(_GRID) * 0.5)), 0, _GRID - 1)
        cell = y * _GRID + x
        du = u - radius * (2.0 * (float(x) + 0.5) / float(_GRID) - 1.0)
        dv = v - radius * (2.0 * (float(y) + 0.5) / float(_GRID) - 1.0)
        score = du * du + dv * dv
        if triangle == picked:
            score = -1.0
    return cell, score, 0.5 * wp.length(normal)


@wp.kernel(enable_backward=False)
def _measure_patch(
    picked: int,
    center: wp.vec3,
    radius: float,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    components: wp.array[int],
    count: wp.array[int],
    frame: wp.array[wp.vec3],
    areas: wp.array[float],
    nearest: wp.array[float],
):
    triangle = wp.tid()
    if triangle < count[0]:
        cell, score, area = _patch_cell(triangle, picked, center, radius, nodes, weights, q, components, frame)
        if cell >= 0:
            wp.atomic_add(areas, cell, area)
            wp.atomic_min(nearest, cell, score)


@wp.kernel(enable_backward=False)
def _choose_patch(
    picked: int,
    center: wp.vec3,
    radius: float,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    components: wp.array[int],
    count: wp.array[int],
    frame: wp.array[wp.vec3],
    nearest: wp.array[float],
    triangles: wp.array[int],
):
    triangle = wp.tid()
    if triangle < count[0]:
        cell, score, _area = _patch_cell(triangle, picked, center, radius, nodes, weights, q, components, frame)
        if cell >= 0 and score == nearest[cell]:
            wp.atomic_min(triangles, cell, triangle)


@wp.kernel(enable_backward=False)
def _capture_patch(
    grip: int,
    picked: int,
    barycentric: wp.vec3,
    center: wp.vec3,
    triangles: wp.array[int],
    areas: wp.array[float],
    surface_nodes: wp.array2d[int],
    surface_weights: wp.array[wp.vec4],
    q: wp.array[wp.vec3],
    parents: wp.array[int],
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    offsets: wp.array[wp.vec3],
    fractions: wp.array[float],
    enabled: wp.array[int],
):
    sample = wp.tid()
    slot = grip * _SAMPLES + sample
    triangle = triangles[sample]
    enabled[slot] = 0
    fractions[slot] = 0.0
    if triangle == 2147483647:
        return
    total = float(0.0)
    for index in range(_SAMPLES):
        total += areas[index]
    if total <= 0.0:
        return
    coordinates = wp.vec4(0.0)
    for vertex in range(3):
        factor = 1.0 / 3.0
        if triangle == picked:
            factor = barycentric[vertex]
        coordinates += factor * surface_weights[3 * triangle + vertex]
    weights[slot] = coordinates
    point = wp.vec3(0.0)
    for corner in range(4):
        node = int(surface_nodes[3 * triangle, corner])
        nodes[slot, corner] = node
        point += coordinates[corner] * q[node]
    offsets[slot] = point - center
    fractions[slot] = areas[sample] / total
    parents[slot] = 2147483647
    enabled[slot] = 1


@wp.kernel(enable_backward=False)
def _capture_parents(
    grip: int, geometry: FragmentArrays, parents: wp.array[int], nodes: wp.array2d[int], enabled: wp.array[int]
):
    fragment = wp.tid()
    if fragment >= geometry.count[0]:
        return
    for sample in range(_SAMPLES):
        slot = grip * _SAMPLES + sample
        if enabled[slot] != 0:
            same = bool(True)
            for corner in range(4):
                if geometry.nodeindices[fragment, corner] != nodes[slot, corner]:
                    same = False
            if same:
                wp.atomic_min(parents, slot, geometry.parent[fragment])


@wp.kernel(enable_backward=False)
def _check_patch(grip: int, parents: wp.array[int], enabled: wp.array[int], status: wp.array[int]):
    slot = grip * _SAMPLES + wp.tid()
    if enabled[slot] != 0:
        if parents[slot] == 2147483647:
            enabled[slot] = 0
            wp.atomic_add(status, 1, 1)
        else:
            wp.atomic_add(status, 0, 1)


@wp.kernel(enable_backward=False)
def _remap(
    geometry: FragmentArrays,
    fragment_nodes: wp.array2d[int],
    previous_nodes: wp.array2d[int],
    use_lineage: bool,
    parents: wp.array[int],
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    enabled: wp.array[int],
):
    slot = wp.tid()
    if enabled[slot] == 0:
        return
    coordinate = wp.vec3d(wp.float64(weights[slot][1]), wp.float64(weights[slot][2]), wp.float64(weights[slot][3]))
    selected, best_score = int(-1), int(-1)
    for fragment in range(geometry.count[0]):
        if geometry.parent[fragment] != parents[slot]:
            continue
        center = wp.vec3d(0.0)
        for vertex in range(geometry.vertex_count[fragment]):
            value = geometry.vertices[fragment, vertex]
            center += wp.vec3d(value[1], value[2], value[3])
        center /= wp.float64(geometry.vertex_count[fragment])
        inside = bool(True)
        for face in range(geometry.face_count[fragment]):
            a4 = geometry.vertices[fragment, geometry.faces[fragment, face, 0]]
            a = wp.vec3d(a4[1], a4[2], a4[3])
            normal = wp.vec3d(0.0)
            for corner in range(1, geometry.face_size[fragment, face] - 1):
                b4 = geometry.vertices[fragment, geometry.faces[fragment, face, corner]]
                c4 = geometry.vertices[fragment, geometry.faces[fragment, face, corner + 1]]
                b, c = wp.vec3d(b4[1], b4[2], b4[3]), wp.vec3d(c4[1], c4[2], c4[3])
                normal += wp.cross(b - a, c - a)
            if wp.dot(normal, a - center) < wp.float64(0.0):
                normal = -normal
            if wp.dot(normal, coordinate - a) > wp.float64(2.0e-7) * wp.length(normal):
                inside = False
        if inside:
            score = int(0)
            for corner in range(4):
                ancestor = int(fragment_nodes[fragment, corner])
                if use_lineage:
                    ancestor = int(previous_nodes[geometry.source[fragment], corner])
                if ancestor == nodes[slot, corner]:
                    score += 1
            # A point exactly on the incision belongs to just one side. Prefer
            # its existing basis, then the first fragment; never clone a hold.
            if score > best_score:
                selected, best_score = fragment, score
    if selected >= 0:
        for corner in range(4):
            nodes[slot, corner] = fragment_nodes[selected, corner]
    else:
        enabled[slot] = 0


@wp.kernel(enable_backward=False)
def _add_matrix(
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    enabled: wp.array[int],
    fractions: wp.array[float],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    scale: float,
):
    slot, row = wp.tid()
    if enabled[slot] == 0:
        return
    particle = nodes[slot, row]
    for column in range(4):
        first, end = int(offsets[particle]), int(offsets[particle + 1])
        target = int(nodes[slot, column])
        while first < end:
            middle = (first + end) // 2
            if columns[middle] < target:
                first = middle + 1
            else:
                end = middle
        if first < offsets[particle + 1] and columns[first] == target:
            stiffness = scale * fractions[slot] * weights[slot][row] * weights[slot][column]
            wp.atomic_add(values, first, wp.diag(wp.vec3(stiffness)))


@wp.kernel(enable_backward=False)
def _advance_targets(requested: wp.array[wp.vec3], driven: wp.array[wp.vec3], dt: float, speed: float):
    grip = wp.tid()
    delta = requested[grip] - driven[grip]
    distance = wp.length(delta)
    if distance > speed * dt:
        delta *= speed * dt / distance
    driven[grip] += delta


@wp.kernel(enable_backward=False)
def _add_forces(
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    enabled: wp.array[int],
    fractions: wp.array[float],
    offsets: wp.array[wp.vec3],
    driven: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    forces: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    dt: float,
):
    slot = wp.tid()
    if enabled[slot] == 0:
        return
    point = wp.vec3(0.0)
    for corner in range(4):
        point += weights[slot][corner] * q[nodes[slot, corner]]
    target = driven[slot // _SAMPLES] + offsets[slot]
    for corner in range(4):
        node = int(nodes[slot, corner])
        weight = float(weights[slot][corner]) * fractions[slot]
        # World attachments have nonzero stiffness row sums. This correction
        # completes the solver's translation-invariant elastic predictor.
        force = weight * (stiffness * (target - point) - (dt * stiffness + damping) * qd[node])
        wp.atomic_add(forces, node, force)


@wp.kernel(enable_backward=False)
def _grip_position(
    grip: int,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    fractions: wp.array[float],
    offsets: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    output: wp.array[wp.vec3],
):
    center = wp.vec3(0.0)
    for sample in range(_SAMPLES):
        slot = grip * _SAMPLES + sample
        point = wp.vec3(0.0)
        for corner in range(4):
            point += weights[slot][corner] * q[nodes[slot, corner]]
        center += fractions[slot] * (point - offsets[slot])
    output[0] = center


@wp.kernel(enable_backward=False)
def _set_target(slot: int, target: wp.vec3, initialize: int, requested: wp.array[wp.vec3], driven: wp.array[wp.vec3]):
    requested[slot] = target
    if initialize != 0:
        driven[slot] = target


@wp.kernel(enable_backward=False)
def _disable(slot: int, enabled: wp.array[int]):
    enabled[slot * _SAMPLES + wp.tid()] = 0


class MaterialHolds:
    """Finite surface grips, with one material owner per sampled anchor.

    An area-weighted 8x8 surface grid distributes each grip's spring and damper
    over existing tetrahedral sparse blocks. Targets are rate limited to avoid
    teleporting tiny elements. No state or surface arrays are downloaded while
    dragging or simulating.
    """

    def __init__(self, device, *, capacity=16, stiffness=10000.0, damping=20.0, maximum_speed=1.0, radius=0.08):
        self.device = wp.get_device(device)
        self.capacity = int(capacity)
        self.samples_per_grip = _SAMPLES
        self.anchor_capacity = self.capacity * _SAMPLES
        self.radius = float(radius)
        self.stiffness, self.damping, self.maximum_speed = float(stiffness), float(damping), float(maximum_speed)
        self.nodes = wp.zeros((self.anchor_capacity, 4), dtype=int, device=device)
        self.parents = wp.zeros(self.anchor_capacity, dtype=int, device=device)
        self.weights = wp.zeros(self.anchor_capacity, dtype=wp.vec4, device=device)
        self.enabled = wp.zeros(self.anchor_capacity, dtype=int, device=device)
        self.fractions = wp.zeros(self.anchor_capacity, dtype=float, device=device)
        self.offsets = wp.zeros(self.anchor_capacity, dtype=wp.vec3, device=device)
        self._backup_nodes = wp.empty_like(self.nodes)
        self._backup_enabled = wp.empty_like(self.enabled)
        self.requested = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self.driven = wp.zeros(capacity, dtype=wp.vec3, device=device)
        self._status = wp.zeros(2, dtype=int, device=device)
        self._frame = wp.zeros(3, dtype=wp.vec3, device=device)
        self._areas = wp.zeros(_SAMPLES, dtype=float, device=device)
        self._nearest = wp.zeros(_SAMPLES, dtype=float, device=device)
        self._triangles = wp.zeros(_SAMPLES, dtype=int, device=device)
        self._position = wp.zeros(1, dtype=wp.vec3, device=device)
        self.count, self.active = 0, None

    def begin(self, surface, geometry, triangle, barycentric, point, q):
        self.cancel()
        if self.count == self.capacity:
            return False
        grip = self.count
        self._areas.zero_()
        self._nearest.fill_(np.finfo(np.float32).max)
        self._triangles.fill_(2147483647)
        self._status.zero_()
        wp.launch(
            _patch_frame,
            dim=1,
            inputs=[triangle, surface.vertex_nodes, surface.weights, q, self._frame],
            device=self.device,
        )
        inputs = [
            triangle,
            wp.vec3(point),
            self.radius,
            surface.vertex_nodes,
            surface.weights,
            q,
            surface.components,
            surface.count,
            self._frame,
        ]
        wp.launch(
            _measure_patch,
            dim=surface.components.size,
            inputs=[*inputs, self._areas, self._nearest],
            device=self.device,
        )
        wp.launch(
            _choose_patch,
            dim=surface.components.size,
            inputs=[*inputs, self._nearest, self._triangles],
            device=self.device,
        )
        wp.launch(
            _capture_patch,
            dim=_SAMPLES,
            inputs=[
                grip,
                triangle,
                wp.vec3(barycentric),
                wp.vec3(point),
                self._triangles,
                self._areas,
                surface.vertex_nodes,
                surface.weights,
                q,
                self.parents,
                self.nodes,
                self.weights,
                self.offsets,
                self.fractions,
                self.enabled,
            ],
            device=self.device,
        )
        wp.launch(
            _capture_parents,
            dim=geometry.capacity,
            inputs=[grip, geometry.arrays, self.parents, self.nodes, self.enabled],
            device=self.device,
        )
        wp.launch(
            _check_patch, dim=_SAMPLES, inputs=[grip, self.parents, self.enabled, self._status], device=self.device
        )
        captured, errors = self._status.numpy()
        if not captured or errors:
            wp.launch(_disable, dim=_SAMPLES, inputs=[grip, self.enabled], device=self.device)
            return False
        wp.launch(_set_target, dim=1, inputs=[grip, wp.vec3(point), 1, self.requested, self.driven], device=self.device)
        self.count += 1
        self.active = grip
        return True

    def grip_position(self, q, grip=0):
        """Read one area-weighted grip center [m] for diagnostics, outside capture."""
        wp.launch(
            _grip_position,
            dim=1,
            inputs=[grip, self.nodes, self.weights, self.fractions, self.offsets, q, self._position],
            device=self.device,
        )
        return self._position.numpy()[0].copy()

    def move(self, point):
        if self.active is not None:
            point = np.asarray(point, dtype=np.float32)
            if point.shape != (3,) or not np.isfinite(point).all():
                raise ValueError("Hold target must be a finite world point")
            wp.launch(
                _set_target,
                dim=1,
                inputs=[self.active, wp.vec3(point), 0, self.requested, self.driven],
                device=self.device,
            )

    def finish(self):
        self.active = None

    def cancel(self):
        if self.active is not None:
            wp.launch(_disable, dim=_SAMPLES, inputs=[self.active, self.enabled], device=self.device)
            self.count -= 1
            self.active = None

    def remove(self, index: int) -> bool:
        """Release one finished grip, replacing its slot with the last grip."""
        if self.active is not None or not 0 <= index < self.count:
            return False
        last = self.count - 1
        if index != last:
            for values in (
                self.nodes,
                self.parents,
                self.weights,
                self.enabled,
                self.fractions,
                self.offsets,
                self._backup_nodes,
                self._backup_enabled,
            ):
                width = 4 if values.ndim == 2 else 1
                array = values.flatten() if values.ndim == 2 else values
                size = _SAMPLES * width
                wp.copy(array, array, dest_offset=index * size, src_offset=last * size, count=size)
            for values in (self.requested, self.driven):
                wp.copy(values, values, dest_offset=index, src_offset=last, count=1)
        for enabled in (self.enabled, self._backup_enabled):
            wp.launch(_disable, dim=_SAMPLES, inputs=[last, enabled], device=self.device)
        self.count = last
        return True

    def clear(self):
        self.enabled.zero_()
        self.count, self.active = 0, None

    def remap(self, geometry, fragment_nodes=None, previous=None):
        wp.launch(
            _remap,
            dim=self.anchor_capacity,
            inputs=[
                geometry.arrays,
                geometry.nodeindices if fragment_nodes is None else fragment_nodes,
                geometry.nodeindices if previous is None else previous.nodeindices,
                previous is not None,
                self.parents,
                self.nodes,
                self.weights,
                self.enabled,
            ],
            device=self.device,
        )

    def stage_remap(self, geometry, fragment_nodes, previous):
        """Stage candidate ownership while retaining the accepted attachment rows."""
        wp.copy(self._backup_nodes, self.nodes)
        wp.copy(self._backup_enabled, self.enabled)
        self.remap(geometry, fragment_nodes, previous)

    def restore(self):
        wp.copy(self.nodes, self._backup_nodes)
        wp.copy(self.enabled, self._backup_enabled)

    def add_matrix(self, solver, dt):
        """Add attachment tangent after the solver refreshes its element matrix."""
        matrix = solver.A_bsr
        wp.launch(
            _add_matrix,
            dim=(self.anchor_capacity, 4),
            inputs=[
                self.nodes,
                self.weights,
                self.enabled,
                self.fractions,
                matrix.offsets,
                matrix.columns,
                matrix.values,
                dt * dt * self.stiffness + dt * self.damping,
            ],
            device=self.device,
        )
        # Refresh in place so the solver's captured block-Jacobi operator sees
        # these additional diagonal blocks without replacing its device arrays.
        solver.refresh_preconditioner()

    def add_forces(self, q, qd, forces, dt):
        wp.launch(
            _advance_targets,
            dim=self.capacity,
            inputs=[self.requested, self.driven, dt, self.maximum_speed],
            device=self.device,
        )
        wp.launch(
            _add_forces,
            dim=self.anchor_capacity,
            inputs=[
                self.nodes,
                self.weights,
                self.enabled,
                self.fractions,
                self.offsets,
                self.driven,
                q,
                qd,
                forces,
                self.stiffness,
                self.damping,
                dt,
            ],
            device=self.device,
        )

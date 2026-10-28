# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Advance a removal tool between boundaries of the current parent tetrahedra.

These virtual cell boundaries schedule edits; they are not collision geometry.
The spherical cutter still subtracts only its swept volume from actual material.
"""

import numpy as np
import warp as wp

from .fragment_device import FragmentArrays


@wp.func
def _ray_coordinates(nodes: wp.vec4i, positions: wp.array[wp.vec3], origin: wp.vec3d, direction: wp.vec3d):
    a = wp.vec3d(positions[nodes[0]])
    b, c, d = wp.vec3d(positions[nodes[1]]) - a, wp.vec3d(positions[nodes[2]]) - a, wp.vec3d(positions[nodes[3]]) - a
    determinant = wp.dot(b, wp.cross(c, d))
    scale = wp.max(wp.length(b), wp.max(wp.length(c), wp.length(d)))
    valid = wp.isfinite(determinant) and wp.abs(determinant) > wp.float64(1.0e-14) * scale * scale * scale
    value, slope = wp.vec4d(0.0), wp.vec4d(0.0)
    if valid:
        relative = origin - a
        value[1] = wp.dot(relative, wp.cross(c, d)) / determinant
        value[2] = wp.dot(relative, wp.cross(d, b)) / determinant
        value[3] = wp.dot(relative, wp.cross(b, c)) / determinant
        slope[1] = wp.dot(direction, wp.cross(c, d)) / determinant
        slope[2] = wp.dot(direction, wp.cross(d, b)) / determinant
        slope[3] = wp.dot(direction, wp.cross(b, c)) / determinant
        value[0] = wp.float64(1.0) - value[1] - value[2] - value[3]
        slope[0] = -slope[1] - slope[2] - slope[3]
    return value, slope, valid


@wp.kernel(enable_backward=False)
def _reset(best: wp.array[wp.float64], selected: wp.array[int], face: wp.array[int]):
    best[0] = wp.float64(1.0e30)
    selected[0] = 2147483647
    face[0] = -1


@wp.kernel(enable_backward=False)
def _scan(
    geometry: FragmentArrays,
    positions: wp.array[wp.vec3],
    origin: wp.array[wp.vec3d],
    direction: wp.array[wp.vec3d],
    epsilon: wp.float64,
    distances: wp.array[wp.float64],
    faces: wp.array[int],
    best: wp.array[wp.float64],
):
    fragment = wp.tid()
    distances[fragment] = wp.float64(1.0e30)
    faces[fragment] = -1
    if fragment >= geometry.count[0] or geometry.error[0] != 0:
        return
    nodes = wp.vec4i(0)
    for i in range(4):
        nodes[i] = geometry.nodeindices[fragment, i]
    value, slope, valid = _ray_coordinates(nodes, positions, origin[0], direction[0])
    if not valid:
        return
    lower, upper = wp.float64(-1.0e30), wp.float64(1.0e30)
    entry, exit_face = int(-1), int(-1)
    for i in range(4):
        if wp.abs(slope[i]) <= wp.float64(1.0e-12):
            if value[i] < wp.float64(-1.0e-8):
                return
        else:
            distance = -value[i] / slope[i]
            if slope[i] > wp.float64(0.0) and distance > lower:
                lower, entry = distance, i
            elif slope[i] < wp.float64(0.0) and distance < upper:
                upper, exit_face = distance, i
    if lower > upper + epsilon or upper <= epsilon:
        return
    distance, face = upper, exit_face
    if lower > epsilon:
        distance, face = lower, entry
    if face >= 0 and wp.isfinite(distance):
        distances[fragment] = distance
        faces[fragment] = face
        wp.atomic_min(best, 0, distance)


@wp.kernel(enable_backward=False)
def _select(distances: wp.array[wp.float64], best: wp.array[wp.float64], selected: wp.array[int]):
    fragment = wp.tid()
    if distances[fragment] == best[0] and best[0] < wp.float64(1.0e29):
        wp.atomic_min(selected, 0, fragment)


@wp.kernel(enable_backward=False)
def _store(
    geometry: FragmentArrays,
    selected: wp.array[int],
    faces: wp.array[int],
    nodes: wp.array[wp.vec4i],
    face: wp.array[int],
):
    fragment = selected[0]
    if fragment < geometry.count[0]:
        result = wp.vec4i(0)
        for i in range(4):
            result[i] = geometry.nodeindices[fragment, i]
        nodes[0] = result
        face[0] = faces[fragment]


@wp.kernel(enable_backward=False)
def _follow(
    positions: wp.array[wp.vec3],
    nodes: wp.array[wp.vec4i],
    face: wp.array[int],
    origin: wp.array[wp.vec3d],
    direction: wp.array[wp.vec3d],
    result: wp.array[wp.float64],
):
    result[0] = wp.float64(1.0e30)
    index = face[0]
    if index < 0:
        return
    value, slope, valid = _ray_coordinates(nodes[0], positions, origin[0], direction[0])
    if valid and wp.abs(slope[index]) > wp.float64(1.0e-12):
        distance = -value[index] / slope[index]
        point = value + distance * slope
        if wp.min(point) >= wp.float64(-1.0e-7) and wp.isfinite(distance):
            result[0] = distance


class TetrahedralTunnelGuide:
    """GPU cell-boundary queries with a four-node guide between edits.

    ``next_boundary`` returns the nearest boundary strictly beyond ``epsilon``
    meters. ``distance`` follows that same moving face and returns a signed
    distance, so crossing the face is observable instead of selecting a new one.
    """

    def __init__(self, device=None, *, epsilon=1.0e-6):
        if not np.isfinite(epsilon) or epsilon <= 0:
            raise ValueError("Guide epsilon must be positive and finite")
        self.device, self.epsilon = wp.get_device(device), float(epsilon)
        self.block_dim = 256 if self.device.is_cuda else 1
        self.origin, self.direction = (wp.zeros(1, dtype=wp.vec3d, device=self.device) for _ in range(2))
        self._cached_origin = self._cached_direction = None
        self.result = wp.zeros(1, dtype=wp.float64, device=self.device)
        self.selected, self.face = (wp.zeros(1, dtype=int, device=self.device) for _ in range(2))
        self.nodes = wp.zeros(1, dtype=wp.vec4i, device=self.device)
        self.face.assign([-1])
        self._distances = self._faces = None
        # Compile during scene setup, before the first interactive stroke.
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def _ray(self, origin, direction):
        origin, direction = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        if (
            origin.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite(origin).all()
            or not np.isfinite(direction).all()
        ):
            raise ValueError("Guide ray must contain finite three-dimensional coordinates")
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Guide direction must have positive finite length")
        direction = direction / length
        # A held stroke keeps the ray fixed while its selected face moves.
        # Avoid uploading those same constants on every physics frame.
        if self._cached_origin is None or not np.array_equal(origin, self._cached_origin):
            self.origin.assign(origin.reshape(1, 3))
            self._cached_origin = origin.copy()
        if self._cached_direction is None or not np.array_equal(direction, self._cached_direction):
            self.direction.assign(direction.reshape(1, 3))
            self._cached_direction = direction.copy()

    def _run(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device, block_dim=self.block_dim)

    def _result(self):
        value = float(self.result.numpy()[0])
        return value if np.isfinite(value) and abs(value) < 1.0e29 else None

    def next_boundary(self, geometry, positions, origin, direction):
        """Select the nearest positive current parent-cell boundary [m]."""
        geometry = geometry.arrays if hasattr(geometry, "arrays") else geometry
        self._ray(origin, direction)
        if self._distances is None or len(self._distances) != geometry.capacity:
            self._distances = wp.empty(geometry.capacity, dtype=wp.float64, device=self.device)
            self._faces = wp.empty(geometry.capacity, dtype=int, device=self.device)
        self._run(_reset, 1, [self.result, self.selected, self.face])
        self._run(
            _scan,
            geometry.capacity,
            [geometry, positions, self.origin, self.direction, self.epsilon, self._distances, self._faces, self.result],
        )
        self._run(_select, geometry.capacity, [self._distances, self.result, self.selected])
        self._run(_store, 1, [geometry, self.selected, self._faces, self.nodes, self.face])
        return self._result()

    def distance(self, positions, origin, direction):
        """Return signed distance [m] to the stored face after deformation."""
        self._ray(origin, direction)
        self._run(_follow, 1, [positions, self.nodes, self.face, self.origin, self.direction, self.result])
        return self._result()

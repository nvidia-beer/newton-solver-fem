# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Pick the frontmost visible material triangle with a small GPU readback."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _intersect(
    points: wp.array[wp.vec3],
    indices: wp.array[int],
    origin: wp.vec3,
    direction: wp.vec3,
    offset: int,
    hits: wp.array[wp.vec3],
    nearest: wp.array[float],
):
    face = wp.tid()
    a = points[indices[3 * face]]
    edge1 = points[indices[3 * face + 1]] - a
    edge2 = points[indices[3 * face + 2]] - a
    cross = wp.cross(direction, edge2)
    determinant = wp.dot(edge1, cross)
    hit = wp.vec3(1.0e30, 0.0, 0.0)
    # Two-sided picking includes newly exposed incision faces.
    if wp.abs(determinant) > 1.0e-8 * wp.length(edge1) * wp.length(edge2):
        relative = origin - a
        u = wp.dot(relative, cross) / determinant
        q = wp.cross(relative, edge1)
        v = wp.dot(direction, q) / determinant
        t = wp.dot(edge2, q) / determinant
        if u >= 0.0 and v >= 0.0 and u + v <= 1.0 and t > 0.0:
            hit = wp.vec3(t, u, v)
            wp.atomic_min(nearest, 0, t)
    hits[offset + face] = hit


@wp.kernel(enable_backward=False)
def _select(hits: wp.array[wp.vec3], nearest: wp.array[float], winner: wp.array[int]):
    face = wp.tid()
    if hits[face][0] == nearest[0] and nearest[0] < 1.0e30:
        # Coincident faces choose one deterministic material support.
        wp.atomic_min(winner, 0, face)


@wp.kernel(enable_backward=False)
def _read_hit(hits: wp.array[wp.vec3], winner: wp.array[int], result: wp.array[float]):
    face = winner[0]
    result[0] = float(face)
    if face >= 0 and face < hits.shape[0]:
        hit = hits[face]
        result[1] = hit[0]
        result[2] = hit[1]
        result[3] = hit[2]


class SurfacePicker:
    """Raycast current triangles on click; keep simulation coordinates on device."""

    def __init__(self, device):
        self.device = wp.get_device(device)
        self._hits = wp.empty(0, dtype=wp.vec3, device=self.device)
        self._nearest = wp.empty(1, dtype=float, device=self.device)
        self._winner = wp.empty(1, dtype=int, device=self.device)
        self._result = wp.zeros(4, dtype=float, device=self.device)

    def pick(self, origin, direction, surfaces):
        """Return (surface, triangle, triangle weights, world point), or None.

        ``surfaces`` contains ``(points, flat_indices, triangle_count)`` tuples.
        Coordinates and ray origin are in meters. The direction is normalized
        here. Only four scalar hit values return to the CPU, once per press.
        """
        origin, direction = np.asarray(origin, dtype=float), np.asarray(direction, dtype=float)
        if origin.shape != (3,) or direction.shape != (3,) or not np.isfinite([origin, direction]).all():
            raise ValueError("Picking requires a finite three-dimensional ray")
        length = np.linalg.norm(direction)
        if length == 0.0:
            raise ValueError("Picking requires a nonzero ray direction")
        direction = direction / length
        sizes = np.asarray([int(surface[2]) for surface in surfaces], dtype=np.int64)
        if any(size < 0 or 3 * size > indices.size for (_, indices, _), size in zip(surfaces, sizes, strict=True)):
            raise ValueError("Picking triangle counts must fit their index buffers")
        count = int(sizes.sum())
        if count == 0:
            return None
        if self._hits.size != count:
            self._hits = wp.empty(count, dtype=wp.vec3, device=self.device)
        self._nearest.fill_(1.0e30)
        self._winner.fill_(count)
        offset = 0
        for (points, indices, _), size in zip(surfaces, sizes, strict=True):
            if size:
                wp.launch(
                    _intersect,
                    dim=int(size),
                    inputs=[points, indices, wp.vec3(*origin), wp.vec3(*direction), offset, self._hits, self._nearest],
                    device=self.device,
                )
            offset += int(size)
        wp.launch(_select, dim=count, inputs=[self._hits, self._nearest, self._winner], device=self.device)
        wp.launch(_read_hit, dim=1, inputs=[self._hits, self._winner, self._result], device=self.device)
        face, distance, u, v = self._result.numpy()
        if int(face) == count:
            return None
        ends = sizes.cumsum()
        surface = int(np.searchsorted(ends, int(face), side="right"))
        local_face = int(face) - (0 if surface == 0 else int(ends[surface - 1]))
        return surface, local_face, np.asarray((1.0 - u - v, u, v)), origin + float(distance) * direction

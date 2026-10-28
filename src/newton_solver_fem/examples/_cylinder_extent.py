# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Conservative through-hole endpoints from the current material boundary."""

import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _reset(result: wp.array[wp.float64]):
    result[0] = wp.float64(1.0e300)
    result[1] = wp.float64(-1.0e300)
    result[2] = wp.float64(1.0e300)
    result[3] = wp.float64(0.0)


@wp.kernel(enable_backward=False)
def _extent(
    points: wp.array[wp.vec3],
    origin: wp.vec3d,
    direction: wp.vec3d,
    result: wp.array[wp.float64],
):
    triangle = wp.tid()
    a = wp.vec3d(points[3 * triangle])
    b = wp.vec3d(points[3 * triangle + 1])
    c = wp.vec3d(points[3 * triangle + 2])
    sa, sb, sc = wp.dot(a - origin, direction), wp.dot(b - origin, direction), wp.dot(c - origin, direction)
    if not wp.isfinite(sa) or not wp.isfinite(sb) or not wp.isfinite(sc):
        wp.atomic_max(result, 3, wp.float64(1.0))
        return
    wp.atomic_min(result, 0, wp.min(sa, wp.min(sb, sc)))
    wp.atomic_max(result, 1, wp.max(sa, wp.max(sb, sc)))

    edge1, edge2 = b - a, c - a
    cross = wp.cross(direction, edge2)
    determinant = wp.dot(edge1, cross)
    scale = wp.length(edge1) * wp.length(edge2)
    if scale <= wp.float64(0.0) or wp.abs(determinant) <= wp.float64(1.0e-12) * scale:
        return
    relative = origin - a
    u = wp.dot(relative, cross) / determinant
    q = wp.cross(relative, edge1)
    v = wp.dot(direction, q) / determinant
    distance = wp.dot(edge2, q) / determinant
    # Shared edges and either winding are valid hits; coplanar rays are not.
    epsilon = wp.float64(1.0e-10)
    if u >= -epsilon and v >= -epsilon and u + v <= wp.float64(1.0) + epsilon and distance >= wp.float64(0.0):
        wp.atomic_min(result, 2, distance)


class CylinderExtent:
    """Find safe flat-cap positions using four scalar GPU reductions.

    The center ray must hit a current surface triangle. Endpoints bound the
    projection of every active boundary vertex, including detached pieces.
    Bounding all vertices instead of only the center-ray intersections clears
    oblique faces across the full cylinder cross-section. Empty space within
    these conservative endpoints does not add material to the cut.
    """

    def __init__(self, device=None):
        self.device = wp.get_device(device)
        self.block_dim = 256 if self.device.is_cuda else 1
        self._result = wp.empty(4, dtype=wp.float64, device=self.device)
        wp.load_module(module=__name__, device=self.device, block_dim=self.block_dim)

    def query(self, points, triangle_count, origin, direction, radius):
        """Return world-space ``(start, end)`` [m], or ``None`` for a missed ray.

        ``points`` contains three consecutive vertices per triangle. Only the
        first ``triangle_count`` triangles are active. The direction need not
        be normalized; radius [m] must be finite and positive. No endpoint lies
        behind the ray origin. Only four scalar values return from the device.
        """
        origin, direction = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
        if (
            origin.shape != (3,)
            or direction.shape != (3,)
            or not np.isfinite(origin).all()
            or not np.isfinite(direction).all()
        ):
            raise ValueError("Cylinder ray must contain finite three-dimensional coordinates")
        length = np.linalg.norm(direction)
        if not np.isfinite(length) or length <= 0:
            raise ValueError("Cylinder ray direction must have positive finite length")
        if not np.isfinite(radius) or radius <= 0:
            raise ValueError("Cylinder radius must be positive and finite")
        if points.device != self.device or points.dtype != wp.vec3 or points.ndim != 1:
            raise ValueError("Cylinder surface must be a vec3 array on the query device")
        if (
            not np.isfinite(triangle_count)
            or int(triangle_count) != triangle_count
            or not 0 <= triangle_count <= points.size // 3
        ):
            raise ValueError("Active triangle count must fit the cylinder surface buffer")
        if triangle_count == 0:
            return None
        direction = direction / length
        wp.launch(_reset, dim=1, inputs=[self._result], device=self.device, block_dim=self.block_dim)
        wp.launch(
            _extent,
            dim=int(triangle_count),
            inputs=[points, wp.vec3d(*origin), wp.vec3d(*direction), self._result],
            device=self.device,
            block_dim=self.block_dim,
        )
        lower, upper, hit, invalid = self._result.numpy()
        if invalid:
            raise ValueError("Cylinder surface contains nonfinite active coordinates")
        if hit == 1.0e300 or upper <= 0.0:
            return None
        # Float32 surface coordinates can lose precision far from the origin.
        coordinate_scale = max(np.max(np.abs(origin)), abs(lower), abs(upper), radius)
        margin = max(1.0e-6, 1.0e-3 * radius, 8.0 * np.finfo(np.float32).eps * coordinate_scale)
        start = origin + max(0.0, lower - margin) * direction
        end = origin + (upper + margin) * direction
        return start, end

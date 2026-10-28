# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Flat-ended cylindrical removal with a fixed, inscribed 16-sided cross section."""

import numpy as np

CYLINDER_SIDES = 16


def cylindrical_cut_planes(start, end, radius):
    """Return the cylinder's outward unit planes in start/radius coordinates.

    The regular polygon is inscribed in the requested circular cross section:
    no removed point lies outside the selected radius. Its two flat end planes
    are exactly at ``start`` and ``end``. This computes tool geometry only; the
    current tissue is intersected and clipped on the GPU.
    """
    start, end = np.asarray(start, dtype=np.float64), np.asarray(end, dtype=np.float64)
    if (
        start.shape != (3,)
        or end.shape != (3,)
        or not np.isfinite(start).all()
        or not np.isfinite(end).all()
        or not np.isscalar(radius)
        or not np.isfinite(radius)
        or radius <= 0
    ):
        raise ValueError("Cylinder endpoints must be finite and radius must be positive")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        delta = (end - start) / radius
        length = float(np.hypot.reduce(delta))
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Cylinder length must be positive and finite relative to its radius")
    axis = delta / length
    reference = np.eye(3)[np.argmin(np.abs(axis))]
    first = np.cross(axis, reference)
    first /= np.linalg.norm(first)
    second = np.cross(axis, first)
    angles = np.arange(CYLINDER_SIDES, dtype=np.float64) * (2.0 * np.pi / CYLINDER_SIDES)
    normals = np.cos(angles)[:, None] * first + np.sin(angles)[:, None] * second
    planes = np.empty((CYLINDER_SIDES + 2, 4), dtype=np.float64)
    planes[:CYLINDER_SIDES, :3] = normals
    planes[:CYLINDER_SIDES, 3] = np.cos(np.pi / CYLINDER_SIDES)
    planes[-2], planes[-1] = (*-axis, 0.0), (*axis, length)
    return planes

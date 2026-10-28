# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Velocity predictor for one linearly implicit elastic step."""

import warp as wp


@wp.kernel(enable_backward=False)
def subtract_elastic_predictor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    velocity: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    node = wp.tid()
    correction = wp.vec3(0.0)
    for block in range(offsets[node], offsets[node + 1]):
        # Element stiffness has zero block-row sum. Differences cancel the
        # mass diagonal exactly and preserve uniform translations in float32.
        correction += values[block] * (velocity[columns[block]] - velocity[node])
    rhs[node] -= correction


@wp.kernel(enable_backward=False)
def restore_spring_damping_predictor(
    indices: wp.array[int],
    damping: wp.array[float],
    dt: float,
    velocity: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    spring = wp.tid()
    a, b = indices[2 * spring], indices[2 * spring + 1]
    if a >= 0 and b >= 0:
        # The matrix contains h*C as well as h²*K, but only the latter enters
        # the position predictor; velocity damping is already in h*f.
        correction = dt * damping[spring] * (velocity[a] - velocity[b])
        wp.atomic_add(rhs, a, correction)
        wp.atomic_sub(rhs, b, correction)

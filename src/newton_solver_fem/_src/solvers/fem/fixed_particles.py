# SPDX-License-Identifier: Apache-2.0

"""Stationary Dirichlet boundaries without changing the physical stiffness."""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator


@wp.kernel(enable_backward=False)
def _restore(indices: wp.array[int], reference: wp.array[wp.vec3], q: wp.array[wp.vec3], v: wp.array[wp.vec3]):
    node = indices[wp.tid()]
    q[node] = reference[node]
    v[node] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _restore_positions(indices: wp.array[int], reference: wp.array[wp.vec3], q: wp.array[wp.vec3]):
    node = indices[wp.tid()]
    q[node] = reference[node]


@wp.kernel(enable_backward=False)
def _zero_fixed(indices: wp.array[int], values: wp.array[wp.vec3]):
    values[indices[wp.tid()]] = wp.vec3(0.0)


@wp.kernel(enable_backward=False)
def _projected_matvec(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    x: wp.array[wp.vec3],
    y: wp.array[wp.vec3],
    z: wp.array[wp.vec3],
    alpha: float,
    beta: float,
):
    row = wp.tid()
    result = wp.vec3(0.0)
    if fixed[row] != 0:
        result = x[row]
    else:
        for block in range(offsets[row], offsets[row + 1]):
            column = columns[block]
            if fixed[column] == 0:
                result += values[block] * x[column]
    result *= alpha
    if beta != 0.0:
        result += beta * y[row]
    z[row] = result


class FixedParticles:
    """Solve (P A P + I-P) dv = P b, retaining A for the elastic predictor.

    P is the diagonal projector onto free nodes. The original matrix must keep
    its constrained columns: the predictor's block-row differences need them
    even when the prescribed boundary velocity is zero.
    """

    def __init__(self, model, matrix, indices):
        self.device = model.device
        self.indices = wp.array(indices, dtype=int, device=self.device)
        mask = np.zeros(model.particle_count, dtype=np.int32)
        mask[indices] = 1
        self.mask = wp.array(mask, dtype=int, device=self.device)
        self.reference = wp.clone(model.particle_q)

        def matvec(x, y, z, alpha, beta):
            wp.launch(
                _projected_matvec,
                model.particle_count,
                inputs=[self.mask, matrix.offsets, matrix.columns, matrix.values, x, y, z, alpha, beta],
                device=self.device,
            )

        self.operator = LinearOperator((model.particle_count, model.particle_count), wp.vec3, self.device, matvec)

    def restore(self, state):
        wp.launch(
            _restore,
            self.indices.size,
            inputs=[self.indices, self.reference, state.particle_q, state.particle_qd],
            device=self.device,
        )

    def project_rhs(self, rhs):
        wp.launch(_zero_fixed, self.indices.size, inputs=[self.indices, rhs], device=self.device)

    def restore_positions(self, positions):
        wp.launch(
            _restore_positions,
            self.indices.size,
            inputs=[self.indices, self.reference, positions],
            device=self.device,
        )

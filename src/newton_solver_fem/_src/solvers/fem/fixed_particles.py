# SPDX-License-Identifier: Apache-2.0

"""Stationary Dirichlet boundaries without changing the physical stiffness."""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, aslinearoperator
from warp.sparse import bsr_copy


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


@wp.kernel(enable_backward=False)
def _project_values(
    fixed: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    projected: wp.array[wp.mat33],
):
    row = wp.tid()
    for block in range(offsets[row], offsets[row + 1]):
        column = columns[block]
        value = wp.mat33(0.0)
        if fixed[row] != 0:
            if row == column:
                value = wp.identity(n=3, dtype=float)
        elif fixed[column] == 0:
            value = values[block]
        projected[block] = value


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
        self._masked_operator = self.operator
        self._projected = None
        self._projected_key = None

    def prepare(self, matrix):
        """Refresh the derived operator once per solve; retain the predictor's A."""
        if not self.device.is_cuda:
            return False
        key = (
            matrix.nrow,
            matrix.ncol,
            matrix.nnz,
            matrix.offsets.ptr,
            matrix.columns.ptr,
            matrix.values.ptr,
            matrix.values.shape,
            None if matrix.row_counts is None else matrix.row_counts.ptr,
        )
        changed = key != self._projected_key
        if changed:
            if self.device.is_capturing:
                raise RuntimeError("Prepare the fixed-particle operator before CUDA graph capture")
            # A diagonal is needed to represent the constrained identity using
            # the same sparse structure. Unusual empty rows retain the old path.
            offsets, columns = matrix.offsets.numpy(), matrix.columns.numpy()
            indices = self.indices.numpy()
            supported = matrix.row_counts is None and all(
                np.any(columns[offsets[row] : offsets[row + 1]] == row) for row in indices
            )
            self._projected = bsr_copy(matrix, structure_only=True) if supported else None
            if self._projected is not None:
                # Retain source capacity for in-place topology updates.
                self._projected.values = wp.empty_like(matrix.values)
                # Share topology so in-place connectivity changes remain visible.
                self._projected.offsets = matrix.offsets
                self._projected.columns = matrix.columns
                self._projected.nnz_sync()
                self.operator = aslinearoperator(self._projected)
            else:
                self.operator = self._masked_operator
            self._projected_key = key
        if self._projected is not None:
            wp.launch(
                _project_values,
                matrix.nrow,
                [self.mask, matrix.offsets, matrix.columns, matrix.values, self._projected.values],
                device=self.device,
            )
        return changed

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

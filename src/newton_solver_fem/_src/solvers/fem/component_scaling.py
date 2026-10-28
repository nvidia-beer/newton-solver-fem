# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Balance disconnected mechanical systems without changing physical masses."""

import warp as wp
from warp.optim.linear import LinearOperator
from warp.sparse import bsr_mv, bsr_zeros

from .kernels import block_jacobi_mv_kernel


@wp.kernel(enable_backward=False)
def _sum_mass(labels: wp.array[int], mass: wp.array[float], totals: wp.array[wp.float64]):
    node = wp.tid()
    component = labels[node]
    if component >= 0 and component < totals.shape[0]:
        wp.atomic_add(totals, component, wp.float64(mass[node]))


@wp.kernel(enable_backward=False)
def _row_scales(labels: wp.array[int], totals: wp.array[wp.float64], scale: wp.array[float]):
    node = wp.tid()
    component = labels[node]
    value = float(1.0)
    if component >= 0 and component < totals.shape[0]:
        if totals[component] > wp.float64(0.0):
            value = float(wp.float64(1.0) / totals[component])
    scale[node] = value


@wp.kernel(enable_backward=False)
def _scale_system(
    offsets: wp.array[int],
    values: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
    rhs: wp.array[wp.vec3],
    scale: wp.array[float],
    scaled_values: wp.array[wp.mat33],
    scaled_inverse: wp.array[wp.mat33],
    scaled_rhs: wp.array[wp.vec3],
):
    row = wp.tid()
    factor = scale[row]
    for slot in range(offsets[row], offsets[row + 1]):
        scaled_values[slot] = factor * values[slot]
    scaled_inverse[row] = inverse[row] / factor
    scaled_rhs[row] = factor * rhs[row]


class ComponentScaling:
    """Scale each disconnected block's equations by its inverse total mass.

    The factor is uniform inside a connected block, so symmetry and the exact
    physical solution are unchanged. Unlike a coordinate-only transformation,
    this balances the blocks' contribution to finite-budget PCG convergence.
    Numerical matrix/preconditioner copies are prepared once per solve; Krylov
    iterations retain the ordinary sparse products and launch count.
    """

    def __init__(self, model, matrix, labels, capacity):
        self.model, self.labels = model, labels
        self.total_mass = wp.zeros(capacity, dtype=wp.float64, device=model.device)
        self.scale = wp.ones(model.particle_count, dtype=float, device=model.device)
        self.rhs = wp.empty(model.particle_count, dtype=wp.vec3, device=model.device)
        self.inverse = wp.empty(model.particle_count, dtype=wp.mat33, device=model.device)
        self.matrix = bsr_zeros(
            matrix.nrow, matrix.ncol, wp.mat33, device=model.device, nnz_capacity=matrix.values.size
        )
        # Like streaming assembly, these products use live device row offsets,
        # never a host-side nnz value. Connectivity storage remains shared.
        self.matrix.offsets, self.matrix.columns = matrix.offsets, matrix.columns

        def matvec(x, y, z, alpha, beta):
            if beta != 0.0 and y.ptr != z.ptr:
                wp.copy(z, y)
            bsr_mv(self.matrix, x, z, alpha=alpha, beta=beta, tile_size=-1)

        def precondition(x, y, z, alpha, beta):
            wp.launch(
                block_jacobi_mv_kernel,
                model.particle_count,
                inputs=[self.inverse, x, y, z, wp.float32(alpha), wp.float32(beta)],
                device=model.device,
            )

        shape = (model.particle_count, model.particle_count)
        self.operator = LinearOperator(shape, wp.vec3, model.device, matvec)
        self.preconditioner = LinearOperator(shape, wp.vec3, model.device, precondition)
        self.refresh()

    def refresh(self):
        self.total_mass.zero_()
        wp.launch(
            _sum_mass,
            self.model.particle_count,
            inputs=[self.labels, self.model.particle_mass, self.total_mass],
            device=self.model.device,
        )
        wp.launch(
            _row_scales,
            self.model.particle_count,
            inputs=[self.labels, self.total_mass, self.scale],
            device=self.model.device,
        )

    def prepare(self, matrix, inverse, rhs):
        wp.launch(
            _scale_system,
            self.model.particle_count,
            inputs=[
                matrix.offsets,
                matrix.values,
                inverse,
                rhs,
                self.scale,
                self.matrix.values,
                self.inverse,
                self.rhs,
            ],
            device=self.model.device,
        )

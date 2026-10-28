# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Owned rollout data and checked sparse adjoints of the production cached system.

The matrix is a fixed input to each recorded linearization. This differentiates
A dv = h f(q,v,a) - h² K v, not a nonlinear equilibrium or truncated Krylov
iteration. Reassembly within a tape is rejected by the solver. Matrix revisions
are shared by recorded substeps, while forces and state copies are owned per step.
"""

from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator, cg, preconditioner
from warp.sparse import bsr_copy, bsr_transposed

from .kernels import (
    block_jacobi_mv_kernel,
    invert_block_diagonal_kernel,
    maximum_rhs_component,
    normalize_linear_rhs,
    restore_linear_increment,
    update_state,
)


def active_tape():
    # Warp 1.17 has public Tape.record_func but no active-tape/suspend accessor.
    return wp._src.context.runtime.tape


@contextmanager
def without_tape():
    tape = active_tape()
    wp._src.context.runtime.tape = None
    try:
        yield tape
    finally:
        wp._src.context.runtime.tape = tape


@wp.kernel
def copy_vectors(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] = source[i]


@wp.kernel
def copy_scalars(source: wp.array[float], target: wp.array[float]):
    i = wp.tid()
    target[i] = source[i]


@wp.kernel
def accumulate_vectors(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    i = wp.tid()
    target[i] += source[i]


@wp.kernel
def sum_forces(
    spring: wp.array[wp.vec3],
    tet: wp.array[wp.vec3],
    hexa: wp.array[wp.vec3],
    tri: wp.array[wp.vec3],
    ground: wp.array[wp.vec3],
    gravity: wp.array[wp.vec3],
    damping: wp.array[wp.vec3],
    external: wp.array[wp.vec3],
    dt: float,
    force: wp.array[wp.vec3],
):
    i = wp.tid()
    value = dt * spring[i]
    value += dt * tet[i]
    value += dt * hexa[i]
    value += dt * tri[i]
    value += dt * ground[i]
    value += dt * gravity[i]
    value += dt * damping[i]
    force[i] = value + dt * external[i]


@wp.kernel
def form_rhs(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    velocity: wp.array[wp.vec3],
    force: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    row = wp.tid()
    correction = wp.vec3(0.0)
    for block in range(offsets[row], offsets[row + 1]):
        correction += values[block] * (velocity[columns[block]] - velocity[row])
    rhs[row] = force[row] - correction


@wp.kernel
def spring_predictor(
    indices: wp.array[int],
    damping: wp.array[float],
    dt: float,
    velocity: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    spring = wp.tid()
    a, b = indices[2 * spring], indices[2 * spring + 1]
    if a >= 0 and b >= 0:
        correction = dt * damping[spring] * (velocity[a] - velocity[b])
        wp.atomic_add(rhs, a, correction)
        wp.atomic_sub(rhs, b, correction)


@wp.kernel(enable_backward=False)
def residual_norms(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    solution: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
    norms: wp.array[wp.float64],
):
    row = wp.tid()
    ax = wp.vec3(0.0)
    for block in range(offsets[row], offsets[row + 1]):
        ax += values[block] * solution[columns[block]]
    residual, value = wp.vec3d(ax - rhs[row]), wp.vec3d(rhs[row])
    wp.atomic_add(norms, 0, wp.dot(residual, residual))
    wp.atomic_add(norms, 1, wp.dot(value, value))


@wp.kernel(enable_backward=False)
def record_residual(norms: wp.array[wp.float64], kind: int, stats: wp.array[wp.float64]):
    residual, scale = wp.sqrt(norms[0]), wp.sqrt(norms[1])
    ratio = residual / wp.max(scale, wp.float64(1.0e-30))
    budget = wp.float64(2.0e-5) * scale + wp.float64(1.0e-10)
    violation = residual / budget
    if not wp.isfinite(residual) or not wp.isfinite(scale):
        ratio = wp.float64(1.0e30)
        violation = wp.float64(1.0e30)
    wp.atomic_max(stats, kind, ratio)
    wp.atomic_max(stats, 2 + kind, violation)


class MatrixSnapshot:
    """One immutable BSR revision and reusable untaped Krylov workspaces."""

    def __init__(self, solver):
        self.revision = solver._differentiation_revision
        self.matrix = bsr_copy(solver.A_bsr)
        self.transpose = bsr_transposed(self.matrix)
        self.n, self.device = solver.model.particle_count, solver.model.device
        self.rhs = wp.empty(self.n, dtype=wp.vec3, device=self.device)
        self.solution = wp.zeros_like(self.rhs)
        self.maximum = wp.zeros(1, dtype=float, device=self.device)
        self.norms = wp.zeros(2, dtype=wp.float64, device=self.device)
        self.stats = solver._differentiation_stats
        self.forward_solver, self.backward_solver = [
            cg(
                matrix,
                self.rhs,
                self.solution,
                tol=1.0e-8,
                atol=0.0,
                maxiter=solver.linear_solver_maxiter,
                M=self.preconditioner(matrix, solver.preconditioner_type),
                check_every=0,
                use_cuda_graph=self.device.is_cuda and wp.is_conditional_graph_supported(),
                run=False,
            )
            for matrix in (self.matrix, self.transpose)
        ]

    def preconditioner(self, matrix, kind):
        if kind != "block_diag":
            return preconditioner(matrix, kind)
        inverse = wp.empty(self.n, dtype=wp.mat33, device=self.device)
        wp.launch(
            invert_block_diagonal_kernel,
            self.n,
            inputs=[matrix.offsets, matrix.columns, matrix.values, inverse],
            device=self.device,
        )

        def multiply(x, y, z, alpha, beta):
            wp.launch(
                block_jacobi_mv_kernel,
                self.n,
                inputs=[inverse, x, y, z, wp.float32(alpha), wp.float32(beta)],
                device=self.device,
            )

        return LinearOperator(shape=(self.n, self.n), dtype=wp.vec3, device=self.device, matvec=multiply)

    def solve(self, rhs, output, transpose=False):
        # A checked solve is a function of this RHS, independent of which
        # forward or adjoint solve used the shared workspace previously.
        # In particular, frame checkpoint recomputation must not inherit an
        # adjoint-sized warm start from the preceding reverse frame.
        self.solution.zero_()
        self.maximum.zero_()
        wp.launch(maximum_rhs_component, self.n, inputs=[rhs, self.maximum], device=self.device)
        wp.launch(normalize_linear_rhs, self.n, inputs=[rhs, self.maximum, self.rhs, self.solution], device=self.device)
        (self.backward_solver if transpose else self.forward_solver)(b=self.rhs)
        wp.launch(restore_linear_increment, self.n, inputs=[self.maximum, self.solution], device=self.device)
        wp.copy(output, self.solution)
        matrix = self.transpose if transpose else self.matrix
        self.norms.zero_()
        wp.launch(
            residual_norms,
            self.n,
            inputs=[matrix.offsets, matrix.columns, matrix.values, output, rhs, self.norms],
            device=self.device,
        )
        wp.launch(record_residual, 1, inputs=[self.norms, int(transpose), self.stats], device=self.device)


def _owned(array, *, grad, target=None):
    owned = wp.empty_like(array, requires_grad=grad) if target is None else target
    kernel = copy_vectors if array.dtype == wp.vec3 else copy_scalars
    wp.launch(kernel, array.size, inputs=[array], outputs=[owned], device=array.device)
    return owned


def _buffer(solver, key, reference, grad):
    if grad:
        return wp.zeros_like(reference, requires_grad=True)
    cache = solver._differentiation_eager_buffers
    if key not in cache:
        cache[key] = wp.zeros_like(reference)
    return cache[key]


def prepare(solver):
    if solver._streaming is not None or solver._component_scaling is not None:
        raise NotImplementedError("Differentiation requires fixed topology without streaming/component scaling")
    if solver._soft_surface_contacts is not None or solver.model.particle_ke > 0.0:
        raise NotImplementedError("Differentiation supports analytic ground contact; disable particle/surface contacts")
    fixed = [
        "particle_mass",
        "particle_inv_mass",
        "particle_radius",
        "tet_poses",
        "tet_materials",
        "tri_poses",
        "tri_materials",
        "spring_stiffness",
        "spring_damping",
        "spring_rest_length",
        "gravity",
    ]
    for name in fixed:
        value = getattr(solver.model, name, None)
        if isinstance(value, wp.array) and value.requires_grad:
            raise ValueError(
                f"{name} is fixed: finalize the model without gradients and enable state/control gradients"
            )
    for name in ["hex_materials", "hex_inv_J0", "hex_det_J0_w"]:
        value = getattr(solver, name, None)
        if isinstance(value, wp.array) and value.requires_grad:
            raise ValueError(f"{name} is fixed and does not support gradients")
    if solver._differentiation_snapshot is None:
        if solver.model.device.is_capturing:
            raise RuntimeError("Call prepare_differentiation() before CUDA graph capture")
        with without_tape():
            solver._differentiation_snapshot = MatrixSnapshot(solver)
    return solver._differentiation_snapshot


def check(solver, *, reset=True):
    if solver.model.device.is_capturing:
        raise RuntimeError("Check differentiable residuals after launching the CUDA graph")
    values = solver._differentiation_stats.numpy()
    result = {"forward_relative_residual": float(values[0]), "backward_relative_residual": float(values[1])}
    solver.last_differentiation_residuals = result
    if reset:
        solver._differentiation_stats.zero_()
    if not np.isfinite(values).all() or np.any(values[2:] > 1.0):
        raise RuntimeError(f"FEM differentiable linear solve did not converge: {result}")
    return result


def step(solver, state_in, state_out, control, contacts, dt, external_forces):
    if dt != solver.dt:
        raise ValueError("Differentiable dt must match the fixed matrix assembly timestep")
    if contacts is not None:
        raise NotImplementedError("Differentiation supports only the analytic ground contact law")
    tape = active_tape()
    grad = tape is not None
    if state_in.particle_q.ptr == state_out.particle_q.ptr or state_in.particle_qd.ptr == state_out.particle_qd.ptr:
        raise ValueError("Use unique input/output states for every taped step")
    if grad:
        if not state_out.particle_q.requires_grad or not state_out.particle_qd.requires_grad:
            raise ValueError("Taped output positions and velocities must require gradients")
        # Tape owns this bookkeeping, so tape.reset() permits a fresh rollout.
        key = f"_fem_written_{id(solver)}"
        recorded = getattr(tape, key, None)
        if recorded is None or not tape.launches:
            recorded = set()
            setattr(tape, key, recorded)
        recorded.update((state_in.particle_q.ptr, state_in.particle_qd.ptr))
        for array in (state_out.particle_q, state_out.particle_qd):
            if array.ptr in recorded:
                raise ValueError("A taped output state was reused; allocate unique states per rollout")
            recorded.add(array.ptr)
    matrix = prepare(solver)
    q = _owned(state_in.particle_q, grad=grad, target=_buffer(solver, "q", state_in.particle_q, grad))
    v = _owned(state_in.particle_qd, grad=grad, target=_buffer(solver, "v", state_in.particle_qd, grad))
    state = SimpleNamespace(particle_q=q, particle_qd=v)
    if control is None:
        control = SimpleNamespace(tet_activations=solver.model.tet_activations)
    if solver.enable_tet_activations:
        control = SimpleNamespace(
            tet_activations=_owned(
                control.tet_activations,
                grad=grad,
                target=_buffer(solver, "activation", control.tet_activations, grad),
            )
        )
    external = _buffer(solver, "external", q, grad)
    if external_forces is None:
        external.zero_()
    else:
        _owned(external_forces, grad=grad, target=external)
    old_buffers, old_recording = solver._force_bufs, solver._recording_forces
    if not grad and "forces" not in solver._differentiation_eager_buffers:
        solver._differentiation_eager_buffers["forces"] = {}
    solver._force_bufs = {} if grad else solver._differentiation_eager_buffers["forces"]
    solver._recording_forces = grad
    try:
        forces = [
            solver.eval_spring_forces(solver.model, state),
            solver.eval_tetrahedral_forces(solver.model, control, state),
            solver.eval_hexahedral_forces(solver.model, control, state),
            solver.eval_triangle_forces(solver.model, control, state),
            solver.eval_particle_ground_contact_forces(solver.model, control, state),
            solver.eval_gravity_forces(solver.model),
            solver.eval_linear_damping_forces(solver.model, state),
        ]
    finally:
        solver._force_bufs, solver._recording_forces = old_buffers, old_recording
    force, rhs, dv = [_buffer(solver, key, q, grad) for key in ("force", "rhs", "dv")]
    wp.launch(sum_forces, q.size, inputs=[*forces, external, dt], outputs=[force], device=q.device)
    if solver.model.tet_count or solver.hex_count or solver.model.spring_count:
        wp.launch(
            form_rhs,
            q.size,
            inputs=[matrix.matrix.offsets, matrix.matrix.columns, matrix.matrix.values, v, force],
            outputs=[rhs],
            device=q.device,
        )
        if solver.model.spring_count:
            wp.launch(
                spring_predictor,
                solver.model.spring_count,
                inputs=[solver.model.spring_indices, solver.model.spring_damping, dt, v],
                outputs=[rhs],
                device=q.device,
            )
    else:
        wp.launch(copy_vectors, q.size, inputs=[force], outputs=[rhs], device=q.device)
    with without_tape():
        matrix.solve(rhs, dv)
        # Keep the public velocity_increment telemetry consistent with both
        # execution modes; the tape retains its separate owned increment.
        wp.copy(solver.dv, dv)
        adjoint = wp.empty_like(dv) if grad else None
    if grad:

        def backward():
            if solver._differentiation_revision != matrix.revision:
                raise RuntimeError("FEM model/matrix changed after recording; rerun the forward tape before backward")
            with without_tape():
                matrix.solve(dv.grad, adjoint, transpose=True)
                wp.launch(accumulate_vectors, rhs.size, inputs=[adjoint], outputs=[rhs.grad], device=rhs.device)
                if not solver.defer_differentiation_checks:
                    check(solver)

        tape.record_func(backward, [rhs, dv])
    wp.launch(
        update_state,
        q.size,
        inputs=[dv, dt, q, v],
        outputs=[state_out.particle_q, state_out.particle_qd],
        device=q.device,
    )
    if not solver.defer_differentiation_checks:
        with without_tape():
            check(solver)
    return state_out

# SPDX-License-Identifier: Apache-2.0
"""Owned affine mechanics for one non-differentiable FEM attempt.

Preparation evaluates the ordinary force terms once. Subsequent RHS assembly,
operator actions and reconstruction are resident operations suitable for CUDA
graphs. This does not alter the material tangent or introduce another model.
"""

import math

import warp as wp
from warp.optim.linear import aslinearoperator

from .differentiation import active_tape
from .implicit_predictor import restore_spring_damping_predictor
from .kernels import maximum_rhs_component, normalize_linear_rhs, restore_linear_increment, update_state


@wp.kernel(enable_backward=False)
def _predictor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    velocity: wp.array[wp.vec3],
    correction: wp.array[wp.vec3],
):
    node = wp.tid()
    value = wp.vec3(0.0)
    for block in range(offsets[node], offsets[node + 1]):
        value += values[block] * (velocity[columns[block]] - velocity[node])
    correction[node] = value


@wp.kernel(enable_backward=False)
def _subtract(source: wp.array[wp.vec3], target: wp.array[wp.vec3]):
    node = wp.tid()
    target[node] -= source[node]


@wp.kernel(enable_backward=False)
def _add_load(source: wp.array[wp.vec3], dt: float, target: wp.array[wp.vec3]):
    node = wp.tid()
    target[node] = target[node] + dt * source[node]


@wp.kernel(enable_backward=False)
def _diagonal(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    fixed: wp.array[int],
    blocks: wp.array[wp.mat33],
):
    row = wp.tid()
    value = wp.mat33(0.0)
    for block in range(offsets[row], offsets[row + 1]):
        if columns[block] == row:
            value += values[block]
    if fixed[row] != 0:
        value = wp.identity(n=3, dtype=float)
    blocks[row] = value


class PreparedFEMStep:
    """A prepared equation ``operator * delta_v = rhs(external_forces)``.

    Obtain this workspace with :meth:`SolverFEM.prepare_step`. Its ``base_rhs``
    is the zero-additional-load impulse [N s], including the existing elastic
    predictor. ``fixed_mask`` and ``diagonal_blocks`` describe the constrained
    operator. All exposed arrays/operators are borrowed, read-only views.

    Input positions/velocities are copied into persistent storage, so trial
    reconstruction cannot overwrite the accepted state. One workspace is
    reused per solver; preparing another attempt invalidates the previous one.
    Reprepare after changes to state, control, contacts, material or supports.
    Call ``refresh_tangent`` separately when changing the chosen linearization
    or timestep, exactly as for ``SolverFEM.step``. Model storage must stay fixed.
    Existing captured graphs require preparation before replay; graph replay
    cannot execute Python revision checks or discover changed model arrays.

    This fast-path API deliberately excludes tapes, differentiation, streaming
    topology and component scaling. The ordinary solver retains those paths.
    """

    def __init__(self, solver):
        self.solver, self.model = solver, solver.model
        self.device = self.model.device
        self._state = self.model.state()
        count = self.model.particle_count
        self.base_rhs = wp.zeros(count, dtype=wp.vec3, device=self.device)
        self._impulse = wp.empty_like(self.base_rhs)
        self._predictor = wp.zeros_like(self.base_rhs)
        self._rhs = wp.empty_like(self.base_rhs)
        self.velocity_increment = wp.zeros_like(self.base_rhs)
        self._normalized_rhs = wp.empty_like(self.base_rhs)
        self._rhs_maximum = wp.zeros(1, dtype=float, device=self.device)
        self.fixed_mask = (
            solver._fixed_particles.mask
            if solver._fixed_particles is not None
            else wp.zeros(count, dtype=int, device=self.device)
        )
        self.diagonal_blocks = wp.empty(count, dtype=wp.mat33, device=self.device)
        self._solvers = {}
        self._operator_signature = None
        self._revision = None
        self.dt = None

    @staticmethod
    def check_supported(solver):
        if solver.differentiable or active_tape() is not None:
            raise RuntimeError("Prepared FEM steps are non-differentiable and cannot run inside a tape")
        if solver._streaming is not None or solver._component_scaling is not None:
            raise ValueError("Prepared FEM steps require assembled, unscaled particle coordinates")

    def _check_current(self):
        self.check_supported(self.solver)
        if self._revision is None:
            raise RuntimeError("Prepare FEM mechanics before using this workspace")
        if self._revision != self.solver._differentiation_revision or self.dt != self.solver.dt:
            raise RuntimeError("FEM mechanics changed; prepare the step again")

    def _vectors(self, array, name):
        if array.dtype != wp.vec3 or array.shape != (self.model.particle_count,) or array.device != self.device:
            raise ValueError(f"{name} must be a particle-sized wp.vec3 array on the model device")

    def prepare(self, state_in, control, contacts, dt, external_forces=None):
        self.check_supported(self.solver)
        if self.device.is_capturing:
            raise RuntimeError("Prepare FEM mechanics before CUDA graph capture")
        dt = float(dt)
        if not math.isfinite(dt) or dt <= 0 or dt != self.solver.dt:
            raise ValueError("Prepared timestep must be finite, positive and match the assembled tangent timestep")
        self._vectors(state_in.particle_q, "particle_q")
        self._vectors(state_in.particle_qd, "particle_qd")
        if external_forces is not None:
            self._vectors(external_forces, "external_forces")
        solver, model = self.solver, self.model
        solver.validate_observables(None, contacts)
        self._revision = None
        self._state.assign(state_in)
        fixed = solver._fixed_particles
        if fixed is not None:
            fixed.restore(self._state)
        control = model.control() if control is None else control
        solver._accumulate_forces(model, self._state, contacts, control, dt)
        wp.copy(self._impulse, self._state.particle_f)
        if external_forces is not None:
            wp.launch(_add_load, model.particle_count, [external_forces, dt, self._impulse], device=self.device)
        matrix = solver.A_bsr
        if model.tet_count or solver.hex_count or model.spring_count:
            wp.launch(
                _predictor,
                model.particle_count,
                [matrix.offsets, matrix.columns, matrix.values, self._state.particle_qd, self._predictor],
                device=self.device,
            )
        else:
            # Preserve the original triangle-only integration approximation.
            self._predictor.zero_()
        if fixed is not None:
            if fixed.prepare(matrix):
                solver._lin_solver = solver._lin_solver_key = None
        self.operator = aslinearoperator(matrix) if fixed is None else fixed.operator
        self.preconditioner = aslinearoperator(solver.M_bsr)
        self.unconstrained_matrix = matrix
        wp.launch(
            _diagonal,
            model.particle_count,
            [matrix.offsets, matrix.columns, matrix.values, self.fixed_mask, self.diagonal_blocks],
            device=self.device,
        )
        signature = (
            id(matrix),
            matrix.offsets.ptr,
            matrix.columns.ptr,
            matrix.values.ptr,
            id(matrix) if fixed is None else id(self.operator),
            id(solver.M_bsr),
        )
        if signature != self._operator_signature:
            self._solvers.clear()
            self._operator_signature = signature
        self.dt, self._revision = dt, solver._differentiation_revision
        wp.copy(self.velocity_increment, solver.velocity_increment)
        self.rhs(out=self.base_rhs)
        self.prepare_solve()
        return self

    def rhs(self, external_forces=None, out=None):
        """Form the constrained impulse; additional nodal loads are in newtons.

        Preserve the ordinary load-before-predictor floating-point order.
        Spring damping's existing atomic correction is retained after predictor
        subtraction; no sparse predictor action or force reevaluation occurs.
        """
        self._check_current()
        out = self._rhs if out is None else out
        self._vectors(out, "out")
        if external_forces is not None:
            self._vectors(external_forces, "external_forces")
            if external_forces.ptr == out.ptr:
                raise ValueError("RHS output must not alias its external-force input")
        wp.copy(out, self._impulse)
        if external_forces is not None:
            wp.launch(_add_load, self.model.particle_count, [external_forces, self.dt, out], device=self.device)
        wp.launch(_subtract, self.model.particle_count, [self._predictor, out], device=self.device)
        if self.model.spring_count:
            wp.launch(
                restore_spring_damping_predictor,
                self.model.spring_count,
                [self.model.spring_indices, self.model.spring_damping, self.dt, self._state.particle_qd, out],
                device=self.device,
            )
        if self.solver._fixed_particles is not None:
            self.solver._fixed_particles.project_rhs(out)
        return out

    def apply_operator(self, vector, out):
        """Write the constrained momentum action without a physical solve."""
        self._check_current()
        self._vectors(vector, "vector")
        self._vectors(out, "out")
        if vector.ptr == out.ptr:
            raise ValueError("Operator action requires distinct input and output storage")
        self.operator.matvec(vector, vector, out, 1.0, 0.0)
        return out

    def apply_preconditioner(self, vector, out):
        """Apply the existing FEM numerical preconditioner."""
        self._check_current()
        self._vectors(vector, "vector")
        self._vectors(out, "out")
        if vector.ptr == out.ptr:
            raise ValueError("Preconditioner action requires distinct input and output storage")
        self.preconditioner.matvec(vector, vector, out, 1.0, 0.0)
        return out

    def prepare_solve(self, tolerance=None):
        """Allocate a persistent reference solve before capture, without running it."""
        self._check_current()
        key = self.solver._linear_solver_settings(tolerance)
        if key not in self._solvers:
            if self.device.is_capturing:
                raise RuntimeError("Prepare this FEM solve tolerance before CUDA graph capture")
            self._solvers[key] = self.solver._make_linear_solver(
                self.operator,
                self.preconditioner,
                key,
                rhs=self._normalized_rhs,
                increment=self.velocity_increment,
            )
        return key

    def solve(self, rhs, out=None, *, tolerance=None):
        """Reference affine solve; changes only this workspace's numerical guess."""
        self._vectors(rhs, "rhs")
        key = self.prepare_solve(tolerance)
        self._rhs_maximum.zero_()
        wp.launch(maximum_rhs_component, self.model.particle_count, [rhs, self._rhs_maximum], device=self.device)
        wp.launch(
            normalize_linear_rhs,
            self.model.particle_count,
            [rhs, self._rhs_maximum, self._normalized_rhs, self.velocity_increment],
            device=self.device,
        )
        self._solvers[key](b=self._normalized_rhs, M=self.preconditioner)
        wp.launch(
            restore_linear_increment,
            self.model.particle_count,
            [self._rhs_maximum, self.velocity_increment],
            device=self.device,
        )
        if out is not None:
            self._vectors(out, "out")
            wp.copy(out, self.velocity_increment)
            return out
        return self.velocity_increment

    def reconstruct(self, delta_v, state_out):
        """Reconstruct only particle q/qd, enforcing the existing fixed supports."""
        self._check_current()
        self._vectors(delta_v, "delta_v")
        self._vectors(state_out.particle_q, "particle_q")
        self._vectors(state_out.particle_qd, "particle_qd")
        if state_out.particle_q.ptr == self._state.particle_q.ptr:
            raise ValueError("Reconstruction must not overwrite the prepared input")
        wp.launch(
            update_state,
            self.model.particle_count,
            [
                delta_v,
                self.dt,
                self._state.particle_q,
                self._state.particle_qd,
                state_out.particle_q,
                state_out.particle_qd,
            ],
            device=self.device,
        )
        if self.solver._fixed_particles is not None:
            self.solver._fixed_particles.restore(state_out)
        return state_out

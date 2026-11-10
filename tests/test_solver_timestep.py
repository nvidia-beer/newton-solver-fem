# SPDX-License-Identifier: Apache-2.0
"""Changed FEM timesteps agree with fresh full matrix assembly."""

import os
import unittest

import numpy as np
import warp as wp

from newton_solver_fem import FEMBuilder, SolverFEM


def dense(matrix):
    offsets, columns, blocks = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for index in range(offsets[row], offsets[row + 1]):
            column = columns[index]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += blocks[index]
    return result


class TestSolverTimestep(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    dt = 0.002

    def model(self, *, mixed=True):
        builder = FEMBuilder(gravity=(0, 0, 0))
        corners = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        for point in 0.1 * corners:
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=0.001)
        builder.add_tetrahedron(0, 1, 3, 4, k_mu=500, k_lambda=1000, k_damp=0)
        if mixed:
            builder.add_soft_hex_mesh(np.arange(8).reshape(1, 8), k_mu=500, k_lambda=1000, k_damp=0)
            builder.add_triangle(0, 1, 3, tri_ke=30, tri_ka=50, tri_kd=0)
            builder.add_spring(2, 6, ke=40, kd=0.2, control=0)
        return builder.finalize(device=self.device)

    def solver(self, model, dt=None, *, differentiable=False, fixed=True):
        return SolverFEM(
            model,
            dt=self.dt if dt is None else dt,
            solver_type="cg",
            preconditioner_type="block_diag",
            linear_solver_maxiter=80,
            linear_solver_tolerance=1e-5,
            self_contact_ke=0,
            differentiable=differentiable,
            fixed_particle_indices=(0, 1) if fixed else None,
        )

    def test_full_matrix_and_physical_update_match_fresh_solvers(self):
        # Exercise both an existing sparse-slot map and a dt change on the very
        # first refresh, when construction has pruned numerically zero blocks.
        for prepare_initial in (False, True):
            with self.subTest(prepare_initial=prepare_initial):
                model = self.model()
                actual = self.solver(model)
                current, following = model.state(), model.state()
                q = model.particle_q.numpy().copy()
                q[2] += (0.002, -0.001, 0.001)
                q[6] += (-0.001, 0.002, -0.003)
                velocity = np.linspace(-0.02, 0.03, 24, dtype=np.float32).reshape(8, 3)
                velocity[:2] = 0
                force = np.linspace(-0.01, 0.02, 24, dtype=np.float32).reshape(8, 3)
                external = wp.array(force, dtype=wp.vec3, device=self.device)
                current.particle_q.assign(q)
                if prepare_initial:
                    actual.refresh_tangent(current.particle_q)
                storage = None
                timesteps = (self.dt, self.dt / 2, self.dt) if prepare_initial else (self.dt / 2, self.dt)
                for dt in timesteps:
                    reference = self.solver(model, dt)
                    reference_in, reference_out = model.state(), model.state()
                    for state, solver in ((current, actual), (reference_in, reference)):
                        state.particle_q.assign(q)
                        state.particle_qd.assign(velocity)
                        solver.velocity_increment.assign(0.25 * velocity)
                        solver.refresh_tangent(state.particle_q, dt)
                    self.assertEqual(actual.dt, dt)
                    np.testing.assert_allclose(
                        actual.bsr_values.numpy(), reference.bsr_values.numpy(), rtol=1e-6, atol=1e-9
                    )
                    np.testing.assert_allclose(dense(actual.A_bsr), dense(reference.A_bsr), rtol=1e-6, atol=1e-9)
                    pointers = (
                        actual.A_bsr.values.ptr,
                        actual.A_bsr.columns.ptr,
                        actual.A_bsr.offsets.ptr,
                        actual._pc_inv_diag.ptr,
                    )
                    if storage is not None:
                        self.assertEqual(pointers, storage)
                    storage = pointers
                    # Timestep changes now clear the old solver's Krylov
                    # guess; compare both assemblies with that same guess.
                    reference.velocity_increment.assign(actual.velocity_increment)
                    actual.step(current, following, None, None, dt, external_forces=external)
                    reference.step(reference_in, reference_out, None, None, dt, external_forces=external)
                    np.testing.assert_allclose(
                        actual.velocity_increment.numpy(), reference.velocity_increment.numpy(), rtol=2e-5, atol=1e-7
                    )
                    np.testing.assert_allclose(
                        following.particle_qd.numpy(), reference_out.particle_qd.numpy(), rtol=2e-5, atol=1e-7
                    )
                    np.testing.assert_allclose(
                        following.particle_q.numpy(), reference_out.particle_q.numpy(), rtol=1e-6, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        current.particle_f.numpy(), reference_in.particle_f.numpy(), rtol=1e-6, atol=1e-9
                    )
                    np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
                    np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
                    # Omitted dt now retains the most recently assembled step.
                    actual.refresh_tangent(current.particle_q)
                    self.assertEqual(actual.dt, dt)

    def test_differentiable_and_tape_timestep_changes_are_rejected(self):
        model = self.model()
        solver = self.solver(model, differentiable=True, fixed=False)
        before = dense(solver.A_bsr)
        with self.assertRaisesRegex(ValueError, "Differentiable"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        np.testing.assert_array_equal(dense(solver.A_bsr), before)
        solver = self.solver(model)
        before = dense(solver.A_bsr)
        with wp.Tape(), self.assertRaisesRegex(RuntimeError, "outside tapes"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        np.testing.assert_array_equal(dense(solver.A_bsr), before)

    def test_changed_timestep_discards_only_the_numerical_warm_start(self):
        model = self.model()
        solver = self.solver(model)
        state = model.state()
        initial_q, initial_qd = state.particle_q.numpy(), state.particle_qd.numpy()
        guess = np.full((model.particle_count, 3), 0.02, dtype=np.float32)
        solver.velocity_increment.assign(guess)
        solver.refresh_tangent(state.particle_q)
        np.testing.assert_array_equal(solver.velocity_increment.numpy(), guess)
        solver.refresh_tangent(state.particle_q, self.dt / 2)
        np.testing.assert_array_equal(solver.velocity_increment.numpy(), np.zeros_like(guess))
        np.testing.assert_array_equal(state.particle_q.numpy(), initial_q)
        np.testing.assert_array_equal(state.particle_qd.numpy(), initial_qd)

    def test_streaming_and_invalid_timestep_changes_are_rejected(self):
        model = self.model(mixed=False)
        solver = self.solver(model, fixed=False)
        solver.prepare_tetrahedral_updates(model.particle_count)
        with self.assertRaisesRegex(ValueError, "Streaming"):
            solver.refresh_tangent(model.particle_q, self.dt / 2)
        self.assertEqual(solver.dt, self.dt)
        solver = self.solver(model)
        before = dense(solver.A_bsr)
        for dt in (0, -1, np.nan, np.inf):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "finite and positive"):
                solver.refresh_tangent(model.particle_q, dt)
            self.assertEqual(solver.dt, self.dt)
            np.testing.assert_array_equal(dense(solver.A_bsr), before)


if __name__ == "__main__":
    unittest.main()

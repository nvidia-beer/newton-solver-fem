# SPDX-License-Identifier: Apache-2.0
"""Prepared mechanics preserve the existing affine FEM equation and state update."""

import os
import unittest
from unittest import mock

import numpy as np
import warp as wp

from newton_solver_fem import FEMBuilder, SolverFEM


def dense(matrix):
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for block in range(offsets[row], offsets[row + 1]):
            column = columns[block]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += values[block]
    return result


class TestPreparedFEMStep(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    dt = 0.002

    def fixture(self, *, mixed=True, fixed=True, differentiable=False):
        builder = FEMBuilder(gravity=(0.0, 0.0, -0.3))
        corners = np.array(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)))
        for point in 0.1 * corners:
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=0.001)
        builder.add_tetrahedron(0, 1, 3, 4, k_mu=500, k_lambda=1000, k_damp=0.05)
        if mixed:
            builder.add_soft_hex_mesh(np.arange(8).reshape(1, 8), k_mu=500, k_lambda=1000, k_damp=0.02)
            builder.add_triangle(0, 1, 3, tri_ke=30, tri_ka=50, tri_kd=0.03)
            builder.add_spring(2, 6, ke=40, kd=0.2, control=0)
        model = builder.finalize(device=self.device)

        def solver():
            return SolverFEM(
                model,
                dt=self.dt,
                preconditioner_type="block_diag",
                linear_solver_maxiter=150,
                linear_solver_tolerance=1e-7,
                self_contact_ke=0,
                linear_damping=0.4,
                min_stretch=0.98,
                max_stretch=1.02,
                differentiable=differentiable,
                fixed_particle_indices=(0, 1) if fixed else None,
            )

        current = model.state()
        q = current.particle_q.numpy()
        q[4] += (0.003, -0.002, 0.009)
        q[6] += (-0.004, 0.003, -0.007)
        current.particle_q.assign(q)
        velocity = np.linspace(-0.02, 0.03, 24, dtype=np.float32).reshape(8, 3)
        if fixed:
            velocity[:2] = 0
        current.particle_qd.assign(velocity)
        current.particle_f.fill_(wp.vec3(7.0, 8.0, 9.0))
        return model, current, solver

    def test_mixed_elements_damping_clamps_external_load_and_fixed_support_parity(self):
        model, current, make_solver = self.fixture()
        actual, reference = make_solver(), make_solver()
        reference_in, reference_out, result = model.state(), model.state(), model.state()
        reference_in.assign(current)
        for solver in (actual, reference):
            solver.refresh_tangent(current.particle_q)
        before = [a.numpy().copy() for a in (current.particle_q, current.particle_qd, current.particle_f)]
        guess = np.full((8, 3), 0.001, dtype=np.float32)
        actual.velocity_increment.assign(guess)
        reference.velocity_increment.assign(guess)
        prepared = actual.prepare_step(current, None, None, self.dt)
        for scale in (0.0, 1.0, -0.5):
            with self.subTest(scale=scale):
                external = wp.array(
                    scale * np.linspace(-0.01, 0.02, 24).reshape(8, 3), dtype=wp.vec3, device=self.device
                )
                # Equal numerical initial guesses isolate the mechanics API.
                prepared.velocity_increment.assign(guess)
                reference.velocity_increment.assign(guess)
                rhs = prepared.rhs(external)
                dv = prepared.solve(rhs)
                prepared.reconstruct(dv, result)
                reference.step(reference_in, reference_out, None, None, self.dt, external_forces=external)
                np.testing.assert_allclose(rhs.numpy(), reference_in.particle_f.numpy(), rtol=2e-6, atol=2e-9)
                np.testing.assert_allclose(dv.numpy(), reference.velocity_increment.numpy(), rtol=3e-5, atol=2e-7)
                np.testing.assert_allclose(
                    result.particle_q.numpy(), reference_out.particle_q.numpy(), rtol=1e-6, atol=1e-8
                )
                np.testing.assert_allclose(
                    result.particle_qd.numpy(), reference_out.particle_qd.numpy(), rtol=3e-5, atol=2e-7
                )
                np.testing.assert_array_equal(result.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
                np.testing.assert_array_equal(result.particle_qd.numpy()[:2], np.zeros((2, 3)))
        for array, expected in zip((current.particle_q, current.particle_qd, current.particle_f), before, strict=True):
            np.testing.assert_array_equal(array.numpy(), expected)
        np.testing.assert_array_equal(actual.velocity_increment.numpy(), guess)

    def test_operator_diagonal_rhs_affinity_and_reconstruction_without_solve(self):
        model, current, make_solver = self.fixture(mixed=False)
        solver = make_solver()
        solver.refresh_tangent(current.particle_q)
        prepared = solver.prepare_step(current, None, None, self.dt)
        matrix = dense(solver.A_bsr)
        fixed = prepared.fixed_mask.numpy().astype(bool)
        scalar_fixed = np.repeat(fixed, 3)
        matrix[scalar_fixed, :] = 0
        matrix[:, scalar_fixed] = 0
        matrix[scalar_fixed, scalar_fixed] = 1
        vector_np = np.linspace(-0.001, 0.002, 24, dtype=np.float32).reshape(8, 3)
        vector = wp.array(vector_np, dtype=wp.vec3, device=self.device)
        out = wp.empty_like(vector)
        prepared.apply_operator(vector, out)
        np.testing.assert_allclose(out.numpy().ravel(), matrix @ vector_np.ravel(), rtol=2e-6, atol=1e-9)
        expected_diagonal = np.array([matrix[3 * i : 3 * i + 3, 3 * i : 3 * i + 3] for i in range(8)])
        np.testing.assert_allclose(prepared.diagonal_blocks.numpy(), expected_diagonal, rtol=1e-7, atol=1e-9)
        prepared.apply_preconditioner(vector, out)
        reference = wp.empty_like(vector)
        solver.M_bsr.matvec(vector, vector, reference, 1.0, 0.0)
        np.testing.assert_array_equal(out.numpy(), reference.numpy())
        rhs = prepared.rhs(vector).numpy()
        difference = self.dt * vector_np
        difference[fixed] = 0
        np.testing.assert_allclose(rhs, prepared.base_rhs.numpy() + difference, rtol=2e-6, atol=1e-9)
        result = model.state()
        with mock.patch.object(solver, "step", side_effect=AssertionError("No solve in reconstruction")):
            prepared.reconstruct(vector, result)
        expected_v = current.particle_qd.numpy() + vector_np
        expected_q = current.particle_q.numpy() + self.dt * expected_v
        expected_v[fixed] = 0
        expected_q[fixed] = model.particle_q.numpy()[fixed]
        np.testing.assert_array_equal(result.particle_q.numpy(), expected_q)
        np.testing.assert_array_equal(result.particle_qd.numpy(), expected_v)

    def test_preparation_owns_input_and_reuses_storage_without_force_reevaluation(self):
        model, current, make_solver = self.fixture(mixed=False)
        solver = make_solver()
        with mock.patch.object(solver, "_accumulate_forces", wraps=solver._accumulate_forces) as forces:
            prepared = solver.prepare_step(current, None, None, self.dt)
            pointers = (prepared.base_rhs.ptr, prepared.velocity_increment.ptr, prepared.diagonal_blocks.ptr)
            q, v = current.particle_q.numpy().copy(), current.particle_qd.numpy().copy()
            current.particle_q.fill_(wp.vec3(3.0))
            current.particle_qd.fill_(wp.vec3(4.0))
            zeros = wp.zeros_like(prepared.base_rhs)
            result = model.state()
            prepared.reconstruct(zeros, result)
            prepared.rhs(zeros)
            prepared.rhs(zeros)
            expected = q + self.dt * v
            expected[:2] = model.particle_q.numpy()[:2]
            np.testing.assert_array_equal(result.particle_q.numpy(), expected)
            self.assertEqual(forces.call_count, 1)
            current.particle_q.assign(q)
            current.particle_qd.assign(v)
            again = solver.prepare_step(current, None, None, self.dt)
            self.assertIs(again, prepared)
            self.assertEqual(pointers, (again.base_rhs.ptr, again.velocity_increment.ptr, again.diagonal_blocks.ptr))
            self.assertEqual(forces.call_count, 2)

    def test_changed_timestep_requires_refresh_and_invalidates_old_preparation(self):
        model, current, make_solver = self.fixture()
        solver = make_solver()
        prepared = solver.prepare_step(current, None, None, self.dt)
        smaller = self.dt / 2
        with self.assertRaisesRegex(ValueError, "assembled tangent"):
            solver.prepare_step(current, None, None, smaller)
        solver.refresh_tangent(current.particle_q, smaller)
        with self.assertRaisesRegex(RuntimeError, "changed"):
            prepared.rhs()
        prepared = solver.prepare_step(current, None, None, smaller)
        result, reference = model.state(), model.state()
        dv = prepared.solve(prepared.base_rhs)
        prepared.reconstruct(dv, result)
        solver.step(current, reference, None, None, smaller)
        np.testing.assert_allclose(result.particle_q.numpy(), reference.particle_q.numpy(), rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(result.particle_qd.numpy(), reference.particle_qd.numpy(), rtol=3e-5, atol=2e-7)

    def test_triangle_only_legacy_predictor_and_prepared_constant_load(self):
        builder = FEMBuilder(gravity=(0.0, 0.0, -0.1))
        for point in ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0)):
            builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=0.002)
        builder.add_triangle(0, 1, 2, tri_ke=30, tri_ka=50, tri_kd=0.02)
        model = builder.finalize(device=self.device)
        solver = SolverFEM(model, dt=self.dt, preconditioner_type="block_diag", self_contact_ke=0)
        current, actual, expected = model.state(), model.state(), model.state()
        current.particle_qd.assign(np.arange(9, dtype=np.float32).reshape(3, 3) * 0.001)
        load = wp.array(np.full((3, 3), 0.002, dtype=np.float32), dtype=wp.vec3, device=self.device)
        prepared = solver.prepare_step(current, None, None, self.dt, external_forces=load)
        prepared.reconstruct(prepared.solve(prepared.base_rhs), actual)
        solver.step(current, expected, None, None, self.dt, external_forces=load)
        np.testing.assert_array_equal(prepared.base_rhs.numpy(), current.particle_f.numpy())
        np.testing.assert_allclose(actual.particle_qd.numpy(), expected.particle_qd.numpy(), rtol=1e-6, atol=1e-8)

    def test_interrupted_preparation_fails_closed_without_mutating_input(self):
        _, current, make_solver = self.fixture(mixed=False)
        solver = make_solver()
        prepared = solver.prepare_step(current, None, None, self.dt)
        before = current.particle_q.numpy().copy()
        with mock.patch.object(solver, "_accumulate_forces", side_effect=RuntimeError("injected failure")):
            with self.assertRaisesRegex(RuntimeError, "injected failure"):
                solver.prepare_step(current, None, None, self.dt)
        with self.assertRaisesRegex(RuntimeError, "Prepare FEM mechanics"):
            prepared.rhs()
        np.testing.assert_array_equal(current.particle_q.numpy(), before)
        solver.prepare_step(current, None, None, self.dt)
        self.assertTrue(np.isfinite(prepared.rhs().numpy()).all())

    def test_unsupported_modes_and_invalid_alias_fail_before_state_change(self):
        model, current, make_solver = self.fixture(differentiable=True, fixed=False)
        with self.assertRaisesRegex(RuntimeError, "non-differentiable"):
            make_solver().prepare_step(current, None, None, self.dt)
        model, current, make_solver = self.fixture(mixed=False, fixed=False)
        solver = make_solver()
        with wp.Tape(), self.assertRaisesRegex(RuntimeError, "tape"):
            solver.prepare_step(current, None, None, self.dt)
        prepared = solver.prepare_step(current, None, None, self.dt)
        vector = wp.zeros_like(prepared.base_rhs)
        with self.assertRaisesRegex(ValueError, "alias"):
            prepared.rhs(vector, vector)
        with self.assertRaisesRegex(ValueError, "distinct"):
            prepared.apply_operator(vector, vector)
        solver = make_solver()
        solver.prepare_tetrahedral_updates(model.particle_count)
        with self.assertRaisesRegex(ValueError, "assembled"):
            solver.prepare_step(current, None, None, self.dt)

    def test_capture_rhs_actions_solve_and_reconstruction(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA capture requires a CUDA test device")
        model, current, make_solver = self.fixture()
        solver = make_solver()
        solver.refresh_tangent(current.particle_q)
        prepared = solver.prepare_step(current, None, None, self.dt)
        external = wp.zeros_like(prepared.base_rhs)
        result = model.state()
        work = wp.empty_like(external)
        # Warm kernels outside capture; actual replay reads updated load values.
        prepared.apply_operator(external, work)
        prepared.apply_preconditioner(external, work)
        prepared.reconstruct(prepared.solve(prepared.rhs(external)), result)
        with wp.ScopedCapture(device=self.device) as capture:
            prepared.reconstruct(prepared.solve(prepared.rhs(external)), result)
        external.fill_(wp.vec3(0.002, -0.001, 0.003))
        prepared.velocity_increment.zero_()
        wp.capture_launch(capture.graph)
        captured = result.particle_qd.numpy()
        prepared.velocity_increment.zero_()
        prepared.reconstruct(prepared.solve(prepared.rhs(external)), result)
        np.testing.assert_array_equal(captured, result.particle_qd.numpy())


if __name__ == "__main__":
    unittest.main()

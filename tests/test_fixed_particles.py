# SPDX-License-Identifier: Apache-2.0
"""Dirichlet solves agree with independent reduced dense systems."""

import os
import unittest
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from warp.sparse import bsr_from_triplets

from newton_solver_fem import SolverFEM
from newton_solver_fem._src.solvers.fem.fixed_particles import FixedParticles


def dense(matrix):
    offsets, columns, values = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for block in range(offsets[row], offsets[row + 1]):
            column = columns[block]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += values[block]
    return result


class TestFixedParticles(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    dt = 0.002

    def model(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=0.001)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=5000, k_lambda=10000, k_damp=0)
        return builder.finalize(device=self.device)

    def solver(self, model, fixed=(0, 1), **kwargs):
        return SolverFEM(
            model,
            dt=self.dt,
            fixed_particle_indices=fixed,
            self_contact_ke=0,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=64,
            **kwargs,
        )

    def test_free_motion_matches_reduced_system_and_preserves_predictor(self):
        model = self.model()
        solver = self.solver(model)
        current, following = model.state(), model.state()
        rest = model.particle_q.numpy().copy()
        q = rest.copy()
        q[2] += (0.002, 0.001, -0.001)
        velocity = np.array(((0, 0, 0), (0, 0, 0), (0.02, -0.03, 0.01), (-0.01, 0.01, 0.02)), np.float32)
        external = np.array(((20, 30, 40), (10, -30, 60), (0.02, 0.04, 0.03), (-0.01, 0.03, 0.02)), np.float32)
        forces = wp.array(external, dtype=wp.vec3, device=self.device)
        current.particle_q.assign(q)
        current.particle_qd.assign(velocity)
        for refresh in (False, True, True):
            if refresh:
                solver.refresh_tangent(current.particle_q)
            matrix = dense(solver.A_bsr)
            elastic = solver.eval_tetrahedral_forces(model, model.control(), current).numpy().copy()
            mass = np.repeat(model.particle_mass.numpy(), 3)
            rhs = self.dt * (elastic + external).ravel() - (matrix - np.diag(mass)) @ velocity.ravel()
            expected = np.zeros(12)
            expected[6:] = np.linalg.solve(matrix[6:, 6:], rhs[6:])
            solver.step(current, following, None, None, self.dt, external_forces=forces)
            np.testing.assert_allclose(solver.velocity_increment.numpy().ravel(), expected, rtol=3e-4, atol=2e-6)
            np.testing.assert_array_equal(following.particle_q.numpy()[:2], rest[:2])
            np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
            np.testing.assert_allclose(following.particle_qd.numpy(), velocity + expected.reshape(4, 3), atol=2e-6)
            # Keeping the physical matrix is essential for next step's predictor.
            np.testing.assert_array_equal(dense(solver.A_bsr), matrix)
            current, following = following, current
            velocity = current.particle_qd.numpy().copy()

    def test_boundary_is_restored_even_after_external_state_changes(self):
        model = self.model()
        solver = self.solver(model, fixed=(0, 0, 1, 2, 3))
        current, following = model.state(), model.state()
        current.particle_q.fill_(wp.vec3(0.5))
        current.particle_qd.fill_(wp.vec3(100.0))
        solver.step(current, following, None, None, self.dt)
        np.testing.assert_array_equal(following.particle_q.numpy(), model.particle_q.numpy())
        np.testing.assert_array_equal(following.particle_qd.numpy(), np.zeros((4, 3)))
        np.testing.assert_array_equal(solver.velocity_increment.numpy(), np.zeros((4, 3)))

    def test_capture_uses_live_forces_and_exact_anchors(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA is unavailable")
        model = self.model()
        solver = self.solver(model)
        current, following = model.state(), model.state()
        forces = wp.zeros(4, dtype=wp.vec3, device=self.device)
        solver.refresh_tangent(current.particle_q)
        matrix = dense(solver.A_bsr)
        # GPU SVD/rest-stress roundoff is small but not exactly zero. Use the
        # independently reduced physical system, including that residual.
        elastic = solver.eval_tetrahedral_forces(model, model.control(), current).numpy().copy()
        solver.step(current, following, None, None, self.dt, external_forces=forces)
        with wp.ScopedCapture(device=self.device) as capture:
            solver.refresh_tangent(current.particle_q)
            solver.step(current, following, None, None, self.dt, external_forces=forces)
        for value in (0.02, -0.03, 0.0):
            force = np.zeros((4, 3), dtype=np.float32)
            force[2, 2] = value
            forces.assign(force)
            wp.capture_launch(capture.graph)
            np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
            np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
            expected = np.zeros(12)
            rhs = self.dt * (elastic + force).ravel()
            expected[6:] = np.linalg.solve(matrix[6:, 6:], rhs[6:])
            np.testing.assert_allclose(following.particle_qd.numpy().ravel(), expected, rtol=5e-4, atol=3e-8)
            if value:
                self.assertGreater(following.particle_qd.numpy()[2, 2] * value, 0)

    def test_reassembly_uses_the_same_fixed_positions_as_force_evaluation(self):
        for method in ("refresh_tangent", "refresh_topology", "rebuild_element_blocks"):
            with self.subTest(method=method):
                model = self.model()
                expected, actual = self.solver(model), self.solver(model)
                reference, changed = model.state(), model.state()
                positions = model.particle_q.numpy().copy()
                positions[2] += (0.002, 0.001, -0.001)
                reference.particle_q.assign(positions)
                positions[:2] += (0.03, -0.02, 0.04)
                changed.particle_q.assign(positions)
                getattr(expected, method)(reference.particle_q, self.dt)
                getattr(actual, method)(changed.particle_q, self.dt)
                np.testing.assert_array_equal(changed.particle_q.numpy(), reference.particle_q.numpy())
                np.testing.assert_array_equal(dense(actual.A_bsr), dense(expected.A_bsr))
                expected_out, actual_out = model.state(), model.state()
                expected.step(reference, expected_out, None, None, self.dt)
                actual.step(changed, actual_out, None, None, self.dt)
                np.testing.assert_allclose(actual_out.particle_qd.numpy(), expected_out.particle_qd.numpy(), atol=1e-7)

    def test_invalid_indices_and_unsupported_modes_rejected(self):
        model = self.model()
        for indices in ((-1,), (4,), (1.0,), ((1, 2),), (True,)):
            with self.subTest(indices=indices), self.assertRaises(ValueError):
                self.solver(model, indices)
        with self.assertRaisesRegex(ValueError, "differentiable"):
            self.solver(model, differentiable=True)
        solver = self.solver(model)
        with self.assertRaisesRegex(ValueError, "Streaming"):
            solver.prepare_tetrahedral_updates(4)
        with self.assertRaisesRegex(ValueError, "Component scaling"):
            solver.prepare_component_scaling(wp.zeros(4, dtype=int, device=self.device), 1)

    def sparse_operator_fixture(self, missing_diagonal=False):
        rows = np.repeat(np.arange(4), 3).astype(np.int32)
        columns = np.array([0, 1, 2, 0, 1, 2, 1, 2, 3, 1, 2, 3], np.int32)
        values = np.random.default_rng(7).normal(size=(len(rows), 3, 3)).astype(np.float32)
        if missing_diagonal:
            rows, columns, values = rows[3:], columns[3:], values[3:]
        matrix = bsr_from_triplets(
            4,
            4,
            wp.array(rows, dtype=int, device=self.device),
            wp.array(columns, dtype=int, device=self.device),
            wp.array(values, dtype=wp.mat33, device=self.device),
        )
        model = SimpleNamespace(
            device=wp.get_device(self.device),
            particle_count=4,
            particle_q=wp.zeros(4, dtype=wp.vec3, device=self.device),
        )
        return matrix, FixedParticles(model, matrix, [0])

    def check_projected_operator(self, matrix, fixed):
        physical = dense(matrix)
        projected = physical.copy()
        projected[:3, :] = 0
        projected[:, :3] = 0
        projected[:3, :3] = np.eye(3)
        x_host = np.random.default_rng(8).normal(size=(4, 3)).astype(np.float32)
        y_host = 0.2 * x_host
        x = wp.array(x_host, dtype=wp.vec3, device=self.device)
        y = wp.empty_like(x)
        z = wp.empty_like(x)
        fixed.prepare(matrix)
        for alpha, beta, alias in ((1.0, 0.0, False), (-0.7, 0.3, False), (0.2, -0.4, True)):
            with self.subTest(alpha=alpha, beta=beta, alias=alias):
                y.assign(y_host if beta else np.full_like(y_host, np.nan))
                output = y if alias else z
                fixed.operator.matvec(x, y, output, alpha, beta)
                expected = alpha * projected @ x_host.ravel() + beta * y_host.ravel()
                np.testing.assert_allclose(output.numpy().ravel(), expected, rtol=2e-6, atol=2e-6)
        np.testing.assert_array_equal(dense(matrix), physical)
        return x, y, z, projected

    def test_projected_operator_preserves_affine_alias_and_cpu_fallback(self):
        for missing_diagonal in (False, True):
            with self.subTest(missing_diagonal=missing_diagonal):
                matrix, fixed = self.sparse_operator_fixture(missing_diagonal)
                original = fixed.operator
                self.check_projected_operator(matrix, fixed)
                if missing_diagonal or not wp.get_device(self.device).is_cuda:
                    self.assertIs(fixed.operator, original)

    def test_projected_operator_refreshes_values_topology_and_storage(self):
        matrix, fixed = self.sparse_operator_fixture()
        self.check_projected_operator(matrix, fixed)
        original = fixed.operator
        columns = matrix.columns.numpy()
        columns[3:6] = (1, 2, 3)
        matrix.columns.assign(columns)
        matrix.values.assign(0.8 * matrix.values.numpy())
        x, y, z, _ = self.check_projected_operator(matrix, fixed)
        self.assertIs(fixed.operator, original)
        if wp.get_device(self.device).is_cuda:
            with wp.ScopedCapture(device=self.device) as capture:
                fixed.prepare(matrix)
                fixed.operator.matvec(x, y, z, 1.0, 0.0)
            matrix.values.assign(0.5 * matrix.values.numpy())
            wp.capture_launch(capture.graph)
            captured = z.numpy()
            _, _, _, projected = self.check_projected_operator(matrix, fixed)
            np.testing.assert_allclose(captured.ravel(), projected @ x.numpy().ravel(), rtol=2e-6, atol=2e-6)
        matrix.offsets = wp.clone(matrix.offsets)
        matrix.columns = wp.clone(matrix.columns)
        matrix.values = wp.clone(matrix.values)
        self.check_projected_operator(matrix, fixed)
        if wp.get_device(self.device).is_cuda:
            self.assertIsNot(fixed.operator, original)


if __name__ == "__main__":
    unittest.main()

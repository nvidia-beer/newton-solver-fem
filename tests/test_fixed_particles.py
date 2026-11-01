# SPDX-License-Identifier: Apache-2.0
"""Dirichlet solves agree with independent reduced dense systems."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_fem import SolverFEM


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


if __name__ == "__main__":
    unittest.main()

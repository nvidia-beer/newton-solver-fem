# SPDX-License-Identifier: Apache-2.0
"""Compatibility checks for mechanics added by the original Newton builder."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_fem import FEMBuilder, SolverLinearFEM


class TestReferenceBuilder(unittest.TestCase):
    def test_unique_tet_edges_and_reference_spring_material(self):
        builder = FEMBuilder(gravity=(0.0, 0.0, 0.0))
        builder.add_soft_mesh(
            pos=wp.vec3(0.0),
            rot=wp.quat_identity(),
            scale=1.0,
            vel=wp.vec3(0.0),
            vertices=[(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 0, -1)],
            indices=[0, 1, 2, 3, 0, 2, 1, 4],
            density=3.0,
            k_mu=100.0,
            k_lambda=120.0,
            k_damp=6.0,
        )
        self.assertEqual(builder.spring_count, 9)
        np.testing.assert_allclose(builder.spring_stiffness, 50.0)
        np.testing.assert_allclose(builder.spring_damping, 3.0)
        model = builder.finalize(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.assertAlmostEqual(float(model.particle_mass.numpy().sum()), 1.0)
        solver = SolverLinearFEM(
            model, dt=0.001, preconditioner_type="block_diag", solver_type="cg", self_contact_ke=0.0
        )
        state, result = model.state(), model.state()
        solver.step(state, result, None, None, 0.001)
        np.testing.assert_allclose(result.particle_q.numpy(), state.particle_q.numpy(), atol=2e-6)
        solver.reset(result)
        solver.notify_model_changed(newton.ModelFlags.MODEL_PROPERTIES)
        self.assertEqual(solver.velocity_increment.shape, (5,))

    def test_hex_reference_quadrature_and_mass_ownership(self):
        builder = FEMBuilder(gravity=(0.0, 0.0, -9.81))
        nodes = np.asarray(
            [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0), (0, 0, 4), (2, 0, 4), (2, 3, 4), (0, 3, 4)], dtype=float
        )
        builder.add_particles(nodes.tolist(), np.zeros_like(nodes).tolist(), [3.0] * 8)
        builder.add_soft_hex_mesh(np.arange(8).reshape(1, 8), k_mu=10.0, k_lambda=20.0, k_damp=0.0)
        model = builder.finalize(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        self.assertEqual(model.fem.hex_count, 1)
        np.testing.assert_allclose(model.fem.hex_det_J0_w.numpy(), 3.0)
        np.testing.assert_allclose(
            model.fem.hex_inv_J0.numpy(), np.broadcast_to(np.diag([1.0, 2 / 3, 0.5]), (1, 8, 3, 3)), atol=2e-7
        )
        self.assertFalse(hasattr(model, "hex_count"))
        solver = SolverLinearFEM(
            model, dt=0.001, preconditioner_type="block_diag", solver_type="cg", self_contact_ke=0.0
        )
        state, result = model.state(), model.state()
        solver.step(state, result, None, None, 0.001)
        np.testing.assert_allclose(result.particle_qd.numpy()[:, 2], -0.00981, atol=1e-7)

    def test_hex_replication_remaps_indices_and_rejects_unsupported_rotation(self):
        local = FEMBuilder()
        nodes = np.asarray(
            [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0), (0, 0, 4), (2, 0, 4), (2, 3, 4), (0, 3, 4)], dtype=float
        )
        local.add_particles(nodes.tolist(), np.zeros_like(nodes).tolist(), [3.0] * 8)
        local.add_soft_hex_mesh(np.arange(8).reshape(1, 8), k_mu=10.0, k_lambda=20.0, k_damp=0.0)
        combined = FEMBuilder()
        combined.add_builder(local)
        transform = wp.transform(wp.vec3(10.0, 20.0, 30.0), wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), 0.7))
        with self.assertRaisesRegex(ValueError, "translation only"):
            combined.add_builder(local, xform=transform)
        transform = wp.transform(wp.vec3(10.0, 20.0, 30.0), wp.quat_identity())
        combined.add_builder(local, xform=transform)
        model = combined.finalize(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        np.testing.assert_array_equal(model.fem.hex_indices.numpy(), np.arange(16).reshape(2, 8))
        np.testing.assert_allclose(model.fem.hex_det_J0_w.numpy(), 3.0, rtol=2e-6)
        expected = np.asarray([list(wp.transform_point(transform, wp.vec3(*point))) for point in nodes])
        np.testing.assert_allclose(model.particle_q.numpy()[8:], expected, atol=2e-6)

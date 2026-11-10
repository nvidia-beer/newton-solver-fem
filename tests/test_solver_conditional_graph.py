# SPDX-License-Identifier: Apache-2.0
"""Conditional linear solves preserve the physical update under graph replay."""

import os
import unittest
from contextlib import contextmanager
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from newton_solver_fem import SolverFEM
from newton_solver_fem._src.solvers.fem import solver as solver_module


@contextmanager
def observe_result(solver_type):
    """Retain the public Warp solver result without synchronizing the solve."""
    factory = getattr(solver_module, solver_type)
    result = []

    def make_solver(*args, **kwargs):
        solve = factory(*args, **kwargs)

        def run(*args, **kwargs):
            result[:] = solve(*args, **kwargs)
            return tuple(result)

        return run

    with patch.object(solver_module, solver_type, side_effect=make_solver):
        yield result


class TestConditionalLinearGraph(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    dt = 0.002

    def model(self, *, device=None, elastic=False):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((0, 0, 0), (0.1, 0, 0), (0, 0.1, 0), (0, 0, 0.1)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=0.001)
        if elastic:
            builder.add_tetrahedron(0, 1, 2, 3, k_mu=5000, k_lambda=10000, k_damp=0)
        return builder.finalize(device=device or self.device)

    def solver(self, model, **kwargs):
        options = dict(
            dt=self.dt,
            fixed_particle_indices=(0, 1),
            self_contact_ke=0,
            preconditioner_type="block_diag",
            solver_type="cg",
            linear_solver_maxiter=80,
            linear_solver_tolerance=1e-5,
        )
        options.update(kwargs)
        return SolverFEM(model, **options)

    def require_conditional_cuda(self):
        if not wp.get_device(self.device).is_cuda or not wp.is_conditional_graph_supported():
            self.skipTest("conditional CUDA graphs are unavailable on the selected device")

    def test_cpu_and_host_checks_retain_existing_paths(self):
        model = self.model(device="cpu")
        solver = self.solver(model)
        current, following = model.state(), model.state()
        with patch.object(wp, "is_conditional_graph_supported", side_effect=AssertionError("CUDA query on CPU")):
            solver.step(current, following, None, None, self.dt)
        self.assertTrue(solver.linear_solver_use_cuda_graph)
        self.assertFalse(solver._lin_solver._use_cuda_graph)
        solver.linear_solver_check_every = 1
        solver.step(current, following, None, None, self.dt)
        self.assertFalse(solver._lin_solver._use_cuda_graph)
        self.assertEqual(solver.last_cg_iters, 0)

    def test_unsupported_cuda_and_public_flag_fall_back(self):
        self.require_conditional_cuda()
        model = self.model()
        solver = self.solver(model)
        current, following = model.state(), model.state()
        with patch.object(wp, "is_conditional_graph_supported", return_value=False):
            solver.step(current, following, None, None, self.dt)
        self.assertFalse(solver._lin_solver._use_cuda_graph)
        fixed_solver = solver._lin_solver
        solver.step(current, following, None, None, self.dt)
        self.assertIsNot(solver._lin_solver, fixed_solver)
        self.assertTrue(solver._lin_solver._use_cuda_graph)
        solver.linear_solver_use_cuda_graph = False
        solver.step(current, following, None, None, self.dt)
        self.assertFalse(solver._lin_solver._use_cuda_graph)
        solver.linear_solver_use_cuda_graph = True
        solver.linear_solver_check_every = 1
        solver.step(current, following, None, None, self.dt)
        self.assertFalse(solver._lin_solver._use_cuda_graph)
        self.assertEqual(solver.last_cg_iters, 0)

    def test_zero_first_capture_replays_changed_loads_and_warm_starts(self):
        self.require_conditional_cuda()
        for solver_type in ("cg", "cr", "bicgstab", "gmres"):
            with self.subTest(solver_type=solver_type):
                model = self.model()
                actual = self.solver(model, solver_type=solver_type)
                reference = self.solver(model, solver_type=solver_type, linear_solver_use_cuda_graph=False)
                current, following = model.state(), model.state()
                reference_in, reference_out = model.state(), model.state()
                external = wp.zeros(4, dtype=wp.vec3, device=self.device)
                with observe_result(solver_type) as result:
                    actual.step(current, following, None, None, self.dt, external_forces=external)
                    self.assertEqual(int(result[0].numpy()[0]), 0)
                    with wp.ScopedCapture(device=self.device) as capture:
                        actual.step(current, following, None, None, self.dt, external_forces=external)
                for scale, warm_scale in ((0.0, 0.0), (1.0, 0.0), (-2.0, 0.5), (0.1, -3.0), (0.0, 1.0)):
                    force = scale * np.array(
                        ((20, 30, 40), (10, -30, 60), (0.02, 0.04, 0.03), (-0.01, 0.03, 0.02)), np.float32
                    )
                    warm = warm_scale * np.full((4, 3), 0.01, dtype=np.float32)
                    # A previous constrained physical update has zero fixed-node
                    # increments; vary the free warm start without violating it.
                    warm[:2] = 0
                    external.assign(force)
                    actual.velocity_increment.assign(warm)
                    reference.velocity_increment.assign(warm)
                    wp.capture_launch(capture.graph)
                    reference.step(reference_in, reference_out, None, None, self.dt, external_forces=external)
                    np.testing.assert_allclose(
                        actual.velocity_increment.numpy(), reference.velocity_increment.numpy(), atol=2e-8, rtol=2e-6
                    )
                    np.testing.assert_allclose(
                        following.particle_qd.numpy(), reference_out.particle_qd.numpy(), atol=2e-8, rtol=2e-6
                    )
                    np.testing.assert_array_equal(current.particle_f.numpy(), reference_in.particle_f.numpy())
                    np.testing.assert_array_equal(following.particle_q.numpy()[:2], model.particle_q.numpy()[:2])
                    np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
                    iterations = int(result[0].numpy()[0])
                    self.assertLess(iterations, actual.linear_solver_maxiter)
                    self.assertLessEqual(float(result[1].numpy()[0]), float(result[2].numpy()[0]))
                    if scale:
                        self.assertGreater(iterations, 0)
                    else:
                        self.assertEqual(iterations, 0)

    def test_captured_elastic_update_matches_fixed_budget_with_live_tangent(self):
        self.require_conditional_cuda()
        model = self.model(elastic=True)
        actual, reference = self.solver(model), self.solver(model, linear_solver_use_cuda_graph=False)
        current, following = model.state(), model.state()
        reference_in, reference_out = model.state(), model.state()
        external = wp.zeros(4, dtype=wp.vec3, device=self.device)
        actual.refresh_tangent(current.particle_q)
        with observe_result("cg") as result:
            actual.step(current, following, None, None, self.dt, external_forces=external)
            with wp.ScopedCapture(device=self.device) as capture:
                actual.refresh_tangent(current.particle_q)
                actual.step(current, following, None, None, self.dt, external_forces=external)
        rest = model.particle_q.numpy().copy()
        for scale in (1.0, -0.5, 2.0):
            q = rest.copy()
            q[2] += scale * np.array((0.002, 0.001, -0.001), np.float32)
            velocity = scale * np.array(((0, 0, 0), (0, 0, 0), (0.02, -0.03, 0.01), (-0.01, 0.01, 0.02)), np.float32)
            force = scale * np.array(((20, 30, 40), (10, -30, 60), (0.02, 0.04, 0.03), (-0.01, 0.03, 0.02)), np.float32)
            external.assign(force)
            for state, solver in ((current, actual), (reference_in, reference)):
                state.particle_q.assign(q)
                state.particle_qd.assign(velocity)
                solver.velocity_increment.assign(0.25 * velocity)
            wp.capture_launch(capture.graph)
            reference.refresh_tangent(reference_in.particle_q)
            reference.step(reference_in, reference_out, None, None, self.dt, external_forces=external)
            np.testing.assert_allclose(
                actual.velocity_increment.numpy(), reference.velocity_increment.numpy(), rtol=2e-6, atol=2e-8
            )
            np.testing.assert_allclose(
                following.particle_q.numpy(), reference_out.particle_q.numpy(), atol=2e-8, rtol=2e-6
            )
            np.testing.assert_allclose(
                following.particle_qd.numpy(), reference_out.particle_qd.numpy(), atol=2e-8, rtol=2e-6
            )
            np.testing.assert_array_equal(current.particle_f.numpy(), reference_in.particle_f.numpy())
            np.testing.assert_array_equal(following.particle_q.numpy()[:2], rest[:2])
            np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
            self.assertLess(int(result[0].numpy()[0]), actual.linear_solver_maxiter)
            self.assertLessEqual(float(result[1].numpy()[0]), float(result[2].numpy()[0]))


if __name__ == "__main__":
    unittest.main()

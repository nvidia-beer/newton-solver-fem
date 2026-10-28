# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Block inversion remains scale invariant for small cut-material masses."""

import os
import unittest
from itertools import product

import newton
import numpy as np
import warp as wp

from newton_solver_fem import SolverLinearFEM
from newton_solver_fem._src.solvers.fem.kernels import invert_block_diagonal_kernel
from newton_solver_fem._src.solvers.fem.streaming import _invert_diagonal


class TestLinearFEMBlockPreconditioner(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def particle_solver(self, mass, dt, method="cg"):
        builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
        builder.add_particle((0, 0, 1), vel=(0, 0, 0), mass=mass)
        model = builder.finalize(device=self.device)
        solver = SolverLinearFEM(
            model,
            dt=dt,
            solver_type=method,
            preconditioner_type="block_diag",
            linear_solver_maxiter=8,
            self_contact_ke=0.0,
        )
        return model, solver

    def invert(self, blocks, streaming):
        count = len(blocks)
        values = wp.array(blocks, dtype=wp.mat33, device=self.device)
        output = wp.empty_like(values)
        slots = wp.array(np.arange(count), dtype=int, device=self.device)
        if streaming:
            wp.launch(_invert_diagonal, count, inputs=[slots, values, output], device=self.device)
        else:
            offsets = wp.array(np.arange(count + 1), dtype=int, device=self.device)
            wp.launch(invert_block_diagonal_kernel, count, inputs=[offsets, slots, values, output], device=self.device)
        return output.numpy()

    def test_spd_off_diagonal_blocks_are_scale_invariant(self):
        base = np.asarray(((4, 0.7, -0.3), (0.7, 2, 0.4), (-0.3, 0.4, 1)), dtype=np.float64)
        scales = np.power(10.0, np.arange(-20, 21, dtype=float))
        blocks = (scales[:, None, None] * base).astype(np.float32)
        expected = np.linalg.inv(blocks.astype(np.float64))
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                inverse = self.invert(blocks, streaming)
                self.assertTrue(np.isfinite(inverse).all())
                np.testing.assert_allclose(inverse, expected, rtol=6e-7, atol=0)
                np.testing.assert_allclose(inverse @ blocks, np.tile(np.eye(3), (len(blocks), 1, 1)), atol=3e-7)

    def test_singular_block_fallback_retains_small_positive_diagonal(self):
        scales = np.power(10.0, np.arange(-20, 21, dtype=float))
        blocks = (scales[:, None, None] * np.diag((1, 0, 2))).astype(np.float32)
        expected = np.zeros_like(blocks)
        expected[:, 0, 0] = 1 / blocks[:, 0, 0]
        expected[:, 1, 1] = 1
        expected[:, 2, 2] = 1 / blocks[:, 2, 2]
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                np.testing.assert_allclose(self.invert(blocks, streaming), expected, rtol=2e-7, atol=0)

    def test_tiny_physical_mass_falls_with_same_gravity(self):
        dt = 1 / 240
        for method, mass in product(("cg", "bicgstab", "gmres", "cr"), (1e-25, 1e-18, 1e-14, 1.0)):
            with self.subTest(method=method, mass=mass):
                model, solver = self.particle_solver(mass, dt, method)
                current, following = model.state(), model.state()
                pointers = solver.dv.ptr, solver._normalized_rhs.ptr, solver._rhs_maximum.ptr
                for step in range(1, 4):
                    solver.step(current, following, model.control(), None, dt)
                    np.testing.assert_allclose(following.particle_qd.numpy()[0], (0, 0, -step * 9.81 * dt), atol=1e-7)
                    np.testing.assert_allclose(solver.dv.numpy()[0], (0, 0, -9.81 * dt), atol=1e-7)
                    np.testing.assert_allclose(
                        current.particle_f.numpy()[0], (0, 0, -mass * 9.81 * dt), rtol=3e-7, atol=0
                    )
                    self.assertEqual(pointers, (solver.dv.ptr, solver._normalized_rhs.ptr, solver._rhs_maximum.ptr))
                    current, following = following, current
                # A zero RHS must clear the previously nonzero warm start.
                model.set_gravity((0, 0, 0))
                solver.step(current, following, model.control(), None, dt)
                np.testing.assert_allclose(following.particle_qd.numpy(), current.particle_qd.numpy(), atol=1e-7)
                np.testing.assert_array_equal(solver.dv.numpy(), np.zeros((1, 3)))

    def test_captured_tiny_rhs_replays_with_live_force_and_stable_buffers(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA is unavailable")
        dt = 1 / 240
        model, solver = self.particle_solver(1e-25, dt)
        current, following = model.state(), model.state()
        solver.step(current, following, model.control(), None, dt)
        pointers = solver.dv.ptr, solver._normalized_rhs.ptr, solver._rhs_maximum.ptr
        with wp.ScopedCapture(device=self.device) as capture:
            solver.step(current, following, model.control(), None, dt)
        for gravity in (-9.81, 0.0, -4.0):
            model.set_gravity((0, 0, gravity))
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(following.particle_qd.numpy()[0], (0, 0, gravity * dt), atol=1e-7)
            np.testing.assert_allclose(solver.dv.numpy()[0], (0, 0, gravity * dt), atol=1e-7)
            self.assertEqual(pointers, (solver.dv.ptr, solver._normalized_rhs.ptr, solver._rhs_maximum.ptr))


if __name__ == "__main__":
    unittest.main()

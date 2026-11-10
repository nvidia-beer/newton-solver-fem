# SPDX-License-Identifier: Apache-2.0
"""CPU reduction batching preserves the independently reduced physical solve."""

import unittest
from unittest.mock import patch

import newton
import numpy as np
import warp as wp

from newton_solver_fem import SolverFEM
from newton_solver_fem._src.solvers.fem import solver as solver_module


class TestCPUCGReduction(unittest.TestCase):
    def test_fixed_physics_with_live_loads_warm_starts_and_preconditioners(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        for point in ((0, 0, 0), (0.125, 0, 0), (0, 0.125, 0), (0, 0, 0.125)):
            builder.add_particle(pos=point, vel=(0, 0, 0), mass=0.001)
        builder.add_tetrahedron(0, 1, 2, 3, k_mu=5000, k_lambda=10000, k_damp=0)
        model = builder.finalize(device="cpu")
        dt = 0.002
        for preconditioner in ("diag", "block_diag"):
            for check_every in (0, 1):
                with self.subTest(preconditioner=preconditioner, check_every=check_every):
                    solver = SolverFEM(
                        model,
                        dt=dt,
                        fixed_particle_indices=(0, 1),
                        self_contact_ke=0,
                        solver_type="cg",
                        preconditioner_type=preconditioner,
                        linear_solver_maxiter=80,
                        linear_solver_tolerance=1e-5,
                    )
                    solver.linear_solver_check_every = check_every
                    matrix = np.zeros((12, 12))
                    offsets, columns, values = (
                        solver.A_bsr.offsets.numpy(),
                        solver.A_bsr.columns.numpy(),
                        solver.A_bsr.values.numpy(),
                    )
                    for row in range(4):
                        for index in range(offsets[row], offsets[row + 1]):
                            column = columns[index]
                            matrix[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += values[index]
                    current, following = model.state(), model.state()
                    elastic = solver.eval_tetrahedral_forces(model, model.control(), current).numpy().copy()
                    forces = wp.zeros(4, dtype=wp.vec3, device="cpu")
                    with patch.object(solver_module, "cg", wraps=solver_module.cg) as construct:
                        for scale, warm_scale in ((0, 2), (1, -3), (-2, 0.5), (0, 4)):
                            force = scale * np.array(
                                ((20, 30, 40), (10, -30, 60), (0.02, 0.04, 0.03), (-0.01, 0.03, 0.02)),
                                dtype=np.float32,
                            )
                            # Cancel rest-state roundoff so scale=0 exercises
                            # an exactly zero physical RHS with a live warm start.
                            force -= elastic
                            warm = np.full((4, 3), warm_scale * 0.01, dtype=np.float32)
                            warm[:2] = 0
                            forces.assign(force)
                            solver.velocity_increment.assign(warm)
                            expected = np.zeros(12)
                            expected[6:] = np.linalg.solve(matrix[6:, 6:], dt * (elastic + force).ravel()[6:])
                            solver.step(current, following, None, None, dt, external_forces=forces)
                            np.testing.assert_allclose(
                                solver.velocity_increment.numpy().ravel(), expected, rtol=3e-5, atol=2e-7
                            )
                            np.testing.assert_array_equal(
                                following.particle_q.numpy()[:2], model.particle_q.numpy()[:2]
                            )
                            np.testing.assert_array_equal(following.particle_qd.numpy()[:2], np.zeros((2, 3)))
                            np.testing.assert_array_equal(forces.numpy(), force)
                            self.assertTrue(np.isfinite(following.particle_q.numpy()).all())
                        self.assertEqual(construct.call_count, 1)
                        operator = construct.call_args.args[0]
                        np.testing.assert_array_equal(operator.batch_offsets.numpy(), [0, 12])
                        self.assertEqual(operator.batch_count, 1)
                    if check_every:
                        self.assertEqual(solver.last_cg_iters, 0)


if __name__ == "__main__":
    unittest.main()

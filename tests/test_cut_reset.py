# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repeated cutting and resetting must replay the current physics buffers."""

import gc
import unittest

import numpy as np
from test_cut_pipeline import _blade
from test_cut_surface_growth import _growth_example


class TestLinearFEMCutReset(unittest.TestCase):
    def test_repeated_reset_matches_fresh_trajectory_after_surface_growth(self):
        with _growth_example() as example:
            initial = example.state_0.particle_q.numpy().copy()
            expected = []
            for _ in range(3):
                example.step()
                expected.append((example.state_0.particle_q.numpy().copy(), example.state_0.particle_qd.numpy().copy()))
            for repetition in range(6):
                with self.subTest(reset=repetition):
                    self.assertTrue(example.cut(_blade()), example.cut_error)
                    example.step()
                    example.test_final()
                    example.reset()
                    gc.collect()
                    self.assertEqual(example.cut_revision, 0)
                    np.testing.assert_array_equal(example.state_0.particle_q.numpy(), initial)
                    for q, qd in expected:
                        example.step()
                        example.test_final()
                        np.testing.assert_allclose(example.state_0.particle_q.numpy(), q, atol=2e-6, rtol=2e-6)
                        np.testing.assert_allclose(example.state_0.particle_qd.numpy(), qd, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    unittest.main()

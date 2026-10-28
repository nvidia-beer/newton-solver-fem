# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cylindrical holes must span the body without moving retained material."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
from test_cut_surface_growth import _growth_example, _mechanical_pointers

from newton_solver_fem.examples import example_linear_fem_baymax_cut as cutting


class TestLinearFEMCylinderCut(unittest.TestCase):
    def test_single_ray_opens_a_constant_diameter_through_hole(self):
        with _growth_example() as example:
            self.assertEqual(example.cutter_mode, "cylinder")
            q = example.state_0.particle_q.numpy()
            velocity = (q * 0.05 + (0.1, 0.2, -0.1)).astype(np.float32)
            example.state_0.particle_qd.assign(velocity)
            pointers = _mechanical_pointers(example)
            mass = example.model.cut_physical_mass.numpy().sum(dtype=np.float64)
            origin, direction, radius = (0.003, 0.004, 0.8), (0.0, 0.0, 1.0), 0.012
            before_time = example.sim_time

            self.assertTrue(example.cut_cylinder_ray(origin, direction, radius), example.cut_error)

            removed = mass - example.model.cut_physical_mass.numpy().sum(dtype=np.float64)
            area = 8.0 * np.sin(np.pi / 8.0) * radius**2
            self.assertAlmostEqual(removed, example.args.density * area * 0.1, delta=5.0e-7)
            self.assertEqual(example.cut_revision, 1)
            self.assertEqual(_mechanical_pointers(example), pointers)
            self.assertEqual(example.sim_time, before_time)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), q)
            np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), velocity)
            self.assertIsNone(example._pick_cut_surface(origin, direction))
            self.assertIsNotNone(example._pick_cut_surface((0.04, 0.004, 0.8), direction))

            count = example._surface_count
            example.contact.update_surface(example.state_0.particle_q)
            triangles = example.contact.points.numpy()[: 3 * count].reshape(count, 3, 3)
            walls = triangles[example.surface.kind.numpy()[:count] == 2].reshape(-1, 3)
            self.assertGreater(len(walls), 0)
            radial = np.linalg.norm(walls[:, :2] - np.asarray(origin)[:2], axis=1)
            self.assertLessEqual(radial.max(), radius + 2.0e-6)
            self.assertGreaterEqual(radial.min(), radius * np.cos(np.pi / 16) - 2.0e-6)
            self.assertAlmostEqual(walls[:, 2].min(), 1.0, delta=2.0e-6)
            self.assertAlmostEqual(walls[:, 2].max(), 1.1, delta=2.0e-6)
            for _ in range(12):
                example.step()
                example.test_final()

    def test_miss_and_repeated_hole_do_not_publish_another_edit(self):
        with _growth_example() as example:
            q = example.state_0.particle_q.numpy()
            self.assertFalse(example.cut_cylinder_ray((2, 2, 0.8), (0, 0, 1), 0.01))
            self.assertEqual(example.cut_revision, 0)
            self.assertEqual(example.cut_error, "")
            self.assertTrue(example.cut_cylinder_ray((0.003, 0.004, 0.8), (0, 0, 1), 0.012), example.cut_error)
            mass = example.model.cut_physical_mass.numpy()
            graph = example.graph
            self.assertFalse(example.cut_cylinder_ray((0.003, 0.004, 0.8), (0, 0, 1), 0.012))
            self.assertEqual(example.cut_revision, 1)
            self.assertEqual(example.cut_error, "")
            self.assertIs(example.graph, graph)
            np.testing.assert_array_equal(example.model.cut_physical_mass.numpy(), mass)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), q)

    def test_one_consumed_click_produces_one_full_depth_request(self):
        example = cutting.Example.__new__(cutting.Example)
        ray = (np.asarray((0, -1, 0)), np.asarray((0, 1, 0)), 0.01)
        example.screen_cylinder = SimpleNamespace(consume=Mock(side_effect=[ray, None]))
        example.cut_cylinder_ray = Mock(return_value=True)
        example._consume_cylinder()
        example._consume_cylinder()
        example.cut_cylinder_ray.assert_called_once_with(*ray)


if __name__ == "__main__":
    unittest.main()

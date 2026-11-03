# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The retained contact experiment must load, deform and release active tissue."""

import importlib.util
import math
import os
import unittest
from contextlib import contextmanager

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples._suturebot_contact_demo import Example

_HAS_RIG_DEPS = all(importlib.util.find_spec(name) is not None for name in ("pxr", "mujoco", "mujoco_warp"))


@contextmanager
def _example(regions):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    arguments = ["--demo", "--regions", *map(str, regions), "--device", device, "--viewer", "null"]
    args = Example.create_parser().parse_args(arguments)
    with wp.ScopedDevice(device):
        viewer = ViewerNull()
        example = None
        try:
            example = Example(viewer, args)
            yield example
        finally:
            if example is not None:
                example.press_demo = None
            viewer.close()


@unittest.skipUnless(_HAS_RIG_DEPS, "Install the examples extra to run USD/MuJoCo rig tests")
class SuturebotPressTests(unittest.TestCase):
    def test_runtime_demo_loads_free_tissue_recovers_and_resets_after_selection(self):
        with _example([1]) as example:
            self.assertIsNone(example.press_demo)
            example.step()
            example.test_final()
            plan = example.press_demo
            self.assertIsNotNone(plan, example.demo_phase)
            face_nodes = example.selection.exterior_surface_triangles[plan.face]
            self.assertFalse(np.isin(face_nodes, example.fixed_vertex_indices).any())
            frames = math.ceil(plan.duration / example.frame_dt)
            pressed_face_displacement = 0.0
            for _ in range(1, frames):
                example.step()
                example.test_final()
                self.assertIs(example.press_demo, plan)
                elapsed = example.sim_time - plan.start_time
                if 2.0 < elapsed <= 5.0 and plan.baseline is not None:
                    actual = example.state_0.particle_q.numpy()[face_nodes]
                    movement = np.linalg.norm(actual - plan.baseline[face_nodes], axis=1).max()
                    pressed_face_displacement = max(pressed_face_displacement, float(movement))
            plan.validate()
            report = plan.report()
            self.assertEqual(report["frames"], frames)
            self.assertEqual(report["selected_region_ids"], [1])
            self.assertGreater(report["maximum_jaw_active_tissue_force_N"], 0.001)
            self.assertGreater(report["maximum_jaw_active_contact_samples"], 0)
            self.assertLessEqual(report["final_jaw_active_tissue_force_N"], 1.0e-4)
            self.assertGreater(report["frames_with_jaw_active_contact"], 0)
            self.assertGreater(pressed_face_displacement, 2.0e-5)
            self.assertGreater(report["maximum_incremental_displacement_m"], 2.0e-5)
            self.assertLess(report["remaining_displacement_m"], 0.5 * report["maximum_incremental_displacement_m"])
            self.assertLess(report["maximum_rcm_error_m"], 2.0e-5)

            # Rebuilding the selection must discard target planning and its
            # baseline; measurements from the old FEM model cannot survive.
            example.select_regions([])
            self.assertIsNone(example.press_demo)
            example.step()
            example.test_final()
            self.assertIsNone(example.press_demo)
            example.select_regions([1])
            self.assertIsNone(example.press_demo)
            example.step()
            example.test_final()
            self.assertIsNot(example.press_demo, plan)
            self.assertEqual(example.press_demo.report()["frames"], 1)

    def test_no_active_tissue_does_not_misreport_fixed_pad_contact_as_a_press(self):
        with _example([]) as example:
            had_fixed_pad_contact = False
            for _ in range(90):
                example.step()
                example.test_final()
                self.assertIsNone(example.press_demo)
                had_fixed_pad_contact |= bool(example.pad_contacts.active_contacts.numpy()[0])
                np.testing.assert_array_equal(example.pad_contacts.active_body_contacts.numpy(), 0)
                np.testing.assert_array_equal(example.pad_contacts.active_body_force.numpy(), 0)
            self.assertIsNone(example.solver)
            self.assertEqual(example.demo_phase, "select active tissue")
            self.assertTrue(had_fixed_pad_contact, "The free needle should contact the fixed pad without an active FEM")


if __name__ == "__main__":
    unittest.main()

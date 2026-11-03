# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Surface force conservation and actual USD/MuJoCo rig dynamics."""

import importlib.util
import os
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples._suturebot_contact_demo import Example
from newton_solver_fem.examples._suturebot_contacts import surface_contact_forces


class PadContactTests(unittest.TestCase):
    def test_rotated_translated_body_contact_conserves_force_and_torque(self):
        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        points = np.array(((0, 0, 0), (0.02, 0, 0), (0, 0.02, 0), (0, 0, -0.02)), dtype=np.float32)
        indices = np.array(((0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)), dtype=np.int32).reshape(-1)
        transform = wp.transform(wp.vec3(-0.01, 0.02, 0.003), wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.7))
        center = wp.vec3(0.005, 0.006, 0.0005)
        local_center = wp.transform_point(wp.transform_inverse(transform), center)
        local_com = wp.vec3(0.002, -0.003, 0.001)
        com = np.asarray(wp.transform_point(transform, local_com))
        with wp.ScopedDevice(device):
            positions = wp.array(points, dtype=wp.vec3)
            triangles = wp.array(indices, dtype=int)
            mesh = wp.Mesh(positions, triangles)
            particle_forces = wp.zeros(4, dtype=wp.vec3)
            body_forces = wp.zeros(1, dtype=wp.spatial_vector)
            count = wp.zeros(1, dtype=int)
            free_force = wp.zeros(1, dtype=wp.vec3)
            free_count = wp.zeros(1, dtype=int)
            reference_force = None
            for name, free, expected_fraction in (
                ("all free", [1, 1, 1, 1], 1.0),
                ("one contacted vertex free", [0, 1, 0, 0], 0.25),
                ("all fixed", [0, 0, 0, 0], 0.0),
                ("only uncontacted vertex free", [0, 0, 0, 1], 0.0),
            ):
                with self.subTest(name=name):
                    for value in (particle_forces, body_forces, count, free_force, free_count):
                        value.zero_()
                    wp.launch(
                        surface_contact_forces,
                        dim=1,
                        inputs=[
                            mesh.id,
                            triangles,
                            positions,
                            wp.zeros(4, dtype=wp.vec3),
                            wp.array(np.arange(4), dtype=int),
                            wp.array(free, dtype=int),
                            wp.array([0], dtype=int),
                            wp.array([local_center], dtype=wp.vec3),
                            wp.array([0.0012], dtype=float),
                            wp.array([1.0], dtype=float),
                            wp.array([transform], dtype=wp.transform),
                            wp.array([[0.02, 0.01, -0.03, 0.2, -0.5, 0.1]], dtype=wp.spatial_vector),
                            wp.array([local_com], dtype=wp.vec3),
                            60.0,
                            0.03,
                            0.4,
                            0.006,
                            particle_forces,
                            body_forces,
                            count,
                            free_force,
                            free_count,
                        ],
                    )
                    pf, bf = particle_forces.numpy(), body_forces.numpy()[0]
                    self.assertEqual(int(count.numpy()[0]), 1)
                    self.assertEqual(int(free_count.numpy()[0]), int(expected_fraction > 0.0))
                    np.testing.assert_allclose(free_force.numpy()[0], expected_fraction * bf[:3], atol=1.0e-8)
                    np.testing.assert_allclose(free_force.numpy()[0], -pf[np.asarray(free, dtype=bool)].sum(axis=0))
                    self.assertGreater(bf[2], 0.04)
                    self.assertLess(bf[0], 0.0)
                    self.assertGreater(np.linalg.norm(bf[3:]), 0)
                    np.testing.assert_allclose(pf.sum(axis=0) + bf[:3], 0, atol=1.0e-8)
                    torque = np.cross(points, pf).sum(axis=0) + np.cross(com, bf[:3]) + bf[3:]
                    np.testing.assert_allclose(torque, 0, atol=2.0e-9)
                    np.testing.assert_array_equal(pf[3], 0)
                    if reference_force is None:
                        reference_force = bf.copy()
                    else:
                        np.testing.assert_array_equal(bf, reference_force)


_HAS_RIG_DEPS = all(importlib.util.find_spec(name) is not None for name in ("pxr", "mujoco", "mujoco_warp"))


@unittest.skipUnless(_HAS_RIG_DEPS, "Install the examples extra to run USD/MuJoCo rig tests")
class SuturebotRigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scope = wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        cls.scope.__enter__()
        cls.viewer = ViewerNull()
        args = Example.create_parser().parse_args(["--regions", "--mujoco-backend", "cpu"])
        # This test imports the packaged preparation result, not generated test
        # geometry or an arbitrary checkout elsewhere on the workstation.
        if not Path(args.rig_asset).is_file():
            cls.viewer.close()
            cls.scope.__exit__(None, None, None)
            raise AssertionError(f"Missing packaged rig asset: {args.rig_asset}")
        cls.example = Example(cls.viewer, args)

    @classmethod
    def tearDownClass(cls):
        cls.viewer.close()
        cls.scope.__exit__(None, None, None)

    def test_imported_drives_needle_support_and_rcm(self):
        example = self.example
        self.assertEqual(example.model.body_count, 18)
        self.assertGreaterEqual(example.model.shape_count, 47)
        sample_bodies = example.pad_contacts.sample_body.numpy()
        sample_weights = example.pad_contacts.sample_weight.numpy()
        self.assertIn(example.needle_body, sample_bodies)
        for body in np.unique(sample_bodies):
            self.assertAlmostEqual(float(sample_weights[sample_bodies == body].sum()), 1.0, places=5)
        initial_needle = example.state_0.body_q.numpy()[example.needle_body, :3].copy()
        initial_fixture = example.state_0.body_q.numpy()[example.body_indices["/SutureBot/RCMFixture"]].copy()
        example.set_tool_targets("PSM1", rcm_yaw=0.08, rcm_pitch=0.04, insertion=-0.004, grip=0.15)
        example.set_tool_targets("PSM2", rcm_yaw=-0.06, rcm_pitch=-0.03, insertion=0.002, grip=0.4)
        had_pad_contact = False
        for _ in range(90):
            example.step()
            had_pad_contact |= bool(example.pad_contacts.active_contacts.numpy()[0])
        example.test_final()
        self.assertTrue(had_pad_contact)
        # This fixture has no selected FEM region. The free needle still rests
        # on the pad, but fixed support must never count as tissue interaction.
        np.testing.assert_array_equal(example.pad_contacts.active_body_contacts.numpy(), 0)
        np.testing.assert_array_equal(example.pad_contacts.active_body_force.numpy(), 0)
        needle = example.state_0.body_q.numpy()[example.needle_body, :3]
        self.assertLess(needle[2], initial_needle[2] - 0.005)
        self.assertGreater(needle[2], -0.008)
        self.assertLess(np.linalg.norm(example.state_0.body_qd.numpy()[example.needle_body, :3]), 0.02)
        self.assertLess(example.rcm_errors().max(), 1.0e-6)
        np.testing.assert_allclose(
            example.state_0.body_q.numpy()[example.body_indices["/SutureBot/RCMFixture"]], initial_fixture, atol=1.0e-7
        )
        coords = example.state_0.joint_q.numpy()
        for tool, sign in (("PSM1", 1), ("PSM2", -1)):
            joint = example.joint_indices[f"/SutureBot/{tool}/Joints/rcm_yaw"]
            self.assertGreater(sign * coords[example.model.joint_q_start.numpy()[joint]], 0.02)
            left = example.joint_indices[f"/SutureBot/{tool}/Joints/jaw_left"]
            right = example.joint_indices[f"/SutureBot/{tool}/Joints/jaw_right"]
            self.assertGreater(coords[example.model.joint_q_start.numpy()[left]], 0.1)
            self.assertLess(coords[example.model.joint_q_start.numpy()[right]], -0.1)

    def test_invalid_tool_control_is_atomic(self):
        example = self.example
        before = example.control.joint_target_q.numpy().copy()
        for values in ({"rcm_yaw": 10}, {"insertion": np.nan}, {"grip": -0.1}, {"nonexistent": 1}):
            with self.assertRaises(ValueError):
                example.set_tool_targets("PSM1", **values)
            np.testing.assert_array_equal(example.control.joint_target_q.numpy(), before)

    def test_needle_has_physical_mujoco_contact_with_a_jaw(self):
        example = self.example
        saved = example.model.state()
        saved.assign(example.state_0)
        try:
            jaw = example.body_indices["/SutureBot/PSM1/jaw_left"]
            bodies = example.model.shape_body.numpy()
            types = example.model.shape_type.numpy()
            jaw_shape = np.flatnonzero((bodies == jaw) & (types == newton.GeoType.CAPSULE))[0]
            needle_shape = np.flatnonzero((bodies == example.needle_body) & (types == newton.GeoType.CAPSULE))[0]
            shape_poses = example.model.shape_transform.numpy()
            body_poses = example.state_0.body_q.numpy()
            point = wp.transform_point(wp.transform(*body_poses[jaw]), wp.vec3(*shape_poses[jaw_shape, :3]))
            needle_origin = np.asarray(point) - shape_poses[needle_shape, :3]
            joint = np.flatnonzero(example.model.joint_child.numpy() == example.needle_body)[0]
            start = example.model.joint_q_start.numpy()[joint]
            q = example.state_0.joint_q.numpy()
            q[start : start + 7] = np.r_[needle_origin, 0, 0, 0, 1]
            example.state_0.joint_q.assign(q)
            example.state_0.joint_qd.zero_()
            newton.eval_fk(example.model, example.state_0.joint_q, example.state_0.joint_qd, example.state_0)
            example.state_0.clear_forces()
            example.rigid_solver.step(example.state_0, example.state_1, example.control, None, example.sim_dt)
            rigid = example.rigid_solver
            body_map = rigid.mjc_body_to_newton.numpy()[0]
            active = []
            for contact in rigid.mj_data.contact:
                pair = {body_map[rigid.mj_model.geom_bodyid[geom]] for geom in (contact.geom1, contact.geom2)}
                if pair == {jaw, example.needle_body} and contact.efc_address >= 0:
                    active.append(contact.dist)
            self.assertTrue(active, "The jaw and needle must have active MuJoCo constraints, not only visual meshes")
            self.assertLess(min(active), 0)
            self.assertGreater(np.linalg.norm(example.state_1.body_qd.numpy()[example.needle_body]), 0)
        finally:
            example.state_0.assign(saved)
            example.state_1.assign(saved)


if __name__ == "__main__":
    unittest.main()

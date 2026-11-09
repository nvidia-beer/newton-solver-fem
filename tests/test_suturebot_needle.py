# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Rigid-needle state transitions, source-clock replay and MuJoCo release."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import newton
import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from newton_solver_fem.examples._suturebot_needle import NeedleReplay, add_needle_to_builder, pose_matrix


class TestSutureBotNeedle(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="suturebot-needle-")
        self.root = Path(self.directory.name)
        stage = Usd.Stage.CreateNew(str(self.root / "needle.usda"))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/TestNeedle")
        stage.SetDefaultPrim(root.GetPrim())
        body = UsdGeom.Xform.Define(stage, "/TestNeedle/Needle")
        body.AddTranslateOp().Set(Gf.Vec3d(0, 0, 0.1))
        UsdPhysics.RigidBodyAPI.Apply(body.GetPrim())
        mass = UsdPhysics.MassAPI.Apply(body.GetPrim())
        mass.CreateMassAttr(0.0003)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(1e-8))
        sphere = UsdGeom.Sphere.Define(stage, "/TestNeedle/Needle/Collision")
        sphere.CreateRadiusAttr(0.0005)
        UsdPhysics.CollisionAPI.Apply(sphere.GetPrim())
        plane = UsdGeom.Cube.Define(stage, "/TestNeedle/Table")
        plane.CreateSizeAttr(1.0)
        plane.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.005))
        plane.AddScaleOp().Set(Gf.Vec3f(1, 1, 0.01))
        UsdPhysics.CollisionAPI.Apply(plane.GetPrim())
        stage.GetRootLayer().Save()
        frames = 61
        poses = np.zeros((frames, 2, 7), np.float32)
        poses[:, :, 6] = 1
        poses[:, 0, 0] = np.arange(frames) * 0.001
        poses[:, 0, 2] = 0.1 + np.minimum(np.arange(frames), 2) * 0.001
        poses[:, 1, 0] = 0.2
        poses[:, 1, 2] = 0.1
        jaw = np.full((frames, 2), 0.3)
        jaw[:2, 0] = [-0.1, 0.08]
        jaw[:, 1] = -0.1  # Already closed; it must not remotely capture a released needle.
        specification = {
            "usd": "needle.usda",
            "inputs_npz": "needle_inputs.npz",
            "body_path": "/TestNeedle/Needle",
            "initial_pose_world_xyz_xyzw": [0, 0, 0.1, 0, 0, 0, 1],
            "initial_held_arm": "psm1",
            "grasp": {
                "close_angle_rad": 0.05,
                "open_angle_rad": 0.18,
                "capture_distance_m": 0.002,
                "tool_body_paths": {"psm1": "arm1", "psm2": "arm2"},
                "tool_point_local_m": {"psm1": [0, 0, 0], "psm2": [0, 0, 0]},
            },
            "simulation": {"maximum_step_s": 1 / 240, "gravity_m_s2": [0, 0, -9.81]},
        }
        self.recording = SimpleNamespace(
            manifest={"needle": specification},
            body_paths=("arm1", "arm2"),
            body_poses=poses,
            frame_count=frames,
            first_valid_frame=0,
            last_valid_frame=frames - 1,
            timestamps_ns=np.arange(frames, dtype=np.int64) * 10_000_000,
            time_s=np.arange(frames) * 0.01,
            asset=lambda relative: self.root / relative,
        )
        self.jaw = jaw
        self._write_inputs()

    def tearDown(self):
        self.directory.cleanup()

    def _write_inputs(self):
        np.savez(
            self.root / "needle_inputs.npz",
            timestamps_ns=self.recording.timestamps_ns,
            jaw_opening_rad=self.jaw,
            centreline_local_m=np.array([[-0.001, 0, 0], [0, 0, 0], [0.001, 0, 0]]),
        )

    def _runtime(self):
        builder = newton.ModelBuilder()
        add_needle_to_builder(builder, self.recording)
        model = builder.finalize(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        state = model.state()
        return NeedleReplay(self.recording, model, state), state

    def test_hysteresis_release_gravity_and_no_far_capture(self):
        needle, state = self._runtime()
        initial_grip = needle.grip_transform.copy()
        held = needle.seek(1, state)
        self.assertEqual(needle.held_arm, "psm1")
        np.testing.assert_allclose(pose_matrix(held), needle.tool_transform(1, "psm1") @ initial_grip, atol=2e-7)
        released = needle.seek(2)
        self.assertIsNone(needle.held_arm)
        fallen = needle.seek(12)
        self.assertIsNone(needle.held_arm)
        self.assertGreater(fallen[0], released[0])  # release inherited tool velocity
        self.assertLess(fallen[2], released[2] - 0.025)  # gravity, not tool following
        settled = needle.seek(60)
        self.assertGreater(settled[2], -0.002)
        self.assertLess(settled[2], 0.003)
        self.assertEqual([event["event"] for event in needle.events], ["initial_grasp", "release"])

    def test_backward_seek_replays_the_same_physics(self):
        needle, _ = self._runtime()
        first = needle.seek(60)
        events = json.dumps(needle.events, sort_keys=True)
        needle.seek(0)
        second = needle.seek(60)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(events, json.dumps(needle.events, sort_keys=True))

    def test_repeated_closures_translate_into_each_grip_without_rotating(self):
        self.recording.manifest["needle"]["simulation"]["gravity_m_s2"] = [0, 0, 0]
        self.jaw[:] = 0.3
        self.jaw[:2, 0] = -0.1
        self.jaw[6:10, 1] = -0.1
        self.jaw[16:20, 1] = -0.1
        self.jaw[24:29, 0] = -0.1
        poses = self.recording.body_poses
        poses[1:, 0, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 0.6))
        poses[:, 1, 0] += np.arange(self.recording.frame_count) * 0.001
        poses[14:, 1, 0] += 0.04
        poses[:, 1, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.1))
        poses[7:, 1, 3:] = np.asarray(wp.quat_from_axis_angle(wp.vec3(0, 0, 1), 1.2))
        self._write_inputs()
        needle, state = self._runtime()
        for frame, arm in ((6, "psm2"), (16, "psm2"), (24, "psm1")):
            with self.subTest(frame=frame, arm=arm):
                before = pose_matrix(needle.seek(frame - 1))
                self.assertIsNone(needle.held_arm)
                after = pose_matrix(needle.seek(frame, state))
                self.assertEqual(needle.held_arm, arm)
                # Release occurs after rotation stops, so free motion has zero
                # angular velocity. A pickup must preserve that orientation.
                np.testing.assert_allclose(after[:3, :3], before[:3, :3], atol=2e-7)
                tool = needle.tool_transform(frame, arm)
                center = tool[:3, :3] @ needle.grasp_points[needle.arm_names.index(arm)] + tool[:3, 3]
                points = needle.centreline @ after[:3, :3].T + after[:3, 3]
                self.assertLess(np.linalg.norm(points - center, axis=1).min(), 2e-7)
                event = needle.events[-1]
                self.assertEqual(event["trigger"], "jaw_close")
                self.assertGreater(np.linalg.norm(event["translation_world_m"]), needle.capture_distance)
                relative = needle.grip_transform.copy()
                following = needle.seek(frame + 2)
                np.testing.assert_allclose(
                    pose_matrix(following), needle.tool_transform(frame + 2, arm) @ relative, atol=2e-7
                )
        final = needle.seek(60)
        events = json.dumps(needle.events, sort_keys=True)
        self.assertEqual([event["frame"] for event in needle.events if event["event"] == "grasp"], [6, 16, 24])
        needle.seek(0)
        np.testing.assert_array_equal(needle.seek(60), final)
        self.assertEqual(json.dumps(needle.events, sort_keys=True), events)

    def test_closure_while_other_arm_holds_does_not_arm_a_later_remote_pickup(self):
        self.jaw[:5, 0] = -0.1
        self.jaw[0, 1] = 0.3
        self.jaw[1:, 1] = -0.1
        self._write_inputs()
        needle, _ = self._runtime()
        needle.seek(60)
        self.assertIsNone(needle.held_arm)
        self.assertEqual([(e["frame"], e["event"]) for e in needle.events], [(0, "initial_grasp"), (5, "release")])

    def test_handoff_requires_close_jaw_and_preserves_pose(self):
        self.jaw[2:, 1] = -0.1
        self.recording.body_poses[2:, 1] = self.recording.body_poses[2:, 0]
        self._write_inputs()
        needle, _ = self._runtime()
        expected = needle.tool_transform(2, "psm1") @ needle.grip_transform
        transferred = needle.seek(2)
        self.assertEqual(needle.held_arm, "psm2")
        np.testing.assert_allclose(pose_matrix(transferred), expected, atol=2e-7)
        relative = needle.grip_transform.copy()
        next_pose = needle.seek(15)
        np.testing.assert_allclose(pose_matrix(next_pose), needle.tool_transform(15, "psm2") @ relative, atol=2e-7)
        self.assertEqual(needle.events[-1]["event"], "grasp")

    def test_mismatched_clock_is_rejected(self):
        self.recording.timestamps_ns = self.recording.timestamps_ns + 1
        with self.assertRaisesRegex(ValueError, "unchanged recording clock"):
            self._runtime()


if __name__ == "__main__":
    wp.init()
    unittest.main()

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
from newton.solvers import SolverMuJoCo
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from newton_solver_fem.examples._suturebot_needle import NeedleReplay, add_needle_to_builder, matrix_pose, pose_matrix
from newton_solver_fem.examples._suturebot_recording import DEFAULT_RECORDING, Recording


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


class TestBundledNeedleBacking(unittest.TestCase):
    def test_one_rectangular_backing_allows_tissue_penetration_and_stops_freefall(self):
        """Exercise the real prepared needle and collider through public MuJoCo APIs."""
        recording = Recording(os.environ.get("SUTUREBOT_TEST_RECORDING", DEFAULT_RECORDING))
        specification = recording.manifest["needle"]
        asset = recording.asset(specification["usd"])
        stage = Usd.Stage.Open(str(asset))
        static_colliders = []
        moving_colliders = []
        for prim in stage.Traverse():
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            ancestor = prim
            while ancestor and not ancestor.HasAPI(UsdPhysics.RigidBodyAPI):
                ancestor = ancestor.GetParent()
            (moving_colliders if ancestor else static_colliders).append(prim)
        self.assertEqual(len(static_colliders), 1, "Only the rectangular pad may collide with the needle")
        backing = static_colliders[0]
        self.assertTrue(backing.IsA(UsdGeom.Cube), "A mesh hull must not bridge the raised tissue")
        self.assertNotIn("TableSupport", str(backing.GetPath()))
        self.assertTrue(moving_colliders)
        self.assertTrue(all(prim.IsA(UsdGeom.Capsule) for prim in moving_colliders))

        matrix = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(backing)).T
        size = float(UsdGeom.Cube(backing).GetSizeAttr().Get())
        lengths = np.linalg.norm(matrix[:3, :3], axis=0)
        rotation = matrix[:3, :3] / lengths
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=2e-7)
        normal = rotation[:, 2]
        top = float(matrix[:3, 3] @ normal + size * lengths[2] / 2)
        labels = json.loads(recording.asset(recording.manifest["point_labels"]["report"]).read_text())
        insertion = next(point for point in labels["points"] if point["label"] == "insertion")
        self.assertEqual(insertion["region_id"], 1)
        insertion_world = np.asarray(insertion["position_world_m"])
        self.assertGreater(float(insertion_world @ normal - top), 0.005)
        with np.load(recording.asset(specification["inputs_npz"]), allow_pickle=False) as archive:
            centreline = archive["centreline_local_m"].copy()

        # Start the entire flat needle below the raised region-1 surface, but
        # above the backing. This detects a retained tissue hull as well as a
        # missing or disabled backing, without relying on collision labels.
        start = np.eye(4)
        start[:3, :3] = rotation
        target = insertion_world + normal * (top + 0.004 - insertion_world @ normal)
        start[:3, 3] = target - rotation @ centreline.mean(axis=0)
        with wp.ScopedDevice("cpu"):
            builder = newton.ModelBuilder()
            builder.default_shape_cfg.ke = specification["simulation"].get("contact_stiffness", 1e6)
            builder.default_shape_cfg.kd = specification["simulation"].get("contact_damping", 2000)
            add_needle_to_builder(builder, recording)
            model = builder.finalize(device="cpu")
            model.set_gravity(tuple(-9.81 * normal))
            self.assertEqual(model.body_count, 1)
            state, following = model.state(), model.state()
            control = model.control()
            solver = SolverMuJoCo(
                model,
                use_mujoco_cpu=True,
                iterations=30,
                ls_iterations=10,
                integrator="implicitfast",
                nconmax=4 * model.shape_count,
                njmax=24 * model.shape_count,
            )
            solver.reset(state)
            state.body_q.assign(np.asarray([matrix_pose(start)], np.float32))
            state.body_qd.zero_()
            newton.eval_ik(model, state, state.joint_q, state.joint_qd)
            initial_height = float(state.body_q.numpy()[0, :3] @ normal)
            for step in range(400):
                state.clear_forces()
                solver.step(state, following, control, None, 0.001)
                state, following = following, state
                if step == 9:
                    drop = initial_height - float(state.body_q.numpy()[0, :3] @ normal)
                    self.assertGreater(drop, 0.0004, "Raised tissue unexpectedly blocks the released needle")
                    self.assertLess(drop, 0.0007)
            final = pose_matrix(state.body_q.numpy()[0])
        heights = (centreline @ final[:3, :3].T + final[:3, 3]) @ normal - top
        self.assertGreater(float(heights.min()), -0.0005, "Needle fell through the rectangular backing")
        self.assertLess(float(heights.min()), 0.0008, "Needle remained suspended above the backing")


class TestBundledNeedleRegrasp(unittest.TestCase):
    def test_source_frame_446_recovers_the_released_needle(self):
        recording = Recording(DEFAULT_RECORDING)
        builder = newton.ModelBuilder()
        add_needle_to_builder(builder, recording)
        model = builder.finalize(device=os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        needle = NeedleReplay(recording, model, model.state())
        needle.seek(445)
        self.assertIsNone(needle.held_arm)
        snapped = pose_matrix(needle.seek(446))
        event = needle.events[-1]
        self.assertEqual((event["frame"], event["arm"], event["trigger"]), (446, "psm1", "jaw_close"))
        self.assertGreater(event["distance_m"], needle.capture_distance)
        tool = needle.tool_transform(446, "psm1")
        center = tool[:3, :3] @ needle.grasp_points[0] + tool[:3, 3]
        points = needle.centreline @ snapped[:3, :3].T + snapped[:3, 3]
        self.assertLess(np.linalg.norm(points - center, axis=1).min(), 2e-7)
        grip = needle.grip_transform.copy()
        later = needle.seek(480)
        np.testing.assert_allclose(pose_matrix(later), needle.tool_transform(480, "psm1") @ grip, atol=2e-7)
        self.assertEqual([e["frame"] for e in needle.events if e["event"] == "grasp"], [446])


if __name__ == "__main__":
    wp.init()
    unittest.main()

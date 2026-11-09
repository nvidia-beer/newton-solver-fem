# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Exact recording clocks, camera/body correspondence and prepared USD replay."""

import hashlib
import importlib.util
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples._suturebot_recording import CAMERAS, DEFAULT_RECORDING, Recording, bundled_recordings
from newton_solver_fem.examples.example_fem_suturebot import Example


class RecordingClockTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.addCleanup(self.directory.cleanup)
        self.timestamps = np.array([1745891580486691183, 1745891580516691183, 1745891580556691183], dtype=np.int64)
        self.times = (self.timestamps - self.timestamps[0]) * 1.0e-9
        self.poses = np.zeros((3, 1, 7))
        self.poses[..., 6] = 1
        self.body_valid = np.ones(3, dtype=bool)
        self.camera_poses = np.broadcast_to(np.eye(4), (3, 4, 4, 4)).copy()
        self.valid = np.ones((3, 4), dtype=bool)
        self.manifest = {
            "schema_version": 1,
            "scene_usd": "scene.usda",
            "motion_npz": "motion.npz",
            "camera_poses_npz": "cameras.npz",
            "body_paths": ["/Rig/Tool"],
            "camera_order": list(CAMERAS),
            "cameras": {name: {"frame_count": 3} for name in CAMERAS},
            "timing": {"nominal_fps": 30, "frame_count": 3},
        }
        (self.root / "scene.usda").write_text("#usda 1.0\n")
        self.save()

    def save(self, camera_timestamps=None):
        np.savez(
            self.root / "motion.npz",
            timestamps_ns=self.timestamps,
            time_s=self.times,
            body_poses=self.poses,
            body_valid=self.body_valid,
        )
        np.savez(
            self.root / "cameras.npz",
            timestamps_ns=self.timestamps if camera_timestamps is None else camera_timestamps,
            camera_poses_cv=self.camera_poses,
            camera_valid=self.valid,
        )
        (self.root / "recording.json").write_text(json.dumps(self.manifest))

    def load(self):
        return Recording(self.root / "recording.json")

    def test_nonuniform_nanosecond_clock_seek_and_loop(self):
        recording = self.load()
        np.testing.assert_array_equal(recording.timestamps_ns, self.timestamps)
        self.assertEqual(recording.frame_at(-1), 0)
        self.assertEqual(recording.frame_at(0.029999), 0)
        self.assertEqual(recording.frame_at(0.030001), 1)
        self.assertEqual(recording.frame_at(0.069999), 1)
        self.assertEqual(recording.frame_at(100), 2)
        self.assertEqual(recording.frame_at(recording.loop_duration + 0.031, loop=True), 1)
        with self.assertRaises(ValueError):
            recording.frame_at(np.nan)

    def test_camera_clock_cannot_drift_even_one_nanosecond(self):
        camera_time = self.timestamps.copy()
        camera_time[1] += 1
        self.save(camera_time)
        with self.assertRaisesRegex(ValueError, "different source timestamps"):
            self.load()

    def test_unsupported_clock_boundary_is_never_replayed(self):
        self.body_valid[0] = False
        self.poses[0] = np.nan
        self.save()
        recording = self.load()
        self.assertEqual(recording.first_valid_frame, 1)
        self.assertEqual(recording.frame_at(0), 1)
        self.assertEqual(recording.frame_at(recording.start_time + recording.loop_duration, loop=True), 1)
        self.assertFalse(any(recording.camera_frame(0)[1].values()))

    def test_internal_clock_gaps_are_rejected(self):
        self.body_valid[1] = False
        self.save()
        with self.assertRaisesRegex(ValueError, "contiguous supported"):
            self.load()

    def test_source_phase_limits_playback_without_rewriting_source_clock(self):
        self.manifest["playback"] = {"first_frame": 1, "last_frame": 2}
        self.save()
        recording = self.load()
        self.assertEqual(recording.first_valid_frame, 0)
        np.testing.assert_array_equal(recording.timestamps_ns, self.timestamps)
        np.testing.assert_array_equal(recording.body_valid, self.body_valid)
        self.assertEqual(recording.frame_at(-1), 1)
        self.assertEqual(recording.frame_at(100), 2)
        self.assertEqual(recording.frame_at(recording.start_time + recording.loop_duration, loop=True), 1)
        for start, end in ((-1, 2), (0, 3), (2, 1), (1, 1), (0.5, 2)):
            with self.subTest(start=start, end=end):
                self.manifest["playback"] = {"first_frame": start, "last_frame": end}
                self.save()
                with self.assertRaisesRegex(ValueError, "Playback window"):
                    self.load()

    def test_float_timestamps_are_rejected(self):
        self.timestamps = self.timestamps.astype(np.float64)
        self.save()
        with self.assertRaisesRegex(ValueError, "int64 timestamps"):
            self.load()

    def test_improper_camera_frames_are_rejected(self):
        self.camera_poses[0, 0, 0, 0] = -1.0
        self.save()
        with self.assertRaisesRegex(ValueError, "proper rigid transforms"):
            self.load()

    def test_missing_camera_pose_stays_explicitly_unavailable(self):
        self.valid[0, 2] = False
        self.camera_poses[0, 2] = np.nan
        self.save()
        recording = self.load()
        poses, validity = recording.camera_frame(0)
        self.assertFalse(validity["psm1"])
        self.assertTrue(np.isnan(poses["psm1"]).all())
        self.assertTrue(validity["left"])


@unittest.skipUnless(importlib.util.find_spec("pxr"), "The examples extra provides the USD importer")
class PreparedReplayTests(unittest.TestCase):
    def test_tool_mesh_alias_uses_the_default_geometry_and_recorded_motion(self):
        original = Recording(DEFAULT_RECORDING)
        preview = Recording("tissue3_215300_mesh_preview")
        self.assertEqual(original.scene_path, preview.scene_path)
        self.assertFalse(preview.manifest["alignment"]["accepted"])
        allowed = {"sample_id", "provenance_file", "playback", "tool_mesh_preview"}
        for key in original.manifest.keys() | preview.manifest.keys():
            if key not in allowed:
                self.assertEqual(original.manifest.get(key), preview.manifest.get(key), key)
        np.testing.assert_array_equal(original.body_poses, preview.body_poses)
        np.testing.assert_array_equal(original.camera_poses, preview.camera_poses)
        self.assertEqual((original.first_frame, original.last_frame), (preview.first_frame, preview.last_frame))

    def test_every_recording_uses_distinct_instrument_assets_with_unchanged_scene_poses(self):
        from pxr import Usd, UsdGeom

        shared_hashes = None
        for name, source in bundled_recordings().items():
            with self.subTest(recording=name):
                recording = Recording(source)
                geometry = recording.manifest["tool_geometry"]
                self.assertFalse(geometry["camera_alignment_validated"])
                files = {arm: recording.asset(path) for arm, path in geometry["files"].items()}
                self.assertNotEqual(files["psm1"], files["psm2"])
                hashes = {arm: hashlib.sha256(path.read_bytes()).hexdigest() for arm, path in files.items()}
                if shared_hashes is None:
                    shared_hashes = hashes
                self.assertEqual(hashes, shared_hashes)
                before = Usd.Stage.Open(str(recording.root / "scene.usda"))
                after = Usd.Stage.Open(str(recording.scene_path))
                self.assertEqual(
                    [str(p.GetPath()) for p in before.Traverse()], [str(p.GetPath()) for p in after.Traverse()]
                )
                old_transforms, new_transforms = UsdGeom.XformCache(), UsdGeom.XformCache()
                changed = set()
                for prim in before.Traverse():
                    path = prim.GetPath()
                    other = after.GetPrimAtPath(path)
                    if prim.IsA(UsdGeom.Xformable):
                        np.testing.assert_array_equal(
                            old_transforms.GetLocalToWorldTransform(prim),
                            new_transforms.GetLocalToWorldTransform(other),
                        )
                    if prim.IsA(UsdGeom.Mesh):
                        a = np.asarray(UsdGeom.Mesh(prim).GetPointsAttr().Get())
                        b = np.asarray(UsdGeom.Mesh(other).GetPointsAttr().Get())
                        if not np.array_equal(a, b):
                            changed.add(str(path))
                self.assertEqual(
                    changed,
                    {
                        f"/SutureBotReplay/{arm}/{part}/CAD"
                        for arm, part in (("PSM1", "roll"), ("PSM2", "roll"), ("PSM2", "jaw1"), ("PSM2", "jaw2"))
                    },
                )
                for arm, file in files.items():
                    asset = Usd.Stage.Open(str(file))
                    self.assertEqual(len([p for p in asset.Traverse() if p.IsA(UsdGeom.Mesh)]), 5)
                    self.assertEqual(len([layer for layer in asset.GetUsedLayers() if layer.realPath]), 1)
                    for link in ("roll", "pitch", "yaw", "jaw1", "jaw2"):
                        mesh = UsdGeom.Mesh(after.GetPrimAtPath(f"/SutureBotReplay/{arm.upper()}/{link}/CAD"))
                        reference = UsdGeom.Mesh(asset.GetPrimAtPath(f"/Tool/{link}/CAD"))
                        for attr in ("points", "faceVertexCounts", "faceVertexIndices"):
                            actual = mesh.GetPrim().GetAttribute(attr)
                            np.testing.assert_array_equal(actual.Get(), reference.GetPrim().GetAttribute(attr).Get())
                            self.assertEqual(Path(actual.GetPropertyStack()[0].layer.realPath).resolve(), file)
                        if link == "roll":
                            points = np.asarray(mesh.GetPointsAttr().Get())
                            original = np.asarray(before.GetPrimAtPath(mesh.GetPath()).GetAttribute("points").Get())
                            np.testing.assert_array_equal(points[:, 2], original[:, 2])
                            ring = points[points[:, 2] == points[:, 2].min()]
                            self.assertAlmostEqual(float(np.ptp(ring[:, 0])), 0.008356937895, places=7)
                for jaw in ("jaw1", "jaw2"):
                    mesh = UsdGeom.Mesh(after.GetPrimAtPath(f"/SutureBotReplay/PSM2/{jaw}/CAD"))
                    points = np.asarray(mesh.GetPointsAttr().Get())
                    self.assertAlmostEqual(float(points[:, 1].max()), 0.012, places=7)
                    side = -1 if jaw == "jaw1" else 1
                    # Distal faces oppose each other without crossing the closed gap.
                    self.assertGreaterEqual(float((side * points[points[:, 1] >= 0.005, 0]).min()), 0.00002499)
                    faces = np.asarray(mesh.GetFaceVertexIndicesAttr().Get()).reshape(-1, 3)
                    triangles = points[faces]
                    centers = triangles.mean(axis=1)
                    inner = (centers[:, 1] > 0.005) & (centers[:, 1] < 0.0115) & (side * centers[:, 0] < 0.00009)
                    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
                    self.assertGreater(float((-side * normals[inner, 0]).sum()), 0.0)

    def test_switch_recordings_preserves_each_source_clock_and_replays_only_prepared_scene(self):
        samples = bundled_recordings()
        self.assertIn("tissue3_215300", samples)
        # A second clock fixture exercises switching even when calibration
        # review rejects additional real recordings. It is deliberately a test
        # copy, never published or presented as another validated dataset clip.
        with tempfile.TemporaryDirectory() as directory, wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            copy = Path(directory) / "different_clock"
            shutil.copytree(DEFAULT_RECORDING.parent, copy)
            for filename in ("motion.npz", "camera_poses.npz"):
                with np.load(copy / filename, allow_pickle=False) as archive:
                    values = dict(archive)
                times = values["timestamps_ns"]
                # Preserve nonuniform intervals while changing both epoch and
                # playback duration. Integer arithmetic retains nanoseconds.
                values["timestamps_ns"] = times[0] + 900_000_000_000 + (times - times[0]) * 3 // 2
                if "time_s" in values:
                    values["time_s"] = (values["timestamps_ns"] - values["timestamps_ns"][0]) * 1.0e-9
                np.savez_compressed(copy / filename, **values)
            path = copy / "recording.json"
            manifest = json.loads(path.read_text())
            manifest["sample_id"] = "synthetic_clock_fixture"
            with np.load(copy / "motion.npz", allow_pickle=False) as motion:
                manifest["timing"]["duration_s"] = float(motion["time_s"][-1])
            path.write_text(json.dumps(manifest))
            viewer = ViewerNull()
            args = Example.create_parser().parse_args(["--no-camera-overlays"])
            example = Example(viewer, args)
            try:
                for source in [*samples.values(), path, DEFAULT_RECORDING]:
                    example.load_recording(source)
                    recording = example.recording
                    self.assertEqual(recording.path, source.resolve())
                    self.assertIn(recording.manifest["region_id"], range(1, 7))
                    self.assertNotIn("needle", recording.manifest)
                    self.assertNotIn("thread", recording.manifest)
                    self.assertFalse(any("needle" in label.lower() for label in example.model.body_label))
                    self.assertFalse(any("thread" in label.lower() for label in example.model.shape_label))
                    for frame in range(recording.first_frame, recording.last_frame + 1):
                        example.seek(frame)
                        example.test_final()
                        self.assertEqual(recording.frame_at(example.elapsed), frame)
                    end = example.state_0.body_q.numpy().copy()
                    example.seek(recording.first_frame)
                    example.seek(recording.last_frame)
                    np.testing.assert_allclose(example.state_0.body_q.numpy(), end, atol=1.0e-7, rtol=0)
            finally:
                example.close()
                viewer.close()

    def test_copied_package_has_no_dependency_on_preparation_checkout(self):
        from pxr import Usd

        with tempfile.TemporaryDirectory() as directory:
            for name, source in bundled_recordings().items():
                with self.subTest(recording=name):
                    root = Path(directory) / name
                    shutil.copytree(source.parent, root)
                    recording = Recording(root / source.name)
                    for path in root.rglob("*.json"):
                        text = path.read_text()
                        self.assertNotIn("/home/", text, str(path.relative_to(root)))
                        self.assertNotIn("file:///", text, str(path.relative_to(root)))
                    stage = Usd.Stage.Open(str(recording.scene_path))
                    for layer in stage.GetUsedLayers():
                        if layer.realPath:
                            self.assertTrue(Path(layer.realPath).resolve().is_relative_to(root), layer.realPath)
                    for camera in recording.cameras.values():
                        recording.asset(camera["ray_directions_cv_path"])
                        recording.asset(camera["ray_valid_path"])
                        for frame in (recording.first_frame, recording.last_frame):
                            recording.asset(camera["frame_path_pattern"].format(frame=frame))
                    with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
                        viewer = ViewerNull()
                        args = Example.create_parser().parse_args(
                            ["--sample", str(recording.path), "--no-camera-overlays"]
                        )
                        example = Example(viewer, args)
                        try:
                            example.seek((recording.first_frame + recording.last_frame) // 2)
                            example.test_final()
                        finally:
                            example.close()
                            viewer.close()

    def test_packaged_usd_and_all_measured_frames_share_the_camera_clock(self):
        self.assertTrue(DEFAULT_RECORDING.is_file(), "Publish the prepared sample before running replay tests")
        with wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")):
            viewer = ViewerNull()
            args = Example.create_parser().parse_args(["--no-camera-overlays"])
            example = Example(viewer, args)
            try:
                self.assertEqual(example.recording.manifest["region_id"], 1)
                self.assertEqual(example.recording.frame_count, 601)
                self.assertGreater(example.model.shape_count, 10)
                self.assertEqual(example.model.particle_count, 0)
                for frame in range(example.recording.first_valid_frame, example.recording.last_valid_frame + 1):
                    example.seek(frame)
                    example.test_final()
                    self.assertEqual(example.recording.frame_at(example.elapsed), frame)
                example.seek(0)
                self.assertEqual(example.frame_index, example.recording.first_valid_frame)
                start = example.state_0.body_q.numpy().copy()
                example.seek(300)
                self.assertGreater(np.linalg.norm(example.state_0.body_q.numpy()[:, :3] - start[:, :3]), 0.001)
                self.assertIsNone(example.overlay)
            finally:
                example.close()
                viewer.close()


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepared material labels stay on actual reference triangles and render through USD UVs."""

import json
import os
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton.sensors import SensorCamera
from PIL import Image
from pxr import Usd, UsdGeom

from newton_solver_fem.examples._suturebot_cameras import camera_rays_cv, camera_transform_cv
from newton_solver_fem.examples._suturebot_recording import DEFAULT_RECORDING, Recording


class TestSutureBotPointLabels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recording = Recording(Path(os.environ.get("SUTUREBOT_RECORDING", DEFAULT_RECORDING)))
        metadata = cls.recording.manifest.get("point_labels")
        if metadata is None:
            raise unittest.SkipTest("This recording has no prepared dataset point labels")
        cls.report = json.loads(cls.recording.asset(metadata["report"]).read_text())
        cls.device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def test_reference_triangle_bindings(self):
        report = self.report
        stage = Usd.Stage.Open(str(self.recording.asset(self.recording.manifest["phantom"]["asset"])))
        self.assertEqual(len(report["points"]), 2)
        self.assertEqual(report["annotation_frame"], self.recording.frame_count - 1)
        self.assertTrue(report["geometry_arrays_unchanged"])
        for row in report["points"]:
            prim = stage.GetPrimAtPath(row["prim_path"])
            points = np.asarray(prim.GetAttribute("newton:referencePoints").Get())
            triangles = np.asarray(prim.GetAttribute("faceVertexIndices").Get()).reshape(-1, 3)
            face = triangles[row["triangle_index"]]
            np.testing.assert_array_equal(face, row["reference_vertex_indices"])
            weights = np.asarray(row["barycentric"])
            self.assertGreaterEqual(weights.min(), 0)
            self.assertAlmostEqual(weights.sum(), 1, places=6)
            transform = np.asarray(UsdGeom.XformCache().GetLocalToWorldTransform(prim))
            position = (np.r_[weights @ points[face], 1] @ transform)[:3] * UsdGeom.GetStageMetersPerUnit(stage)
            np.testing.assert_allclose(position, row["position_world_m"], atol=2e-8, rtol=0)
            coordinates = UsdGeom.PrimvarsAPI(prim).GetPrimvar("st")
            self.assertEqual(coordinates.GetInterpolation(), UsdGeom.Tokens.faceVarying)
            uv = np.asarray(coordinates.Get()).reshape(-1, 3, 2)[row["triangle_index"]]
            np.testing.assert_allclose(weights @ uv, row["uv"], atol=1e-7, rtol=0)

    def test_actual_texture_at_material_points(self):
        with wp.ScopedDevice(self.device):
            builder = newton.ModelBuilder()
            builder.add_usd(str(self.recording.asset(self.recording.manifest["phantom"]["asset"])))
            model = builder.finalize(device=self.device)
            sensor = SensorCamera(
                model,
                load_textures=True,
                default_render_config=SensorCamera.RenderConfig(
                    enable_textures=True, enable_particles=False, enable_backface_culling=False
                ),
            )
            pose = self.recording.camera_poses[self.report["annotation_frame"], 0]
            points = np.array([row["position_world_m"] for row in self.report["points"]] + [[0.0, 0.0, 0.0]])
            directions = (points - pose[:3, 3]) @ pose[:3, :3]
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
            rays, _ = camera_rays_cv(directions[None].astype(np.float32), device=self.device)
            albedo = sensor.allocate_image_albedo(1, 3, 1)
            sensor.update(
                model.state(),
                wp.array([camera_transform_cv(pose)], dtype=wp.transform, device=self.device),
                rays,
                albedo_image=albedo,
            )
            colors = SensorCamera.Utils.to_rgba_from_color(albedo).numpy()[0, 0, :, :3]
            # Check the visible display color and an unlabelled pad point. A disabled texture
            # renders the whole pad white and must fail this check.
            source = np.asarray(Image.open(self.recording.asset(self.report["source_mask"])).convert("RGB"))
            expected = []
            for row in self.report["points"]:
                x, y = np.rint(row["pixel_original"]).astype(int)
                self.assertTrue(np.any(source[y, x] > 0))
                expected.append(self.report["display_color_rgb"])
            self.assertEqual(self.report["display_color_rgb"], [0, 255, 0])
            np.testing.assert_allclose(colors[:2], expected, atol=10, rtol=0)
            atlas = np.asarray(Image.open(self.recording.asset(self.report["texture"])).convert("RGB"))
            np.testing.assert_allclose(colors[2], atlas[0, 0], atol=2, rtol=0)
            self.assertTrue(
                any(
                    source.texture is not None and source.uvs is not None
                    for source in model.shape_source
                    if source is not None
                )
            )


if __name__ == "__main__":
    unittest.main()

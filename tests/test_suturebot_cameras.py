# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrated ray geometry and actual depth-tested recording overlays."""

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

import newton
import numpy as np
import warp as wp

from newton_solver_fem.examples._suturebot_cameras import (
    CAMERA_NAMES,
    FourCameraOverlay,
    camera_rays_cv,
    camera_transform_cv,
    composite_overlay,
    visible_surface_edges,
)


class CalibratedCameraTests(unittest.TestCase):
    def test_cv_rays_and_pose_preserve_world_projection(self):
        direction = np.array([0.2, -0.3, 1.0])
        direction /= np.linalg.norm(direction)
        rays, valid = camera_rays_cv(np.array([[direction, [np.nan, np.nan, np.nan]]]), [[True, False]], device="cpu")
        angle = 0.43
        pose = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0, 0.12],
                [np.sin(angle), np.cos(angle), 0, -0.04],
                [0, 0, 1, 0.3],
                [0, 0, 0, 1],
            ]
        )
        transform = camera_transform_cv(pose)
        world_ray = wp.transform_vector(transform, wp.vec3(*rays.numpy()[0, 0, 0, 1]))
        np.testing.assert_allclose(world_ray, pose[:3, :3] @ direction, atol=1.0e-7)
        np.testing.assert_allclose(np.asarray(transform)[:3], pose[:3, 3], atol=1.0e-7)
        np.testing.assert_array_equal(rays.numpy()[0, 1], 0)
        np.testing.assert_array_equal(valid, [[True, False]])
        with self.assertRaises(ValueError):
            camera_rays_cv(np.ones((1, 1, 3)), device="cpu")
        with self.assertRaises(ValueError):
            camera_transform_cv(np.diag([-1, 1, 1, 1]))

    def test_composite_preserves_uncovered_and_invalid_pixels(self):
        recorded = np.full((6, 8, 3), (20, 40, 60), dtype=np.uint8)
        rendered = np.full_like(recorded, (200, 100, 80))
        depth = np.zeros((6, 8), dtype=np.float32)
        depth[1:5, 1:7] = 0.2
        valid = np.ones((6, 8), dtype=bool)
        valid[2:4, 5:7] = False
        normals = np.zeros((6, 8, 3), dtype=np.float32)
        normals[..., 2] = -1.0
        image = composite_overlay(recorded, rendered, depth, normals, valid, opacity=0.25)
        mask = (depth > 0) & valid
        np.testing.assert_array_equal(image[~mask], recorded[~mask])
        np.testing.assert_array_equal(image[2, 3], (65, 55, 65))
        self.assertTrue(np.any(image[mask] != recorded[mask]))

    def test_edges_ignore_coplanar_partitions_but_preserve_geometry(self):
        depth = np.full((8, 12), 0.1, dtype=np.float32)
        normals = np.zeros((8, 12, 3), dtype=np.float32)
        normals[..., 2] = -1.0
        valid = np.ones((8, 12), dtype=bool)
        # Region/shape ownership is deliberately not an input. A plane split
        # into separate USD meshes must have no artificial interior outline.
        np.testing.assert_array_equal(visible_surface_edges(depth, normals, valid), False)
        # A continuous crease at equal depth remains visible from its normals.
        normals[:, 6:] = (0.8, 0.0, -0.6)
        edges = visible_surface_edges(depth, normals, valid)
        self.assertTrue(edges[:, 5:7].all())
        self.assertFalse(edges[:, :5].any())
        self.assertFalse(edges[:, 7:].any())
        # Parallel separated surfaces still outline their occlusion boundary.
        normals[..., 0], normals[..., 2] = 0.0, -1.0
        depth[:, 6:] = 0.12
        self.assertTrue(visible_surface_edges(depth, normals, valid)[:, 5:7].all())

    @unittest.skipUnless(importlib.util.find_spec("PIL"), "Install the examples extra for prepared image decoding")
    def test_sensor_projection_occlusion_motion_and_unavailable_frames(self):
        from PIL import Image

        device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
        width, height = 64, 48
        fx, fy, cx, cy = 80.0, 75.0, 19.0, 14.0
        y, x = np.indices((height, width))
        directions = np.stack(((x - cx) / fx, (y - cy) / fy, np.ones_like(x)), axis=-1)
        directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
        angle = 0.35
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        camera_pose = np.eye(4)
        camera_pose[:3, :3], camera_pose[:3, 3] = rotation, (0.12, -0.06, 0.03)

        def body_transform(camera_point):
            return wp.transform(
                wp.vec3(*(rotation @ camera_point + camera_pose[:3, 3])),
                wp.quat_from_matrix(wp.mat33(rotation.astype(np.float32))),
            )

        with wp.ScopedDevice(device), tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            np.save(path / "rays.npy", directions.astype(np.float32))
            recorded = np.full((height, width, 3), (30, 50, 70), dtype=np.uint8)
            Image.fromarray(recorded).save(path / "000000.png")
            cameras = {
                name: dict(
                    overlay_size=[width, height],
                    ray_directions_cv_path="rays.npy",
                    frame_path_pattern="{frame:06d}.png",
                    frame_count=1,
                )
                for name in CAMERA_NAMES
            }
            builder = newton.ModelBuilder()
            body = builder.add_body(xform=body_transform((0.014, 0.009, 0.10)))
            front = builder.add_shape_box(body, hx=0.006, hy=0.004, hz=0.002, color=(0.8, 0.3, 0.1))
            back = builder.add_body(xform=body_transform((0.021, 0.0135, 0.15)))
            builder.add_shape_box(back, hx=0.01, hy=0.008, hz=0.002, color=(0.1, 0.3, 0.8))
            model = builder.finalize()
            state = model.state()
            overlay = FourCameraOverlay(model, cameras, path)
            poses = dict.fromkeys(CAMERA_NAMES, camera_pose)
            try:
                result = overlay.render(state, 0, poses)
                front_mask = overlay.shape_indices["left"] == front
                py, px = np.nonzero(front_mask)
                self.assertGreater(len(px), 12)
                expected = (fx * 0.014 / 0.098 + cx, fy * 0.009 / 0.098 + cy)
                np.testing.assert_allclose((px.mean(), py.mean()), expected, atol=0.8)
                u, v = np.rint(expected).astype(int)
                self.assertEqual(overlay.shape_indices["left"][v, u], front)
                self.assertAlmostEqual(float(overlay.depth["left"][v, u]), 0.098 / directions[v, u, 2], places=5)
                for name in CAMERA_NAMES:
                    self.assertGreater(overlay.coverage[name], 0)
                    np.testing.assert_array_equal(result[name][0, 0], recorded[0, 0])

                # Change actual body state at the same image index: rendering
                # must refit public model bounds, not reuse an old silhouette.
                transforms = state.body_q.numpy()
                transforms[body] = np.asarray(body_transform((-0.014, 0.009, 0.10)))
                state.body_q.assign(transforms)
                overlay.render(state, 0, poses)
                _, moved_x = np.nonzero(overlay.shape_indices["left"] == front)
                self.assertLess(moved_x.mean(), px.mean() - 18)
                self.assertNotEqual(overlay.shape_indices["left"][v, u], front)

                overlay.show_overlay = False
                overlay.render(state, 0, poses)
                np.testing.assert_array_equal(overlay.images["left"], recorded)
                overlay.render(state, 0, poses, dict.fromkeys(CAMERA_NAMES, False))
                for name in CAMERA_NAMES:
                    np.testing.assert_array_equal(overlay.images[name], 0)
                    self.assertEqual(overlay.coverage[name], 0)
                overlay.render(state, 1, poses)
                np.testing.assert_array_equal(overlay.images["left"], 0)
                self.assertIn("Unavailable", overlay.status["left"])
            finally:
                overlay.close()


if __name__ == "__main__":
    unittest.main()

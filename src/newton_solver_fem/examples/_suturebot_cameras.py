# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Calibrated model overlays on four prepared, exactly indexed recording views.

Preparation owns lens inversion, image resizing and camera trajectories. This
module reads those assets, renders the actual Newton model through the supplied
rays and poses, and composites depth-tested surfaces onto the recorded pixels.
It neither estimates alignment nor changes calibration or installed Newton code.
"""

from pathlib import Path

import numpy as np
import warp as wp
from newton.sensors import SensorCamera

CAMERA_NAMES = ("left", "right", "psm1", "psm2")
_CV_TO_GL = np.diag((1.0, -1.0, -1.0))


def camera_transform_cv(pose):
    """Convert a proper camera-to-world CV matrix to Newton's GL camera pose."""
    pose = np.asarray(pose, dtype=np.float64)
    if (
        pose.shape != (4, 4)
        or not np.isfinite(pose).all()
        or not np.allclose(pose[3], (0.0, 0.0, 0.0, 1.0), atol=1.0e-7, rtol=0.0)
        or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1.0e-5, rtol=0.0)
        or not np.isclose(np.linalg.det(pose[:3, :3]), 1.0, atol=1.0e-5, rtol=0.0)
    ):
        raise ValueError("A finite proper T_world_from_camera_cv transform is required")
    rotation = wp.mat33((pose[:3, :3] @ _CV_TO_GL).astype(np.float32))
    return wp.transform(wp.vec3(*pose[:3, 3]), wp.quat_from_matrix(rotation))


def camera_rays_cv(directions, valid=None, *, device=None):
    """Wrap prepared CV inverse-lens rays in the public SensorCamera format.

    Rays correspond to the prepared image pixel centers, including the original
    distortion. Invalid rays stay zero; they are never repaired or clamped.
    """
    directions = np.asarray(directions, dtype=np.float32)
    if directions.ndim != 3 or directions.shape[2] != 3 or min(directions.shape[:2]) < 1:
        raise ValueError("Prepared camera rays must have shape (height, width, 3)")
    if valid is None:
        valid = np.isfinite(directions).all(axis=2) & (np.linalg.norm(directions, axis=2) > 0.0)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape != directions.shape[:2]:
        raise ValueError("Camera ray validity must match the prepared image dimensions")
    selected = directions[valid]
    if not np.isfinite(selected).all() or not np.allclose(np.linalg.norm(selected, axis=1), 1.0, atol=1.0e-4):
        raise ValueError("Valid prepared camera rays must be finite unit directions")
    rays = np.zeros((*valid.shape, 1, 2, 3), dtype=np.float32)
    rays[:, :, 0, 1] = np.where(valid[..., None], directions, 0.0) @ _CV_TO_GL
    return wp.array(rays, dtype=wp.vec3, device=device), valid


def visible_surface_edges(depth, normals, valid):
    """Find visible geometric discontinuities, independent of USD partition IDs.

    Adjacent coplanar pad regions are a continuous surface. A normal turn of
    more than 40 degrees or a substantial depth jump marks a geometric edge;
    the thresholds suppress tiny triangle faceting in the prepared scan.
    """
    depth, normals = np.asarray(depth), np.asarray(normals)
    mask = (depth > 0.0) & np.asarray(valid, dtype=bool)
    edges = np.zeros_like(mask)
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis], second[axis] = slice(1, None), slice(None, -1)
        first, second = tuple(first), tuple(second)
        both = mask[first] & mask[second]
        turn = np.sum(normals[first] * normals[second], axis=-1) < np.cos(np.deg2rad(40.0))
        jump = np.abs(depth[first] - depth[second]) > np.maximum(
            0.0015, 0.025 * np.minimum(depth[first], depth[second])
        )
        boundary = (mask[first] != mask[second]) | (both & (turn | jump))
        edges[first] |= boundary & mask[first]
        edges[second] |= boundary & mask[second]
    return edges


def composite_overlay(recorded, rendered, depth, normals, valid, *, opacity=0.18):
    """Blend visible model surfaces and outlines; preserve every background pixel.

    Depth ordering is between model surfaces. The recording has no measured
    depth and cannot occlude a model surface merely because it looks foreground.
    """
    image = np.asarray(recorded, dtype=np.uint8).copy()
    mask = (np.asarray(depth) > 0.0) & np.asarray(valid, dtype=bool)
    if image.shape != (*mask.shape, 3) or np.asarray(rendered).shape != image.shape:
        raise ValueError("Rendered and recorded images must have identical RGB dimensions")
    if not 0.0 <= opacity <= 1.0:
        raise ValueError("Overlay opacity must be in [0, 1]")
    image[mask] = np.rint((1.0 - opacity) * image[mask] + opacity * np.asarray(rendered)[mask]).astype(np.uint8)
    edges = visible_surface_edges(depth, normals, valid)
    image[edges] = (55, 235, 245)
    return image


class FourCameraOverlay:
    """Render one model through four prepared camera views, with optional ImGui UI.

    ``cameras`` maps ``left``, ``right``, ``psm1`` and ``psm2`` to records with
    ``overlay_size`` [width,height], ``ray_directions_cv_path`` (.npy), optional
    ``ray_valid_path`` (.npy), ``frame_path_pattern`` (e.g.
    ``images/left/{frame:06d}.jpg``) and ``frame_count``. Paths are relative to
    ``assets_root`` unless absolute. K, D and native video paths can remain in
    the records for provenance; only the prepared exact inverse rays are used.

    ``render`` accepts the current model state and camera poses for the same
    video frame. It returns four RGB arrays and publishes ``coverage``,
    ``depth`` and ``shape_indices`` for rendering validation. A false camera
    validity flag or out-of-range frame is visibly unavailable, never clamped.
    """

    def __init__(self, model, cameras, assets_root):
        if set(cameras) != set(CAMERA_NAMES):
            raise ValueError(f"Exactly these four camera views are required: {CAMERA_NAMES}")
        self.model, self.cameras = model, cameras
        self.assets_root = Path(assets_root)
        self.sensor = SensorCamera(
            model,
            default_render_config=SensorCamera.RenderConfig(
                enable_particles=False, enable_backface_culling=False, enable_textures=True, max_distance=3.0
            ),
            load_textures=True,
        )
        self.show_overlay, self.opacity = True, 0.18
        self.frame_index = -1
        self.images, self.depth, self.normals, self.shape_indices, self.coverage, self.status = {}, {}, {}, {}, {}, {}
        self._rays, self._valid, self._buffers, self._recorded = {}, {}, {}, {}
        self._textures, self._uploaded_generation = {}, -1
        self._generation = 0
        self._ui_viewport_size = None
        self._pose = wp.empty(1, dtype=wp.transform, device=model.device)
        self._world = wp.zeros(1, dtype=int, device=model.device)
        for name in CAMERA_NAMES:
            camera = cameras[name]
            width, height = map(int, camera["overlay_size"])
            if min(width, height, int(camera["frame_count"])) < 1:
                raise ValueError(f"Invalid prepared image dimensions or frame count for {name}")
            directions = np.load(self.assets_root / camera["ray_directions_cv_path"], allow_pickle=False)
            validity = (
                np.load(self.assets_root / camera["ray_valid_path"], allow_pickle=False)
                if camera.get("ray_valid_path")
                else None
            )
            rays, validity = camera_rays_cv(directions, validity, device=model.device)
            if validity.shape != (height, width):
                raise ValueError(f"Prepared ray dimensions differ from {name} overlay_size")
            self._rays[name], self._valid[name] = rays, validity
            self._buffers[name] = (
                self.sensor.allocate_image_albedo(1, width, height),
                self.sensor.allocate_image_depth(1, width, height),
                self.sensor.allocate_image_shape_index(1, width, height),
                self.sensor.allocate_image_normal(1, width, height),
            )
            self.images[name] = np.zeros((height, width, 3), dtype=np.uint8)

    def _read_frame(self, name, frame):
        from PIL import Image

        cached = self._recorded.get(name)
        if cached is not None and cached[0] == frame:
            return cached[1]
        camera = self.cameras[name]
        path = self.assets_root / camera["frame_path_pattern"].format(frame=frame)
        with Image.open(path) as source:
            image = np.asarray(source.convert("RGB"))
        if image.shape != self.images[name].shape:
            raise ValueError(f"Prepared frame {path} dimensions differ from its calibrated ray grid")
        self._recorded[name] = frame, image
        return image

    def render(self, state, frame_index, camera_poses_cv, camera_valid=None):
        """Render the selected original video index using its exact prepared poses."""
        if isinstance(frame_index, bool) or int(frame_index) != frame_index:
            raise ValueError("The recording frame index must be an integer")
        self.frame_index = int(frame_index)
        camera_valid = dict.fromkeys(CAMERA_NAMES, True) if camera_valid is None else camera_valid
        self.model.bvh_refit_shapes(state)
        self.model.bvh_refit_particles(state)
        self.sensor.sync_deformable_meshes(state)
        for name in CAMERA_NAMES:
            camera = self.cameras[name]
            available = 0 <= self.frame_index < int(camera["frame_count"])
            available &= bool(camera_valid.get(name, False))
            if not available:
                self.images[name].fill(0)
                self.depth[name] = np.zeros(self.images[name].shape[:2], dtype=np.float32)
                self.normals[name] = np.zeros(self.images[name].shape, dtype=np.float32)
                self.shape_indices[name] = np.full(self.images[name].shape[:2], 0xFFFFFFFF, dtype=np.uint32)
                self.coverage[name], self.status[name] = 0, "Unavailable camera/frame"
                continue
            recorded = self._read_frame(name, self.frame_index)
            transform = camera_transform_cv(camera_poses_cv[name])
            self._pose.assign(np.asarray([transform], dtype=np.float32))
            albedo, depth, labels, normals = self._buffers[name]
            self.sensor.update(
                state,
                self._pose,
                self._rays[name],
                albedo_image=albedo,
                depth_image=depth,
                shape_index_image=labels,
                normal_image=normals,
                world_indices=self._world,
                sync_deformables=False,
            )
            self.depth[name], self.shape_indices[name] = depth.numpy()[0], labels.numpy()[0]
            self.normals[name] = normals.numpy()[0]
            self.coverage[name] = int(np.count_nonzero((self.depth[name] > 0.0) & self._valid[name]))
            if self.show_overlay:
                rendered = SensorCamera.Utils.to_rgba_from_color(albedo).numpy()[0, ..., :3]
                # An oblique camera-relative light reveals actual scanned
                # groove relief; no fitted traces or region boundaries enter
                # either the shading or edge computation.
                light_cv = np.array((-0.35, -0.45, -1.0))
                light_world = np.asarray(camera_poses_cv[name])[:3, :3] @ (light_cv / np.linalg.norm(light_cv))
                diffuse = np.maximum(np.einsum("ijk,k->ij", self.normals[name], light_world), 0.0)
                rendered = np.rint(rendered * (0.3 + 0.7 * diffuse[..., None])).clip(0, 255).astype(np.uint8)
                self.images[name] = composite_overlay(
                    recorded,
                    rendered,
                    self.depth[name],
                    self.normals[name],
                    self._valid[name],
                    opacity=self.opacity,
                )
            else:
                self.images[name] = recorded.copy()
            self.status[name] = "Model overlay" if self.show_overlay else "Recorded image"
        self._generation += 1
        return self.images

    def _upload_images(self):
        from OpenGL import GL

        if self._uploaded_generation == self._generation:
            return
        previous = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)
        alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        try:
            for name in CAMERA_NAMES:
                if name not in self._textures:
                    self._textures[name] = int(GL.glGenTextures(1))
                GL.glBindTexture(GL.GL_TEXTURE_2D, self._textures[name])
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
                GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
                pixels = np.ascontiguousarray(self.images[name])
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGB,
                    pixels.shape[1],
                    pixels.shape[0],
                    0,
                    GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE,
                    pixels,
                )
        finally:
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, alignment)
            GL.glBindTexture(GL.GL_TEXTURE_2D, previous)
        self._uploaded_generation = self._generation

    def draw(self, ui):
        """Public viewer free-UI callback: four small movable calibrated views."""
        viewport = ui.get_main_viewport()
        width = 2 * max(image.shape[1] for image in self.images.values()) + 34
        height = (
            sum(max(self.images[name].shape[0] for name in row) for row in (CAMERA_NAMES[:2], CAMERA_NAMES[2:])) + 165
        )
        viewport_size = (int(viewport.work_size.x), int(viewport.work_size.y))
        if self._ui_viewport_size != viewport_size:
            # Saved ImGui positions can belong to a larger monitor. Place all
            # four views inside this viewport on opening or resizing; users
            # can still move the window during an ordinary viewing session.
            ui.set_next_window_pos(
                ui.ImVec2(
                    max(viewport.work_pos.x, viewport.work_pos.x + viewport.work_size.x - width - 12),
                    max(viewport.work_pos.y, viewport.work_pos.y + viewport.work_size.y - height - 12),
                ),
                ui.Cond_.always,
            )
            ui.set_next_window_size(ui.ImVec2(width, height), ui.Cond_.always)
            self._ui_viewport_size = viewport_size
        expanded, _ = ui.begin("Recorded camera alignment")
        if expanded:
            _, self.show_overlay = ui.checkbox("Model overlay", self.show_overlay)
            ui.same_line()
            ui.set_next_item_width(140)
            _, self.opacity = ui.slider_float("Fill opacity", self.opacity, 0.0, 0.7)
            ui.text(f"Original frame {self.frame_index} | cyan: visible model edges")
            self._upload_images()
            if ui.begin_table("##suturebot_camera_views", 2):
                for name in CAMERA_NAMES:
                    ui.table_next_column()
                    ui.text(f"{name.upper()} | {self.status.get(name, 'Not rendered')}")
                    image = self.images[name]
                    ui.image(ui.ImTextureRef(self._textures[name]), ui.ImVec2(image.shape[1], image.shape[0]))
                ui.end_table()
        ui.end()

    def close(self):
        """Release UI textures, or forget IDs after the viewer destroyed its context."""
        if self._textures:
            from OpenGL import GL

            # Newton's ordinary runner closes the viewer before an example's
            # outer finally block. Context destruction already releases these
            # objects; GL deletion is only meaningful with a live context.
            if GL.glGetString(GL.GL_VERSION) is not None:
                GL.glDeleteTextures(list(self._textures.values()))
            self._textures.clear()

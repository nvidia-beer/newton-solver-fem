# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay measured SutureBot motion with four source-video camera overlays.

Loads a prepared recording.json, USD geometry and timestamped poses. All camera
fitting, frame extraction and asset preparation live in the MSc workbench.
Prepared point labels remain fixed to the phantom's UV surface. Source-defined
approach clips retain their original frame numbers. The prepared scene contains
the tools, cameras, fixture and phantom; motion is replayed without dynamics.

Run: python -m newton_solver_fem.examples fem_suturebot
"""

import argparse

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem.examples._suturebot_recording import DEFAULT_RECORDING, Recording, bundled_recordings


class Example:
    def __init__(self, viewer, args=None):
        self.args = newton.examples.default_args(self.create_parser()) if args is None else args
        self.viewer, self.device = viewer, wp.get_device()
        if not np.isfinite(self.args.playback_speed) or self.args.playback_speed <= 0:
            raise ValueError("Playback speed must be positive and finite")
        self.samples = bundled_recordings(include_aliases=False)
        self.pending_sample = None
        self.pending_frame = None
        self.overlay = None
        self.show_overlays = not self.args.no_camera_overlays
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 60.0
        self.load_recording(self.args.sample or DEFAULT_RECORDING)
        self.reset_camera()
        if hasattr(self.viewer, "register_ui_callback"):
            self.viewer.register_ui_callback(self.camera_gui, position="free")

    def load_recording(self, path):
        self.pending_frame = None
        self.recording = Recording(path)
        self._build_scene()
        self.elapsed = 0.0
        self.playing = True
        self.seek(0)

    def _build_scene(self):
        """Import the prepared geometry when switching recordings."""
        if self.overlay is not None:
            self.overlay.close()
            self.overlay = None
        builder = newton.ModelBuilder()
        builder.add_usd(str(self.recording.scene_path), collapse_fixed_joints=False, hide_collision_shapes=True)
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0, 0, 0))
        self.state_0 = self.model.state()
        labels = {label: index for index, label in enumerate(self.model.body_label)}
        missing = set(self.recording.body_paths) - set(labels)
        if missing:
            raise ValueError(f"Prepared motion references bodies absent from the scene USD: {sorted(missing)}")
        self.body_indices = np.asarray([labels[path] for path in self.recording.body_paths])
        self.body_poses = self.state_0.body_q.numpy().copy()
        self.viewer.set_model(self.model)
        self.frame_index = -1

    def seek(self, frame):
        frame = int(np.clip(frame, self.recording.first_frame, self.recording.last_frame))
        self.elapsed = float(self.recording.time_s[frame])
        self._show_frame(frame)

    def _show_frame(self, frame):
        if frame == self.frame_index:
            return
        self.frame_index = frame
        self.body_poses[self.body_indices] = self.recording.body_poses[frame]
        self.state_0.body_q.assign(self.body_poses)
        self.state_0.body_qd.zero_()

    def step(self):
        if self.pending_sample is not None:
            self.load_recording(self.pending_sample)
            self.pending_sample = None
        if self.pending_frame is not None:
            self.seek(self.pending_frame)
            self.pending_frame = None
        if self.playing:
            self.elapsed += self.frame_dt * self.args.playback_speed
            self._show_frame(self.recording.frame_at(self.elapsed, loop=self.args.loop))
            if not self.args.loop and self.elapsed >= self.recording.end_time:
                self.playing = False
        self.sim_time += self.frame_dt

    def reset_camera(self):
        self.viewer.camera_speed = 0.12
        # Lettering-edge view of the fixture/table, leaving room for overlays.
        self.viewer.set_camera(pos=(-0.568, 0.551, 0.426), pitch=-24.0, yaw=-55.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot((0.0, 0.0, 0.08))

    def frame_phantom(self):
        self.viewer.set_camera(pos=(-0.22, 0.21, 0.20), pitch=-32.0, yaw=-44.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot((0.0, 0.0, 0.02))

    def render(self):
        if self.show_overlays:
            if self.overlay is None:
                from newton_solver_fem.examples._suturebot_cameras import FourCameraOverlay

                self.overlay = FourCameraOverlay(self.model, self.recording.cameras, self.recording.root)
            poses, valid = self.recording.camera_frame(self.frame_index)
            self.overlay.render(self.state_0, self.frame_index, poses, valid)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("SutureBot recording replay")
        ui.text_wrapped(self.recording.manifest["sample_id"])
        names = list(self.samples)
        if len(names) > 1:
            current = next(
                (index for index, name in enumerate(names) if self.samples[name].resolve() == self.recording.path), -1
            )
            changed, selected = ui.combo("Recording", current, names)
            if changed:
                self.pending_sample = self.samples[names[selected]]
        ui.text(f"Region {self.recording.manifest['region_id']} | {self.recording.duration:.2f} s")
        if "playback" in self.recording.manifest:
            ui.text_wrapped(self.recording.manifest["playback"].get("description", "Selected source phase"))
        changed, self.playing = ui.checkbox("Play recording", self.playing)
        changed, self.args.loop = ui.checkbox("Loop", self.args.loop)
        changed, frame = ui.slider_int("Frame", self.frame_index, self.recording.first_frame, self.recording.last_frame)
        if changed:
            self.playing = False
            self.pending_frame = frame
        ui.text(f"Source time: {self.recording.time_s[self.frame_index]:.3f} s")
        if ui.button("Restart recording"):
            self.pending_frame = self.recording.first_frame
            self.playing = True
        changed, self.show_overlays = ui.checkbox("Four camera overlays", self.show_overlays)
        if ui.button("Frame scene"):
            self.reset_camera()
        if ui.button("Frame phantom"):
            self.frame_phantom()
        ui.text_wrapped("Measured motion replay. Camera and phantom alignment are fitted estimates.")
        alignment = self.recording.manifest.get("alignment", {})
        if alignment.get("failure_reason"):
            ui.text_wrapped(f"Alignment check failed: {alignment['failure_reason']}")
        elif not alignment.get("accepted", False):
            ui.text_wrapped("Full tool/pad geometry alignment is not validated.")
        if "point_labels" in self.recording.manifest:
            if self.recording.manifest["point_labels"].get("binding_valid") is False:
                ui.text_wrapped("Green marker bindings are superseded and need refitting.")
            else:
                ui.text("Dataset point labels on phantom surface")

    def camera_gui(self, ui):
        if self.show_overlays and self.overlay is not None:
            self.overlay.draw(ui)

    def test_final(self):
        actual = self.state_0.body_q.numpy()[self.body_indices]
        np.testing.assert_allclose(actual, self.recording.body_poses[self.frame_index], atol=2.0e-7, rtol=0)
        if not np.isfinite(self.state_0.body_q.numpy()).all():
            raise AssertionError("Recording replay produced nonfinite body poses")

    def close(self):
        if self.overlay is not None:
            self.overlay.close()

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.description = __doc__
        parser.add_argument(
            "--sample", help="Bundled sample name or recording.json path; omit for a terminal recording menu"
        )
        parser.add_argument("--list-samples", action="store_true", help="List bundled recordings and exit")
        parser.add_argument(
            "--choose-sample", action="store_true", help="Choose a recording before starting the viewer"
        )
        parser.add_argument("--playback-speed", type=float, default=1.0)
        parser.add_argument("--loop", action="store_true")
        parser.add_argument("--no-camera-overlays", action="store_true")
        parser.add_argument("--demo", action="store_true", help=argparse.SUPPRESS)
        return parser


if __name__ == "__main__":
    from newton_solver_fem.examples._suturebot_cli import configure_sample

    parser = Example.create_parser()
    configure_sample(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        example.close()

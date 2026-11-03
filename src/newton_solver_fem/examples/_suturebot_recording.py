# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Read prepared recording assets; fitting and conversion belong in MSc."""

import json
from pathlib import Path

import numpy as np

CAMERAS = ("left", "right", "psm1", "psm2")
RECORDINGS = Path(__file__).parent / "assets/suturebot/recordings"
DEFAULT_RECORDING = RECORDINGS / "tissue3_215300/recording.json"


def bundled_recordings():
    """Discover complete prepared samples without maintaining a second catalog."""
    return {path.parent.name: path for path in sorted(RECORDINGS.glob("*/recording.json"))}


class Recording:
    def __init__(self, path):
        path = Path(path).expanduser()
        if not path.exists() and len(path.parts) == 1:
            path = bundled_recordings().get(str(path), path)
        self.path = path.resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"Missing prepared recording: {self.path}. See docs/usage.md for preparation.")
        self.root = self.path.parent
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get("schema_version") != 1:
            raise ValueError("Unsupported SutureBot recording schema")
        self.scene_path = self.asset(self.manifest["scene_usd"])
        self.body_paths = tuple(self.manifest["body_paths"])
        if not self.body_paths or len(set(self.body_paths)) != len(self.body_paths):
            raise ValueError("Recording body paths must be nonempty and unique")
        self.cameras = self.manifest["cameras"]
        if set(self.cameras) != set(CAMERAS) or tuple(self.manifest["camera_order"]) != CAMERAS:
            raise ValueError("Recording must identify left, right, psm1 and psm2 cameras in that order")
        with np.load(self.asset(self.manifest["motion_npz"]), allow_pickle=False) as data:
            self.timestamps_ns = data["timestamps_ns"].copy()
            self.time_s = data["time_s"].copy()
            self.body_poses = data["body_poses"].copy()
            self.body_valid = data["body_valid"].copy()
        self.frame_count = len(self.timestamps_ns)
        if (
            self.frame_count < 2
            or self.timestamps_ns.dtype != np.dtype("int64")
            or self.timestamps_ns.shape != (self.frame_count,)
            or np.any(np.diff(self.timestamps_ns) <= 0)
            or self.time_s.shape != (self.frame_count,)
            or not np.allclose(self.time_s, (self.timestamps_ns - self.timestamps_ns[0]) * 1.0e-9, atol=1.0e-9, rtol=0)
        ):
            raise ValueError("Recording must preserve increasing int64 timestamps and their exact relative clock")
        if self.body_valid.shape != (self.frame_count,) or self.body_valid.dtype != np.dtype(bool):
            raise ValueError("Body availability must be a boolean mask on the source clock")
        supported = np.flatnonzero(self.body_valid)
        if len(supported) < 2 or np.any(np.diff(supported) != 1):
            raise ValueError("Recording must have a contiguous supported playback interval")
        self.first_valid_frame, self.last_valid_frame = int(supported[0]), int(supported[-1])
        if (
            self.body_poses.shape != (self.frame_count, len(self.body_paths), 7)
            or not np.isfinite(self.body_poses[self.body_valid]).all()
            or not np.allclose(
                np.linalg.norm(self.body_poses[self.body_valid, :, 3:], axis=-1), 1.0, atol=2.0e-6, rtol=0
            )
        ):
            raise ValueError("Recording body poses must be finite xyz / unit quaternion xyzw")
        with np.load(self.asset(self.manifest["camera_poses_npz"]), allow_pickle=False) as data:
            if not np.array_equal(data["timestamps_ns"], self.timestamps_ns):
                raise ValueError("Camera and body trajectories have different source timestamps")
            self.camera_poses = data["camera_poses_cv"].copy()
            self.camera_valid = data["camera_valid"].copy()
        if self.camera_poses.shape != (self.frame_count, 4, 4, 4) or self.camera_valid.shape != (self.frame_count, 4):
            raise ValueError("Camera pose/availability arrays do not match the four-view recording")
        if self.camera_valid.dtype != np.dtype(bool):
            raise ValueError("Camera availability must be a boolean mask")
        valid = self.camera_poses[self.camera_valid]
        rotation = valid[:, :3, :3]
        if (
            not np.isfinite(valid).all()
            or not np.allclose(valid[:, 3], [0, 0, 0, 1], atol=1.0e-7, rtol=0)
            or not np.allclose(rotation.swapaxes(1, 2) @ rotation, np.eye(3), atol=2.0e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotation), 1.0, atol=2.0e-6, rtol=0)
        ):
            raise ValueError("Available camera poses must be proper rigid transforms in OpenCV axes")
        self.nominal_fps = float(self.manifest["timing"]["nominal_fps"])
        if not np.isfinite(self.nominal_fps) or self.nominal_fps <= 0:
            raise ValueError("Recording frame rate must be positive and finite")
        if int(self.manifest["timing"]["frame_count"]) != self.frame_count:
            raise ValueError("Manifest frame count differs from its trajectory")
        for key in ("first_valid_frame", "last_valid_frame"):
            if key in self.manifest["timing"] and self.manifest["timing"][key] != getattr(self, key):
                raise ValueError("Manifest supported interval differs from its trajectory")
        for view in CAMERAS:
            if int(self.cameras[view]["frame_count"]) != self.frame_count:
                raise ValueError(f"{view} frame count differs from its trajectory")
        playback = self.manifest.get("playback", {})
        self.first_frame = playback.get("first_frame", self.first_valid_frame)
        self.last_frame = playback.get("last_frame", self.last_valid_frame)
        if (
            type(self.first_frame) is not int
            or type(self.last_frame) is not int
            or not self.first_valid_frame <= self.first_frame < self.last_frame <= self.last_valid_frame
        ):
            raise ValueError("Playback window must contain at least two frames inside source support")
        # A source-defined phase limits playback without relabeling measured
        # availability, renumbering source images, or rewriting timestamps.
        self.start_time = float(self.time_s[self.first_frame])
        self.end_time = float(self.time_s[self.last_frame])
        self.duration = self.end_time - self.start_time
        self.loop_duration = self.duration + 1.0 / self.nominal_fps

    def asset(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or not path.is_file():
            raise ValueError(f"Prepared asset must exist inside the recording package: {relative}")
        return path

    def frame_at(self, elapsed, *, loop=False):
        """Hold the last measured frame until its successor's timestamp.

        The same index is used for source pixels, geometry and optical poses;
        no independent rounding or nominal-FPS remapping is applied to motion.
        """
        if not np.isfinite(elapsed):
            raise ValueError("Replay time must be finite")
        elapsed = (elapsed - self.start_time) % self.loop_duration + self.start_time if loop else elapsed
        elapsed = np.clip(elapsed, self.start_time, self.end_time)
        return int(np.clip(np.searchsorted(self.time_s, elapsed, side="right") - 1, self.first_frame, self.last_frame))

    def camera_frame(self, frame):
        return (
            {view: self.camera_poses[frame, index] for index, view in enumerate(CAMERAS)},
            {
                view: bool(self.body_valid[frame] and self.camera_valid[frame, index])
                for index, view in enumerate(CAMERAS)
            },
        )

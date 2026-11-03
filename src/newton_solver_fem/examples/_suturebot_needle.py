# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tool-driven grasp and free MuJoCo dynamics for a prepared rigid needle.

Only the initial placement is image fitted. No recorded needle trajectory is
loaded. Held poses use one constant transform per grasp; released poses come
from MuJoCo. A fresh jaw closure can translate a released needle into the grip
without rotating it. Backward seeks reset the same source-clock decisions.
"""

import math

import newton
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


def pose_matrix(pose):
    """Newton xyz / quaternion xyzw to a column-vector rigid transform."""
    matrix = np.eye(4)
    matrix[:3, 3] = pose[:3]
    x, y, z, w = pose[3:]
    matrix[:3, :3] = [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]
    return matrix


def matrix_pose(matrix):
    rotation = wp.mat33(*np.asarray(matrix[:3, :3], np.float32).reshape(-1))
    quaternion = wp.quat_from_matrix(rotation)
    return np.array([*matrix[:3, 3], *quaternion], dtype=np.float32)


def body_twist(previous, current, dt):
    """COM velocity and world angular velocity from two rigid poses."""
    relative = current[:3, :3] @ previous[:3, :3].T
    quaternion = matrix_pose(np.block([[relative, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]]))[3:]
    if quaternion[3] < 0:
        quaternion = -quaternion
    sine = np.linalg.norm(quaternion[:3])
    omega = quaternion[:3] * (2 * math.atan2(sine, quaternion[3]) / max(sine, 1.0e-12) / dt)
    return np.r_[(current[:3, 3] - previous[:3, 3]) / dt, omega]


def add_needle_to_builder(builder, recording):
    """Import prepared visual and collision geometry before model finalization."""
    specification = recording.manifest.get("needle")
    if specification is not None:
        builder.add_usd(
            str(recording.asset(specification["usd"])),
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
        )


class NeedleReplay:
    """An isolated rigid solver keeps measured tool poses entirely unchanged."""

    def __init__(self, recording, render_model, render_state, *, backend="cpu"):
        self.recording = recording
        self.specification = recording.manifest["needle"]
        specification = self.specification
        self.close_angle = float(specification["grasp"]["close_angle_rad"])
        self.open_angle = float(specification["grasp"]["open_angle_rad"])
        self.capture_distance = float(specification["grasp"]["capture_distance_m"])
        self.max_dt = float(specification["simulation"]["maximum_step_s"])
        if not 0 <= self.close_angle < self.open_angle or self.capture_distance <= 0 or self.max_dt <= 0:
            raise ValueError("Invalid needle jaw hysteresis, capture distance or physics timestep")
        with np.load(recording.asset(specification["inputs_npz"]), allow_pickle=False) as archive:
            self.jaw = archive["jaw_opening_rad"].copy()
            self.centreline = archive["centreline_local_m"].copy()
            if not np.array_equal(archive["timestamps_ns"], recording.timestamps_ns):
                raise ValueError("Needle jaw decisions must use the unchanged recording clock")
        if self.jaw.shape != (recording.frame_count, 2) or not np.isfinite(self.jaw).all():
            raise ValueError("Needle jaw inputs must contain two finite source-angle arrays")
        if (
            self.centreline.ndim != 2
            or self.centreline.shape[1] != 3
            or not len(self.centreline)
            or not np.isfinite(self.centreline).all()
        ):
            raise ValueError("Needle centreline must be finite local xyz geometry")
        self.arm_names = ("psm1", "psm2")
        self.tool_indices = [
            recording.body_paths.index(specification["grasp"]["tool_body_paths"][a]) for a in self.arm_names
        ]
        self.grasp_points = np.asarray([specification["grasp"]["tool_point_local_m"][a] for a in self.arm_names])
        self.render_index = list(render_model.body_label).index(specification["body_path"])
        if backend not in ("cpu", "warp", "auto"):
            raise ValueError("Needle backend must be cpu, warp or auto")
        use_cpu = backend == "cpu" or (backend == "auto" and not render_model.device.is_cuda)
        if not use_cpu and not render_model.device.is_cuda:
            raise ValueError("MuJoCo Warp needle simulation requires a CUDA device")
        # The CPU backend avoids allocating a second scene-wide CUDA solver for
        # one small rigid body. Rendering can independently use CUDA.
        self.device = wp.get_device("cpu") if use_cpu else render_model.device
        builder = newton.ModelBuilder()
        # Small metal geometry needs a shorter contact time constant than the
        # scene defaults; otherwise a thin support can be traversed on impact.
        builder.default_shape_cfg.ke = float(specification["simulation"].get("contact_stiffness", 1.0e6))
        builder.default_shape_cfg.kd = float(specification["simulation"].get("contact_damping", 2000.0))
        builder.add_usd(
            str(recording.asset(specification["usd"])), collapse_fixed_joints=False, hide_collision_shapes=True
        )
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity(tuple(specification["simulation"]["gravity_m_s2"]))
        self.index = list(self.model.body_label).index(specification["body_path"])
        if self.model.body_count != 1 or self.model.joint_coord_count != 7:
            raise ValueError("Prepared needle physics must contain exactly one free rigid body")
        self.state = self.model.state()
        self.next_state = self.model.state()
        self.control = self.model.control()
        self.solver = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=use_cpu,
            iterations=30,
            ls_iterations=10,
            integrator="implicitfast",
            # Up to two contacts per arc segment, each with several friction
            # rows, can appear together when the curved needle lands flat.
            # The initial airborne pose otherwise yields too small a Warp
            # allocation for later support contacts.
            nconmax=4 * self.model.shape_count,
            njmax=24 * self.model.shape_count,
            deterministic=wp.DeterministicMode.RUN_TO_RUN,
        )
        self.initial = np.asarray(specification["initial_pose_world_xyz_xyzw"], dtype=np.float32)
        self.events = []
        self.frame = -1
        self.held_arm = None
        self.grip_transform = None
        self.reset()
        self.seek(recording.first_valid_frame, render_state)

    def tool_transform(self, frame, arm):
        return pose_matrix(self.recording.body_poses[frame, self.tool_indices[self.arm_names.index(arm)]])

    def _set_pose(self, matrix, velocity=None):
        self.state.body_q.assign(np.asarray([matrix_pose(matrix)], dtype=np.float32))
        self.state.body_qd.assign(np.asarray([np.zeros(6) if velocity is None else velocity], dtype=np.float32))
        newton.eval_ik(self.model, self.state, self.state.joint_q, self.state.joint_qd)

    def reset(self):
        self.solver.reset(self.state)
        self._set_pose(pose_matrix(self.initial))
        self.frame = self.recording.first_valid_frame
        self.held_arm = self.specification.get("initial_held_arm")
        self.events = []
        self.grip_transform = None
        # A distant regrasp needs a complete open -> close cycle. Consume each
        # closure even while the other arm holds the needle, so a stale closed
        # jaw cannot later teleport a released needle to itself.
        self.grasp_armed = self.jaw[self.frame] >= self.open_angle
        if self.held_arm is not None:
            index = self.arm_names.index(self.held_arm)
            if self.jaw[self.frame, index] > self.open_angle:
                raise ValueError("Initial needle holder has an open recorded jaw")
            self.grip_transform = np.linalg.inv(self.tool_transform(self.frame, self.held_arm)) @ pose_matrix(
                self.initial
            )
            self.events.append({"frame": self.frame, "event": "initial_grasp", "arm": self.held_arm})

    def _try_grasp(self, frame, closing):
        needle = pose_matrix(self.state.body_q.numpy()[self.index])
        points = self.centreline @ needle[:3, :3].T + needle[:3, 3]
        candidates = []
        for index, arm in enumerate(self.arm_names):
            if self.jaw[frame, index] > self.close_angle:
                continue
            tool = self.tool_transform(frame, arm)
            center = tool[:3, :3] @ self.grasp_points[index] + tool[:3, 3]
            distances = np.linalg.norm(points - center, axis=1)
            nearest = int(np.argmin(distances))
            distance = float(distances[nearest])
            if closing[index] or distance <= self.capture_distance:
                translation = center - points[nearest] if closing[index] else np.zeros(3)
                candidates.append((distance, arm, tool, bool(closing[index]), translation))
        if candidates:
            distance, self.held_arm, tool, just_closed, translation = min(candidates, key=lambda item: item[0])
            # Translate a material point on the needle to the jaw center. Its
            # current world orientation survives the assisted pickup; only
            # subsequent holding-tool motion rotates it.
            needle[:3, 3] += translation
            self.grip_transform = np.linalg.inv(tool) @ needle
            self.solver.reset(self.state, flags=0)
            self._set_pose(needle)
            self.events.append(
                {
                    "frame": frame,
                    "event": "grasp",
                    "arm": self.held_arm,
                    "distance_m": distance,
                    "trigger": "jaw_close" if just_closed else "proximity",
                    "translation_world_m": translation.tolist(),
                }
            )

    def _advance(self, frame):
        dt = float(self.recording.time_s[frame] - self.recording.time_s[self.frame])
        angles = self.jaw[frame]
        closing = self.grasp_armed & (angles <= self.close_angle)
        self.grasp_armed[angles >= self.open_angle] = True
        self.grasp_armed[angles <= self.close_angle] = False
        if self.held_arm is not None:
            before = pose_matrix(self.state.body_q.numpy()[self.index])
            after = self.tool_transform(frame, self.held_arm) @ self.grip_transform
            self._set_pose(after, body_twist(before, after, dt))
            index = self.arm_names.index(self.held_arm)
            if self.jaw[frame, index] >= self.open_angle:
                self.events.append({"frame": frame, "event": "release", "arm": self.held_arm})
                self.held_arm = None
                self.grip_transform = None
                self.solver.reset(self.state, flags=0)
                self._try_grasp(frame, closing)
        else:
            steps = max(1, math.ceil(dt / self.max_dt))
            for _ in range(steps):
                self.state.clear_forces()
                self.solver.step(self.state, self.next_state, self.control, None, dt / steps)
                self.state, self.next_state = self.next_state, self.state
            self._try_grasp(frame, closing)
        self.frame = frame

    def seek(self, frame, render_state=None):
        """Reach a source frame reproducibly, independent of viewer framerate."""
        frame = int(np.clip(frame, self.recording.first_valid_frame, self.recording.last_valid_frame))
        if frame < self.frame:
            self.reset()
        for following in range(self.frame + 1, frame + 1):
            self._advance(following)
        pose = self.state.body_q.numpy()[self.index]
        if not np.isfinite(pose).all():
            raise RuntimeError("Needle rigid simulation produced a nonfinite pose")
        if render_state is not None:
            poses = render_state.body_q.numpy()
            poses[self.render_index] = pose
            render_state.body_q.assign(poses)
        return pose.copy()

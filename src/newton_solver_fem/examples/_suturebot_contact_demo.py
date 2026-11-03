# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Legacy contact experiment for the approximate USD rig and regional FEM.

The authored rig is an approximate laboratory layout, not calibrated dVRK
hardware. MuJoCo advances its articulated tools, needle and rigid contacts;
FEM advances one selected dataset region. Sampled capsule/surface penalties
transfer equal-and-opposite forces between the actual pad surface and rig.
Inactive tissue remains fixed. No puncture, thread or automated suturing model
is implied. All geometry preparation lives in the original MSc workbench.

The dataset replay entry point is example_fem_suturebot.py. This separate
experiment is retained for force-conservation and tissue-contact validation.
"""

from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo

from newton_solver_fem.examples._suturebot_contacts import PadContacts
from newton_solver_fem.examples._suturebot_press import TissuePress
from newton_solver_fem.examples.example_fem_phantom_regions import Example as PhantomExample

TOOL_JOINTS = ("rcm_yaw", "rcm_pitch", "insertion", "roll", "wrist_pitch", "wrist_yaw")
_ROOT = "/SutureBot"


class Example(PhantomExample):
    def __init__(self, viewer, args=None):
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        values = (args.contact_stiffness, args.contact_damping, args.contact_friction)
        if not np.isfinite(values).all() or args.contact_stiffness <= 0 or min(values[1:]) < 0:
            raise ValueError("Contact stiffness must be positive; damping and friction must be nonnegative and finite")
        self.rig_path = Path(args.rig_asset).expanduser().resolve()
        if not self.rig_path.is_file():
            raise FileNotFoundError(f"Missing prepared rig USD: {self.rig_path}; see docs/usage.md")
        self.rigid_solver = None
        self.demo_phase = "manual"
        super().__init__(viewer, args)

    def configure_builder(self, builder):
        builder.add_usd(
            str(self.rig_path),
            enable_self_collisions=True,
            collapse_fixed_joints=False,
            hide_collision_shapes=True,
            force_position_velocity_actuation=True,
        )

    def select_regions(self, region_ids):
        super().select_regions(region_ids)
        self.model.set_gravity((0.0, 0.0, -9.81))
        self.control = self.model.control()
        self._external_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state_0)
        self.state_1.assign(self.state_0)
        self.joint_indices = {label: index for index, label in enumerate(self.model.joint_label)}
        self.body_indices = {label: index for index, label in enumerate(self.model.body_label)}
        self._dof_start = self.model.joint_qd_start.numpy()
        self._target_start = self.model.joint_target_q_start.numpy()
        self._lower, self._upper = self.model.joint_limit_lower.numpy(), self.model.joint_limit_upper.numpy()
        self._targets = self.control.joint_target_q.numpy()
        self.tool_targets = {tool: dict.fromkeys((*TOOL_JOINTS, "grip"), 0.0) for tool in ("PSM1", "PSM2")}
        self.needle_body = self.body_indices[f"{_ROOT}/Needle"]
        self.shaft_bodies = [self.body_indices[f"{_ROOT}/{tool}/insertion"] for tool in ("PSM1", "PSM2")]
        initial = self.state_0.body_q.numpy()
        self.rcm_pivots = np.asarray(
            [initial[self.body_indices[f"{_ROOT}/{tool}/yaw"], :3] for tool in ("PSM1", "PSM2")]
        )
        contact_bodies = [
            index
            for label, index in self.body_indices.items()
            if label == f"{_ROOT}/Needle" or label.startswith((f"{_ROOT}/PSM1/", f"{_ROOT}/PSM2/"))
        ]
        self.pad_contacts = PadContacts(
            self.model,
            self.reference,
            self.selection,
            self.display_points,
            contact_bodies,
            stiffness=self.args.contact_stiffness,
            damping=self.args.contact_damping,
            friction=self.args.contact_friction,
            fixed_particle_indices=self.fixed_vertex_indices,
        )
        backend = self.args.mujoco_backend
        self.use_mujoco_cpu = backend == "cpu" or (backend == "auto" and not self.device.is_cuda)
        if backend == "warp" and not self.device.is_cuda:
            raise ValueError("The MuJoCo Warp backend requires a CUDA device; choose --mujoco-backend cpu")
        self.rigid_solver = SolverMuJoCo(
            self.model,
            use_mujoco_cpu=self.use_mujoco_cpu,
            iterations=40,
            ls_iterations=10,
            integrator="implicitfast",
        )
        self.set_tool_targets("PSM1", grip=0.35)
        self.set_tool_targets("PSM2", grip=0.35)
        self.sim_time = 0.0
        self.press_demo = None
        self.demo_phase = "manual"

    def _capture_physics(self):
        # The base constructs FEM before MuJoCo and the deforming surface BVH.
        self.graph = None

    def set_tool_targets(self, tool, **targets):
        """Set physical joint targets (radians, metres); grip is a half angle.

        ``tool`` is ``PSM1`` or ``PSM2``; accepted keys are ``TOOL_JOINTS``
        and ``grip``. Both jaws are commanded symmetrically. A rejected update
        leaves all targets unchanged. Positions are reached by MuJoCo drives.
        """
        if tool not in self.tool_targets:
            raise ValueError("tool must be PSM1 or PSM2")
        unknown = set(targets) - set(self.tool_targets[tool])
        if unknown:
            raise ValueError(f"Unknown tool controls: {sorted(unknown)}")
        updated = self.tool_targets[tool] | targets
        if not np.isfinite(list(updated.values())).all() or updated["grip"] < 0:
            raise ValueError("Tool targets must be finite and grip nonnegative")
        joint_values = {key: updated[key] for key in TOOL_JOINTS}
        joint_values.update(jaw_left=updated["grip"], jaw_right=-updated["grip"])
        changes = []
        for joint_name, value in joint_values.items():
            joint = self.joint_indices[f"{_ROOT}/{tool}/Joints/{joint_name}"]
            dof = self._dof_start[joint]
            if not self._lower[dof] <= value <= self._upper[dof]:
                raise ValueError(f"{tool}/{joint_name} target {value} outside [{self._lower[dof]}, {self._upper[dof]}]")
            changes.append((self._target_start[joint], float(value)))
        for dof, value in changes:
            self._targets[dof] = value
        self.tool_targets[tool] = updated
        self.control.joint_target_q.assign(self._targets)

    def _demo_targets(self):
        if self.solver is None:
            self.demo_phase = "select active tissue"
            return
        if self.press_demo is None:
            try:
                self.press_demo = TissuePress(self)
            except ValueError as error:
                self.demo_phase = str(error)
                self.args.demo = False
                return
        self.press_demo.apply()
        self.demo_phase = self.press_demo.phase

    def simulate(self):
        if self.solver is not None:
            self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
            self.holds.add_matrix(self.solver, self.sim_dt)
        for _ in range(self.args.substeps):
            self.state_0.clear_forces()
            self._external_forces.zero_()
            self.pad_contacts.add_forces(self.state_0, self._external_forces)
            if self.solver is not None:
                self.holds.add_forces(
                    self.state_0.particle_q, self.state_0.particle_qd, self._external_forces, self.sim_dt
                )
                self.solver.step(
                    self.state_0,
                    self.state_1,
                    self.control,
                    None,
                    self.sim_dt,
                    external_forces=self._external_forces,
                )
            self.rigid_solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        if self.args.demo:
            self._demo_targets()
        self.simulate()
        self.sim_time += self.frame_dt
        if self.args.demo and self.press_demo is not None:
            self.press_demo.observe()

    def reset_camera(self):
        self.viewer.camera_speed = 0.12
        self.viewer.set_camera(pos=(-0.25, 0.36, 0.25), pitch=-24.0, yaw=-55.0)
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot((0.0, 0.0, 0.05))

    def frame_tissue_contact(self, blend=1.0):
        """View millimetre-scale contact from the native lettering edge."""
        if self.press_demo is None:
            return
        point = self.press_demo.surface
        start = np.array([-0.25, 0.36, 0.25])
        stop = point + np.array([-0.040, 0.085, 0.056])
        target = point + np.array([0.010, 0.005, 0.0])
        direction = target - stop
        direction /= np.linalg.norm(direction)
        pitch = np.degrees(np.arcsin(direction[2]))
        yaw = np.degrees(np.arctan2(direction[1], direction[0]))
        blend = np.clip(blend, 0.0, 1.0)
        self.viewer.set_camera(
            pos=tuple(start + blend * (stop - start)),
            pitch=float(-24.0 + blend * (pitch + 24.0)),
            yaw=float(-55.0 + blend * (yaw + 55.0)),
        )
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(target)

    def rcm_errors(self):
        """Distance from each fixed pivot to the actual simulated shaft axis [m]."""
        transforms = self.state_0.body_q.numpy()[self.shaft_bodies]
        errors = []
        for transform, pivot in zip(transforms, self.rcm_pivots, strict=True):
            axis = np.asarray(wp.quat_rotate(wp.quat(*transform[3:]), wp.vec3(0.0, 0.0, 1.0)))
            errors.append(np.linalg.norm(np.cross(pivot - transform[:3], axis)))
        return np.asarray(errors)

    def gui(self, ui):
        ui.text("SutureBot: USD rig / MuJoCo + regional FEM")
        ui.text_wrapped(
            "Approximate laboratory layout. RCM joints constrain the shafts; the needle is a free rigid body."
        )
        changed, self.args.demo = ui.checkbox("Press and release active tissue", self.args.demo)
        if changed:
            self.press_demo = None
            self.demo_phase = "planning tissue press" if self.args.demo else "manual"
        ui.text_wrapped(self.demo_phase)
        if self.press_demo is not None:
            ui.text(f"Jaw / active tissue: {self.press_demo.force * 1000:.1f} mN")
            ui.text(f"Tissue motion from baseline: {self.press_demo.displacement * 1000:.2f} mm")
            if ui.button("View tissue contact"):
                self.frame_tissue_contact()
        for tool in ("PSM1", "PSM2"):
            if ui.collapsing_header(f"{tool} controls"):
                for name in TOOL_JOINTS:
                    joint = self.joint_indices[f"{_ROOT}/{tool}/Joints/{name}"]
                    dof = self._dof_start[joint]
                    scale = 1000.0 if name == "insertion" else 180.0 / np.pi
                    unit = "mm" if name == "insertion" else "deg"
                    changed, value = ui.slider_float(
                        f"{name} ({unit})##{tool}",
                        self.tool_targets[tool][name] * scale,
                        float(self._lower[dof] * scale),
                        float(self._upper[dof] * scale),
                    )
                    if changed:
                        self.args.demo = False
                        self.press_demo = None
                        self.demo_phase = "manual"
                        self.set_tool_targets(tool, **{name: value / scale})
                changed, value = ui.slider_float(
                    f"Jaw half angle (deg)##{tool}", np.degrees(self.tool_targets[tool]["grip"]), 0.0, 25.0
                )
                if changed:
                    self.args.demo = False
                    self.press_demo = None
                    self.demo_phase = "manual"
                    self.set_tool_targets(tool, grip=np.radians(value))
        ui.text_wrapped("Surface penalties include the active FEM and fixed pad regions. No puncture or thread model.")
        ui.separator()
        super().gui(ui)

    def test_final(self):
        super().test_final()
        for array in (self.state_0.body_q, self.state_0.body_qd, self.state_0.joint_q, self.state_0.joint_qd):
            if not np.isfinite(array.numpy()).all():
                raise AssertionError("MuJoCo rig produced a nonfinite state")
        if self.rcm_errors().max() > 2.0e-5:
            raise AssertionError(f"Tool shaft left its RCM: {self.rcm_errors()} m")

    @staticmethod
    def create_parser():
        parser = PhantomExample.create_parser()
        parser.description = __doc__
        parser.set_defaults(regions=[1], substeps=16)
        parser.add_argument("--rig-asset", default=str(Path(__file__).parent / "assets/suturebot/suturebot_rig.usda"))
        parser.add_argument("--mujoco-backend", choices=("auto", "cpu", "warp"), default="auto")
        parser.add_argument(
            "--demo", action="store_true", help="Press and release the selected active FEM tissue with PSM1"
        )
        parser.add_argument("--contact-stiffness", type=float, default=60.0, help="Total per-body pad penalty [N/m]")
        parser.add_argument("--contact-damping", type=float, default=0.03, help="Per-body pad contact damping [kg/s]")
        parser.add_argument("--contact-friction", type=float, default=0.4)
        return parser

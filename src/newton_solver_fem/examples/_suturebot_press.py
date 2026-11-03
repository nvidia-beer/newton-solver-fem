# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drive an imported jaw into free FEM surface nodes, then measure recovery.

FK is evaluated only on scratch states to plan joint targets. All displayed
motion and tissue deformation come from the ordinary coupled dynamics.
"""

from weakref import proxy

import newton
import numpy as np
import warp as wp


class TissuePress:
    duration = 9.0
    controls = ("rcm_yaw", "rcm_pitch", "insertion")

    def __init__(self, example):
        self.example = proxy(example)
        self.start_time = example.sim_time
        self.body = example.body_indices["/SutureBot/PSM1/jaw_left"]
        self.jaw_bodies = [example.body_indices[f"/SutureBot/PSM1/{jaw}"] for jaw in ("jaw_left", "jaw_right")]
        model = example.model
        shapes = np.flatnonzero(
            (model.shape_body.numpy() == self.body) & (model.shape_type.numpy() == newton.GeoType.CAPSULE)
        )
        shape = wp.transform(*model.shape_transform.numpy()[shapes[0]])
        radius, half = model.shape_scale.numpy()[shapes[0], :2]
        ends = [wp.transform_point(shape, wp.vec3(0, 0, float(sign * half))) for sign in (-1, 1)]
        starts = model.joint_q_start.numpy()
        joints = [example.joint_indices[f"/SutureBot/PSM1/Joints/{name}"] for name in self.controls]
        self.indices = starts[joints]
        self.base = example.state_0.joint_q.numpy().copy()
        for name in ("roll", "wrist_pitch", "wrist_yaw", "jaw_left", "jaw_right"):
            self.base[starts[example.joint_indices[f"/SutureBot/PSM1/Joints/{name}"]]] = 0.0
        self.initial = self.base[self.indices].copy()
        self.scratch = model.state()
        self.coordinates = wp.array(self.base, dtype=float, device=example.device)
        # Select the distal endpoint in the neutral wrist/jaw pose used for
        # planning, including when restarting after manual wrist rotation.
        newton.eval_fk(model, self.coordinates, model.joint_qd, self.scratch)
        pose = wp.transform(*self.scratch.body_q.numpy()[self.body])
        self.local_tip = min(ends, key=lambda point: wp.transform_point(pose, point)[2])
        dofs = model.joint_qd_start.numpy()[joints]
        self.lower = model.joint_limit_lower.numpy()[dofs]
        self.upper = model.joint_limit_upper.numpy()[dofs]

        faces = example.selection.exterior_surface_triangles
        points = example.state_0.particle_q.numpy()
        corners = points[faces]
        normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
        lengths = np.linalg.norm(normals, axis=1)
        normals /= np.maximum(lengths[:, None], 1.0e-20)
        eligible = ~np.isin(faces, example.fixed_vertex_indices).any(axis=1)
        eligible &= (normals[:, 2] > 0.9) & (lengths > 1.0e-12)
        choices = np.flatnonzero(eligible)
        if not len(choices):
            raise ValueError("Select a region with a free upward-facing tissue surface for the press demo")
        centers = corners.mean(axis=1)
        # Wound 1's upper lip is clear of its slit and supports a local press.
        # Other selections use their nearest free face, keeping the same IDs.
        probe = np.array([0.028, -0.0465, 0.001]) if example.selected_region_ids == (1,) else self.fk(self.initial)
        choices = choices[np.argsort(np.linalg.norm(centers[choices] - probe, axis=1))]
        for face in choices[:64]:
            surface, normal = centers[face], normals[face]
            try:
                hover = self.solve(surface + normal * (radius + 0.004), self.initial)
                press = self.solve(surface + normal * (radius - 0.0015), hover)
                retract = self.solve(surface + normal * (radius + 0.006), hover)
            except ValueError:
                continue
            self.face, self.surface, self.normal = int(face), surface, normal
            self.hover, self.press, self.retract = hover, press, retract
            break
        else:
            raise ValueError("The selected free tissue surface is outside the tool's joint limits")
        self.phase = "approach active tissue"
        self.baseline = None
        self.maximum_force = self.maximum_displacement = self.remaining_displacement = 0.0
        self.maximum_contacts = self.contact_frames = self.frames = 0
        self.displacement = self.force = 0.0
        self.maximum_rcm_error = 0.0

    def fk(self, values):
        q = self.base.copy()
        q[self.indices] = values
        self.coordinates.assign(q)
        newton.eval_fk(self.example.model, self.coordinates, self.example.model.joint_qd, self.scratch)
        pose = wp.transform(*self.scratch.body_q.numpy()[self.body])
        return np.asarray(wp.transform_point(pose, self.local_tip), dtype=float)

    def solve(self, target, guess):
        values = np.asarray(guess, dtype=float).copy()
        steps = np.array([1.0e-4, 1.0e-4, 1.0e-5])
        for _ in range(16):
            point = self.fk(values)
            error = target - point
            if np.linalg.norm(error) < 2.0e-7:
                return values
            jacobian = np.column_stack(
                [(self.fk(values + np.eye(3)[j] * steps[j]) - point) / steps[j] for j in range(3)]
            )
            update = np.linalg.lstsq(jacobian, error, rcond=None)[0]
            values = np.clip(values + update, self.lower, self.upper)
        raise ValueError("No joint-limited FK solution for this tissue target")

    def apply(self):
        t = self.example.sim_time - self.start_time
        if t < 1.0:
            self.phase, start, stop, blend = "approach active tissue", self.initial, self.hover, t
        elif t < 2.0:
            self.phase, start, stop, blend = "settle above tissue", self.hover, self.hover, 0.0
        elif t < 3.5:
            self.phase, start, stop, blend = "press active tissue", self.hover, self.press, (t - 2.0) / 1.5
        elif t < 5.0:
            self.phase, start, stop, blend = "hold tissue pressure", self.press, self.press, 0.0
        elif t < 6.5:
            self.phase, start, stop, blend = "release tissue", self.press, self.retract, (t - 5.0) / 1.5
        else:
            self.phase, start, stop, blend = "tissue recovery", self.retract, self.retract, 0.0
        blend = np.clip(blend, 0.0, 1.0)
        blend = blend * blend * (3 - 2 * blend)
        values = start + (stop - start) * blend
        self.example.set_tool_targets(
            "PSM1", **dict(zip(self.controls, values, strict=True)), roll=0, wrist_pitch=0, wrist_yaw=0, grip=0
        )

    def observe(self):
        self.frames += 1
        t = self.example.sim_time - self.start_time
        points = self.example.state_0.particle_q.numpy()
        if t <= 2.0 + 1.0e-8:
            self.baseline = points.copy()
        if self.baseline is not None:
            self.displacement = float(np.linalg.norm(points - self.baseline, axis=1).max())
        contacts = self.example.pad_contacts
        self.force = float(np.linalg.norm(contacts.active_body_force.numpy()[self.jaw_bodies], axis=1).sum())
        count = int(contacts.active_body_contacts.numpy()[self.jaw_bodies].sum())
        if 2.0 < t <= 5.0 + 1.0e-8:
            self.maximum_force = max(self.maximum_force, self.force)
            self.maximum_contacts = max(self.maximum_contacts, count)
            self.contact_frames += int(count > 0)
            self.maximum_displacement = max(self.maximum_displacement, self.displacement)
        if t >= 6.5:
            self.remaining_displacement = self.displacement
        self.maximum_rcm_error = max(self.maximum_rcm_error, float(self.example.rcm_errors().max()))

    def validate(self):
        if self.example.sim_time - self.start_time < self.duration - 1.0e-7:
            raise AssertionError("Complete the press and recovery sequence before validating it")
        if self.maximum_force <= 0.001 or self.maximum_contacts == 0 or self.maximum_displacement <= 2.0e-5:
            raise AssertionError("The driven jaws must load free FEM nodes and deform the selected tissue")
        if self.remaining_displacement >= 0.5 * self.maximum_displacement:
            raise AssertionError("Active tissue did not recover half its incremental displacement after release")
        if self.force > 1.0e-4:
            raise AssertionError("The jaw is still loading the active tissue after retraction")
        self.example.test_final()

    def report(self):
        return {
            "selected_region_ids": list(self.example.selected_region_ids),
            "face": self.face,
            "surface_point_m": self.surface.tolist(),
            "target_order": list(self.controls),
            "hover_targets": self.hover.tolist(),
            "press_targets": self.press.tolist(),
            "retract_targets": self.retract.tolist(),
            "maximum_jaw_active_tissue_force_N": self.maximum_force,
            "maximum_jaw_active_contact_samples": self.maximum_contacts,
            "frames_with_jaw_active_contact": self.contact_frames,
            "force_sampling": "PSM1 jaws, force delivered to free FEM nodes, final substep per frame",
            "maximum_incremental_displacement_m": self.maximum_displacement,
            "remaining_displacement_m": self.remaining_displacement,
            "final_jaw_active_tissue_force_N": self.force,
            "maximum_rcm_error_m": self.maximum_rcm_error,
            "active_tetrahedra": len(self.example.selection.indices),
            "active_nodes": len(self.example.selection.vertices),
            "frames": self.frames,
            "device": str(self.example.device),
            "performance_benchmark": False,
        }

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Select one FEM region on a clickable map and right-drag the pink suture pad.

Only the selected region contributes particles, tetrahedra and sparse FEM
blocks. Other regions remain visible at rest. Cut interfaces against inactive
material and the pad underside are fixed. Switching regions resets deformation
and mouse holds; None disables the FEM solve.

Right-drag an active region, release to leave a spring hold, and middle-click
its marker to release it. Use Clear holds to watch elastic recovery.

Run: python -m newton_solver_fem.examples fem_phantom_regions
"""

from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem import SolverFEM, list_mesh_regions, load_mesh_regions
from newton_solver_fem.examples._region_map import RegionMap
from newton_solver_fem.examples._screen_pick import ScreenPickTool
from newton_solver_fem.examples._surface_pick import SurfacePicker
from newton_solver_fem.examples._tet_surface_holds import TetSurfaceHolds

# Warm pink inspired by the manufacturer's pink soft-tissue suture pad.
PAD_COLOR = (0.86, 0.48, 0.40)


@wp.kernel(enable_backward=False)
def _scatter_selected(q: wp.array[wp.vec3], indices: wp.array[int], display: wp.array[wp.vec3]):
    node = wp.tid()
    display[indices[node]] = q[node]


def underside_vertices(vertices, exterior_triangles):
    """Identify the authored planar underside without moving the reference mesh."""
    faces = vertices[exterior_triangles]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    areas = np.linalg.norm(normals, axis=1)
    downward = normals[:, 2] < -0.95 * areas
    downward &= faces.mean(axis=1)[:, 2] <= np.quantile(vertices[:, 2], 0.3) + 1.0e-12
    candidates = np.flatnonzero(downward)
    if not len(candidates):
        raise ValueError("The pad asset needs a flat, downward-facing underside")
    face = candidates[np.argmax(areas[candidates])]
    normal = normals[face] / areas[face]
    distance = (vertices - faces[face, 0]) @ normal
    return np.flatnonzero(np.abs(distance) < 1.0e-7)


class Example:
    def __init__(self, viewer, args=None):
        self.args = newton.examples.default_args(self.create_parser()) if args is None else args
        args = self.args
        if min(args.substeps, args.solver_iterations) < 1:
            raise ValueError("Substeps and solver iterations must be positive")
        parameters = (args.young_modulus, args.density, args.grip_stiffness, args.grip_radius, args.grip_speed)
        if not np.isfinite(parameters).all() or min(parameters) <= 0:
            raise ValueError("Material and grip parameters must be finite and positive")
        damping = (args.material_damping, args.grip_damping, args.linear_damping)
        if not 0 < args.poisson_ratio < 0.5 or not np.isfinite(damping).all() or min(damping) < 0:
            raise ValueError("Poisson ratio must lie in (0, 0.5) and damping must be nonnegative")
        self.viewer, self.device = viewer, wp.get_device()
        self.asset_path = Path(args.asset).expanduser().resolve()
        self.regions = list_mesh_regions(self.asset_path)
        self.reference = load_mesh_regions(self.asset_path, [region.id for region in self.regions])
        self.region_map = RegionMap(
            self.reference.vertices,
            self.reference.exterior_surface_triangles,
            self.reference.exterior_surface_region_ids,
            self.regions,
        )
        self.full_tet_count = len(self.reference.indices)
        self.frame_dt, self.sim_time = 1 / 60, 0.0
        self.sim_dt = self.frame_dt / args.substeps
        self._underside_global = self.reference.global_vertex_indices[
            underside_vertices(self.reference.vertices, self.reference.exterior_surface_triangles)
        ]
        self.display_points = wp.array(self.reference.vertices, dtype=wp.vec3, device=self.device)
        self._render_meshes = []
        for region in self.regions:
            triangles = self.reference.exterior_surface_triangles[
                self.reference.exterior_surface_region_ids == region.id
            ]
            self._render_meshes.append((region, wp.array(triangles.reshape(-1), dtype=int, device=self.device)))
        self.surface_picker = SurfacePicker(self.device)
        self.screen_pick = None
        self.solver, self.model, self.selection, self.holds = None, None, None, None
        self.selected_region_ids = ()
        self.selection_revision = 0
        self.selection_error = ""
        self.select_regions(args.regions)
        self.screen_pick = ScreenPickTool(
            viewer,
            self.model,
            begin=self.begin_grab,
            move=self.move_grab,
            finish=self.finish_grab,
            cancel=self.cancel_grab,
            remove=self.remove_hold,
            can_start=lambda: self.solver is not None,
        )
        self.reset_camera()

    def reset_camera(self):
        center = (self.reference.vertices.min(axis=0) + self.reference.vertices.max(axis=0)) / 2
        span = np.ptp(self.reference.vertices, axis=0).max()
        offset = span * np.array((0.95, -1.10, 1.20))
        self.view_direction = -offset / np.linalg.norm(offset)
        right = np.cross(self.view_direction, (0.0, 0.0, 1.0))
        right /= np.linalg.norm(right)
        # Leave the left sidebar clear while filling the remaining viewport.
        target = center - 0.38 * span * right
        self.view_position = target + offset
        self.view_up = np.array((0.0, 0.0, 1.0)) - self.view_direction[2] * self.view_direction
        self.view_up /= np.linalg.norm(self.view_up)
        pitch = np.degrees(np.arcsin(self.view_direction[2]))
        yaw = np.degrees(np.arctan2(self.view_direction[1], self.view_direction[0]))
        # The pad is only 13.5 cm wide; viewer defaults target metre-scale scenes.
        self.viewer.camera_speed = 0.5 * span
        self.viewer.set_camera(pos=tuple(self.view_position), pitch=float(pitch), yaw=float(yaw))
        camera = getattr(self.viewer, "camera", None)
        if camera is not None and hasattr(camera, "set_pivot"):
            camera.set_pivot(target)

    def select_regions(self, region_ids):
        """Activate at most one region as a compact FEM body, resetting to rest."""
        ids = tuple(sorted(set(int(value) for value in region_ids)))
        if len(ids) > 1:
            raise ValueError("Choose at most one active FEM region; use an empty selection to disable FEM")
        unknown = set(ids) - {region.id for region in self.regions}
        if unknown:
            raise ValueError(f"Unknown pad regions: {sorted(unknown)}")
        selection = load_mesh_regions(self.asset_path, ids)
        if self.screen_pick is not None:
            self.screen_pick.reset()
        self.selection_error = ""
        builder = newton.ModelBuilder()
        self.holds, self.solver, self.control, self.graph = None, None, None, None
        self.selection = selection
        self.selected_region_ids = ids
        self.fixed_vertex_indices = np.union1d(
            selection.fixed_vertex_indices,
            np.flatnonzero(np.isin(selection.global_vertex_indices, self._underside_global)),
        ).astype(np.int32)
        if len(selection.indices):
            mu = self.args.young_modulus / (2 * (1 + self.args.poisson_ratio))
            lam = (
                self.args.young_modulus
                * self.args.poisson_ratio
                / ((1 + self.args.poisson_ratio) * (1 - 2 * self.args.poisson_ratio))
            )
            builder.add_soft_mesh(
                pos=wp.vec3(0.0),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0),
                scale=1.0,
                vertices=selection.vertices,
                indices=selection.indices.reshape(-1),
                density=self.args.density,
                k_mu=mu,
                k_lambda=lam,
                k_damp=self.args.material_damping,
                particle_radius=0.00015,
                add_surface_mesh_edges=False,
            )
        self.model = builder.finalize(device=self.device)
        self.model.set_gravity((0.0, 0.0, 0.0))
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self.viewer.set_model(self.model)
        for attribute in ("show_particles", "show_triangles", "show_springs", "picking_enabled"):
            if hasattr(self.viewer, attribute):
                setattr(self.viewer, attribute, False)
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.spotlight_enabled = False
        if self.screen_pick is not None:
            self.screen_pick.model = self.model
            # ViewerGL clears example callbacks when changing models. Initial
            # registration belongs to newton.examples.run and ScreenPickTool;
            # subsequent selections restore each callback exactly once here.
            if hasattr(self.viewer, "register_ui_callback"):
                self.viewer.register_ui_callback(self.gui, position="side")
                self.viewer.register_ui_callback(self.screen_pick.draw_overlay, position="free")
        self._active_indices = wp.array(selection.exterior_surface_triangles.reshape(-1), dtype=int, device=self.device)
        inactive_triangles = self.reference.exterior_surface_triangles[
            ~np.isin(self.reference.exterior_surface_region_ids, ids)
        ]
        self._inactive_indices = wp.array(inactive_triangles.reshape(-1), dtype=int, device=self.device)
        self._active_to_display = wp.array(
            np.searchsorted(self.reference.global_vertex_indices, selection.global_vertex_indices).astype(np.int32),
            dtype=int,
            device=self.device,
        )
        self.display_points.assign(self.reference.vertices)
        if len(selection.indices):
            self.control = self.model.control()
            self.solver = SolverFEM(
                self.model,
                dt=self.sim_dt,
                preconditioner_type="block_diag",
                solver_type="cg",
                linear_solver_maxiter=self.args.solver_iterations,
                ground_plane=None,
                self_contact_ke=0.0,
                self_contact_kd=0.0,
                linear_damping=self.args.linear_damping,
                min_stretch=0.2,
                max_stretch=5.0,
                fixed_particle_indices=self.fixed_vertex_indices,
            )
            self.holds = TetSurfaceHolds(
                self.device,
                selection.indices,
                selection.exterior_surface_triangles,
                self.fixed_vertex_indices,
                capacity=8,
                stiffness=self.args.grip_stiffness,
                damping=self.args.grip_damping,
                maximum_speed=self.args.grip_speed,
                radius=self.args.grip_radius,
            )
            self._external_forces = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
            self._capture_physics()
        self.selection_revision += 1
        print(
            f"[phantom regions] active={list(ids)}: {len(selection.indices):,}/{self.full_tet_count:,} tetrahedra, "
            f"{len(selection.vertices):,} nodes, {len(self.fixed_vertex_indices):,} fixed",
            flush=True,
        )

    def begin_grab(self, origin, direction):
        if self.holds is None:
            return None
        hit = self.surface_picker.pick(
            origin,
            direction,
            [
                (self.state_0.particle_q, self._active_indices, len(self.selection.exterior_surface_triangles)),
                (self.display_points, self._inactive_indices, self._inactive_indices.size // 3),
            ],
        )
        if hit is None or hit[0] != 0:
            return None
        _, triangle, barycentric, point = hit
        if self.holds.begin_surface(triangle, barycentric, point, self.state_0.particle_q):
            return point
        return None

    def move_grab(self, point):
        if self.holds is not None:
            self.holds.move(point)

    def finish_grab(self):
        if self.holds is not None:
            self.holds.finish()

    def cancel_grab(self):
        if self.holds is not None:
            self.holds.cancel()

    def remove_hold(self, index):
        return self.holds is not None and self.holds.remove(index)

    def clear_holds(self):
        if self.screen_pick is not None:
            self.screen_pick.reset()
        if self.holds is not None:
            self.holds.clear()

    def reset(self):
        self.select_regions(self.selected_region_ids)

    def simulate(self):
        """A stable pair of state buffers supports both eager and captured frames."""
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.holds.add_matrix(self.solver, self.sim_dt)
        for substep in range(self.args.substeps):
            current, following = (self.state_0, self.state_1) if substep % 2 == 0 else (self.state_1, self.state_0)
            self._external_forces.zero_()
            self.holds.add_forces(current.particle_q, current.particle_qd, self._external_forces, self.sim_dt)
            self.solver.step(current, following, self.control, None, self.sim_dt, external_forces=self._external_forces)
        if self.args.substeps % 2:
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)

    def _capture_physics(self):
        self.graph = None
        if not self.device.is_cuda:
            return
        snapshots = (self.model.state(), self.model.state())
        for target, source in zip(snapshots, (self.state_0, self.state_1), strict=True):
            target.assign(source)
        persistent = (self.solver.velocity_increment, self.holds.driven)
        saved = tuple(wp.clone(array) for array in persistent)

        def restore():
            self.state_0.assign(snapshots[0])
            self.state_1.assign(snapshots[1])
            for target, source in zip(persistent, saved, strict=True):
                wp.copy(target, source)

        try:
            # Allocate the sparse structure, CG scratch and generated kernels first.
            self.simulate()
            wp.synchronize_device(self.device)
            restore()
            with wp.ScopedCapture(device=self.device) as capture:
                self.simulate()
            self.graph = capture.graph
        except (RuntimeError, ValueError) as error:
            print(f"[phantom regions] CUDA capture unavailable; using eager FEM: {error}", flush=True)
        finally:
            restore()

    def step(self):
        if self.solver is not None:
            if self.graph is None:
                self.simulate()
            else:
                wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        if self.solver is not None:
            wp.launch(
                _scatter_selected,
                dim=self.model.particle_count,
                inputs=[self.state_0.particle_q, self._active_to_display, self.display_points],
                device=self.device,
            )
        self.viewer.log_state(self.state_0)
        for region, indices in self._render_meshes:
            self.viewer.log_mesh(
                f"/phantom/region_{region.id:02d}",
                points=self.display_points,
                indices=indices,
                color=PAD_COLOR,
                roughness=0.75,
                metallic=0.0,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Phantom: active FEM region")
        ui.text_wrapped(
            "Click one region in the top-view map to simulate it. The pink region is active; all others stay fixed. "
            "Switching regions resets deformation and holds."
        )
        chosen = set(self.selected_region_ids)
        clicked = self.region_map.draw(ui, self.selected_region_ids)
        changed = clicked is not None and self.selected_region_ids != (clicked,)
        if changed:
            chosen = {clicked}
        ui.text("Top view (+Z) | 0 = base and lettering")
        if ui.button("None"):
            chosen, changed = set(), bool(self.selected_region_ids)
        button_min, button_max = ui.get_item_rect_min(), ui.get_item_rect_max()
        self.region_none_button_position = (
            (button_min.x + button_max.x) / 2,
            (button_min.y + button_max.y) / 2,
        )
        if changed:
            try:
                self.select_regions(chosen)
            except ValueError as error:
                self.selection_error = str(error)
        if self.selection_error:
            ui.text_wrapped(f"Selection unavailable: {self.selection_error}")
        ui.text(f"Active: {len(self.selection.indices):,} / {self.full_tet_count:,} tetrahedra")
        ui.text(f"Fixed cut / underside nodes: {len(self.fixed_vertex_indices):,}")
        ui.separator()
        ScreenPickTool.gui(ui)
        if ui.button("Clear holds"):
            self.clear_holds()
        ui.same_line()
        if ui.button("Reset deformation"):
            self.reset()
        if ui.button("Frame pad"):
            self.reset_camera()
        ui.text("SI units: metres, kilograms, seconds; no mass rescaling")

    def test_final(self):
        if self.solver is None:
            return
        positions = self.state_0.particle_q.numpy()
        if not np.isfinite(positions).all() or not np.isfinite(self.state_0.particle_qd.numpy()).all():
            raise AssertionError("Regional FEM produced nonfinite state")
        current = positions[self.selection.indices].astype(np.float64)
        rest = self.selection.vertices[self.selection.indices]
        current_det = np.linalg.det((current[:, 1:] - current[:, :1]).transpose(0, 2, 1))
        rest_det = np.linalg.det((rest[:, 1:] - rest[:, :1]).transpose(0, 2, 1))
        if np.any(current_det <= 0):
            raise AssertionError(f"Regional FEM inverted an element: minimum J={np.min(current_det / rest_det):.6g}")
        expected = self.selection.vertices[self.fixed_vertex_indices].astype(np.float32)
        if not np.array_equal(positions[self.fixed_vertex_indices], expected):
            raise AssertionError("Fixed cut/underside vertices moved")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--asset", default=str(Path(__file__).parent / "assets/phantom_pad/phantom_regions.usd"))
        parser.add_argument(
            "--regions",
            type=int,
            nargs="*",
            default=[9],
            help="Zero or one initially active region ID; empty disables FEM",
        )
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--solver-iterations", type=int, default=80)
        parser.add_argument("--young-modulus", type=float, default=30000.0, help="Young's modulus [Pa]")
        parser.add_argument("--poisson-ratio", type=float, default=0.35)
        parser.add_argument("--density", type=float, default=1000.0, help="Physical density [kg/m^3]")
        parser.add_argument("--material-damping", type=float, default=0.01)
        parser.add_argument("--linear-damping", type=float, default=1.0)
        parser.add_argument(
            "--grip-stiffness", type=float, default=20.0, help="Total finite-patch grip stiffness [N/m]"
        )
        parser.add_argument("--grip-damping", type=float, default=0.03, help="Total grip damping [kg/s]")
        parser.add_argument("--grip-radius", type=float, default=0.002, help="Grip patch radius [m]")
        parser.add_argument("--grip-speed", type=float, default=0.015, help="Maximum moving grip speed [m/s]")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

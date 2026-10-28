# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Drill holes, slice or carve a soft Baymax; use the mouse to lift, hold and release material.

Baymax falls under gravity and deforms against the floor. Cylinder is the
default cutter: left-click to drill through the body. Select Line and
left-drag, then release to make an incision. Select Sphere and hold the
left button to carve along the initial viewing direction.

Right-drag to lift material. Releasing the mouse leaves a spring hold at
a green marker. Middle-click a marker to release that hold, or use
Clear holds to release them all.

The original tetrahedral asset supplies a position-only elastic field. Cuts
partition its material into convex fragments and duplicate disconnected basis
nodes. Fragment volumes weight the original element energy; barycentric first
moments give the nodal masses. Contact and drawing use the actual material
boundary, including the new incision faces.

Run: python -m newton_solver_fem.examples linear_fem_baymax_cut
"""

import argparse
from pathlib import Path
from time import perf_counter

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem import (
    CutModel,
    CutSurfaceContact,
    CutVolumeGuard,
    LinearTetCut,
    MaterialHolds,
    SolverFEM,
    TetrahedralTunnelGuide,
    load_asset,
)
from newton_solver_fem.examples._cut_debug_view import CutDebugView
from newton_solver_fem.examples._cut_diagnostics import rejection_message
from newton_solver_fem.examples._cylinder_extent import CylinderExtent
from newton_solver_fem.examples._hold_debug import HoldDebug
from newton_solver_fem.examples._screen_cut import ScreenCutTool
from newton_solver_fem.examples._screen_cylinder_cut import ScreenCylinderCutTool
from newton_solver_fem.examples._screen_pick import ScreenPickTool
from newton_solver_fem.examples._screen_sphere_cut import ScreenSphereCutTool
from newton_solver_fem.examples._surface_pick import SurfacePicker


@wp.kernel
def _clone_nodes(
    sources: wp.array2d[int],
    count: wp.array[int],
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
    copied_q: wp.array[wp.vec3],
    copied_qd: wp.array[wp.vec3],
):
    i = wp.tid()
    if i < count[0]:
        target, source = sources[i, 0], sources[i, 1]
        copied_q[target] = q[source]
        copied_qd[target] = qd[source]


@wp.kernel(enable_backward=False)
def _node_components(
    elements: wp.array2d[int],
    fractions: wp.array[float],
    components: wp.array[int],
    node_components: wp.array[int],
):
    element = wp.tid()
    if fractions[element] > 0.0:
        for corner in range(4):
            wp.atomic_max(node_components, elements[element, corner], components[element])


def load_baymax():
    """Load the original filled Baymax mesh, rotated from Y-up to Z-up [m]."""

    asset = load_asset(str(Path(__file__).parent / "assets/tet_baymax.usda"))
    points = asset["vertices"].astype(np.float64)[:, (0, 2, 1)] * (1.0, -1.0, 1.0)
    points[:, 2] -= points[:, 2].min()
    elements = asset["tets"].copy()
    vertices = points[elements]
    negative = np.linalg.det((vertices[:, 1:] - vertices[:, :1]).transpose(0, 2, 1)) < 0.0
    elements[negative, 1], elements[negative, 2] = elements[negative, 2].copy(), elements[negative, 1].copy()
    return points, elements


class Example:
    def __init__(self, viewer, args=None):
        args = newton.examples.default_args(self.create_parser()) if args is None else args
        if min(args.substeps, args.solver_iterations, args.node_copies) < 1:
            raise ValueError("Substeps, solver iterations and node copies must be positive")
        if not np.isfinite([args.young_modulus, args.density, args.drop_height, args.gravity]).all():
            raise ValueError("Material and drop parameters must be finite")
        if args.young_modulus <= 0 or args.density <= 0 or args.drop_height < 0 or args.gravity < 0:
            raise ValueError("Material values must be positive and drop height/gravity nonnegative")
        self.viewer, self.args = viewer, args
        self.device = wp.get_device()
        self.frame_dt, self.sim_time = 1.0 / 60.0, 0.0
        self.sim_dt = self.frame_dt / args.substeps
        self.reference_positions, self.reference_elements = load_baymax()
        self._pool_size = len(self.reference_positions) * args.node_copies
        builder = newton.ModelBuilder()
        pool = np.tile(self.reference_positions, (args.node_copies, 1))
        builder.add_particles(
            pool.tolist(), np.zeros_like(pool).tolist(), np.ones(len(pool)).tolist(), radius=[0.0] * len(pool)
        )
        builder.add_ground_plane()
        self._display_model = builder.finalize(device=self.device)
        self._display_model.set_gravity((0.0, 0.0, -args.gravity))
        self._display_state = self._display_model.state()
        self._model_factory = CutModel(
            self._display_model,
            self.reference_positions,
            self.reference_elements,
            density=args.density,
            young_modulus=args.young_modulus,
            poisson_ratio=0.3,
            damping=args.material_damping,
        )
        self.viewer.set_model(self._display_model)
        if hasattr(viewer, "show_particles"):
            viewer.show_particles = False
        self.viewer.set_camera(pos=(1.0, -1.7, 0.95), pitch=-12.0, yaw=120.0)
        self.cutter_mode = args.cutter_mode
        self.screen_cut = ScreenCutTool(
            viewer,
            self._display_model,
            can_start=lambda: self.cutter_mode == "line" and not self.screen_pick.active,
        )
        self.surface_picker = SurfacePicker(self.device)
        self.cylinder_extent = CylinderExtent(self.device)
        self.tunnel_guide = TetrahedralTunnelGuide(self.device)
        self.screen_pick = ScreenPickTool(
            viewer,
            self._display_model,
            begin=self._begin_pick,
            move=self._move_pick,
            finish=self._finish_pick,
            cancel=self._cancel_pick,
            remove=self._remove_pick,
            can_start=lambda: (
                self.screen_cut.start is None and not self.screen_sphere.active and not self.screen_cylinder.active
            ),
        )
        self.screen_sphere = ScreenSphereCutTool(
            viewer,
            self._display_model,
            begin=self._pick_cut_surface,
            finish=self._finish_sphere,
            can_start=lambda: not self.screen_pick.active and self.screen_cut.start is None,
            speed=args.sphere_speed,
            enabled=self.cutter_mode == "sphere",
        )
        self.screen_cylinder = ScreenCylinderCutTool(
            viewer,
            self._display_model,
            begin=self._pick_cut_surface,
            can_start=lambda: not self.screen_pick.active and self.screen_cut.start is None,
            enabled=self.cutter_mode == "cylinder",
            diameter_pixels=args.cylinder_diameter_pixels,
        )
        self.connection_debug = CutDebugView(viewer, self.device, self._debug_topology, kind="linear")
        self._hold_debug = None
        self._hold_debug_revision = None
        self._hold_revision = 0
        self.reset()

    def _debug_topology(self):
        return self.cutter.geometry.arrays, self.cutter.material_components, self.model.tet_indices

    def _make_solver(self, model):
        solver = SolverFEM(
            model,
            tet_volume_weights=model.cut_volume_fractions,
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
        )
        if self.args.streaming_topology:
            solver.prepare_tetrahedral_updates(len(self.reference_positions))
        return solver

    def _install_physics(self, model, state, surface):
        self.model, self.state_0, self.state_1 = model, state, model.state()
        self.state_1.assign(state)
        self.control = model.control()
        self.solver = self._make_solver(model)
        self._mechanical_components = wp.clone(self.cutter.material_components)
        self._node_components = wp.empty(model.particle_count, dtype=int, device=self.device)
        self._update_mechanical_components(self.cutter.material_components)
        self.solver.prepare_component_scaling(self._node_components, model.tet_count)
        self.volume_guard = CutVolumeGuard(model, components=self._mechanical_components)
        self.contact = CutSurfaceContact(
            surface,
            state.particle_q,
            model.particle_inv_mass,
            ground_stiffness=self.args.ground_stiffness,
            piece_stiffness=self.args.piece_contact_stiffness,
        )
        self.surface = surface
        self._surface_count = int(surface.count.numpy()[0])
        # GL uploads indices once. Reserved triangles stay in the draw list;
        # inactive point triples collapse to zero until a cut activates them.
        self._render_indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        self.holds.remap(self.cutter.geometry)
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.graph = self._capture_physics_graph()

    def _update_mechanical_components(self, components):
        # A growing surface warms the new model before the geometry transaction
        # commits. Its component labels must already match those new elements.
        wp.copy(self._mechanical_components, components)
        self._node_components.fill_(-1)
        wp.launch(
            _node_components,
            dim=self.model.tet_count,
            inputs=[
                self.model.tet_indices,
                self.model.cut_volume_fractions,
                self._mechanical_components,
                self._node_components,
            ],
            device=self.device,
        )

    def _capture_physics_graph(self):
        """Warm and capture the current buffers without advancing physical state."""
        if not self.device.is_cuda:
            return None
        states = (self.state_0, self.state_1)
        saved_states = tuple(self.model.state() for _ in states)
        for saved, state in zip(saved_states, states, strict=True):
            saved.assign(state)
        persistent = (self.solver.velocity_increment, self.holds.driven, self.volume_guard.limited_steps)
        saved_persistent = tuple(wp.clone(array) for array in persistent)
        try:
            self.simulate()
            for state, saved in zip(states, saved_states, strict=True):
                state.assign(saved)
            for array, saved in zip(persistent, saved_persistent, strict=True):
                wp.copy(array, saved)
            with wp.ScopedCapture(device=self.device) as capture:
                self.simulate()
            return capture.graph
        finally:
            for state, saved in zip(states, saved_states, strict=True):
                state.assign(saved)
            for array, saved in zip(persistent, saved_persistent, strict=True):
                wp.copy(array, saved)

    def _grow_surface(self, triangle_count):
        """Replace contact and drawing storage only when the boundary outgrows it."""
        capacity = max(1, self.contact.capacity)
        while capacity < triangle_count:
            capacity *= 2
        capacity = min(capacity, self.cutter.surface.components.size)
        surface = CutSurfaceContact.reserve_surface(self.cutter.surface, triangle_count, capacity)
        previous = self.contact
        contact = CutSurfaceContact(
            surface,
            self.state_0.particle_q,
            self.model.particle_inv_mass,
            ground_height=previous.ground_height,
            ground_stiffness=previous.ground_stiffness,
            piece_stiffness=previous.piece_stiffness,
            damping_ratio=previous.damping_ratio,
            search_radius=previous.search_radius,
        )
        indices = wp.array(np.arange(surface.indices.size, dtype=np.int32), device=self.device)
        self.contact, self.surface, self._render_indices = contact, surface, indices
        # The FEM model, states, solver and sparse storage retain their addresses.
        # Only the graph's references to surface/contact buffers need replacing.
        self.graph = self._capture_physics_graph()

    def simulate(self):
        self.solver.refresh_tangent(self.state_0.particle_q, self.sim_dt)
        self.holds.add_matrix(self.solver, self.sim_dt)
        for substep in range(self.args.substeps):
            current, following = (self.state_0, self.state_1) if substep % 2 == 0 else (self.state_1, self.state_0)
            forces = self.contact.evaluate(current.particle_q, current.particle_qd, self.sim_dt, piece_contact=True)
            self.holds.add_forces(current.particle_q, current.particle_qd, forces, self.sim_dt)
            self.solver.step(current, following, self.control, None, self.sim_dt, external_forces=forces)
            self.volume_guard.apply(
                following.particle_q, following.particle_qd, self.sim_dt, previous_q=current.particle_q
            )
        if self.args.substeps % 2:
            wp.copy(self.state_0.particle_q, self.state_1.particle_q)
            wp.copy(self.state_0.particle_qd, self.state_1.particle_qd)

    def reset(self):
        self.screen_cut.cancel()
        self.screen_sphere.cancel()
        self.screen_cylinder.cancel()
        self._sphere_clock = perf_counter()
        self._sphere_segment_start = None
        self._sphere_gesture = -1
        self.screen_pick.reset()
        self.connection_debug.invalidate()
        if self._hold_debug is not None:
            self._hold_debug.hide(self.viewer)
        self._hold_debug, self._hold_debug_revision = None, None
        self.holds = MaterialHolds(self.device)
        self.cut_revision = 0
        source_count = len(self.reference_elements)
        self.cutter = LinearTetCut(
            self.reference_positions,
            self.reference_elements,
            fragment_capacity=self.args.fragment_capacity or source_count + max(4096, (source_count + 3) // 4),
            node_copies=self.args.node_copies,
            vertex_capacity=32,
            face_capacity=24,
            device=self.device,
        )
        self.cutter.prepare(patch_counts=(1,))
        self.cutter.prepare_sphere()
        pool = np.tile(self.reference_positions, (self.args.node_copies, 1))
        pool[:, 2] += self.args.drop_height
        q = wp.array(pool, dtype=wp.vec3, device=self.device)
        qd = wp.zeros_like(q)
        self._cut_positions = wp.empty_like(q)
        self.fragment_count = len(self.reference_elements)
        model, state = self._build_model(self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd)
        self._staged_model, self._staged_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        self._backup_model, self._backup_state = self._build_model(
            self.cutter.geometry, self.cutter.geometry.nodeindices, q, qd
        )
        surface_count = int(self.cutter.surface.count.numpy()[0])
        surface = CutSurfaceContact.reserve_surface(
            self.cutter.surface,
            surface_count,
            self.args.surface_capacity or max(2 * surface_count, surface_count + 256),
        )
        self._install_physics(model, state, surface)
        # Keep the display model bound: rebinding would clear cutting overlays
        # and UI callbacks. The viewer resizes the drawing storage as needed.
        self.sim_time, self.cut_revision = 0.0, 0
        self.last_cut_ms, self.last_geometry_ms, self.last_rebuild_ms = 0.0, 0.0, 0.0
        self.cut_error = ""

    def _build_model(self, geometry, nodes, q, qd):
        return self._model_factory.build(
            geometry.parent,
            nodes,
            self.cutter.volume_fractions,
            self.cutter.mass_integrals,
            self.fragment_count,
            q,
            qd,
            capacity=self.cutter.capacity,
        )

    def cut(self, triangles):
        """Apply a finite screen-projected incision; leave a rejected edit unpublished."""
        return self._apply_edit(lambda positions: self.cutter.propose_triangles(positions, triangles))

    def cut_sphere(self, start, end, radius):
        """Remove the spherical tool's swept material; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_sphere(positions, start, end, radius))

    def cut_cylinder(self, start, end, radius):
        """Remove one straight, constant-radius cylinder; distances are in metres."""
        return self._apply_edit(lambda positions: self.cutter.propose_cylinder(positions, start, end, radius))

    def cut_cylinder_ray(self, origin, direction, radius):
        """Drill through the current material along a screen ray in one transaction."""
        self.contact.update_surface(self.state_0.particle_q)
        try:
            extent = self.cylinder_extent.query(self.contact.points, self._surface_count, origin, direction, radius)
        except ValueError as error:
            self.cut_error = f"Cut rejected: {error}"
            return False
        if extent is None:
            self.cut_error = ""
            return False
        return self.cut_cylinder(*extent, radius)

    def _consume_cylinder(self):
        request = self.screen_cylinder.consume()
        if request is not None:
            self.cut_cylinder_ray(*request)

    def _apply_edit(self, propose):
        start = perf_counter()
        published = False
        holds_staged = False
        previous_count, previous_surface_count = self.fragment_count, self._surface_count
        previous_surface_buffers = self.contact, self.surface, self._render_indices, self.graph
        try:
            wp.copy(self._cut_positions, self.state_0.particle_q)
            diagnostics = propose(self._cut_positions)
            if not diagnostics["changed_parents"]:
                self.cutter.discard()
                self.cut_error = ""
                return False
            surface_count = diagnostics["triangle_count"]
            self.last_geometry_ms = 1000.0 * (perf_counter() - start)
            self._staged_state.assign(self.state_0)
            wp.launch(
                _clone_nodes,
                dim=self._pool_size,
                inputs=[
                    self.cutter.candidate.clone_sources,
                    self.cutter.candidate.clone_count,
                    self.state_0.particle_q,
                    self.state_0.particle_qd,
                    self._staged_state.particle_q,
                    self._staged_state.particle_qd,
                ],
                device=self.device,
            )
            self._model_factory.update(
                self._staged_model,
                self.cutter.pending_geometry.parent,
                self.cutter.candidate.nodes,
                self.cutter.volume_fractions,
                self.cutter.mass_integrals,
                diagnostics["fragment_count"],
            )
            holds_staged = True
            self.holds.stage_remap(self.cutter.pending_geometry, self.cutter.candidate.nodes, self.cutter.geometry)
            self._model_factory.copy_into(self._backup_model, self.model)
            self._backup_state.assign(self.state_0)
            published = True
            self._model_factory.copy_into(self.model, self._staged_model)
            self.state_0.assign(self._staged_state)
            self.state_1.assign(self._staged_state)
            self._update_mechanical_components(self.cutter.candidate.material_components)
            self.solver.refresh_component_scaling()
            self.solver.refresh_topology(
                self.state_0.particle_q,
                self.sim_dt,
                element_sources=self.cutter.pending_geometry.source if self.args.streaming_topology else None,
            )
            if surface_count > self.contact.capacity:
                self._grow_surface(surface_count)
            else:
                self.contact.update_topology(self.cutter.surface, surface_count, self.state_0.particle_q)
            if self.device.is_cuda:
                wp.synchronize_device(self.device)
            self.cutter.commit()
            self.fragment_count, self._surface_count = diagnostics["fragment_count"], surface_count
            self.cut_revision += 1
        except (MemoryError, ValueError, RuntimeError) as error:
            self.cutter.discard()
            if holds_staged:
                self.holds.restore()
            self.contact, self.surface, self._render_indices, self.graph = previous_surface_buffers
            if published:
                self._model_factory.copy_into(self.model, self._backup_model)
                self.state_0.assign(self._backup_state)
                self.state_1.assign(self._backup_state)
                self._update_mechanical_components(self.cutter.material_components)
                self.solver.refresh_component_scaling()
                self.solver.refresh_topology(self.state_0.particle_q, self.sim_dt)
                self.contact.update_topology(self.cutter.surface, previous_surface_count, self.state_0.particle_q)
            self.fragment_count, self._surface_count = previous_count, previous_surface_count
            self.cut_error = rejection_message(error, self.cutter.geometry.capacity)
            return False
        self.last_cut_ms = 1000.0 * (perf_counter() - start)
        self.last_rebuild_ms = self.last_cut_ms - self.last_geometry_ms
        self.cut_error = ""
        return True

    def _pick_cut_surface(self, origin, direction):
        self.contact.update_surface(self.state_0.particle_q)
        hit = self.surface_picker.pick(
            origin, direction, [(self.contact.points, self._render_indices, self._surface_count)]
        )
        return None if hit is None else hit[3]

    def _advance_sphere(self):
        now = perf_counter()
        # Avoid a long catch-up sweep after a paused debugger or blocked window.
        dt = min(now - self._sphere_clock, 0.1)
        self._sphere_clock = now
        if not self.screen_sphere.active:
            self._sphere_segment_start = None
            return
        if self._sphere_gesture != self.screen_sphere.gesture_id:
            self._sphere_gesture = self.screen_sphere.gesture_id
            self._sphere_segment_start = self.screen_sphere.center.copy()
            self._next_sphere_boundary()
            dt = 0.0
        segment = self.screen_sphere.advance(dt)
        if segment is not None:
            _, end, radius = segment
            boundary = self.tunnel_guide.distance(
                self.state_0.particle_q, self._sphere_segment_start, self.screen_sphere.direction
            )
            if boundary is None:
                boundary = self._next_sphere_boundary()
            # Existing tetrahedral faces set the update frequency, independent
            # of rendering FPS. Geometry still clips the current swept volume;
            # the guide neither deletes whole cells nor redirects the tunnel.
            if boundary is None or np.linalg.norm(end - self._sphere_segment_start) < boundary:
                return
            self.cut_sphere(self._sphere_segment_start, end, radius)
            self._sphere_segment_start = end.copy()
            if self.cut_error:
                self.screen_sphere.cancel()
            else:
                self._next_sphere_boundary()

    def _next_sphere_boundary(self):
        return self.tunnel_guide.next_boundary(
            self.cutter.geometry,
            self.state_0.particle_q,
            self._sphere_segment_start,
            self.screen_sphere.direction,
        )

    def _finish_sphere(self):
        """Publish the last partial cell at the displayed tip, without advancing."""
        start, end = self._sphere_segment_start, self.screen_sphere.center
        if start is not None and end is not None and np.linalg.norm(end - start) > self.cutter.tolerance:
            self.cut_sphere(start, end, self.screen_sphere.radius)
        self._sphere_segment_start = None

    def set_cutter_mode(self, mode):
        if mode not in ("line", "cylinder", "sphere"):
            raise ValueError("Cutter mode must be line, cylinder or sphere")
        self.screen_cut.cancel()
        self.screen_sphere.cancel()
        self.screen_cylinder.cancel()
        self.cutter_mode = mode
        self.screen_sphere.enabled = mode == "sphere"
        self.screen_cylinder.enabled = mode == "cylinder"
        self._sphere_clock = perf_counter()
        self._sphere_segment_start = None

    def _begin_pick(self, origin, direction):
        self.contact.update_surface(self.state_0.particle_q)
        hit = self.surface_picker.pick(
            origin, direction, [(self.contact.points, self._render_indices, self._surface_count)]
        )
        if hit is None:
            return None
        _, triangle, barycentric, point = hit
        if self.holds.begin(self.surface, self.cutter.geometry, triangle, barycentric, point, self.state_0.particle_q):
            self._hold_revision += 1
            return point
        return None

    def _move_pick(self, point):
        self.holds.move(point)

    def _finish_pick(self):
        self.holds.finish()

    def _remove_pick(self, index):
        removed = self.holds.remove(index)
        if removed:
            self._hold_revision += 1
        return removed

    def _cancel_pick(self):
        if hasattr(self, "holds"):
            self.holds.cancel()

    def clear_holds(self):
        self.screen_pick.reset()
        self.holds.clear()
        self._hold_revision += 1

    def step(self):
        self._consume_cylinder()
        self._advance_sphere()
        triangles = self.screen_cut.consume()
        if len(triangles):
            self.cut(triangles)
        if self.graph is None:
            self.simulate()
        else:
            wp.capture_launch(self.graph)
        self.sim_time += self.frame_dt

    def render(self):
        if self.viewer.is_paused():
            self._consume_cylinder()
            self._advance_sphere()
            if self.screen_cut.pending:
                self.cut(self.screen_cut.consume())
        self.contact.update_surface(self.state_0.particle_q)
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self._display_state)
        self.viewer.log_mesh(
            "baymax_linear_fem_cut",
            self.contact.points,
            self._render_indices,
            color=(0.93, 0.94, 0.97),
            hidden=self.connection_debug.hide_skin,
        )
        self.connection_debug.render(self.state_0.particle_q, self.cut_revision)
        if self.connection_debug.enabled:
            if self._hold_debug is None:
                self._hold_debug = HoldDebug(self.holds, self.device)
            revision = (self.cut_revision, self._hold_revision, self.holds.count)
            if revision != self._hold_debug_revision:
                self._hold_debug.rebuild(self.model.tet_indices, self.cutter.material_components, self.fragment_count)
                self._hold_debug_revision = revision
            self._hold_debug.render(self.viewer, self.state_0.particle_q)
        elif self._hold_debug is not None:
            self._hold_debug.hide(self.viewer)
        self.viewer.end_frame()

    def gui(self, ui):
        ui.text("Baymax: SolverFEM cutting")
        self.connection_debug.gui(ui)
        if self.connection_debug.enabled and self._hold_debug is not None:
            ui.text_wrapped("Green lines: individual grip anchors to their world targets.")
            if self._hold_debug.spanning_grips:
                grips = ", ".join(str(index + 1) for index in self._hold_debug.spanning_grips)
                ui.text_wrapped(
                    f"Holds spanning multiple pieces: {grips}. Release those green markers to free both sides."
                )
        modes = ("line", "cylinder", "sphere")
        changed, mode = ui.combo("Cutter", modes.index(self.cutter_mode), ["Line", "Cylinder", "Sphere"])
        if changed:
            self.set_cutter_mode(modes[mode])
        if self.cutter_mode == "cylinder":
            self.screen_cylinder.gui(ui)
        elif self.cutter_mode == "sphere":
            self.screen_sphere.gui(ui)
        else:
            self.screen_cut.gui(ui)
        self.screen_pick.gui(ui)
        if ui.button("Clear holds"):
            self.clear_holds()
        ui.text(f"Holds: {self.holds.count}/{self.holds.capacity}")
        if ui.button("Reset body and cuts"):
            self.reset()
        ui.text(f"Accepted cuts: {self.cut_revision}")
        ui.text(f"Physics: {self.args.substeps} substeps, {self.args.solver_iterations} CG iterations each")
        if self.cut_revision:
            ui.text(f"Last cut: {self.last_cut_ms:.1f} ms")
        if self.cut_error:
            ui.text_wrapped(self.cut_error)

    def test_final(self):
        q, qd = self.state_0.particle_q.numpy(), self.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise AssertionError("Linear FEM Baymax contains non-finite state")
        indices = self.model.tet_indices.numpy()[: self.fragment_count]
        corners = q[indices]
        gradients = (corners[:, 1:] - corners[:, :1]).transpose(0, 2, 1) @ self.model.tet_poses.numpy()[
            : self.fragment_count
        ]
        if self.fragment_count and np.min(np.linalg.det(gradients)) <= 0.0:
            raise AssertionError("Linear FEM Baymax contains inverted material")

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.set_defaults(num_frames=1200)
        parser.add_argument("--cutter_mode", choices=("line", "cylinder", "sphere"), default="cylinder")
        parser.add_argument(
            "--cylinder_diameter_pixels", type=float, default=8.0, help="Cylinder entry diameter [framebuffer pixels]."
        )
        parser.add_argument("--sphere_speed", type=float, default=1.0, help="Held sphere advance [diameters/s].")
        parser.add_argument("--substeps", type=int, default=4)
        parser.add_argument("--solver_iterations", type=int, default=30)
        parser.add_argument(
            "--streaming_topology",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Update cut matrix entries in persistent GPU storage; disable to compare with sparse rebuilding.",
        )
        parser.add_argument("--young_modulus", type=float, default=25000.0, help="Elastic modulus [Pa].")
        parser.add_argument("--density", type=float, default=200.0, help="Material density [kg/m^3].")
        parser.add_argument("--material_damping", type=float, default=0.1, help="Viscous material damping [Pa s].")
        parser.add_argument("--linear_damping", type=float, default=5.0, help="Mass-proportional damping [1/s].")
        parser.add_argument("--drop_height", type=float, default=0.2, help="Initial clearance above the floor [m].")
        parser.add_argument("--gravity", type=float, default=9.81, help="Gravity magnitude [m/s^2].")
        parser.add_argument("--ground_stiffness", type=float, default=2.0e7, help="Surface floor penalty [N/m^3].")
        parser.add_argument(
            "--piece_contact_stiffness", type=float, default=2.0e7, help="Cut-piece contact penalty [N/m^3]."
        )
        parser.add_argument("--node_copies", type=int, default=4)
        parser.add_argument(
            "--fragment_capacity", type=int, default=0, help="Reserved material fragments; zero selects automatically."
        )
        parser.add_argument(
            "--surface_capacity",
            type=int,
            default=0,
            help="Initial boundary triangle reserve; grows automatically. Zero selects the initial size.",
        )
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

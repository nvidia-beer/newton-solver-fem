# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example: Linear-element FEM Demo
#
# Three modes selected via --mode:
#   tet                — tet FEM soft bodies loaded from USD assets
#   hex                — Q1 hex FEM soft bodies
#   showcase (default) — six Tet4/Hex8 soft bodies
#
# Commands:
#   uv run -m newton_solver_fem.examples linear_fem
#   uv run -m newton_solver_fem.examples linear_fem --mode hex
#   uv run -m newton_solver_fem.examples linear_fem --mode showcase
###########################################################################

"""Six deformable solids drop onto a floor: Tet4 box, sphere and Baymax; Hex8 cube, tire and pyramid.

The default showcase loads all six bodies from USD assets and simulates their
elastic deformation under gravity and contact. Tet4 uses four-node tetrahedra;
Q1 Hex8 uses eight-node hexahedra.

Use --mode tet or --mode hex to view an individual element family.
"""

from __future__ import annotations

import json
import os

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem import (
    FEMBuilder,
    SolverFEM,
    detect_asset_type,
    load_asset,
    load_hex_assembly,
)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------


_EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))


def _resolve_path(path: str) -> str:
    """Resolve an asset path: absolute paths are used as-is; relative paths
    are resolved relative to the example directory."""
    if os.path.isabs(path):
        return path
    return os.path.join(_EXAMPLE_DIR, path)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_surface_tris(builder, local_tris: np.ndarray, p_off: int) -> None:
    """Add local surface triangles to the builder with a global offset (zero stiffness)."""
    for a, b, c in local_tris:
        builder.add_triangle(int(a) + p_off, int(b) + p_off, int(c) + p_off, tri_ke=0.0, tri_ka=0.0, tri_kd=0.0)


# ---------------------------------------------------------------------------
# _SoftBodyExample — shared base for tet and hex modes
# ---------------------------------------------------------------------------


class _SoftBodyExample:
    """Shared simulate / step / gui base for tet and hex FEM modes.

    Subclasses must populate before calling any of these methods:
        self.viewer, self.solver, self.model
        self.state_0, self.state_1, self.control
        self.sim_substeps, self.sim_dt, self.frame_dt
        self.sim_time  (float, initialised to 0.0)
        self.graph     (wp.Graph | None)
    """

    def simulate(self) -> None:
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self) -> None:
        # Refresh element Hessian from the current deformed positions before the graph.
        # update_preconditioner=False when a CUDA graph is active: bsr_set_from_triplets
        # reuses the same A_bsr buffer pointer (graph reads fresh values), but
        # preconditioner() would allocate a new inv_diag and freeing the old one would
        # invalidate the pointer captured by the graph.
        self.solver.rebuild_element_blocks(
            self.state_0.particle_q,
            self.sim_dt,
            update_preconditioner=(self.graph is None),
        )

        # Rebuild surface-contact BVHs from the current state before the CUDA graph
        # launches.  mesh.refit() goes through the Warp C++ runtime and may not be
        # captured in a CUDA graph; an explicit host-side call here guarantees the BVH
        # topology is up-to-date every frame.
        sc = self.solver.soft_surface_contacts
        if sc is not None:
            sc.update(self.state_0.particle_q)

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self.sim_time += self.frame_dt

    def gui(self, ui) -> None:
        ui.text("Linear FEM: nonlinear elastic soft bodies")


# ---------------------------------------------------------------------------
# Example — public factory (defined before subclasses so they can inherit it)
# ---------------------------------------------------------------------------


class Example:
    """Select the showcase, tetrahedral, or hexahedral scene.

    ``Example(viewer, args)`` dispatches according to ``args.mode``.
    The bundled configuration selects the six-body showcase by default.
    """

    def __new__(cls, viewer, args):
        if cls is Example:
            mode = getattr(args, "mode", "tet")
            target = {
                "tet": _TetExample,
                "hex": _HexExample,
                "showcase": _ShowcaseExample,
            }[mode]
            return object.__new__(target)
        return object.__new__(cls)

    @staticmethod
    def create_parser():
        """Return an argument parser for the three scene modes."""
        parser = newton.examples.create_parser()

        # Mode
        parser.add_argument(
            "--mode",
            choices=["tet", "hex", "showcase"],
            default="tet",
            help="Simulation mode: tet FEM / hex FEM / showcase",
        )

        # Simulation timing
        parser.add_argument("--substeps", type=int, default=5)
        parser.add_argument("--fps", type=int, default=60)
        parser.add_argument("--gravity", type=float, default=9.81)

        # FEM material (tet and hex)
        parser.add_argument("--k-mu", type=float, default=1e5, dest="k_mu")
        parser.add_argument("--k-lambda", type=float, default=1e5, dest="k_lambda")
        parser.add_argument("--k-damp", type=float, default=1.0, dest="k_damp")
        parser.add_argument("--density", type=float, default=1.0)
        parser.add_argument("--particle-radius", type=float, default=0.008, dest="particle_radius")
        parser.add_argument("--linear-damping", type=float, default=0.0, dest="linear_damping")

        # Solver
        parser.add_argument("--solver-type", default="cg", dest="solver_type")
        parser.add_argument("--solver-maxiter", type=int, default=100, dest="solver_maxiter")
        parser.add_argument(
            "--preconditioner",
            default="diag",
            choices=["diag", "diag_abs", "block_diag", "id"],
            dest="preconditioner",
            help="CG preconditioner. 'diag' (default) is Warp point-Jacobi; "
            "'block_diag' inverts the full 3x3 per-particle block, which "
            "usually cuts the iteration count substantially.",
        )
        parser.add_argument(
            "--cg-check-every",
            type=int,
            default=0,
            dest="cg_check_every",
            help="0 (default) = fixed iteration count, CUDA-graph capturable. "
            ">0 = host-side residual checks with early exit; disables graph "
            "capture. Use to measure how many iterations the solve needs.",
        )

        # Ground
        parser.add_argument("--ground-ke", type=float, default=1e5, dest="ground_ke")
        parser.add_argument("--ground-kd", type=float, default=1e2, dest="ground_kd")
        parser.add_argument("--ground-kf", type=float, default=1e3, dest="ground_kf")
        parser.add_argument("--ground-mu", type=float, default=0.5, dest="ground_mu")

        # Self-contact (tet and hex)
        parser.add_argument("--contact-ke", type=float, default=1e3, dest="contact_ke")
        parser.add_argument("--contact-kd", type=float, default=1e2, dest="contact_kd")

        # Tet-mode specific
        parser.add_argument(
            "--soft-objects", default="[]", dest="soft_objects", help="JSON list of per-object configs (tet mode only)"
        )
        parser.add_argument("--debug-interval", type=int, default=0, dest="debug_interval")

        # Showcase-mode specific — no default; must be supplied via the JSON config
        parser.add_argument(
            "--objects",
            default=None,
            dest="objects",
            help="JSON list of per-object configs for showcase mode. "
            'Each entry: {"path": "...", "position": [x,y,z], "color": [r,g,b], '
            '"k-mu", "k-lambda", "k-damp", "density", "particle-radius"}. '
            "Set in examples/linear_fem.json.",
        )

        with open(os.path.join(_EXAMPLE_DIR, "linear_fem.json")) as stream:
            defaults = json.load(stream)["args"]
        parser.set_defaults(
            **{
                k.replace("-", "_"): json.dumps(v) if isinstance(v, list) and k == "objects" else v
                for k, v in defaults.items()
                if k not in {"viewer", "headless", "num-frames"}
            }
        )
        return parser

    def test_final(self) -> None:
        if (
            not np.isfinite(self.state_0.particle_q.numpy()).all()
            or not np.isfinite(self.state_0.particle_qd.numpy()).all()
        ):
            raise AssertionError("FEM produced nonfinite particle state")


# ---------------------------------------------------------------------------
# _TetExample — tet FEM, multi-object, CUDA graph
# ---------------------------------------------------------------------------


class _TetExample(_SoftBodyExample, Example):
    """Multi-object tet FEM demo (``--mode tet``)."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        gravity = float(getattr(args, "gravity", 9.81))
        self._debug_interval = max(0, int(getattr(args, "debug_interval", 0)))
        self._frame_idx = 0
        self._rest_particle_q_np: np.ndarray | None = None

        # ── Config (from CLI args) ────────────────────────────────────────
        self.sim_substeps = int(args.substeps)
        self.sim_dt = self.frame_dt / self.sim_substeps

        solver_type = str(args.solver_type)
        solver_maxiter = int(args.solver_maxiter)
        ground_ke = float(args.ground_ke)
        ground_kd = float(args.ground_kd)
        ground_kf = float(args.ground_kf)
        ground_mu = float(args.ground_mu)
        contact_ke = float(args.contact_ke)
        contact_kd = float(args.contact_kd)
        linear_damping = float(args.linear_damping)

        soft_objects_raw: list = json.loads(args.soft_objects) if args.soft_objects else []
        if not soft_objects_raw and args.objects:
            soft_objects_raw = [
                {**obj, "name": obj.get("name", os.path.splitext(os.path.basename(obj["path"]))[0])}
                for obj in json.loads(args.objects)
                if obj.get("active", True) and detect_asset_type(_resolve_path(obj["path"])) == "tet"
            ]
        active_items = [(obj["name"], obj) for obj in soft_objects_raw if obj.get("active", True)]
        if not active_items:
            raise ValueError('No soft objects have "active": true in --soft-objects. Pass at least one active object.')

        # ── Build model ───────────────────────────────────────────────────
        builder = FEMBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(ke=ground_ke, kd=ground_kd, kf=ground_kf, mu=ground_mu)
        )

        particle_ranges: list[tuple[int, int]] = []
        surface_triangles_list: list = []
        for name, obj in active_items:
            if "path" not in obj:
                raise ValueError(f"Object '{name}' in --soft-objects is missing a 'path' field")
            path = _resolve_path(obj["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"Asset not found: {path}. Install the bundled examples/assets files or set an existing path in --soft-objects."
                )
            asset = load_asset(path)
            verts = asset["vertices"]
            tets = asset["tets"]
            if "position" not in obj:
                raise ValueError(f"Object '{obj.get('name', path)}' is missing 'position'")
            pos = obj["position"]
            p_start = builder.particle_count
            builder.add_soft_mesh(
                pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                vertices=verts,
                indices=tets.flatten(),
                scale=1.0,
                density=float(obj.get("density", 1.0)),
                k_mu=float(obj.get("k-mu", 1.0e5)),
                k_lambda=float(obj.get("k-lambda", 1.0e5)),
                k_damp=float(obj.get("k-damp", 1.0)),
                particle_radius=float(obj.get("particle-radius", 0.008)),
            )
            particle_ranges.append((p_start, builder.particle_count - p_start))
            surface_triangles_list.append(asset.get("surface_triangles"))

        self.model = builder.finalize()
        self.model.set_gravity((0.0, 0.0, -float(gravity)))

        # Use density-value directly as per-particle mass [kg].
        # The builder computes particle_mass = density * tet_volume / 4 (micro-grams for
        # small tets), but ground_ke / contact_ke were tuned for mass ~ density_val.
        _mass_np = np.empty(self.model.particle_count, dtype=np.float32)
        for (p_start, p_count), (_, obj) in zip(particle_ranges, active_items, strict=False):
            _mass_np[p_start : p_start + p_count] = float(obj.get("density", 1.0))
        self.model.particle_mass = wp.array(_mass_np, dtype=wp.float32, device=self.model.device)

        self.solver = SolverFEM(
            model=self.model,
            dt=self.sim_dt,
            solver_type=solver_type,
            linear_solver_maxiter=solver_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
            self_contact_ke=contact_ke,
            self_contact_kd=contact_kd,
            linear_damping=linear_damping,
        )

        if len(particle_ranges) > 1:
            self.solver.set_soft_surface_contacts(
                particle_ranges=particle_ranges,
                surface_triangles_list=surface_triangles_list,
                ke=contact_ke,
                kd=contact_kd,
            )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "show_particles"):
            self.viewer.show_particles = True

        if self._debug_interval > 0:
            self._rest_particle_q_np = self.state_0.particle_q.numpy().copy()

        self.capture()

    def capture(self) -> None:
        """Warm up kernels, snapshot initial state, then capture a CUDA graph.

        Pressure updates are applied in :meth:`step` *before* the graph
        launch — not inside :meth:`simulate` — so the graph captures only
        the solver substep loop.

        When inter-body surface contacts are active, graph capture is skipped.
        ``mesh.refit()`` goes through the Warp C++ runtime and is not reliably
        captured; using a graph would freeze the BVH at frame-start positions
        across all substeps, causing objects to pass through each other.
        Eager mode calls ``solver.step()`` (and therefore ``sc.update()``) once
        per substep, so the BVH is always current.
        """
        self.graph = None
        if not wp.get_device().is_cuda:
            return
        if self.solver.soft_surface_contacts is not None:
            # Warm up kernels but do not capture — see docstring above.
            snap_0 = self.model.state()
            snap_0.assign(self.state_0)
            snap_1 = self.model.state()
            snap_1.assign(self.state_1)
            try:
                self.simulate()
                wp.synchronize_device()
            except Exception as exc:
                print(f"   [capture] warmup failed ({exc}).", flush=True)
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            return
        snap_0 = self.model.state()
        snap_0.assign(self.state_0)
        snap_1 = self.model.state()
        snap_1.assign(self.state_1)
        try:
            self.simulate()
            wp.synchronize_device()
        except Exception as exc:
            print(f"   [capture] warmup failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            return
        if self.sim_substeps % 2 == 1:
            self.state_0, self.state_1 = self.state_1, self.state_0
        self.state_0.assign(snap_0)
        self.state_1.assign(snap_1)
        try:
            with wp.ScopedCapture() as capture:
                self.simulate()
            self.graph = capture.graph
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
        except Exception as exc:
            print(f"   [capture] ScopedCapture failed ({exc}); running eager.", flush=True)
            self.state_0.assign(snap_0)
            self.state_1.assign(snap_1)
            self.graph = None

    def step(self) -> None:
        super().step()
        self._frame_idx += 1
        if self._debug_interval > 0 and self._frame_idx % self._debug_interval == 0:
            self._dump_debug_state()

    def _dump_debug_state(self) -> None:
        """Print one line of particle bbox / NaN / drift state."""
        try:
            q = self.state_0.particle_q.numpy()
            nan_n = int(np.isnan(q).any(axis=1).sum())
            parts = [f"[dbg t={self.sim_time:7.3f}s f={self._frame_idx:6d}]"]
            if nan_n:
                parts.append(f"NaN={nan_n}/{q.shape[0]}")
            else:
                qmin = q.min(axis=0)
                qmax = q.max(axis=0)
                bbox = qmax - qmin
                parts.append(
                    f"qmin=({qmin[0]:+.3f},{qmin[1]:+.3f},{qmin[2]:+.3f}) qmax=({qmax[0]:+.3f},{qmax[1]:+.3f},{qmax[2]:+.3f}) "
                    f"bbox=({bbox[0]:.3f},{bbox[1]:.3f},{bbox[2]:.3f})"
                )
                if self._rest_particle_q_np is not None:
                    drift = np.linalg.norm(q - self._rest_particle_q_np, axis=1)
                    parts.append(f"drift max={float(drift.max()):.4f} mean={float(drift.mean()):.4f}")
            print(" ".join(parts), flush=True)
        except Exception as exc:
            print(f"[linear FEM tet][dbg] dump failed: {exc}", flush=True)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-5.0, -5.0, -0.5)
        p_upper = wp.vec3(5.0, 5.0, 10.0)
        newton.examples.test_particle_state(
            self.state_0,
            "particles stay in workspace",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


# ---------------------------------------------------------------------------
# _HexExample — hex FEM, no CUDA graph
# ---------------------------------------------------------------------------


class _HexExample(_SoftBodyExample, Example):
    """Three Q1 hex FEM soft bodies (``--mode hex``)."""

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        k_mu = args.k_mu
        k_lambda = args.k_lambda
        k_damp = args.k_damp
        density = args.density
        ground_ke = args.ground_ke
        ground_kd = args.ground_kd
        ground_kf = args.ground_kf
        ground_mu = args.ground_mu
        contact_ke = args.contact_ke
        contact_kd = args.contact_kd
        cg_maxiter = args.solver_maxiter

        # mass floor: ensures kd_ground * dt / m < 2 for all nodes (safety factor 1.5)
        _m_floor = 10.0 * self.sim_dt * 1.5  # ≈ 0.05 kg at substeps=5

        # ── Load hex objects; type auto-detected from each USDA path ─────
        if not args.objects:
            raise ValueError("--objects is required for hex mode. Set it in examples/linear_fem.json.")
        all_objects = json.loads(args.objects)
        hex_cfgs, hex_objects = [], []
        for o in all_objects:
            if "path" not in o:
                raise ValueError(f"Object entry missing 'path': {o}")
            path = _resolve_path(o["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Asset not found: {path}")
            kind = o.get("type") or detect_asset_type(path)
            if kind != "hex":
                continue
            objs = load_hex_assembly(path)
            if not objs:
                raise RuntimeError(f"No hex mesh prim found in {path}")
            usd_obj = objs[0]
            usd_obj["_cfg"] = o
            hex_cfgs.append(o)
            hex_objects.append(usd_obj)
        if not hex_objects:
            raise ValueError("--objects contains no hex assets")

        # ── Assemble Newton model ─────────────────────────────────────────
        builder = FEMBuilder()
        builder.add_ground_plane()

        particle_ranges: list[tuple[int, int]] = []
        surface_triangles_list: list[np.ndarray] = []

        for obj in hex_objects:
            verts = obj["vertices"]  # (V,3) float32 — world-space
            hexes = obj["hex_indices"]  # (H,8) int32   — local
            local_tris = obj["surface_triangles"]  # (T,3) int32 — local

            H_obj = hexes.shape[0]
            n_verts = verts.shape[0]

            # Lumped mass: approximate element volume from node-0 / node-6
            # bounding box.  Overestimates ~2-3× for curved tire elements
            # (safe: more mass → more stable).
            node_mass = np.zeros(n_verts, dtype=np.float64)
            for e in range(H_obj):
                n0, n6 = int(hexes[e, 0]), int(hexes[e, 6])
                dx = abs(float(verts[n6, 0]) - float(verts[n0, 0]))
                dy = abs(float(verts[n6, 1]) - float(verts[n0, 1]))
                dz = abs(float(verts[n6, 2]) - float(verts[n0, 2]))
                em = density * max(dx * dy * dz, 1e-10)
                for a in range(8):
                    node_mass[int(hexes[e, a])] += em / 8.0

            p_off = builder.particle_count
            p_radius = float(obj["_cfg"].get("particle-radius", 0.008))
            for i in range(n_verts):
                pos = verts[i]
                builder.add_particle(
                    pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    mass=float(max(node_mass[i], _m_floor)),
                    radius=p_radius,
                )

            shifted_hexes = hexes + p_off
            builder.add_soft_hex_mesh(
                hex_indices=shifted_hexes,
                k_mu=k_mu,
                k_lambda=k_lambda,
                k_damp=k_damp,
            )
            _add_surface_tris(builder, local_tris, p_off)
            particle_ranges.append((p_off, n_verts))
            surface_triangles_list.append(local_tris)

        self.model = builder.finalize()

        # ── Create solver ─────────────────────────────────────────────────
        self.solver = SolverFEM(
            model=self.model,
            dt=self.sim_dt,
            preconditioner_type="diag",
            solver_type="cg",
            linear_solver_maxiter=cg_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=ground_ke,
            ground_kd=ground_kd,
            ground_kf=ground_kf,
            ground_mu=ground_mu,
            self_contact_ke=0.0,
            self_contact_kd=0.0,
        )

        self.solver.set_soft_surface_contacts(
            particle_ranges=particle_ranges,
            surface_triangles_list=surface_triangles_list,
            ke=contact_ke,
            kd=contact_kd,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.control = self.model.control()

        # Hex mode always runs eager — no CUDA graph.
        self.graph = None

        self.viewer.set_model(self.model)

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.log_contacts(self.model.contacts(), self.state_0)
        self.viewer.end_frame()

    def test_final(self) -> None:
        p_lower = wp.vec3(-2.0, -2.0, -0.5)
        p_upper = wp.vec3(2.0, 2.0, 15.0)
        newton.examples.test_particle_state(
            self.state_0,
            "hex-demo particles stay within bounds",
            lambda q, _qd: newton.math.vec_inside_limits(q, p_lower, p_upper),
        )


# ---------------------------------------------------------------------------
# _ShowcaseExample — configured Tet4 and Hex8 soft bodies
# ---------------------------------------------------------------------------


class _ShowcaseExample(Example):
    """Drop the active objects from the supplied scene configuration.

    The default row contains a Tet4 box, Hex8 cube, Tet4 sphere, Hex8 tire,
    Tet4 Baymax, and Hex8 pyramid.
    """

    def __init__(self, viewer, args):
        self.viewer = viewer
        self.sim_time = 0.0
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_substeps = args.substeps
        self.sim_dt = self.frame_dt / self.sim_substeps

        # Parse per-object config; filter by "active", auto-detect type
        if not args.objects:
            raise ValueError("--objects is required for showcase mode. Set it in examples/linear_fem.json.")
        all_objects = json.loads(args.objects)
        tet_objs, hex_objs = [], []
        for o in all_objects:
            if not o.get("active", True):
                continue
            if "path" not in o:
                raise ValueError(f"Object entry missing 'path': {o}")
            path = _resolve_path(o["path"])
            if not os.path.isfile(path):
                raise FileNotFoundError(f"Asset not found: {path}")
            kind = o.get("type") or detect_asset_type(path)
            name = o.get("name") or os.path.splitext(os.path.basename(path))[0]
            obj = {**o, "type": kind, "name": name, "_resolved_path": path}
            if kind == "tet":
                tet_objs.append(obj)
            elif kind == "hex":
                hex_objs.append(obj)
            else:
                raise ValueError(f"Unknown asset type '{kind}' for {path}")

        if not tet_objs and not hex_objs:
            raise ValueError('No active objects in --objects. Set at least one entry to "active": true')

        # ----- load tet assets -----
        tet_assets = [load_asset(o["_resolved_path"]) for o in tet_objs]

        # ----- load hex assets -----
        hex_objects = []
        for o in hex_objs:
            ho = load_hex_assembly(o["_resolved_path"])
            if not ho:
                raise RuntimeError(f"No hex mesh prim in {o['_resolved_path']}")
            hex_objects.append(ho[0])

        # ----- build unified model -----
        builder = FEMBuilder()
        builder.add_ground_plane(
            cfg=newton.ModelBuilder.ShapeConfig(
                ke=args.ground_ke,
                kd=args.ground_kd,
                kf=args.ground_kf,
                mu=args.ground_mu,
            )
        )

        # --- Tet objects at their configured positions ---
        tet_particle_ranges = []
        tet_surface_tris = []
        # (name, tri_start, tri_end, color) for per-object mesh coloring
        _mesh_color_ranges: list[tuple[str, int, int, tuple]] = []
        for asset, obj in zip(tet_assets, tet_objs, strict=False):
            p0 = builder.particle_count
            tri0 = builder.tri_count
            builder.add_soft_mesh(
                pos=wp.vec3(*[float(v) for v in obj["position"]]),
                rot=wp.quat_identity(),
                vel=wp.vec3(0.0, 0.0, 0.0),
                scale=1.0,
                vertices=asset["vertices"],
                indices=asset["tets"].reshape(-1).tolist(),
                density=float(obj.get("density", 1.0)),
                k_mu=float(obj.get("k-mu", 1e5)),
                k_lambda=float(obj.get("k-lambda", 1e5)),
                k_damp=float(obj.get("k-damp", 1.0)),
                particle_radius=float(obj.get("particle-radius", 0.008)),
            )
            p1 = builder.particle_count
            tet_particle_ranges.append((p0, p1 - p0))
            surf_tris = asset.get("surface_triangles")
            tet_surface_tris.append(surf_tris)
            # Add surface triangles to builder so they appear in model.tri_indices
            if surf_tris is not None:
                _add_surface_tris(builder, surf_tris, p0)
            c = obj.get("color", [0.7, 0.6, 0.4])
            _mesh_color_ranges.append(
                (
                    obj.get("name", f"tet_{len(tet_particle_ranges) - 1}"),
                    tri0,
                    builder.tri_count,
                    (float(c[0]), float(c[1]), float(c[2])),
                )
            )

        # --- Hex objects at their configured positions ---
        _m_floor = 10.0 * self.sim_dt * 1.5
        hex_particle_ranges = []
        hex_surface_tris = []

        hex_usda_by_name = {o["name"]: o for o in hex_objects}
        ordered_hex_usd = [hex_usda_by_name.get(o["name"], hex_objects[i]) for i, o in enumerate(hex_objs)]

        for i, (cfg, usd_obj) in enumerate(zip(hex_objs, ordered_hex_usd, strict=False)):
            if "position" not in cfg:
                raise ValueError(f"Hex object '{cfg.get('name', cfg['path'])}' is missing 'position'")
            pos = cfg["position"]
            verts = usd_obj["vertices"].copy()
            # Apply X and Y from position; Z: if specified use it as floor, else place on ground
            verts[:, 0] += float(pos[0]) - verts[:, 0].mean()
            verts[:, 1] += float(pos[1]) - verts[:, 1].mean()
            # z in position is the floor height (bottom of mesh). 0.0 = rest on ground.
            verts[:, 2] += float(pos[2]) - verts[:, 2].min()
            hexes = usd_obj["hex_indices"]
            local_tris = usd_obj["surface_triangles"]
            H_obj = hexes.shape[0]
            n_verts = verts.shape[0]
            obj_density = float(cfg.get("density", 1000.0))

            node_mass = np.zeros(n_verts, dtype=np.float64)
            for e in range(H_obj):
                n0, n6 = int(hexes[e, 0]), int(hexes[e, 6])
                dx = abs(float(verts[n6, 0]) - float(verts[n0, 0]))
                dy = abs(float(verts[n6, 1]) - float(verts[n0, 1]))
                dz = abs(float(verts[n6, 2]) - float(verts[n0, 2]))
                em = obj_density * max(dx * dy * dz, 1e-10)
                for a in range(8):
                    node_mass[int(hexes[e, a])] += em / 8.0

            p_off = builder.particle_count
            for vi in range(n_verts):
                pos = verts[vi]
                builder.add_particle(
                    pos=wp.vec3(float(pos[0]), float(pos[1]), float(pos[2])),
                    vel=wp.vec3(0.0, 0.0, 0.0),
                    mass=float(max(node_mass[vi], _m_floor)),
                    radius=float(cfg.get("particle-radius", 0.008)),
                )
            shifted_hexes = hexes + p_off
            builder.add_soft_hex_mesh(
                hex_indices=shifted_hexes,
                k_mu=float(cfg.get("k-mu", 5e5)),
                k_lambda=float(cfg.get("k-lambda", 5e5)),
                k_damp=float(cfg.get("k-damp", 0.01)),
            )
            _hex_tri0 = builder.tri_count
            # Flip winding so outward normals face the viewer (CCW convention).
            _add_surface_tris(builder, local_tris[:, [0, 2, 1]], p_off)
            hc = cfg.get("color", [0.5, 0.5, 0.5])
            _mesh_color_ranges.append(
                (cfg.get("name", f"hex_{i}"), _hex_tri0, builder.tri_count, (float(hc[0]), float(hc[1]), float(hc[2])))
            )
            hex_particle_ranges.append((p_off, n_verts))
            hex_surface_tris.append(local_tris)

        model = builder.finalize()
        model.set_gravity((0.0, 0.0, -float(getattr(args, "gravity", 9.81))))
        self.model = model

        # ----- Create solver -----
        self.solver = SolverFEM(
            model=model,
            dt=self.sim_dt,
            preconditioner_type=args.preconditioner,
            solver_type=args.solver_type,
            linear_solver_maxiter=args.solver_maxiter,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=args.ground_ke,
            ground_kd=args.ground_kd,
            ground_kf=args.ground_kf,
            ground_mu=args.ground_mu,
            self_contact_ke=args.contact_ke,
            self_contact_kd=args.contact_kd,
            linear_damping=args.linear_damping,
        )

        # Surface contacts for all active particle groups
        all_particle_ranges = tet_particle_ranges + hex_particle_ranges
        all_surface_tris = tet_surface_tris + hex_surface_tris
        # ke <= 0 means inter-body contact is off: skip the BVH build/refit and
        # the per-substep surface-contact kernel entirely rather than computing
        # forces that are identically zero (~5.7% of GPU time in the showcase).
        if all_particle_ranges and args.contact_ke > 0.0:
            self.solver.set_soft_surface_contacts(
                particle_ranges=all_particle_ranges,
                surface_triangles_list=all_surface_tris,
                ke=args.contact_ke,
                kd=args.contact_kd,
            )

        # ----- States -----
        self.state_0 = model.state()
        self.state_1 = model.state()
        self.control = model.control()

        # ----- Debug -----
        self._debug_interval = getattr(args, "debug_interval", 0)
        self._frame_idx = 0
        # Store per-object particle index ranges for per-object reporting
        self._particle_ranges_labeled = [
            (o.get("name", f"tet_{i}"), p0, p0 + n)
            for i, (o, (p0, n)) in enumerate(zip(tet_objs, tet_particle_ranges, strict=False))
        ] + [
            (o.get("name", f"hex_{i}"), p0, p0 + n)
            for i, (o, (p0, n)) in enumerate(zip(hex_objs, hex_particle_ranges, strict=False))
        ]

        # ----- Per-object render meshes -----
        # Build each object's triangle index buffer ONCE as an owned GPU array.
        # Slicing a per-frame temporary here would hand the viewer views that
        # dangle as soon as the temporary is freed (CUDA 700 on the next frame).
        self._render_meshes: list[tuple[str, wp.array, tuple]] = []
        if model.tri_count > 0 and model.tri_indices is not None:
            _tri_np = model.tri_indices.numpy().reshape(-1)
            for name, tri0, tri1, color in _mesh_color_ranges:
                if tri1 <= tri0:
                    continue
                self._render_meshes.append(
                    (
                        name,
                        wp.array(_tri_np[tri0 * 3 : tri1 * 3].copy(), dtype=wp.int32, device=model.device),
                        color,
                    )
                )
        # Suppress the viewer's default single-color unified triangle mesh;
        # render() draws per-object colored meshes instead.
        self.viewer.show_triangles = False
        # The GL viewer's spotlight is camera-anchored with a fixed 30/45 degree
        # cone, so objects at the ends of the showcase row fall outside it and
        # render with only the (very dim) ambient term, i.e. black.
        renderer = getattr(self.viewer, "renderer", None)
        if renderer is not None:
            renderer.spotlight_enabled = False

        self.solver.linear_solver_check_every = int(getattr(args, "cg_check_every", 0))

        self.graph = None
        self.viewer.set_model(self.model)
        self.capture()

    def capture(self) -> None:
        """Capture the FEM substep loop as a CUDA graph.

        Only ``simulate()`` is captured — ``_rebuild_element_blocks``
        (``bsr_set_from_triplets``) runs outside the graph each frame.
        """
        self.graph = None
        if not wp.get_device().is_cuda:
            return
        if self.solver.linear_solver_check_every > 0:
            print(
                "   [showcase] --cg-check-every > 0 — host-side residual checks are not capturable; running eager.",
                flush=True,
            )
            return
        try:
            with wp.ScopedCapture() as cap:
                self.simulate()
            self.graph = cap.graph
            print("   [showcase] CUDA graph captured.", flush=True)
        except Exception as exc:
            print(f"   [showcase] ScopedCapture failed: {exc}", flush=True)
            self.graph = None

    def simulate(self) -> None:
        """Advance the deformable bodies with alternating input and output states."""
        for _ in range(self.sim_substeps):
            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)

            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)

            self.state_0, self.state_1 = self.state_1, self.state_0

    def _dump_debug(self) -> None:
        """Print per-object particle bounding box, max velocity, and NaN count."""
        q = self.state_0.particle_q.numpy()
        qd = self.state_0.particle_qd.numpy()
        cg_n = self.solver.last_cg_iters
        cg_txt = f"  cg_iters={cg_n}/{self.solver.linear_solver_maxiter}" if cg_n is not None else ""
        print(f"\n[showcase frame {self._frame_idx}  t={self.sim_time:.3f}s]{cg_txt}")
        for name, p0, p1 in self._particle_ranges_labeled:
            pos = q[p0:p1]
            vel = qd[p0:p1]
            nan_pos = int(np.isnan(pos).any(axis=1).sum())
            nan_vel = int(np.isnan(vel).any(axis=1).sum())
            if nan_pos > 0 or nan_vel > 0:
                print(f"  {name:20s}  NaN pos={nan_pos}/{p1 - p0}  NaN vel={nan_vel}/{p1 - p0}  *** DIVERGED ***")
            else:
                lo = pos.min(axis=0)
                hi = pos.max(axis=0)
                vmax = float(np.linalg.norm(vel, axis=1).max()) if len(vel) else 0.0
                print(f"  {name:20s}  z=[{lo[2]:.3f},{hi[2]:.3f}]  vmax={vmax:.3f} m/s")

    def step(self) -> None:
        # Rebuild Hessian blocks + BSR assembly (must stay outside the graph —
        # bsr_set_from_triplets uses CUB temp alloc that can't be captured).
        # Skip preconditioner rebuild when graph is active: preconditioner()
        # allocates a new inv-diag buffer, which would invalidate the captured
        # graph's pointer.  The graph carries the preconditioner from capture.
        self.solver.rebuild_element_blocks(
            self.state_0.particle_q,
            self.sim_dt,
            update_preconditioner=(self.graph is None),
        )

        if self.graph is not None:
            wp.capture_launch(self.graph)
            if self.sim_substeps % 2 == 1:
                self.state_0, self.state_1 = self.state_1, self.state_0
        else:
            self.simulate()
        self.sim_time += self.frame_dt
        self._frame_idx += 1
        if self._debug_interval > 0 and self._frame_idx % self._debug_interval == 0:
            self._dump_debug()

    def gui(self, ui) -> None:
        ui.text("Linear FEM: tet and hex soft bodies")

    def render(self) -> None:
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        # Draw per-object colored meshes (show_triangles=False suppresses the default).
        for name, indices, color in self._render_meshes:
            self.viewer.log_mesh(
                f"/model/mesh/{name}",
                points=self.state_0.particle_q,
                indices=indices,
                color=color,
                backface_culling=False,
            )
        self.viewer.end_frame()

    def test_final(self) -> None:
        if (
            not np.isfinite(self.state_0.particle_q.numpy()).all()
            or not np.isfinite(self.state_0.particle_qd.numpy()).all()
        ):
            raise AssertionError("FEM produced nonfinite particle state")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)

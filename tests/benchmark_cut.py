# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Replay line incisions or a held sphere and measure complete simulation frames.

Run: python tests/benchmark_cut.py --frames 300 --output /tmp/fem_cut.json
"""

import argparse
import json
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples import example_linear_fem_baymax_cut as cutting

Example = cutting.Example


def _stroke(axis, coordinate, lower, upper):
    other = [a for a in range(3) if a != axis]
    quad = np.empty((4, 3))
    quad[:, axis] = coordinate
    quad[:, other] = np.asarray((lower, (upper[0], lower[1]), upper, (lower[0], upper[1])))
    return quad[[[0, 1, 2], [0, 2, 3]]]


def matrix_info(solver):
    """Report sparse storage and cut-update counters outside measured intervals."""
    matrix = solver.A_bsr
    entries = int(matrix.offsets.numpy()[-1])
    stream = getattr(solver, "_streaming", None)
    return {
        "candidate_blocks": entries,
        "value_capacity_blocks": matrix.values.size,
        "active_blocks": entries if stream is None else int(stream.active_counts.numpy().sum()),
        "streaming_update": None if stream is None else dict(stream.last_update),
    }


def _statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return None
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(values.max()),
        "over_16_67_ms": int(np.count_nonzero(values > 1000 / 60)),
    }


class _SphereReplay:
    """Run the actual boundary controller with a deterministic held tool."""

    def __init__(self, example, radius, speed, cuts):
        self.example, self.radius, self.speed, self.cuts = example, radius, speed, cuts
        self.frame = 0
        self.original_cut = example.cut_sphere
        self.graph = example.graph
        points = example.reference_positions + np.asarray((0.0, 0.0, example.args.drop_height))
        lower, upper = points.min(axis=0), points.max(axis=0)
        self.origin = np.asarray(
            (
                (lower[0] + upper[0]) / 2,
                lower[1] - max(upper[1] - lower[1], 2 * radius),
                lower[2] + 0.45 * (upper[2] - lower[2]),
            )
        )
        self.direction = np.asarray((0.0, 1.0, 0.0))
        # Warm the real visible-surface picker before timing the controller.
        if example._pick_cut_surface(self.origin, self.direction) is None:
            raise ValueError("The benchmark's forward belly ray missed the source mesh")
        example.cut_sphere = self._edit

    def _edit(self, start, end, radius):
        # The controller clock is synthetic, but geometry/publication timing
        # must retain real wall time inside the example's transaction.
        began = perf_counter()
        with patch.object(cutting, "perf_counter", perf_counter):
            with wp.ScopedTimer("spherical_cut", print=False, use_nvtx=False):
                accepted = self.original_cut(start, end, radius)
                wp.synchronize_device(self.example.device)
        elapsed = 1000 * (perf_counter() - began)
        self.cuts.append(
            {
                "frame": self.frame,
                "tool": "sphere",
                "accepted": accepted,
                "wall_ms": elapsed,
                "geometry_ms": self.example.last_geometry_ms if accepted else None,
                "rebuild_ms": self.example.last_rebuild_ms if accepted else None,
                "error": self.example.cut_error,
                "fragments": self.example.fragment_count,
                "surface_triangles": self.example._surface_count,
                "revision": self.example.cut_revision,
                "physics_graph_reused": self.example.graph is self.graph,
                "start": np.asarray(start).tolist(),
                "end": np.asarray(end).tolist(),
                "radius": radius,
            }
        )
        return accepted

    def advance(self, frame, *, release=False):
        self.frame = frame
        tool = self.example.screen_sphere
        if frame == 30:
            hit = self.example._pick_cut_surface(self.origin, self.direction)
            if hit is None:
                raise ValueError("The benchmark's forward belly ray missed the current body")
            tool.center = np.asarray(hit) - self.radius * self.direction
            tool.direction, tool.radius, tool.speed = self.direction.copy(), self.radius, self.speed
            tool.gesture_id += 1
            tool._pressed = True
        synthetic_time = frame * self.example.frame_dt
        with patch.object(cutting, "perf_counter", lambda: synthetic_time):
            self.example._advance_sphere()
        if release and tool.active:
            tool.on_mouse_release(0, 0, tool._left, 0)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--solver_iterations", type=int, default=30)
    parser.add_argument("--profile", action="store_true", help="Mark the measured loop for Nsight Systems capture.")
    parser.add_argument("--no_cuts", action="store_true")
    parser.add_argument("--legacy", action="store_true", help="Compare with full sparse reconstruction after cuts.")
    parser.add_argument("--validate_every_frame", action="store_true")
    parser.add_argument("--cutter", choices=("line", "sphere"), default="line")
    parser.add_argument("--sphere_radius", type=float, default=0.006, help="Sphere radius [m].")
    parser.add_argument("--sphere_speed", type=float, default=1.0, help="Sphere advance [diameters/s].")
    parser.add_argument("--output", type=Path, default=Path("/tmp/fem_cut.json"))
    args = parser.parse_args(argv)
    if args.frames < 1 or args.warmup < 0:
        parser.error("frames must be positive and warmup must be nonnegative")
    if (
        not np.isfinite([args.sphere_radius, args.sphere_speed]).all()
        or min(args.sphere_radius, args.sphere_speed) <= 0
    ):
        parser.error("sphere radius and speed must be positive and finite")
    wp.init()
    viewer = ViewerNull()
    options = Example.create_parser().parse_args(
        [
            "--substeps",
            str(args.substeps),
            "--solver_iterations",
            str(args.solver_iterations),
        ]
        + (["--no-streaming_topology"] if args.legacy else [])
        + (["--gravity", "0", "--cutter_mode", "sphere"] if args.cutter == "sphere" else [])
    )
    started = perf_counter()
    example = Example(viewer, options)
    wp.synchronize_device(example.device)
    setup_ms = (perf_counter() - started) * 1000
    initial_matrix = matrix_info(example.solver)
    times, cuts, samples, epochs = [], [], [], {}
    control_times, total_times = [], []
    initial_graph = example.graph
    replay = _SphereReplay(example, args.sphere_radius, args.sphere_speed, cuts) if args.cutter == "sphere" else None
    sequence = {
        30: _stroke(2, 0.3, (-0.12, -1.0), (0.12, 1.0)),
        90: _stroke(0, -0.3, (-1.0, -0.05), (1.0, 1.0)),
        150: _stroke(0, 0.3, (-1.0, -0.05), (1.0, 1.0)),
    }
    failure = ""
    if args.profile:
        wp.cuda_profiler_start()
    try:
        for frame in range(args.frames):
            frame_cut_start = len(cuts)
            control_ms = 0.0
            if replay is not None and not args.no_cuts:
                start = perf_counter()
                with wp.ScopedTimer("sphere_control", print=False, use_nvtx=False):
                    replay.advance(frame, release=frame == args.frames - 1)
                wp.synchronize_device(example.device)
                control_ms = 1000 * (perf_counter() - start)
                if example.cut_error:
                    raise RuntimeError(f"Spherical edit at frame {frame} failed: {example.cut_error}")
            if replay is None and not args.no_cuts and frame in sequence:
                example.test_final()
                previous_revision = example.cut_revision
                start = perf_counter()
                with wp.ScopedTimer("finite_cut", print=False, use_nvtx=False):
                    accepted = example.cut(sequence[frame])
                wp.synchronize_device(example.device)
                cuts.append(
                    {
                        "frame": frame,
                        "accepted": accepted,
                        "wall_ms": 1000 * (perf_counter() - start),
                        # Rejections/misses do not overwrite these example
                        # diagnostics: never attribute a previous cut's time.
                        "geometry_ms": example.last_geometry_ms if accepted else None,
                        "rebuild_ms": example.last_rebuild_ms if accepted else None,
                        "error": example.cut_error,
                        "fragments": example.fragment_count,
                        "revision": example.cut_revision,
                        "physics_graph_reused": example.graph is initial_graph,
                        "matrix": matrix_info(example.solver),
                    }
                )
                control_ms = cuts[-1]["wall_ms"]
                print(json.dumps({"cut": cuts[-1]}), flush=True)
                if not accepted or example.cut_revision != previous_revision + 1:
                    reason = example.cut_error or "stroke missed the current material or published no topology change"
                    raise RuntimeError(f"Intended cut at frame {frame} did not commit: {reason}")
            start = perf_counter()
            with wp.ScopedTimer("physics_frame", print=False, use_nvtx=False):
                if replay is None:
                    example.step()
                else:
                    # Input was already replayed above. This is the same
                    # physics portion of Example.step, without a second clock.
                    if example.graph is None:
                        example.simulate()
                    else:
                        wp.capture_launch(example.graph)
                    example.sim_time += example.frame_dt
            wp.synchronize_device(example.device)
            times.append(1000 * (perf_counter() - start))
            control_times.append(control_ms)
            total_times.append(control_ms + times[-1])
            epoch = epochs.setdefault(
                example.cut_revision,
                {
                    "revision": example.cut_revision,
                    "fragments": example.fragment_count,
                    "first_frame": frame,
                    "frame_indices": [],
                    "frame_times_ms": [],
                },
            )
            epoch["frame_indices"].append(frame)
            epoch["frame_times_ms"].append(times[-1])
            for event in cuts[frame_cut_start:]:
                event["cut_frame_ms"] = total_times[-1]
                if replay is not None:
                    print(json.dumps({"cut": event}), flush=True)
            if args.validate_every_frame:
                example.test_final()
            if frame % 30 == 0 or frame == args.frames - 1:
                q = example.state_0.particle_q.numpy()
                tets = q[example.model.tet_indices.numpy()[: example.fragment_count]]
                gradients = (tets[:, 1:] - tets[:, :1]).transpose(0, 2, 1) @ example.model.tet_poses.numpy()[
                    : example.fragment_count
                ]
                determinant = np.linalg.det(gradients)
                example.contact.update_surface(example.state_0.particle_q)
                skin = example.contact.points.numpy()[: 3 * example._surface_count]
                samples.append(
                    {
                        "frame": frame,
                        "min_J": float(determinant.min()) if len(determinant) else None,
                        "max_J": float(determinant.max()) if len(determinant) else None,
                        "min_z": float(skin[:, 2].min()) if len(skin) else None,
                        "max_speed": float(np.linalg.norm(example.state_0.particle_qd.numpy(), axis=1).max()),
                        "mass": float(example.model.cut_physical_mass.numpy().sum()),
                        "surface_triangles": example._surface_count,
                        "contact_rows": example.contact.active_rows.numpy().tolist(),
                    }
                )
                print(json.dumps({"sample": samples[-1]}), flush=True)
                example.test_final()
        example.test_final()
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
    finally:
        if args.profile:
            wp.cuda_profiler_stop()
        axis_hit = None if replay is None else example._pick_cut_surface(replay.origin, replay.direction)
        warm = np.asarray(times[args.warmup :] or times)
        for epoch in epochs.values():
            indices = np.asarray(epoch["frame_indices"])
            values = np.asarray(epoch["frame_times_ms"])
            measured = values[indices >= args.warmup]
            epoch["step_ms_after_warmup"] = (
                {
                    "count": len(measured),
                    "median": float(np.median(measured)),
                    "p95": float(np.percentile(measured, 95)),
                    "max": float(measured.max()),
                    "over_16_67_ms": int(np.count_nonzero(measured > 1000 / 60)),
                }
                if len(measured)
                else None
            )
        report = {
            "device": str(example.device),
            "setup_ms": setup_ms,
            "source_nodes": len(example.reference_positions),
            "source_tets": len(example.reference_elements),
            "substeps": args.substeps,
            "cg_iterations": args.solver_iterations,
            "streaming_topology": bool(options.streaming_topology),
            "cutter": args.cutter,
            "gravity": options.gravity,
            "sphere_radius": args.sphere_radius if replay is not None else None,
            "sphere_speed_diameters_per_second": args.sphere_speed if replay is not None else None,
            "sphere_start_frame": 30 if replay is not None else None,
            "sphere_axis_clear": axis_hit is None if replay is not None else None,
            "sphere_ray_origin": replay.origin.tolist() if replay is not None else None,
            "sphere_direction": replay.direction.tolist() if replay is not None else None,
            "physics_graph_captured": initial_graph is not None,
            "physics_graph_reused": example.graph is initial_graph,
            "initial_matrix": initial_matrix,
            "validated_every_frame": args.validate_every_frame,
            "frames_completed": len(times),
            "step_ms": (
                {
                    "median": float(np.median(warm)),
                    "p95": float(np.percentile(warm, 95)),
                    "max": float(warm.max()),
                }
                if len(warm)
                else None
            ),
            "topology_epochs": list(epochs.values()),
            "frame_ms": _statistics(total_times[args.warmup :] or total_times),
            "normal_frame_ms": _statistics(
                [
                    value
                    for frame, value in enumerate(total_times)
                    if frame >= args.warmup and not any(event["frame"] == frame for event in cuts)
                ]
            ),
            "cut_plus_physics_frame_ms": _statistics(
                [
                    value
                    for frame, value in enumerate(total_times)
                    if frame >= args.warmup and any(event["frame"] == frame for event in cuts)
                ]
            ),
            "controller_times_ms": control_times,
            "frame_times_ms": total_times,
            "cuts": cuts,
            "samples": samples,
            "failure": failure,
            "timing_scope": (
                "Synchronized headless physics. Complete frame times include controller/guide, edits and physics; "
                "rendering, reporting and validation downloads are excluded. Sphere motion uses deterministic "
                "1/60 s clock intervals; release publishes only the already displayed segment."
                if replay is not None
                else "Synchronized headless physics; cut wall times include geometry and publication into the existing physics graph."
            ),
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
        viewer.close()
    if failure:
        raise RuntimeError(failure)


if __name__ == "__main__":
    main()

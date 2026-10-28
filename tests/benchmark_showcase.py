# SPDX-License-Identifier: Apache-2.0
"""Measure warmed showcase physics, excluding setup, rendering and validation."""

import argparse
import json
from time import perf_counter

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples.example_linear_fem import Example


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--snapshot")
    args = parser.parse_args()
    with wp.ScopedDevice(args.device):
        settings = Example.create_parser().parse_args([])
        settings.debug_interval = 0
        scene = Example(ViewerNull(num_frames=args.frames), settings)
        for _ in range(args.warmup):
            scene.step()
        wp.synchronize_device()
        start = perf_counter()
        for _ in range(args.frames):
            scene.step()
        wp.synchronize_device()
        elapsed = perf_counter() - start
        q, qd = scene.state_0.particle_q.numpy(), scene.state_0.particle_qd.numpy()
        if not np.isfinite(q).all() or not np.isfinite(qd).all():
            raise RuntimeError("Nonfinite simulation state")
        report = {
            "device": str(wp.get_device()),
            "particles": scene.model.particle_count,
            "tetrahedra": scene.model.tet_count,
            "hexahedra": scene.solver.hex_count,
            "springs": scene.model.spring_count,
            "substeps": scene.sim_substeps,
            "iterations": scene.solver.linear_solver_maxiter,
            "warmup_frames": args.warmup,
            "measured_frames": args.frames,
            "wall_seconds": elapsed,
            "frame_ms": 1000 * elapsed / args.frames,
            "simulated_seconds": args.frames * scene.frame_dt,
            "realtime_factor": args.frames * scene.frame_dt / elapsed,
            "cuda_graph": scene.graph is not None,
            "max_velocity": float(np.linalg.norm(qd, axis=1).max()),
        }
        if args.snapshot:
            np.savez(args.snapshot, q=q, qd=qd, mass=scene.model.particle_mass.numpy())
        print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

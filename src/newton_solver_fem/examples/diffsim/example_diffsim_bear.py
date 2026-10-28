# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Optimize a full-resolution bear gait through sparse FEM, then simulate the learned controller.

The 1,986-vertex, 5,354-tet bear uses eight sinusoidal phase inputs,
per-tet tanh activations, seeded weights and Adam, 60 Hz frames and 80 substeps.
Training saves frame checkpoints and recomputes substeps during backward.

``--train-iters 1 --sim-steps 300`` trains for one full training trajectory;
``--inference --load-weights bear-controller.npz`` runs a saved controller.
The viewer always shows newly simulated inference frames, not training replay.
"""

from pathlib import Path

import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem.examples.diffsim._bear_training import FRAME_DT, BearProblem


class Example:
    def __init__(self, viewer, args):
        if args.train_iters < 0:
            raise ValueError("--train-iters must be nonnegative")
        self.viewer = viewer
        self.problem = BearProblem(args.sim_steps, use_graph=not args.no_graph, verbose=args.verbose)
        if args.load_weights:
            self.problem.load_weights(args.load_weights)
        self.viewer.set_model(self.problem.model)
        self.viewer.set_camera(pos=wp.vec3(25.0, -20.0, 10.0), pitch=-20.0, yaw=130.0)
        self.trained = not args.inference and args.train_iters > 0
        if self.trained:
            print(
                f"Training full bear: {self.problem.model.particle_count} particles, "
                f"{self.problem.model.tet_count} tets, {args.sim_steps} frames x 80 substeps; "
                f"{args.train_iters} Adam update(s).",
                flush=True,
            )
            self.problem.train(args.train_iters)
            if args.save_weights:
                self.problem.save_weights(args.save_weights)
                print(f"Saved controller: {Path(args.save_weights).expanduser().resolve()}", flush=True)
        elif args.save_weights:
            self.problem.save_weights(args.save_weights)
        self.state = self.problem.live_states[0]

    def step(self):
        self.state = self.problem.advance()

    def render(self):
        self.viewer.begin_frame(self.problem.live_frame * FRAME_DT)
        self.viewer.log_state(self.state)
        self.viewer.end_frame()

    def test_final(self):
        self.problem.solver.check_differentiation()
        positions = self.state.particle_q.numpy()
        velocities = self.state.particle_qd.numpy()
        assert np.isfinite(positions).all() and np.isfinite(velocities).all(), "Nonfinite live bear state"
        assert self.problem.model.particle_count == 1986 and self.problem.model.tet_count == 5354
        if self.trained:
            losses = self.problem.loss_history
            assert min(losses[1:]) < losses[0], f"Adam did not improve the locomotion objective: {losses}"

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--sim-steps", type=int, default=300, help="Frames per training trajectory (80 substeps each)."
        )
        parser.add_argument(
            "--train-iters", type=int, default=1, help="Adam updates before live inference (default: 1)."
        )
        parser.add_argument(
            "--inference", action="store_true", help="Skip optimization and simulate the controller live."
        )
        parser.add_argument("--load-weights", type=Path, help="Load a controller NPZ saved by this example.")
        parser.add_argument(
            "--save-weights", type=Path, help="Write controller weights to this user-selected NPZ path."
        )
        parser.add_argument("--no-graph", action="store_true", help="Disable CUDA frame graphs for debugging.")
        parser.add_argument("--verbose", action="store_true", help="Print training progress every 50 frames.")
        return parser


if __name__ == "__main__":
    viewer, args = newton.examples.init(Example.create_parser())
    newton.examples.run(Example(viewer, args), args)

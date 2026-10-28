# SPDX-FileCopyrightText: Copyright (c) 2025 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Original bear controller and bounded-memory FEM trajectory differentiation.

Adapted from Newton's ``examples/diffsim/example_diffsim_bear.py``. The full
reference mesh, material, edge springs, controller and velocity objective are
preserved. The package's linearly implicit sparse FEM replaces SemiImplicit;
its cached rest matrix and analytic ground-contact law define this rollout.
"""

import json
from pathlib import Path
from time import perf_counter

import newton
import numpy as np
import warp as wp
import warp.optim
from pxr import Usd, UsdGeom

from newton_solver_fem import FEMBuilder, SolverFEM

PHASE_COUNT = 8
PHASE_FREQUENCY = 5.0  # Angular frequency [rad/s], as in the reference.
ACTIVATION_STRENGTH = 0.3
FRAME_DT = 1.0 / 60.0
SUBSTEPS = 80
ASSET_PATH = Path(__file__).resolve().parents[1] / "assets" / "bear.usd"
ASSET_SHA256 = "a2b1dbe891e8ed0d8bb6cd12a3907b37b6abbab84f64a5c398e8ebf0ea31658d"
CONTROLLER_METADATA = {
    "mesh_sha256": ASSET_SHA256,
    "particle_count": 1986,
    "tet_count": 5354,
    "frame_dt": FRAME_DT,
    "substeps": SUBSTEPS,
    "phase_count": PHASE_COUNT,
    "phase_frequency_radians_per_second": PHASE_FREQUENCY,
    "activation_strength": ACTIVATION_STRENGTH,
    "density": 1.0,
    "lame_mu": 2000.0,
    "lame_lambda": 2000.0,
    "tet_damping": 2.0,
    "edge_spring_stiffness": 1000.0,
    "edge_spring_damping": 1.0,
    "ground": {"plane": [0, 0, 1, 0], "ke": 2000.0, "kd": 0.1, "kf": 10.0, "mu": 0.7},
    "solver": "sparse-fem-frozen-rest-matrix-cg-block-diag",
    "linear_solver_maxiter": 100,
    "stretch_clamping": False,
    "ground_friction": "weight-capped-coulomb",
}


@wp.kernel
def controller(weights: wp.array2d[float], time: wp.array[float], activations: wp.array[float]):
    tet = wp.tid()
    value = float(0.0)
    for phase in range(8):
        value += weights[tet, phase] * wp.sin(5.0 * time[0] + float(phase) * (2.0 * wp.pi / 8.0))
    activations[tet] = 0.3 * wp.tanh(value)


@wp.kernel
def mean_velocity(velocities: wp.array[wp.vec3], mean: wp.array[wp.vec3]):
    i = wp.tid()
    wp.atomic_add(mean, 0, velocities[i] / float(velocities.shape[0]))


@wp.kernel
def velocity_loss(mean: wp.array[wp.vec3], loss: wp.array[float]):
    velocity = mean[0]
    wp.atomic_add(loss, 0, wp.abs(velocity[1]) + wp.abs(velocity[2]) - velocity[0])


@wp.kernel
def add_gradient(source: wp.array2d[float], destination: wp.array2d[float]):
    i, j = wp.tid()
    destination[i, j] += source[i, j]


def build_bear_model(device=None):
    """Build all 1,986 original vertices and 5,354 original tetrahedra."""
    stage = Usd.Stage.Open(str(ASSET_PATH))
    if stage is None:
        raise FileNotFoundError(f"Bundled bear geometry is missing: {ASSET_PATH}")
    mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
    vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
    indices = np.asarray(mesh.GetPrim().GetAttribute("tetraIndices").Get(), dtype=np.int32)
    if vertices.shape != (1986, 3) or indices.size != 5354 * 4:
        raise ValueError("The bear example requires the unchanged full-resolution reference mesh")
    builder = FEMBuilder()
    builder.add_soft_mesh(
        pos=wp.vec3(0.0, 0.0, 0.5),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0, 0.0, 0.0),
        vertices=vertices,
        indices=indices,
        density=1.0,
        k_mu=2000.0,
        k_lambda=2000.0,
        k_damp=2.0,
        tri_ke=0.0,
        tri_ka=1.0e-8,
        tri_kd=0.0,
        tri_drag=0.0,
        tri_lift=0.0,
    )
    builder.add_ground_plane(cfg=newton.ModelBuilder.ShapeConfig(ke=2000.0, kf=10.0, kd=0.1, mu=0.7))
    # Only states and controller weights are differentiated, not model data.
    model = builder.finalize(device=device, requires_grad=False)
    model.soft_contact_ke, model.soft_contact_kf = 2000.0, 10.0
    model.soft_contact_kd, model.soft_contact_mu = 0.1, 0.7
    return model


class BearProblem:
    """Controller optimization through the production FEM solver.

    A forward pass saves only frame-boundary positions and velocities. Reverse
    mode recomputes one frame (80 steps) at a time, carries its final-state
    adjoint backwards, and sums the controller gradient. CUDA graphs reuse this
    one-frame workspace; CPU follows exactly the same checkpoint schedule.
    """

    def __init__(self, sim_steps=300, *, device=None, use_graph=True, verbose=False):
        if sim_steps <= 0:
            raise ValueError("sim_steps must be positive")
        self.sim_steps = int(sim_steps)
        self.verbose = verbose
        self.model = build_bear_model(device)
        self.device = self.model.device
        self.sim_dt = FRAME_DT / SUBSTEPS
        self.solver = SolverFEM(
            self.model,
            dt=self.sim_dt,
            differentiable=True,
            enable_tet_activations=True,
            defer_differentiation_checks=True,
            solver_type="cg",
            preconditioner_type="block_diag",
            linear_solver_maxiter=100,
            min_stretch=-1.0,
            max_stretch=-1.0,
            self_contact_ke=0.0,
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=2000.0,
            ground_kd=0.1,
            ground_kf=10.0,
            ground_mu=0.7,
        )
        self.solver.prepare_differentiation()
        rng = np.random.default_rng(42)
        padded_tets = (self.model.tet_count + 7) // 8 * 8
        weights = rng.uniform(-np.sqrt(1.0 / 8), np.sqrt(1.0 / 8), (padded_tets, PHASE_COUNT))
        self.weights = wp.array(weights, dtype=float, device=self.device, requires_grad=True)
        self.gradient = wp.zeros_like(self.weights, requires_grad=False)
        self.optimizer = warp.optim.Adam([self.weights.flatten()], lr=0.025)
        self.control = self.model.control(requires_grad=True)
        self.time = wp.zeros(1, dtype=float, device=self.device)
        self.live_states = [self.model.state(), self.model.state()]
        self.segment_states = [self.model.state(requires_grad=True) for _ in range(SUBSTEPS + 1)]
        shape = (self.sim_steps + 1) * self.model.particle_count
        self.checkpoint_q = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.checkpoint_v = wp.empty(shape, dtype=wp.vec3, device=self.device)
        self.carry_q = wp.zeros(self.model.particle_count, dtype=wp.vec3, device=self.device)
        self.carry_v = wp.zeros_like(self.carry_q)
        self.mean = wp.zeros(1, dtype=wp.vec3, device=self.device, requires_grad=True)
        self.loss = wp.zeros(1, dtype=float, device=self.device, requires_grad=True)
        self.frame_loss = wp.zeros_like(self.loss)
        self.forward_graph = None
        self.backward_graph = None
        self.tape = None
        self.timings = {}
        self.live_frame = 0
        self.loss_history = []
        self.use_graph = bool(use_graph and self.device.is_cuda)
        self.reset_live()

    def reset_live(self):
        """Reset live inference to the original reference pose and zero velocity."""
        for state in self.live_states:
            wp.copy(state.particle_q, self.model.particle_q)
            wp.copy(state.particle_qd, self.model.particle_qd)
        self.live_frame = 0

    def _network(self):
        wp.launch(
            controller,
            self.model.tet_count,
            inputs=[self.weights, self.time, self.control.tet_activations],
            device=self.device,
        )

    def _forward_frame(self):
        self._network()
        for _ in range(SUBSTEPS):
            self.live_states[0].clear_forces()
            self.solver.step(self.live_states[0], self.live_states[1], self.control, None, self.sim_dt)
            self.live_states.reverse()
        self.mean.zero_()
        wp.launch(
            mean_velocity,
            self.model.particle_count,
            inputs=[self.live_states[0].particle_qd, self.mean],
            device=self.device,
        )
        wp.launch(velocity_loss, 1, inputs=[self.mean, self.loss], device=self.device)

    def _backward_frame(self):
        self.mean.zero_()
        self.frame_loss.zero_()
        self.tape = wp.Tape()
        with self.tape:
            self._network()
            for step in range(SUBSTEPS):
                self.segment_states[step].clear_forces()
                self.solver.step(
                    self.segment_states[step], self.segment_states[step + 1], self.control, None, self.sim_dt
                )
            wp.launch(
                mean_velocity,
                self.model.particle_count,
                inputs=[self.segment_states[-1].particle_qd, self.mean],
                device=self.device,
            )
            wp.launch(velocity_loss, 1, inputs=[self.mean, self.frame_loss], device=self.device)
        self.tape.backward(
            self.frame_loss,
            grads={
                self.segment_states[-1].particle_q: self.carry_q,
                self.segment_states[-1].particle_qd: self.carry_v,
            },
        )
        wp.copy(self.carry_q, self.segment_states[0].particle_q.grad)
        wp.copy(self.carry_v, self.segment_states[0].particle_qd.grad)
        wp.launch(add_gradient, self.weights.shape, inputs=[self.weights.grad, self.gradient], device=self.device)
        self.tape.zero()

    def _save_frame(self, frame):
        n = self.model.particle_count
        wp.copy(self.checkpoint_q, self.live_states[0].particle_q, dest_offset=frame * n, count=n)
        wp.copy(self.checkpoint_v, self.live_states[0].particle_qd, dest_offset=frame * n, count=n)

    def _prepare_forward_graph(self):
        if self.use_graph and self.forward_graph is None:
            # Compile outside capture and restore the initial state afterwards.
            self._forward_frame()
            self.reset_live()
            self.loss.zero_()
            with wp.ScopedCapture(device=self.device) as capture:
                self._forward_frame()
            self.forward_graph = capture.graph

    def forward(self):
        """Evaluate the exact reference velocity objective and save checkpoints."""
        started = perf_counter()
        self.reset_live()
        self.loss.zero_()
        self._save_frame(0)
        self.time.zero_()
        self._prepare_forward_graph()
        for frame in range(self.sim_steps):
            self.time.assign(np.asarray([frame * FRAME_DT], dtype=np.float32))
            if self.forward_graph is not None:
                wp.capture_launch(self.forward_graph)
            else:
                self._forward_frame()
            self._save_frame(frame + 1)
            if self.verbose and (frame + 1) % 50 == 0:
                print(f"  forward {frame + 1}/{self.sim_steps} frames", flush=True)
        self.solver.check_differentiation()
        result = float(self.loss.numpy()[0])
        if not np.isfinite(result):
            raise FloatingPointError("Bear rollout produced a nonfinite loss")
        self.timings["forward_seconds"] = perf_counter() - started
        return result

    def backward(self):
        """Differentiate the last forward rollout using frame checkpoints."""
        started = perf_counter()
        self.carry_q.zero_()
        self.carry_v.zero_()
        self.gradient.zero_()
        n = self.model.particle_count
        for frame in reversed(range(self.sim_steps)):
            self.time.assign(np.asarray([frame * FRAME_DT], dtype=np.float32))
            wp.copy(self.segment_states[0].particle_q, self.checkpoint_q, src_offset=frame * n, count=n)
            wp.copy(self.segment_states[0].particle_qd, self.checkpoint_v, src_offset=frame * n, count=n)
            if self.use_graph and self.backward_graph is None:
                self._backward_frame()
                self.carry_q.zero_()
                self.carry_v.zero_()
                self.gradient.zero_()
                with wp.ScopedCapture(device=self.device) as capture:
                    self._backward_frame()
                self.backward_graph = capture.graph
            if self.backward_graph is not None:
                wp.capture_launch(self.backward_graph)
            else:
                self._backward_frame()
            try:
                self.solver.check_differentiation()
            except RuntimeError as error:
                raise RuntimeError(f"Bear backward recomputation failed at frame {frame}: {error}") from error
            if self.verbose and frame % 50 == 0:
                print(f"  backward {self.sim_steps - frame}/{self.sim_steps} frames", flush=True)
        self.solver.check_differentiation()
        wp.synchronize_device(self.device)
        self.timings["backward_seconds"] = perf_counter() - started
        return self.gradient

    def train(self, iterations):
        """Run actual Adam updates; return losses before and after each update."""
        if iterations < 0:
            raise ValueError("iterations must be nonnegative")
        self.loss_history = [self.forward()]
        for iteration in range(iterations):
            self.backward()
            gradient = self.gradient.numpy()
            if not np.isfinite(gradient).all():
                raise FloatingPointError("Bear controller gradient is nonfinite")
            self.optimizer.step([self.gradient.flatten()])
            self.loss_history.append(self.forward())
            print(
                f"Adam {iteration + 1}/{iterations}: loss {self.loss_history[-2]:.9g} -> "
                f"{self.loss_history[-1]:.9g}; |gradient|={np.linalg.norm(gradient):.6g}; "
                f"forward={self.timings['forward_seconds']:.3f}s, backward={self.timings['backward_seconds']:.3f}s",
                flush=True,
            )
        self.reset_live()
        return self.loss_history

    def advance(self):
        """Simulate one new 1/60-second live frame; never replay saved states."""
        self.time.assign(np.asarray([self.live_frame * FRAME_DT], dtype=np.float32))
        self._prepare_forward_graph()
        if self.forward_graph is not None:
            wp.capture_launch(self.forward_graph)
        else:
            self._forward_frame()
        self.solver.check_differentiation()
        self.live_frame += 1
        return self.live_states[0]

    def save_weights(self, path):
        """Save a portable controller; no device pointers or solver state."""
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as output:
            np.savez_compressed(
                output,
                format=np.asarray("newton-fem-bear-v1"),
                metadata=np.asarray(json.dumps(CONTROLLER_METADATA, sort_keys=True)),
                weights=self.weights.numpy(),
            )

    def load_weights(self, path):
        with np.load(Path(path).expanduser(), allow_pickle=False) as archive:
            if str(archive["format"]) != "newton-fem-bear-v1":
                raise ValueError("Unsupported bear controller format")
            if "metadata" not in archive or json.loads(str(archive["metadata"])) != CONTROLLER_METADATA:
                raise ValueError("Bear controller mesh, model or solver settings do not match this example")
            weights = np.asarray(archive["weights"], dtype=np.float32)
        if weights.shape != self.weights.shape or not np.isfinite(weights).all():
            raise ValueError(f"Expected finite bear controller weights with shape {self.weights.shape}")
        self.weights.assign(weights)
        self.optimizer.reset_internal_state()
        self.reset_live()

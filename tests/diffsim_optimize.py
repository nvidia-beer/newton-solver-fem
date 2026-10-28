# SPDX-License-Identifier: Apache-2.0
"""Fit a tip load from a deforming cantilever trajectory with Warp autodiff."""

import argparse
import json
from pathlib import Path

import numpy as np
import warp as wp

from newton_solver_fem import FEMBuilder, SolverLinearFEM, load_mesh

HEADLESS = True


@wp.kernel
def apply_load(amplitude: wp.array[float], weights: wp.array[float], force: wp.array[wp.vec3]):
    i = wp.tid()
    force[i] = wp.vec3(0.0, 0.0, amplitude[0] * weights[i])


@wp.kernel
def target_loss(q: wp.array[wp.vec3], target: wp.array[wp.vec3], loss: wp.array[float]):
    i = wp.tid()
    delta = q[i] - target[i]
    wp.atomic_add(loss, 0, 10000.0 * wp.dot(delta, delta) / float(q.shape[0]))


@wp.kernel
def project_supports(
    rest: wp.array[wp.vec3],
    inverse_mass: wp.array[float],
    raw_q: wp.array[wp.vec3],
    raw_v: wp.array[wp.vec3],
    q: wp.array[wp.vec3],
    v: wp.array[wp.vec3],
):
    """Apply the fixture's fixed end after each production FEM substep."""
    i = wp.tid()
    if inverse_mass[i] == 0.0:
        q[i] = rest[i]
        v[i] = wp.vec3(0.0)
    else:
        q[i] = raw_q[i]
        v[i] = raw_v[i]


def make_beam(device="cpu"):
    builder = FEMBuilder(gravity=(0.0, 0.0, 0.0))
    mesh = load_mesh(Path(__file__).parents[1] / "src/newton_solver_fem/examples/assets/tet_cantilever.usda")[0]
    builder.add_soft_mesh(
        pos=wp.vec3(0.0),
        rot=wp.quat_identity(),
        scale=1.0,
        vel=wp.vec3(0.0),
        vertices=mesh.vertices,
        indices=mesh.indices.flatten(),
        density=100.0,
        k_mu=500.0,
        k_lambda=1000.0,
        k_damp=0.1,
        add_surface_mesh_edges=False,
    )
    for i, point in enumerate(builder.particle_q):
        if abs(point[0]) < 1.0e-8:
            builder.particle_mass[i] = 0.0
    model = builder.finalize(device=device, requires_grad=False)
    tip = model.particle_q.numpy()[:, 0] > 0.199
    weights = wp.array(tip.astype(np.float32) / tip.sum(), dtype=float, device=device)
    return model, weights


def rollout(model, solver, weights, amplitude, steps=12):
    states = [model.state(requires_grad=amplitude.requires_grad) for _ in range(steps + 1)]
    control = model.control()
    for k in range(steps):
        force = wp.zeros(
            model.particle_count, dtype=wp.vec3, device=model.device, requires_grad=amplitude.requires_grad
        )
        wp.launch(
            apply_load,
            model.particle_count,
            inputs=[amplitude, weights],
            outputs=[force],
            device=model.device,
        )
        raw = model.state(requires_grad=amplitude.requires_grad)
        solver.step(states[k], raw, control, None, solver.dt, external_forces=force)
        wp.launch(
            project_supports,
            model.particle_count,
            inputs=[model.particle_q, model.particle_inv_mass, raw.particle_q, raw.particle_qd],
            outputs=[states[k + 1].particle_q, states[k + 1].particle_qd],
            device=model.device,
        )
    return states


def optimize_load(*, iterations=15, device="cpu"):
    if iterations < 1:
        raise ValueError("iterations must be positive")
    model, weights = make_beam(device)
    solver = SolverLinearFEM(
        model,
        dt=0.01,
        linear_solver_maxiter=64,
        solver_type="cg",
        preconditioner_type="block_diag",
        differentiable=True,
        min_stretch=-1.0,
        self_contact_ke=0.0,
    )
    reference = wp.array([-0.2], dtype=float, device=device, requires_grad=True)
    target = wp.clone(rollout(model, solver, weights, reference)[-1].particle_q, requires_grad=False)
    amplitude = wp.array([-0.06], dtype=float, device=device, requires_grad=True)
    history = []
    previous = None
    accepted_updates = 0
    for iteration in range(iterations + 1):
        loss = wp.zeros(1, device=device, requires_grad=True)
        with wp.Tape() as tape:
            states = rollout(model, solver, weights, amplitude)
            wp.launch(
                target_loss, model.particle_count, inputs=[states[-1].particle_q, target], outputs=[loss], device=device
            )
        value = float(loss.numpy()[0])
        history.append(value)
        if iteration == iterations or value < 1e-11:
            break
        tape.backward(loss)
        gradient = float(amplitude.grad.numpy()[0])
        current = float(amplitude.numpy()[0])
        if not np.isfinite(gradient):
            raise RuntimeError("The production FEM load gradient is nonfinite")
        # Scalar secant scaling uses only successive native gradients. A
        # backtracking forward solve verifies each proposed optimization step.
        rate = 0.08 / max(abs(gradient), 1e-12)
        if previous is not None:
            change = current - previous[0]
            curvature = (gradient - previous[1]) / change if abs(change) > 1e-10 else 0.0
            if curvature > 0.0:
                rate = 1.0 / curvature
        candidate = wp.empty(1, dtype=float, device=device)
        accepted = False
        for _ in range(12):
            candidate.assign([current - rate * gradient])
            trial = rollout(model, solver, weights, candidate)
            trial_loss = wp.zeros(1, device=device)
            wp.launch(
                target_loss,
                model.particle_count,
                inputs=[trial[-1].particle_q, target],
                outputs=[trial_loss],
                device=device,
            )
            if float(trial_loss.numpy()[0]) < value:
                previous = current, gradient
                amplitude.assign(candidate)
                accepted_updates += 1
                accepted = True
                break
            rate *= 0.5
        tape.zero()
        if not accepted:
            break
    return {
        "initial_loss": history[0],
        "final_loss": history[-1],
        "load_N": float(amplitude.numpy()[0]),
        "target_load_N": -0.2,
        "history": history,
        "accepted_updates": accepted_updates,
        "tip_displacement_m": (target.numpy() - model.particle_q.numpy())[weights.numpy() > 0].mean(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()
    result = optimize_load(iterations=args.iterations, device=args.device)
    print(json.dumps(result, indent=2))
    if not np.isfinite(result["final_loss"]) or result["final_loss"] >= result["initial_loss"]:
        raise RuntimeError("The force fit did not reduce its loss")


if __name__ == "__main__":
    main()

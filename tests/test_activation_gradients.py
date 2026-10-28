# SPDX-License-Identifier: Apache-2.0
"""Differentiate the production FEM tet controls and sinusoidal bear controller."""

import unittest

import numpy as np
import warp as wp
from test_solver import DEVICE, DT, cube, evaluate, make_solver, objective


@wp.kernel
def phase_network(weights: wp.array2d[float], time: float, activations: wp.array[float]):
    """The reference bear's eight phases and one tanh layer, without padding."""
    tet = wp.tid()
    total = float(0.0)
    for phase in range(8):
        total += weights[tet, phase] * wp.sin(5.0 * time + float(phase) * (2.0 * wp.pi / 8.0))
    activations[tet] = 0.3 * wp.tanh(total)


def scene(*, contact=False, clamp=False):
    model = cube(gravity=(0, 0, -9.81) if contact else (0, 0, 0))
    options = {"enable_tet_activations": True}
    if contact:
        options.update(ground_plane=(0, 0, 1, 0), ground_ke=10.0, ground_kd=0.02, ground_kf=0.01, ground_mu=0.5)
    if clamp:
        options.update(min_stretch=0.05, max_stretch=20.0)
    solver = make_solver(model, **options)
    q = model.particle_q.numpy()
    origin = q.min(axis=0)
    q = origin + (q - origin) * [1.03, 0.97, 1.01]
    if contact:
        q[:, 2] -= 0.298
    initial_velocity = (0.4, -0.3, -0.3) if contact else (0.04, -0.03, -0.04)
    velocity = np.tile(initial_velocity, (len(q), 1)).astype(np.float32)
    return model, solver, q.astype(np.float32), velocity


def network_rollout(model, solver, q, velocity, values, *, backward=False):
    states = [model.state(requires_grad=True) for _ in range(4)]
    states[0].particle_q.assign(q)
    states[0].particle_qd.assign(velocity)
    weights = wp.array(values, dtype=float, device=DEVICE, requires_grad=True)
    activations = [wp.empty(model.tet_count, dtype=float, device=DEVICE, requires_grad=True) for _ in range(3)]
    controls = [model.control() for _ in range(3)]
    loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
    with wp.Tape() as tape:
        for i, control in enumerate(controls):
            wp.launch(phase_network, model.tet_count, inputs=[weights, i * DT], outputs=[activations[i]], device=DEVICE)
            control.tet_activations = activations[i]
            solver.step(states[i], states[i + 1], control, None, DT)
        wp.launch(
            objective,
            model.particle_count,
            inputs=[states[-1].particle_q, states[-1].particle_qd],
            outputs=[loss],
            device=DEVICE,
        )
    if backward:
        tape.backward(loss)
    return float(loss.numpy()[0]), weights, states


class TestActivationGradients(unittest.TestCase):
    def assert_contact_branch(self, model, states, baseline, reference=None):
        signatures = []
        for index, state in enumerate(states):
            q, velocity = state.particle_q.numpy(), state.particle_qd.numpy()
            np.testing.assert_array_equal(q[:, 2] < model.particle_radius.numpy(), baseline)
            self.assertTrue(np.all(velocity[baseline, 2] < -0.01))
            self.assertTrue(np.all(np.linalg.norm(velocity[baseline, :2], axis=1) > 0.01))
            normal = 10.0 * (model.particle_radius.numpy() - q[:, 2]) - 0.02 * velocity[:, 2]
            weight = model.particle_mass.numpy() * 9.81
            speed = np.linalg.norm(velocity[:, :2], axis=1)
            signature = np.stack((normal > weight, 0.01 * speed > 0.5 * np.minimum(normal, weight)))[:, baseline]
            signatures.append(signature)
            if reference is not None:
                np.testing.assert_array_equal(signature, reference[index])
        return signatures

    def test_tet_activation_direction_and_component_derivatives(self):
        for contact in (False, True):
            with self.subTest(contact=contact):
                model, solver, q, velocity = scene(contact=contact)
                forces = np.zeros((3, *q.shape), dtype=np.float32)
                activation = np.random.default_rng(11).uniform(-0.02, 0.02, (3, model.tet_count)).astype(np.float32)
                _, states, _, controls, _, _ = evaluate(
                    model, solver, q, velocity, forces, activations=activation, backward=True
                )
                derivative = np.array([control.tet_activations.grad.numpy() for control in controls])
                self.assertTrue(np.isfinite(derivative).all())
                self.assertGreater(np.linalg.norm(derivative), 1e-5)
                direction = np.random.default_rng(12).normal(size=activation.shape).astype(np.float32)
                direction /= np.linalg.norm(direction)
                component = np.zeros_like(direction)
                component.flat[np.argmax(np.abs(derivative))] = 1.0
                branch = q[:, 2] < model.particle_radius.numpy()
                branches = None
                if contact:
                    branches = self.assert_contact_branch(model, states, branch)
                for delta in (direction, component):
                    exact = float(np.sum(derivative * delta))
                    for epsilon in (0.004, 0.002):
                        losses = []
                        for sign in (-1, 1):
                            result = evaluate(
                                model,
                                solver,
                                q,
                                velocity,
                                forces,
                                record=False,
                                activations=activation + sign * epsilon * delta,
                            )
                            losses.append(result[0])
                            if contact:
                                self.assert_contact_branch(model, result[1], branch, branches)
                        np.testing.assert_allclose(
                            exact, (losses[1] - losses[0]) / (2 * epsilon), rtol=0.015, atol=2e-5
                        )

    def test_network_weights_multistep_gradient_with_floor_contact(self):
        model, solver, q, velocity = scene(contact=True)
        values = (
            np.random.default_rng(42).uniform(-np.sqrt(1 / 8), np.sqrt(1 / 8), (model.tet_count, 8)).astype(np.float32)
        )
        _, weights, states = network_rollout(model, solver, q, velocity, values, backward=True)
        gradient = weights.grad.numpy().copy()
        self.assertTrue(np.isfinite(gradient).all())
        self.assertGreater(np.linalg.norm(gradient), 1e-5)
        branch = q[:, 2] < model.particle_radius.numpy()
        branches = self.assert_contact_branch(model, states, branch)
        direction = np.random.default_rng(6).normal(size=values.shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        component = np.zeros_like(direction)
        component.flat[np.argmax(np.abs(gradient))] = 1.0
        for delta in (direction, component):
            exact = float(np.sum(gradient * delta))
            for epsilon in (0.008, 0.004):
                losses = []
                for sign in (-1, 1):
                    value, _, trajectory = network_rollout(model, solver, q, velocity, values + sign * epsilon * delta)
                    losses.append(value)
                    self.assert_contact_branch(model, trajectory, branch, branches)
                np.testing.assert_allclose(exact, (losses[1] - losses[0]) / (2 * epsilon), rtol=0.02, atol=2e-5)

    def test_recorded_snapshots_survive_unrelated_forward_rollout(self):
        model, solver, q, velocity = scene()
        forces = np.zeros((3, *q.shape), dtype=np.float32)
        activation = np.full((3, model.tet_count), 0.03, dtype=np.float32)
        first = evaluate(model, solver, q, velocity, forces, activations=activation)
        # These steps overwrite every persistent force/Krylov buffer before the
        # first reverse pass. Retained snapshots must own all required primals.
        evaluate(model, solver, q + [0.01, 0.02, 0], -velocity, forces + 0.001, record=False, activations=-activation)
        first[4].backward(first[5])
        repeated = evaluate(model, solver, q, velocity, forces, activations=activation, backward=True)
        for original, later in zip(first[1], repeated[1], strict=True):
            np.testing.assert_allclose(
                original.particle_q.grad.numpy(), later.particle_q.grad.numpy(), rtol=2e-6, atol=5e-6
            )
            np.testing.assert_allclose(
                original.particle_qd.grad.numpy(), later.particle_qd.grad.numpy(), rtol=2e-6, atol=5e-6
            )
        for original, later in zip(first[3], repeated[3], strict=True):
            np.testing.assert_allclose(
                original.tet_activations.grad.numpy(), later.tet_activations.grad.numpy(), rtol=1e-5, atol=1e-7
            )

    def test_default_stretch_clamp_has_finite_state_derivative_in_smooth_region(self):
        model, solver, q, velocity = scene(clamp=True)
        force = np.zeros((2, *q.shape), dtype=np.float32)
        _, states, *_ = evaluate(model, solver, q, velocity, force, backward=True)
        gradient = states[0].particle_q.grad.numpy()
        self.assertTrue(np.isfinite(gradient).all())
        direction = np.random.default_rng(5).normal(size=q.shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        exact = float(np.sum(gradient * direction))
        for epsilon in (2e-4, 1e-4):
            plus = evaluate(model, solver, q + epsilon * direction, velocity, force, record=False)[0]
            minus = evaluate(model, solver, q - epsilon * direction, velocity, force, record=False)[0]
            np.testing.assert_allclose(exact, (plus - minus) / (2 * epsilon), rtol=0.015, atol=1e-4)

    def test_tet_hex_clamp_derivatives_at_repeated_rest_and_active_limits(self):
        for kind in ("tet", "hex"):
            for stretch in ((1.0, 1.0, 1.0), (0.03, 0.8, 1.2), (21.0, 1.1, 0.9)):
                with self.subTest(kind=kind, stretch=stretch):
                    model = cube(kind)
                    solver = make_solver(model, dt=1e-4, min_stretch=0.05, max_stretch=20.0)
                    q = model.particle_q.numpy()
                    origin = q.min(axis=0)
                    q = (origin + (q - origin) * stretch).astype(np.float32)
                    velocity = np.tile((0.03, -0.02, 0.04), (len(q), 1)).astype(np.float32)
                    forces = np.zeros((1, *q.shape), dtype=np.float32)
                    _, states, *_ = evaluate(model, solver, q, velocity, forces, backward=True)
                    direction = np.random.default_rng(23).normal(size=q.shape).astype(np.float32)
                    direction /= np.linalg.norm(direction)
                    for index, array in enumerate((states[0].particle_q, states[0].particle_qd)):
                        gradient = array.grad.numpy()
                        self.assertTrue(np.isfinite(gradient).all())
                        exact = float(np.sum(gradient * direction))
                        for epsilon in (4e-5, 2e-5) if index == 0 else (4e-3, 2e-3):
                            plus, minus = [q.copy(), velocity.copy()], [q.copy(), velocity.copy()]
                            plus[index] += epsilon * direction
                            minus[index] -= epsilon * direction
                            numeric = (
                                evaluate(model, solver, *plus, forces, record=False)[0]
                                - evaluate(model, solver, *minus, forces, record=False)[0]
                            ) / (2 * epsilon)
                            np.testing.assert_allclose(exact, numeric, rtol=0.02, atol=2e-4)

    def test_matrix_refresh_and_state_alias_are_rejected_during_recording(self):
        model, solver, q, velocity = scene()
        current, following = model.state(requires_grad=True), model.state(requires_grad=True)
        current.particle_q.assign(q)
        current.particle_qd.assign(velocity)
        with self.assertRaisesRegex((ValueError, RuntimeError), "tape|record|fixed|refresh"):
            with wp.Tape():
                solver.refresh_tangent(current.particle_q)
        with self.assertRaisesRegex((ValueError, RuntimeError), "distinct|alias|reus|unique"):
            with wp.Tape():
                solver.step(current, current, None, None, DT)
        with self.assertRaisesRegex((ValueError, RuntimeError), "distinct|alias|reus|unique"):
            with wp.Tape():
                solver.step(current, following, None, None, DT)
                solver.step(following, current, None, None, DT)

    def test_matrix_revision_change_invalidates_recorded_backward(self):
        model, solver, q, velocity = scene()
        forces = np.zeros((2, *q.shape), dtype=np.float32)
        result = evaluate(model, solver, q, velocity, forces)
        solver.refresh_tangent(model.particle_q)
        with self.assertRaisesRegex((ValueError, RuntimeError), "revision|changed|reassembl|invalid"):
            result[4].backward(result[5])


if __name__ == "__main__":
    unittest.main()

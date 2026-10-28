# SPDX-License-Identifier: Apache-2.0
"""Production FEM mechanics, sparse solves, and multistep state/force derivatives."""

import os
import unittest

import newton
import numpy as np
import warp as wp
from diffsim_optimize import optimize_load
from fem_fixture import ASSETS, add_mesh

from newton_solver_fem import FEMBuilder, SolverFEM, SolverLinearFEM, hex_quadrature, load_mesh

DEVICE = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
DT = 0.002


@wp.kernel
def objective(q: wp.array[wp.vec3], v: wp.array[wp.vec3], loss: wp.array[wp.float64]):
    i = wp.tid()
    weight = wp.vec3(float(i + 1) / float(q.shape[0]), 0.13, -0.41)
    position, velocity = wp.vec3d(q[i]), wp.vec3d(v[i])
    wp.atomic_add(loss, 0, wp.dot(position, wp.vec3d(weight)) + wp.float64(0.03) * wp.dot(velocity, velocity))


@wp.kernel
def velocity_sum(v: wp.array[wp.vec3], loss: wp.array[wp.float64]):
    wp.atomic_add(loss, 0, wp.float64(v[wp.tid()][0]))


def dense(matrix):
    """Independent small-test oracle; production storage remains BSR."""
    offsets, columns, blocks = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((3 * matrix.nrow, 3 * matrix.ncol))
    for row in range(matrix.nrow):
        for index in range(offsets[row], offsets[row + 1]):
            col = columns[index]
            result[3 * row : 3 * row + 3, 3 * col : 3 * col + 3] += blocks[index]
    return result


def cube(kind="tet", density=100.0, *, gravity=(0.0, 0.0, 0.0)):
    builder = FEMBuilder(gravity=gravity)
    mesh = load_mesh(ASSETS / f"{kind}_grid.usda")[0]
    vertices = mesh.vertices - mesh.vertices.min(axis=0)
    vertices *= 0.1 / np.max(np.ptp(vertices, axis=0))
    vertices += (0.0, 0.0, 0.3)
    add_mesh(builder, mesh, vertices=vertices, density=density, material=(500.0, 1000.0, 0.0))
    return builder.finalize(device=DEVICE, requires_grad=False)


def make_solver(model, *, differentiable=True, dt=DT, **options):
    args = dict(
        dt=dt,
        differentiable=differentiable,
        linear_solver_maxiter=128,
        solver_type="cg",
        preconditioner_type="block_diag",
        self_contact_ke=0.0,
        min_stretch=-1.0,
    )
    args.update(options)
    return SolverLinearFEM(model, **args)


def elastic_force(solver, state):
    model, control = solver.model, solver.model.control()
    return (
        solver.eval_tetrahedral_forces(model, control, state).numpy().copy()
        + solver.eval_hexahedral_forces(model, control, state).numpy().copy()
    )


def energy(model, q):
    """Independent smooth Stable Neo-Hookean energy in reference quadrature."""
    result = 0.0
    samples = []
    if model.tet_count:
        for ids, pose, material in zip(
            model.tet_indices.numpy(), model.tet_poses.numpy(), model.tet_materials.numpy(), strict=True
        ):
            gradient = np.vstack((-pose.sum(axis=0), pose)).astype(np.float64)
            samples.append((ids, gradient, 1.0 / (6.0 * np.linalg.det(pose.astype(float))), material))
    if model.fem.hex_count:
        ids = model.fem.hex_indices.numpy()
        _, gradients, volumes = hex_quadrature(model.particle_q.numpy().astype(float), ids)
        for nodes, gs, vs, material in zip(ids, gradients, volumes, model.fem.hex_materials.numpy(), strict=True):
            samples.extend((nodes, g, v, material) for g, v in zip(gs, vs, strict=True))
    for ids, gradients, volume, material in samples:
        deformation = q[ids].T @ gradients
        ic, jacobian = np.sum(deformation * deformation), np.linalg.det(deformation)
        mu, lame = material[:2]
        alpha = 1.0 + 0.75 * mu / lame
        result += volume * (0.5 * mu * (ic - 3.0) - 0.5 * mu * np.log(ic + 1.0) + 0.5 * lame * (jacobian - alpha) ** 2)
    return result


def evaluate(model, solver, q, v, forces, *, backward=False, record=True, activations=None):
    steps = len(forces)
    states = [model.state(requires_grad=True) for _ in range(steps + 1)]
    states[0].particle_q.assign(q)
    states[0].particle_qd.assign(v)
    inputs = [wp.array(force, dtype=wp.vec3, device=DEVICE, requires_grad=True) for force in forces]
    controls = []
    for i in range(steps):
        control = model.control()
        if activations is not None:
            control.tet_activations = wp.array(activations[i], dtype=float, device=DEVICE, requires_grad=True)
        controls.append(control)
    loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
    tape = wp.Tape()

    def forward():
        for i in range(steps):
            solver.step(states[i], states[i + 1], controls[i], None, solver.dt, external_forces=inputs[i])
        wp.launch(
            objective,
            model.particle_count,
            inputs=[states[-1].particle_q, states[-1].particle_qd],
            outputs=[loss],
            device=DEVICE,
        )

    if record:
        with tape:
            forward()
    else:
        forward()
    if backward:
        tape.backward(loss)
    return float(loss.numpy()[0]), states, inputs, controls, tape, loss


class TestFEM(unittest.TestCase):
    def test_public_solver_names_are_one_backend(self):
        self.assertIs(SolverFEM, SolverLinearFEM)

    def test_forward_is_independent_of_interleaved_large_adjoint_solves(self):
        model = cube()
        solver = make_solver(model)
        q = model.particle_q.numpy()
        q[0] += np.array((0.001, -0.002, 0.001), dtype=np.float32)
        velocity = np.tile((0.1, -0.04, 0.03), (model.particle_count, 1)).astype(np.float32)
        forces = [np.full_like(q, 0.03)]
        baseline = evaluate(model, solver, q, velocity, forces, record=False)[1][-1]
        expected_q = baseline.particle_q.numpy().copy()
        expected_v = baseline.particle_qd.numpy().copy()
        for seed in (1.0e4, 1.0e8):
            with self.subTest(adjoint_seed=seed):
                recorded = evaluate(model, solver, q, velocity, forces)
                recorded[4].backward(grads={recorded[5]: wp.array([seed], dtype=wp.float64, device=DEVICE)})
                repeated = evaluate(model, solver, q, velocity, forces, record=False)[1][-1]
                np.testing.assert_allclose(repeated.particle_q.numpy(), expected_q, rtol=0, atol=1e-7)
                np.testing.assert_allclose(repeated.particle_qd.numpy(), expected_v, rtol=5e-6, atol=1e-7)

    def test_rest_rigid_motion_and_hex_patch(self):
        rotation = np.array([[0.8, -0.6, 0.0], [0.6, 0.8, 0.0], [0.0, 0.0, 1.0]])
        for kind in ("tet", "hex"):
            with self.subTest(kind=kind):
                model = cube(kind)
                solver = make_solver(model, differentiable=False)
                state = model.state()
                state.particle_q.assign(model.particle_q.numpy() @ rotation.T + [0.1, 0.2, -0.15])
                np.testing.assert_allclose(elastic_force(solver, state), 0, atol=3e-4)
                matrix = dense(solver.A_bsr)
                np.testing.assert_allclose(matrix, matrix.T, atol=2e-8)
                self.assertGreater(np.linalg.eigvalsh(matrix).min(), 0)
                self.assertAlmostEqual(solver.compute_volume(state), 0.1**3, places=8)

    def test_force_is_energy_gradient(self):
        rng = np.random.default_rng(7)
        for kind in ("tet", "hex"):
            model = cube(kind)
            solver = make_solver(model, differentiable=False)
            state = model.state()
            q = model.particle_q.numpy().astype(float)
            q[:, 0] *= 1.08
            q[:, 1] += 0.04 * q[:, 2]
            state.particle_q.assign(q)
            force = elastic_force(solver, state)
            direction = rng.normal(size=q.shape)
            direction /= np.linalg.norm(direction)
            for epsilon in (1e-4, 5e-5):
                difference = (energy(model, q + epsilon * direction) - energy(model, q - epsilon * direction)) / (
                    2 * epsilon
                )
                np.testing.assert_allclose(float(-(force * direction).sum()), difference, rtol=0.001, atol=3e-5)
            np.testing.assert_allclose(force.sum(axis=0), 0, atol=2e-5)

    def test_step_matches_dense_solve_and_force_input_is_preserved(self):
        model = cube()
        solver = make_solver(model)
        initial, output = model.state(), model.state()
        q = initial.particle_q.numpy()
        q[-1, 2] -= 0.001
        initial.particle_q.assign(q)
        velocity = np.random.default_rng(2).normal(0, 0.03, q.shape).astype(np.float32)
        initial.particle_qd.assign(velocity)
        force = np.zeros_like(q)
        force[-1] = (0.1, 0.05, -0.2)
        external = wp.array(force, dtype=wp.vec3, device=DEVICE)
        mass = np.repeat(model.particle_mass.numpy(), 3)
        matrix = dense(solver.A_bsr)
        rhs = (
            DT * (force + elastic_force(solver, initial)).flatten()
            - matrix @ velocity.flatten()
            + mass * velocity.flatten()
        )
        increment = np.linalg.solve(matrix, rhs).reshape(q.shape)
        solver.step(initial, output, None, None, DT, external_forces=external)
        np.testing.assert_allclose(output.particle_qd.numpy(), velocity + increment, rtol=3e-4, atol=3e-6)
        np.testing.assert_allclose(output.particle_q.numpy(), q + DT * (velocity + increment), rtol=1e-6, atol=1e-7)
        np.testing.assert_array_equal(initial.particle_q.numpy(), q)
        np.testing.assert_array_equal(initial.particle_qd.numpy(), velocity)
        np.testing.assert_array_equal(external.numpy(), force)

    def test_multistep_state_force_gradients_and_reuse(self):
        for kind in ("tet", "hex"):
            with self.subTest(kind=kind):
                model = cube(kind)
                solver = make_solver(model)
                q = model.particle_q.numpy()
                q[:, 0] *= 1.03
                velocity = np.tile((0.02, -0.01, 0.03), (len(q), 1)).astype(np.float32)
                forces = np.zeros((3, *q.shape), dtype=np.float32)
                forces[:, -1, 2] = -0.02
                direction = np.random.default_rng(1).normal(size=q.shape).astype(np.float32)
                direction /= np.linalg.norm(direction)
                base, states, force_arrays, _, _, _ = evaluate(model, solver, q, velocity, forces, backward=True)
                exact = [
                    float(np.sum(states[0].particle_q.grad.numpy() * direction)),
                    float(np.sum(states[0].particle_qd.grad.numpy() * direction)),
                    float(sum(np.sum(f.grad.numpy() * direction) for f in force_arrays)),
                ]
                for index, epsilons in enumerate(((2e-4, 1e-4), (2e-3, 1e-3), (2e-3, 1e-3))):
                    for epsilon in epsilons:
                        plus, minus = (
                            [q.copy(), velocity.copy(), forces.copy()],
                            [q.copy(), velocity.copy(), forces.copy()],
                        )
                        plus[index] += epsilon * direction
                        minus[index] -= epsilon * direction
                        numeric = (
                            evaluate(model, solver, *plus, record=False)[0]
                            - evaluate(model, solver, *minus, record=False)[0]
                        ) / (2 * epsilon)
                        np.testing.assert_allclose(
                            exact[index],
                            numeric,
                            rtol=0.015,
                            atol=1e-4,
                            err_msg=f"{kind} input={index} epsilon={epsilon}",
                        )
                repeated, second, _, _, _, _ = evaluate(model, solver, q, velocity, forces, backward=True)
                self.assertAlmostEqual(base, repeated, places=6)
                np.testing.assert_allclose(
                    second[0].particle_q.grad.numpy(), states[0].particle_q.grad.numpy(), rtol=2e-6, atol=1e-6
                )

    def test_freefall_across_mass_scales_and_zero_rhs(self):
        for density in (0.1, 1000.0):
            model = cube(density=density)
            solver = make_solver(model)
            initial, output = model.state(), model.state()
            solver.step(initial, output, None, None, DT)
            np.testing.assert_allclose(output.particle_q.numpy(), initial.particle_q.numpy(), atol=2e-7)
            model.set_gravity((0, 0, -9.81))
            solver.step(initial, output, None, None, DT)
            np.testing.assert_allclose(output.particle_qd.numpy()[:, 2], -DT * 9.81, atol=2e-5)
            self.assertTrue(np.isfinite(output.particle_q.numpy()).all())

    def test_zero_rhs_has_nonzero_force_derivative(self):
        builder = newton.ModelBuilder(gravity=(0, 0, 0))
        builder.add_particle((0, 0, 1), (0, 0, 0), mass=2.0)
        model = builder.finalize(device=DEVICE, requires_grad=False)
        solver = make_solver(model)
        initial, output = model.state(requires_grad=True), model.state(requires_grad=True)
        external = wp.zeros(1, dtype=wp.vec3, device=DEVICE, requires_grad=True)
        loss = wp.zeros(1, dtype=wp.float64, device=DEVICE, requires_grad=True)
        with wp.Tape() as tape:
            solver.step(initial, output, None, None, DT, external_forces=external)
            wp.launch(velocity_sum, 1, inputs=[output.particle_qd], outputs=[loss], device=DEVICE)
        tape.backward(loss)
        np.testing.assert_allclose(external.grad.numpy(), [[DT / 2, 0, 0]], atol=1e-8)

    def test_floor_force_and_fixed_contact_state_velocity_derivatives(self):
        model = cube(gravity=(0, 0, -9.81))
        solver = make_solver(
            model, ground_plane=(0, 0, 1, 0), ground_ke=10.0, ground_kd=0.02, ground_kf=0.01, ground_mu=0.5
        )
        q = model.particle_q.numpy()
        q[:, 2] -= 0.298
        velocity = np.tile((0.04, -0.03, -0.04), (len(q), 1)).astype(np.float32)
        state = model.state()
        state.particle_q.assign(q)
        state.particle_qd.assign(velocity)
        force = solver.eval_particle_ground_contact_forces(model, model.control(), state).numpy()
        penetration = np.minimum(q[:, 2] - model.particle_radius.numpy(), 0)
        expected_z = np.where(penetration < 0, -10.0 * penetration + 0.02 * 0.04, 0)
        np.testing.assert_allclose(force[:, 2], expected_z, atol=1e-7)
        forces = np.zeros((3, *q.shape), dtype=np.float32)
        _, states, _, _, _, _ = evaluate(model, solver, q, velocity, forces, backward=True)
        active = q[:, 2] < model.particle_radius.numpy()
        for state in states:
            np.testing.assert_array_equal(state.particle_q.numpy()[:, 2] < model.particle_radius.numpy(), active)
            self.assertTrue(np.all(state.particle_qd.numpy()[active, 2] < -0.01))
        direction = np.random.default_rng(19).normal(size=q.shape).astype(np.float32)
        direction /= np.linalg.norm(direction)
        for index, array in enumerate((states[0].particle_q, states[0].particle_qd)):
            exact = float(np.sum(array.grad.numpy() * direction))
            for epsilon in (1e-4, 5e-5) if index == 0 else (2e-3, 1e-3):
                plus, minus = [q.copy(), velocity.copy()], [q.copy(), velocity.copy()]
                plus[index] += epsilon * direction
                minus[index] -= epsilon * direction
                values = []
                for inputs in (plus, minus):
                    value, perturbed, *_ = evaluate(model, solver, *inputs, forces, record=False)
                    for state in perturbed:
                        np.testing.assert_array_equal(
                            state.particle_q.numpy()[:, 2] < model.particle_radius.numpy(), active
                        )
                    values.append(value)
                np.testing.assert_allclose(exact, (values[0] - values[1]) / (2 * epsilon), rtol=0.015, atol=1e-4)

    def test_hex_quadrature_and_material_storage(self):
        model = cube("hex")
        solver = make_solver(model)
        self.assertEqual(solver.hex_count, 8)
        self.assertEqual(model.fem.hex_materials.shape, (8, 3))
        np.testing.assert_allclose(model.fem.hex_materials.numpy(), np.tile((500, 1000, 0), (8, 1)))
        self.assertAlmostEqual(float(model.fem.hex_det_J0_w.numpy().sum()), 0.1**3, places=8)
        self.assertAlmostEqual(float(model.particle_mass.numpy().sum()), 100 * 0.1**3, places=6)

    def test_checked_and_fast_paths_use_same_physics_and_sparse_matrix(self):
        model = cube()
        checked = make_solver(model, defer_differentiation_checks=True)
        fast = make_solver(model, differentiable=False)
        q = model.particle_q.numpy()
        q[:, 0] *= 1.02
        velocity = np.tile((0.02, -0.03, 0.01), (len(q), 1)).astype(np.float32)
        forces = np.zeros((3, *q.shape), dtype=np.float32)
        forces[:, -1, 2] = -0.01
        _, taped, *_ = evaluate(model, checked, q, velocity, forces, backward=True)
        _, eager, *_ = evaluate(model, checked, q, velocity, forces, record=False)
        _, ordinary, *_ = evaluate(model, fast, q, velocity, forces, record=False)
        for state_a, state_b, state_c in zip(taped, eager, ordinary, strict=True):
            np.testing.assert_allclose(state_a.particle_q.numpy(), state_b.particle_q.numpy(), atol=1e-7)
            np.testing.assert_allclose(state_a.particle_qd.numpy(), state_b.particle_qd.numpy(), atol=2e-6)
            np.testing.assert_allclose(state_a.particle_q.numpy(), state_c.particle_q.numpy(), atol=2e-6)
            # The production fast solve stops at its original 1% residual;
            # compare whole-state error, including velocities near zero.
            checked_v, fast_v = state_a.particle_qd.numpy(), state_c.particle_qd.numpy()
            self.assertLess(np.linalg.norm(checked_v - fast_v) / np.linalg.norm(checked_v), 0.01)
        matrix = dense(fast.A_bsr)
        for state_in, state_out in zip(ordinary[:-1], ordinary[1:], strict=True):
            delta = (state_out.particle_qd.numpy() - state_in.particle_qd.numpy()).flatten()
            rhs = state_in.particle_f.numpy().flatten()
            self.assertLess(np.linalg.norm(matrix @ delta - rhs) / np.linalg.norm(rhs), 0.0101)
        residuals = checked.check_differentiation()
        self.assertLess(residuals["forward_relative_residual"], 2e-5)
        self.assertLess(residuals["backward_relative_residual"], 2e-5)
        np.testing.assert_array_equal(checked.A_bsr.offsets.numpy(), fast.A_bsr.offsets.numpy())
        active = int(checked.A_bsr.offsets.numpy()[-1])
        np.testing.assert_array_equal(checked.A_bsr.columns.numpy()[:active], fast.A_bsr.columns.numpy()[:active])
        np.testing.assert_allclose(checked.A_bsr.values.numpy()[:active], fast.A_bsr.values.numpy()[:active], atol=0)
        self.assertLess(active, model.particle_count**2)
        # Triplet reserve includes duplicated element blocks and boundary
        # triangles before sparse summation; on tiny meshes it can exceed n².
        # Its bound follows mesh connectivity, rather than all node pairs.
        reserve = model.particle_count + 16 * model.tet_count + 64 * model.fem.hex_count
        reserve += 3 * model.tri_count + 2 * model.spring_count
        self.assertLessEqual(checked.A_bsr.values.size, reserve)
        for value in vars(checked).values():
            if isinstance(value, wp.array):
                self.assertNotEqual(value.shape, (3 * model.particle_count, 3 * model.particle_count))

    def test_load_optimization(self):
        result = optimize_load(iterations=15, device=DEVICE)
        self.assertGreater(result["accepted_updates"], 0)
        self.assertLess(result["final_loss"], 0.01 * result["initial_loss"])
        self.assertAlmostEqual(result["load_N"], result["target_load_N"], delta=0.01)
        self.assertGreater(abs(result["tip_displacement_m"][2]), 1e-5)


if __name__ == "__main__":
    unittest.main()

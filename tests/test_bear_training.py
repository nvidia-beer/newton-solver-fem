# SPDX-License-Identifier: Apache-2.0
"""Full bear model provenance and checkpointed-versus-full-tape differentiation."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom
from test_solver import DEVICE, cube

from newton_solver_fem.examples.diffsim import _bear_training as bear


class TestBearTraining(unittest.TestCase):
    def test_controller_persistence_and_compatibility_checks(self):
        with patch.object(bear, "build_bear_model", return_value=cube()):
            problem = bear.BearProblem(sim_steps=2, device=DEVICE, use_graph=False)
        original = problem.weights.numpy().copy()
        with tempfile.TemporaryDirectory(prefix="fem-bear-controller-") as directory:
            path = Path(directory) / "controller.npz"
            problem.save_weights(path)
            problem.weights.assign(original + 0.1)
            problem.live_frame = 7
            problem.load_weights(path)
            np.testing.assert_array_equal(problem.weights.numpy(), original)
            self.assertEqual(problem.live_frame, 0)
            with np.load(path, allow_pickle=False) as archive:
                saved = {name: archive[name].copy() for name in archive.files}
            for field, value in (("mesh_sha256", "wrong-mesh"), ("substeps", 79)):
                with self.subTest(metadata=field):
                    metadata = json.loads(str(saved["metadata"]))
                    metadata[field] = value
                    malformed = dict(saved, metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
                    np.savez_compressed(path, **malformed)
                    with self.assertRaisesRegex(ValueError, "mesh, model or solver settings"):
                        problem.load_weights(path)
            np.savez_compressed(path, **dict(saved, weights=original[:-1]))
            with self.assertRaisesRegex(ValueError, "shape"):
                problem.load_weights(path)
            malformed_weights = original.copy()
            malformed_weights[0, 0] = np.nan
            np.savez_compressed(path, **dict(saved, weights=malformed_weights))
            with self.assertRaisesRegex(ValueError, "finite"):
                problem.load_weights(path)

    def test_full_original_mesh_materials_and_edge_springs(self):
        self.assertEqual(hashlib.sha256(bear.ASSET_PATH.read_bytes()).hexdigest(), bear.ASSET_SHA256)
        stage = Usd.Stage.Open(str(bear.ASSET_PATH))
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
        vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float32)
        indices = np.asarray(mesh.GetPrim().GetAttribute("tetraIndices").Get(), dtype=np.int32).reshape(-1, 4)
        model = bear.build_bear_model(device=DEVICE)
        self.assertEqual((model.particle_count, model.tet_count), (1986, 5354))
        np.testing.assert_array_equal(model.particle_q.numpy(), vertices + np.asarray((0, 0, 0.5), dtype=np.float32))
        np.testing.assert_array_equal(model.tet_indices.numpy(), indices)
        np.testing.assert_array_equal(model.tet_materials.numpy(), np.tile((2000, 2000, 2), (5354, 1)))
        unique_edges = {
            tuple(sorted((int(nodes[i]), int(nodes[j]))))
            for nodes in indices
            for i in range(4)
            for j in range(i + 1, 4)
        }
        self.assertEqual(model.spring_count, len(unique_edges))
        np.testing.assert_array_equal(model.spring_stiffness.numpy(), np.full(model.spring_count, 1000))
        np.testing.assert_array_equal(model.spring_damping.numpy(), np.full(model.spring_count, 1))
        edges = vertices.astype(float)[indices[:, 1:]] - vertices.astype(float)[indices[:, :1]]
        volume = np.linalg.det(edges).sum() / 6.0
        np.testing.assert_allclose(model.particle_mass.numpy().sum(), volume, rtol=2e-6)
        self.assertFalse(model.tet_materials.requires_grad)

    def test_checkpointed_controller_gradient_matches_full_tape(self):
        # Preserve the exact two-frame/80-substep schedule, controller and
        # objective. A small solid isolates checkpoint correctness from the
        # cost of the separate full-resolution benchmark.
        model = cube(gravity=(0, 0, -9.81))
        model.particle_qd.assign(np.tile((0.12, 0.09, 0.4), (model.particle_count, 1)).astype(np.float32))
        with patch.object(bear, "build_bear_model", return_value=model):
            problem = bear.BearProblem(sim_steps=2, device=DEVICE, use_graph=False)
        checkpoint_loss = problem.forward()
        checkpoint_gradient = problem.backward().numpy().copy()
        self.assertEqual(len(problem.live_states), 2)
        self.assertEqual(len(problem.segment_states), bear.SUBSTEPS + 1)
        self.assertEqual(problem.checkpoint_q.size, 3 * model.particle_count)
        self.assertTrue(np.isfinite(checkpoint_gradient).all())
        self.assertGreater(np.linalg.norm(checkpoint_gradient), 1e-6)

        weights = wp.array(problem.weights.numpy(), dtype=float, device=DEVICE, requires_grad=True)
        states = [model.state(requires_grad=True) for _ in range(2 * bear.SUBSTEPS + 1)]
        activations = [wp.empty(model.tet_count, dtype=float, device=DEVICE, requires_grad=True) for _ in range(2)]
        times = [wp.array([i * bear.FRAME_DT], dtype=float, device=DEVICE) for i in range(2)]
        means = [wp.zeros(1, dtype=wp.vec3, device=DEVICE, requires_grad=True) for _ in range(2)]
        controls = [model.control() for _ in range(2)]
        loss = wp.zeros(1, dtype=float, device=DEVICE, requires_grad=True)
        with wp.Tape() as tape:
            for frame in range(2):
                wp.launch(
                    bear.controller,
                    model.tet_count,
                    inputs=[weights, times[frame]],
                    outputs=[activations[frame]],
                    device=DEVICE,
                )
                controls[frame].tet_activations = activations[frame]
                for step in range(bear.SUBSTEPS):
                    index = frame * bear.SUBSTEPS + step
                    problem.solver.step(states[index], states[index + 1], controls[frame], None, problem.sim_dt)
                wp.launch(
                    bear.mean_velocity,
                    model.particle_count,
                    inputs=[states[(frame + 1) * bear.SUBSTEPS].particle_qd],
                    outputs=[means[frame]],
                    device=DEVICE,
                )
                wp.launch(bear.velocity_loss, 1, inputs=[means[frame]], outputs=[loss], device=DEVICE)
        tape.backward(loss)
        residuals = problem.solver.check_differentiation()
        # check_differentiation enforces 2e-5 relative plus 1e-10 absolute.
        # Near-zero late-frame RHS values can be dominated by the absolute
        # term, so a pure relative threshold would reject valid solves.
        self.assertTrue(all(np.isfinite(value) for value in residuals.values()))
        np.testing.assert_allclose(checkpoint_loss, float(loss.numpy()[0]), rtol=2e-6, atol=1e-7)
        np.testing.assert_allclose(checkpoint_gradient, weights.grad.numpy(), rtol=2e-4, atol=2e-6)
        np.testing.assert_allclose(
            problem.checkpoint_q.numpy().reshape(3, model.particle_count, 3)[-1],
            states[-1].particle_q.numpy(),
            rtol=0,
            atol=2e-6,
        )
        # A subsequent live frame is computed from the original starting pose,
        # and agrees with the saved forward frame rather than advancing a replay.
        problem.reset_live()
        live = problem.advance()
        np.testing.assert_allclose(live.particle_q.numpy(), states[bear.SUBSTEPS].particle_q.numpy(), atol=2e-6)


if __name__ == "__main__":
    unittest.main()

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Surface-buffer growth must preserve accepted material and mechanical storage."""

import os
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull
from test_cut_pipeline import (
    _blade,
    _cube,
    _persistent_objects,
    _persistent_pointers,
    _surface_faces,
)

from newton_solver_fem.examples import example_linear_fem_baymax_cut as cutting


@contextmanager
def _growth_example():
    args = cutting.Example.create_parser().parse_args(
        [
            "--drop_height",
            "1",
            "--linear_damping",
            "0",
            "--material_damping",
            "0",
            "--ground_stiffness",
            "0",
            "--piece_contact_stiffness",
            "0",
            "--fragment_capacity",
            "512",
            "--surface_capacity",
            "12",
        ]
    )
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")
    with wp.ScopedDevice(device), patch.object(cutting, "load_baymax", side_effect=_cube):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _mechanical_pointers(example):
    return {name: value for name, value in _persistent_pointers(example).items() if not name.startswith("surface.")}


def _state_arrays(state):
    return {name: getattr(state, name).numpy().copy() for name in ("particle_q", "particle_qd", "particle_f")}


class TestLinearFEMSurfaceGrowth(unittest.TestCase):
    def check_capture_component_labels(self, *, fail):
        with _growth_example() as example:
            guard = example.volume_guard
            component_pointer = guard.components.ptr
            before = guard.components.numpy().copy()
            original_capture = example._capture_physics_graph
            checked = []

            def capture():
                active = example.model.cut_volume_fractions.numpy() > 0
                expected = example.cutter.candidate.material_components.numpy()
                actual = guard.components.numpy()
                self.assertTrue(np.all(actual[active] >= 0))
                self.assertTrue(np.all(actual[active] < len(actual)))
                np.testing.assert_array_equal(actual[active], expected[active])
                self.assertEqual(guard.components.ptr, component_pointer)
                checked.append(True)
                if fail:
                    raise RuntimeError("injected capture after checking component labels")
                return original_capture()

            with patch.object(example, "_capture_physics_graph", side_effect=capture):
                self.assertEqual(example.cut(_blade()), not fail, example.cut_error)
            self.assertEqual(len(checked), 1)
            self.assertIs(example.volume_guard, guard)
            self.assertEqual(guard.components.ptr, component_pointer)
            if fail:
                np.testing.assert_array_equal(guard.components.numpy(), before)
            else:
                np.testing.assert_array_equal(guard.components.numpy(), example.cutter.material_components.numpy())
            example.step()
            example.test_final()

    def test_capture_uses_new_model_component_labels_before_geometry_publication(self):
        self.check_capture_component_labels(fail=False)

    def test_failed_capture_restores_guard_component_labels(self):
        self.check_capture_component_labels(fail=True)

    def add_pending_hold(self, example):
        points = example.contact.update_surface(example.state_0.particle_q).numpy()
        point = points[:3].mean(axis=0)
        self.assertTrue(
            example.holds.begin(
                example.surface,
                example.cutter.geometry,
                0,
                (1.0 / 3.0,) * 3,
                point,
                example.state_0.particle_q,
            )
        )
        example.holds.move(point + np.asarray((0, 0, 0.25), dtype=np.float32))
        example.holds.finish()
        return example.holds.driven.numpy().copy(), example.holds.requested.numpy().copy()

    def assert_mechanics_retained(self, example, objects, pointers):
        for name, previous in objects.items():
            if name not in ("contact", "surface", "graph"):
                self.assertIs(_persistent_objects(example)[name], previous, name)
        self.assertEqual(_mechanical_pointers(example), pointers)

    def assert_surface_grown(self, example, old_contact, old_surface, old_indices, old_graph):
        self.assertGreater(example._surface_count, old_surface.components.size)
        self.assertGreaterEqual(example.surface.components.size, example._surface_count)
        self.assertIsNot(example.surface, old_surface)
        self.assertIsNot(example.contact, old_contact)
        self.assertIs(example.contact.surface, example.surface)
        self.assertIsNot(example._render_indices, old_indices)
        self.assertGreater(example._render_indices.size, old_indices.size)
        self.assertEqual(example._render_indices.size, example.contact.points.size)
        self.assertEqual(example.contact.capacity, example.surface.components.size)
        self.assertEqual(int(example.surface.count.numpy()[0]), example._surface_count)
        np.testing.assert_array_equal(
            example._render_indices.numpy(), np.arange(example._render_indices.size, dtype=np.int32)
        )
        if example.device.is_cuda:
            self.assertIsNotNone(example.graph)
            self.assertIsNot(example.graph, old_graph)
        else:
            self.assertIsNone(example.graph)

    def test_plane_growth_preserves_state_mass_and_mechanical_storage(self):
        with _growth_example() as example:
            self.assertEqual(example.surface.components.size, 12)
            example.step()
            q = example.state_0.particle_q.numpy()
            velocity = (q * (0.2, 0.1, -0.3) + (0.1, 0.2, -0.1)).astype(np.float32)
            example.state_0.particle_qd.assign(velocity)
            mass = example.model.cut_physical_mass.numpy()
            center, momentum = mass @ q, mass @ velocity
            objects, pointers = _persistent_objects(example), _mechanical_pointers(example)
            old_indices, sim_time = example._render_indices, example.sim_time

            self.assertTrue(example.cut(_blade()), example.cut_error)

            self.assert_surface_grown(example, objects["contact"], objects["surface"], old_indices, objects["graph"])
            self.assert_mechanics_retained(example, objects, pointers)
            after_mass = example.model.cut_physical_mass.numpy()
            np.testing.assert_allclose(after_mass.sum(), mass.sum(), rtol=3.0e-6)
            np.testing.assert_allclose(after_mass @ example.state_0.particle_q.numpy(), center, atol=2.0e-7)
            np.testing.assert_allclose(after_mass @ example.state_0.particle_qd.numpy(), momentum, atol=2.0e-7)
            self.assertEqual(example.sim_time, sim_time)
            example.step()
            example.test_final()

    def test_following_edit_within_reserve_reuses_surface_and_graph(self):
        with _growth_example() as example:
            example.step()
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            objects, pointers = _persistent_objects(example), _persistent_pointers(example)
            indices = example._render_indices

            with patch.object(example, "_capture_physics_graph", wraps=example._capture_physics_graph) as capture:
                self.assertTrue(example.cut(_blade()), example.cut_error)

            capture.assert_not_called()
            self.assertLessEqual(example._surface_count, objects["surface"].components.size)
            for name, previous in objects.items():
                self.assertIs(_persistent_objects(example)[name], previous, name)
            self.assertEqual(_persistent_pointers(example), pointers)
            self.assertIs(example._render_indices, indices)
            example.step()
            example.test_final()

    def test_capture_preserves_both_complete_state_buffers(self):
        with _growth_example() as example:
            example.step()
            for index, state in enumerate((example.state_0, example.state_1)):
                q = state.particle_q.numpy()
                q[:, 0] += 0.001 * index
                state.particle_q.assign(q)
                state.particle_qd.assign(q * 0.1 + (0.1 * index, 0.2, -0.3))
                state.particle_f.assign(q * (index + 1) + 0.7)
            before = [_state_arrays(state) for state in (example.state_0, example.state_1)]
            warm_start = example.solver.dv.numpy().copy()
            warm_start[:] = np.arange(warm_start.size, dtype=np.float32).reshape(warm_start.shape) * 1.0e-4
            example.solver.dv.assign(warm_start)
            driven, requested = self.add_pending_hold(example)
            example.volume_guard.limited_steps.assign([17])
            pointers, sim_time = _mechanical_pointers(example), example.sim_time

            graph = example._capture_physics_graph()

            self.assertEqual(example.sim_time, sim_time)
            self.assertEqual(_mechanical_pointers(example), pointers)
            np.testing.assert_array_equal(example.solver.dv.numpy(), warm_start)
            np.testing.assert_array_equal(example.holds.driven.numpy(), driven)
            np.testing.assert_array_equal(example.holds.requested.numpy(), requested)
            np.testing.assert_array_equal(example.volume_guard.limited_steps.numpy(), [17])
            if example.device.is_cuda:
                self.assertIsNotNone(graph)
            else:
                self.assertIsNone(graph)
            for state, expected in zip((example.state_0, example.state_1), before, strict=True):
                for name, values in expected.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), values, err_msg=name)

    def check_cpu_capture_preserves_state(self, *, fail):
        # Exercise the capture orchestration on CPU without creating a CUDA
        # context: only graph recording is mocked; warmup runs actual physics.
        with patch.dict(os.environ, {"NEWTON_TEST_DEVICE": "cpu"}), _growth_example() as example:
            driven, requested = self.add_pending_hold(example)
            example.volume_guard.limited_steps.assign([17])
            before = [_state_arrays(state) for state in (example.state_0, example.state_1)]
            warm_start = example.solver.dv.numpy().copy()
            actual_simulate = example.simulate
            simulated = []
            graph = object()

            def simulate():
                simulated.append(True)
                if len(simulated) == 1:
                    actual_simulate()
                    self.assertFalse(np.array_equal(example.holds.driven.numpy(), driven))
                    # A warmup can trigger this diagnostic even when the
                    # accepted state is restored before recording.
                    example.volume_guard.limited_steps.assign([23])
                else:
                    np.testing.assert_array_equal(example.holds.driven.numpy(), driven)
                    np.testing.assert_array_equal(example.volume_guard.limited_steps.numpy(), [17])
                    if fail:
                        raise RuntimeError("injected capture failure")

            @contextmanager
            def capture(**kwargs):
                yield SimpleNamespace(graph=graph)

            with (
                patch.object(example, "device", SimpleNamespace(is_cuda=True)),
                patch.object(example, "simulate", side_effect=simulate),
                patch.object(wp, "ScopedCapture", side_effect=capture),
            ):
                if fail:
                    with self.assertRaisesRegex(RuntimeError, "injected capture failure"):
                        example._capture_physics_graph()
                else:
                    self.assertIs(example._capture_physics_graph(), graph)
            self.assertEqual(len(simulated), 2)
            np.testing.assert_array_equal(example.holds.driven.numpy(), driven)
            np.testing.assert_array_equal(example.holds.requested.numpy(), requested)
            np.testing.assert_array_equal(example.volume_guard.limited_steps.numpy(), [17])
            np.testing.assert_array_equal(example.solver.dv.numpy(), warm_start)
            for state, expected in zip((example.state_0, example.state_1), before, strict=True):
                for name, values in expected.items():
                    np.testing.assert_array_equal(getattr(state, name).numpy(), values, err_msg=name)

    def test_cpu_warmup_restores_persistent_hold_and_guard_on_capture_success(self):
        self.check_cpu_capture_preserves_state(fail=False)

    def test_cpu_warmup_restores_persistent_hold_and_guard_on_capture_failure(self):
        self.check_cpu_capture_preserves_state(fail=True)

    def test_failed_capture_restores_old_surface_graph_and_material(self):
        with _growth_example() as example:
            example.step()
            objects, pointers = _persistent_objects(example), _persistent_pointers(example)
            indices, sim_time = example._render_indices, example.sim_time
            before_state = _state_arrays(example.state_0)
            before_model = {
                name: getattr(example.model, name).numpy() for name in example._model_factory._array_names()
            }
            before_geometry = {
                name: getattr(example.cutter.geometry, name).numpy()
                for name in ("parent", "nodeindices", "vertices", "faces", "vertex_count", "face_count", "count")
            }
            faces = _surface_faces(example)
            count, surface_count, revision = example.fragment_count, example._surface_count, example.cut_revision

            with patch.object(
                example, "_capture_physics_graph", side_effect=RuntimeError("injected surface graph capture failure")
            ) as capture:
                self.assertFalse(example.cut(_blade()))

            capture.assert_called_once()
            self.assertIn("injected surface graph capture failure", example.cut_error)
            for name, previous in objects.items():
                self.assertIs(_persistent_objects(example)[name], previous, name)
            self.assertEqual(_persistent_pointers(example), pointers)
            self.assertIs(example._render_indices, indices)
            self.assertEqual(
                (example.fragment_count, example._surface_count, example.cut_revision), (count, surface_count, revision)
            )
            self.assertEqual(example.sim_time, sim_time)
            self.assertEqual(_surface_faces(example), faces)
            for name, values in before_state.items():
                np.testing.assert_array_equal(getattr(example.state_0, name).numpy(), values, err_msg=name)
            for name, values in before_model.items():
                np.testing.assert_array_equal(getattr(example.model, name).numpy(), values, err_msg=name)
            for name, values in before_geometry.items():
                np.testing.assert_array_equal(getattr(example.cutter.geometry, name).numpy(), values, err_msg=name)
            example.step()
            example.test_final()
            self.assertTrue(example.cut(_blade()), example.cut_error)
            example.step()
            example.test_final()

    def test_sphere_growth_removes_material_without_advancing_state(self):
        with _growth_example() as example:
            example.step()
            q = example.state_0.particle_q.numpy()
            velocity = (q * (0.2, 0.1, -0.3) + (0.1, 0.2, -0.1)).astype(np.float32)
            example.state_0.particle_qd.assign(velocity)
            mass = example.model.cut_physical_mass.numpy().sum(dtype=np.float64)
            objects, pointers = _persistent_objects(example), _mechanical_pointers(example)
            indices, sim_time = example._render_indices, example.sim_time
            center, radius = (0.003, 0.004, 1.05), 0.018

            self.assertTrue(example.cut_sphere(center, center, radius), example.cut_error)

            self.assert_surface_grown(example, objects["contact"], objects["surface"], indices, objects["graph"])
            self.assert_mechanics_retained(example, objects, pointers)
            removed_mass = mass - example.model.cut_physical_mass.numpy().sum(dtype=np.float64)
            self.assertAlmostEqual(removed_mass, example.args.density * 2.53615071012041 * radius**3, delta=5.0e-7)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), q)
            np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), velocity)
            self.assertEqual(example.sim_time, sim_time)
            example.step()
            example.test_final()


if __name__ == "__main__":
    unittest.main()

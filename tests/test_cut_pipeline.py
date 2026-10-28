# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Small physical end-to-end checks for the interactive Linear FEM cutter."""

import os
import unittest
from contextlib import contextmanager
from itertools import permutations
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem import SolverLinearFEM
from newton_solver_fem.examples import example_linear_fem_baymax_cut as cutting


def _cube():
    points = np.asarray([(x, y, z) for x in (-0.05, 0.05) for y in (-0.05, 0.05) for z in (0.0, 0.1)])
    elements = []
    for first, second, _ in permutations(range(3)):
        nodes = [0, 1 << first, (1 << first) | (1 << second), 7]
        corners = points[nodes]
        if np.linalg.det((corners[1:] - corners[0]).T) < 0:
            nodes[1], nodes[2] = nodes[2], nodes[1]
        elements.append(nodes)
    return points, np.asarray(elements, dtype=np.int32)


@contextmanager
def _example(device=None):
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
            "64",
        ]
    )
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu") if device is None else device
    with wp.ScopedDevice(device), patch.object(cutting, "load_baymax", side_effect=_cube):
        viewer = ViewerNull()
        try:
            yield cutting.Example(viewer, args)
        finally:
            viewer.close()


def _blade(ymax=0.2):
    polygon = np.asarray(((0.003, -0.2, 0.8), (0.003, ymax, 0.8), (0.003, ymax, 1.3), (0.003, -0.2, 1.3)))
    return polygon[np.asarray(((0, 1, 2), (0, 2, 3)))]


def _persistent_objects(example):
    names = ("model", "state_0", "state_1", "control", "solver", "contact", "surface", "graph")
    objects = {name: getattr(example, name) for name in names}
    objects["linear_solver"] = example.solver._lin_solver
    objects["geometry"] = example.cutter.geometry
    return objects


def _persistent_pointers(example):
    arrays = {f"model.{name}": getattr(example.model, name) for name in example._model_factory._array_names()}
    for state_name in ("state_0", "state_1"):
        for name in ("particle_q", "particle_qd", "particle_f"):
            arrays[f"{state_name}.{name}"] = getattr(getattr(example, state_name), name)
    for name in ("offsets", "columns", "values"):
        arrays[f"matrix.{name}"] = getattr(example.solver.A_bsr, name)
    for name in ("_tangent_triplet_slots", "_pc_inv_diag", "dv"):
        arrays[f"solver.{name}"] = getattr(example.solver, name)
    for name in ("vertex_nodes", "weights", "indices", "components", "kind", "reference_area", "count"):
        arrays[f"surface.{name}"] = getattr(example.surface, name)
    return {name: None if array is None else array.ptr for name, array in arrays.items()}


def _surface_faces(example):
    """Compare physical triangles without relying on atomic append ordering."""
    count = int(example.surface.count.numpy()[0])
    nodes = example.surface.vertex_nodes.numpy()[: 3 * count]
    weights = example.surface.weights.numpy()[: 3 * count]
    points = example.state_0.particle_q.numpy()
    triangles = np.einsum("ni,nij->nj", weights, points[nodes]).reshape(count, 3, 3)
    components = example.surface.components.numpy()[:count]
    kinds = example.surface.kind.numpy()[:count]
    result = []
    for triangle, component, kind in zip(triangles, components, kinds, strict=True):
        vertices = [tuple(point) for point in np.round(triangle, decimals=7)]
        first = vertices.index(min(vertices))
        result.append((int(component), int(kind), *(vertices[first:] + vertices[:first])))
    return sorted(result)


class TestCutPipeline(unittest.TestCase):
    def test_debug_view_rebuilds_and_renders_after_cut(self):
        with _example() as example:
            view = example.connection_debug
            view.connections.enabled = True
            view.render(example.state_0.particle_q, example.cut_revision)
            self.assertEqual(view.connections.fragment_count, example.fragment_count)
            self.assertTrue(example.cut(_blade()), example.cut_error)
            view.render(example.state_0.particle_q, example.cut_revision)
            self.assertEqual(view.connections.fragment_count, example.fragment_count)
            self.assertGreaterEqual(view.connections.component_count, 2)

    def assert_persistent(self, example, objects, pointers):
        for name, value in _persistent_objects(example).items():
            self.assertIs(value, objects[name], name)
        self.assertEqual(_persistent_pointers(example), pointers)

    def test_accepted_cuts_reuse_physics_graph_and_buffers(self):
        with _example() as example:
            # CPU construction does not warm the iterative solver until step.
            example.step()
            objects, pointers = _persistent_objects(example), _persistent_pointers(example)
            if example.device.is_cuda:
                self.assertIsNotNone(example.graph)
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            self.assert_persistent(example, objects, pointers)
            self.assertEqual(example.model.cut_fragment_count, example.fragment_count)
            example.step()
            self.assertTrue(example.cut(_blade()), example.cut_error)
            self.assert_persistent(example, objects, pointers)
            for _ in range(3):
                example.step()
            example.test_final()

    def test_failed_publication_restores_geometry_state_mass_and_surface(self):
        with _example() as example:
            example.step()
            objects, pointers = _persistent_objects(example), _persistent_pointers(example)
            geometry = example.cutter.geometry
            geometry_arrays = {
                name: getattr(geometry, name).numpy()
                for name in ("parent", "nodeindices", "vertices", "faces", "vertex_count", "face_count", "count")
            }
            model_arrays = {
                name: getattr(example.model, name).numpy() for name in example._model_factory._array_names()
            }
            q, qd = example.state_0.particle_q.numpy(), example.state_0.particle_qd.numpy()
            volume = example.solver.compute_volume(example.state_0)
            surface, surface_count = _surface_faces(example), int(example.surface.count.numpy()[0])
            fragment_count, revision = example.fragment_count, example.cut_revision
            original_update = example.contact.update_topology
            calls = 0

            def fail_after_update(*args, **kwargs):
                nonlocal calls
                calls += 1
                original_update(*args, **kwargs)
                if calls == 1:
                    raise RuntimeError("injected publication failure after contact mutation")

            with patch.object(example.contact, "update_topology", side_effect=fail_after_update):
                self.assertFalse(example.cut(_blade()))
            self.assertEqual(calls, 2)
            self.assertIn("injected publication failure", example.cut_error)
            self.assert_persistent(example, objects, pointers)
            self.assertEqual(example.fragment_count, fragment_count)
            self.assertEqual(example.model.cut_fragment_count, fragment_count)
            self.assertEqual(example.cut_revision, revision)
            self.assertEqual(int(example.surface.count.numpy()[0]), surface_count)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), q)
            np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), qd)
            self.assertEqual(example.solver.compute_volume(example.state_0), volume)
            for name, expected in geometry_arrays.items():
                np.testing.assert_array_equal(getattr(geometry, name).numpy(), expected, err_msg=name)
            for name, expected in model_arrays.items():
                np.testing.assert_array_equal(getattr(example.model, name).numpy(), expected, err_msg=name)
            self.assertEqual(_surface_faces(example), surface)
            self.assertTrue(example.cut(_blade()), example.cut_error)
            self.assert_persistent(example, objects, pointers)
            example.step()
            example.test_final()

    def test_finite_incisions_preserve_mass_state_and_reset(self):
        with _example() as example:
            self.assertIsInstance(example.solver, SolverLinearFEM)
            original_q = example.state_0.particle_q.numpy()
            velocity = original_q * (0.2, 0.1, -0.3) + (1.0, 0.5, -0.1)
            example.state_0.particle_qd.assign(velocity)
            old_mass = example.model.cut_physical_mass.numpy()
            old_volume = example.solver.compute_volume(example.state_0)
            self.assertAlmostEqual(old_volume, 0.001, places=8)
            self.assertAlmostEqual(example.solver.get_initial_volume(), old_volume, places=8)
            old_momentum = old_mass @ velocity
            old_center = old_mass @ original_q
            self.assertTrue(example.cut(_blade(ymax=0.013)), example.cut_error)
            self.assertGreater(example.fragment_count, len(example.reference_elements))
            mass = example.model.cut_physical_mass.numpy()
            np.testing.assert_allclose(mass.sum(), old_mass.sum(), rtol=3e-6)
            self.assertAlmostEqual(example.solver.compute_volume(example.state_0), old_volume, places=8)
            np.testing.assert_allclose(mass @ example.state_0.particle_q.numpy(), old_center, atol=2e-7)
            np.testing.assert_allclose(mass @ example.state_0.particle_qd.numpy(), old_momentum, atol=2e-7)
            example.test_final()
            example.reset()
            self.assertEqual(example.cut_revision, 0)
            self.assertEqual(example.fragment_count, len(example.reference_elements))
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), original_q)
            np.testing.assert_array_equal(example.state_0.particle_qd.numpy(), 0)

    def test_separated_pieces_both_free_fall(self):
        with _example() as example:
            self.assertTrue(example.cut(_blade()), example.cut_error)
            labels = example.cutter.candidate.material_components.numpy()[: example.fragment_count]
            nodes = example.model.tet_indices.numpy()[: example.fragment_count]
            components = [np.unique(nodes[labels == label]) for label in np.unique(labels)]
            self.assertEqual(len(components), 2)
            self.assertEqual(len(np.intersect1d(*components)), 0)
            points = example.state_0.particle_q.numpy()
            for index, component in enumerate(components):
                points[component, 0] += -0.02 if index == 0 else 0.02
            example.state_0.particle_q.assign(points)
            mass = example.model.cut_physical_mass.numpy()
            initial = [mass[ids] @ points[ids] / mass[ids].sum() for ids in components]
            for _ in range(12):
                example.step()
            after = example.state_0.particle_q.numpy()
            expected_drop = 0.5 * example.args.gravity * example.sim_time * (example.sim_time + example.sim_dt)
            for ids, center in zip(components, initial, strict=True):
                displacement = mass[ids] @ after[ids] / mass[ids].sum() - center
                np.testing.assert_allclose(displacement, (0, 0, -expected_drop), atol=5e-4)
            example.test_final()

    def test_rejected_patch_keeps_live_state_and_topology(self):
        with _example() as example:
            before = example.state_0.particle_q.numpy()
            model, geometry = example.model, example.cutter.geometry
            self.assertFalse(example.cut(np.zeros((2, 3, 3))))
            self.assertTrue(example.cut_error)
            self.assertIs(example.model, model)
            self.assertIs(example.cutter.geometry, geometry)
            self.assertEqual(example.cut_revision, 0)
            np.testing.assert_array_equal(example.state_0.particle_q.numpy(), before)
            self.assertTrue(example.cut(_blade()), example.cut_error)


if __name__ == "__main__":
    unittest.main()

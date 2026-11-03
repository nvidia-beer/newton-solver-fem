# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Physical selected-region interaction checks on two adjoining small boxes."""

import json
import os
import unittest
from contextlib import contextmanager
from itertools import permutations
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem import MeshRegion, RegionMesh, boundary_faces
from newton_solver_fem.examples import example_fem_phantom_regions as phantom


def _regions():
    points = np.array([(x, y, z) for x in range(3) for y in range(2) for z in range(2)], dtype=float) * 0.01
    tets, labels = [], []
    for region in range(2):
        for first, second, _ in permutations(range(3)):
            nodes = np.asarray([0, 1 << first, (1 << first) | (1 << second), 7]) + 4 * region
            corners = points[nodes]
            if np.linalg.det((corners[1:] - corners[0]).T) < 0:
                nodes[1], nodes[2] = nodes[2], nodes[1]
            tets.append(nodes)
            labels.append(region + 1)
    return points, np.asarray(tets, dtype=np.int32), np.asarray(labels)


_POINTS, _TETS, _LABELS = _regions()
_CATALOG = tuple(MeshRegion(i, f"feature_{i}", f"/Pad/region_{i}", (0.5, 0.7, 0.2), 6, 8) for i in (1, 2))
_FULL_FACES, _ = boundary_faces(_TETS)
_EXTERIOR_KEYS = {tuple(sorted(face)) for face in _FULL_FACES}


def _selection(_path, region_ids):
    selected = np.flatnonzero(np.isin(_LABELS, region_ids))
    tets = _TETS[selected]
    gids = np.unique(tets)
    local = np.searchsorted(gids, tets).astype(np.int32)
    if len(selected):
        faces, owners = boundary_faces(local)
    else:
        faces, owners = np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int32)
    is_cut = np.asarray([tuple(sorted(gids[face])) not in _EXTERIOR_KEYS for face in faces], dtype=bool)
    return RegionMesh(
        _POINTS[gids],
        local,
        faces,
        _LABELS[selected],
        _LABELS[selected][owners],
        faces[~is_cut],
        _LABELS[selected][owners][~is_cut],
        np.unique(faces[is_cut]),
        gids,
        selected,
        tuple(region for region in _CATALOG if region.id in region_ids),
    )


@contextmanager
def _example(substeps=2, viewer_cls=ViewerNull, numbering=None):
    with (
        TemporaryDirectory() as directory,
        wp.ScopedDevice(os.environ.get("NEWTON_TEST_DEVICE", "cpu")),
        patch.object(phantom, "list_mesh_regions", return_value=_CATALOG),
        patch.object(phantom, "load_mesh_regions", side_effect=_selection),
    ):
        asset = Path(directory) / "phantom_regions.usd"
        if numbering is not None:
            asset.with_name("suturebot_region_map.json").write_text(json.dumps({"regions": numbering}))
        args = phantom.Example.create_parser().parse_args(
            ["--asset", str(asset), "--regions", "1", "--substeps", str(substeps)]
        )
        viewer = viewer_cls()
        try:
            yield phantom.Example(viewer, args)
        finally:
            viewer.close()


class PhantomRegionsExampleTests(unittest.TestCase):
    def test_one_numbering_selects_original_geometry_without_changing_membership(self):
        numbering = [
            {"id": 1, "source_region_id": 2, "name": "first_wound"},
            {"id": 2, "source_region_id": 1, "name": "remaining_pad"},
        ]
        with _example(numbering=numbering) as example:
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertEqual(example.selection.regions[0].name, "first_wound")
            np.testing.assert_array_equal(example.reference.vertices, _POINTS)
            np.testing.assert_array_equal(example.reference.indices, _TETS)
            np.testing.assert_array_equal(example.selection.global_tetrahedron_indices, np.arange(6, 12))
            self.assertTrue(np.all(example.selection.tetrahedron_region_ids == 1))
            self.assertTrue(np.all(example.selection.vertices[:, 0] >= 0.01))
            example.select_regions([2])
            np.testing.assert_array_equal(example.selection.global_tetrahedron_indices, np.arange(6))
            self.assertTrue(np.all(example.selection.tetrahedron_region_ids == 2))
            for region in example.regions:
                self.assertEqual(example.region_map.region_at(*example.region_map.centers[region.id]), region.id)

    def test_selection_compacts_physics_and_rejects_multiple_active_regions(self):
        with _example() as example:
            self.assertEqual(example.model.tet_count, 6)
            self.assertEqual(example.model.particle_count, 8)
            original_model = example.model
            with self.assertRaisesRegex(ValueError, "Unknown pad regions"):
                example.select_regions([99])
            self.assertIs(example.model, original_model)
            with self.assertRaisesRegex(ValueError, "at most one active FEM region"):
                example.select_regions([1, 2])
            self.assertIs(example.model, original_model)
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertEqual(len(example.fixed_vertex_indices), 6)
            example.step()
            example.test_final()
            example.select_regions([])
            self.assertEqual(example.model.tet_count, 0)
            self.assertIsNone(example.solver)
            example.step()
            self.assertIsNone(example.begin_grab((0.005, 0.005, 0.1), (0, 0, -1)))
            example.select_regions([2])
            self.assertEqual(example.model.tet_count, 6)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )

    def test_inactive_rejects_grab_and_active_spring_deforms_fem(self):
        with _example() as example:
            self.assertIsNone(example.begin_grab((0.015, 0.005, 0.1), (0, 0, -1)))
            hit = example.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
            self.assertIsNotNone(hit)
            example.move_grab(hit + (0, 0, 0.001))
            for _ in range(20):
                example.step()
            example.test_final()
            displacement = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
            self.assertGreater(displacement.max(), 1.0e-5)
            example.render()
            displayed = example.display_points.numpy()
            inactive = ~np.isin(example.reference.global_vertex_indices, example.selection.global_vertex_indices)
            np.testing.assert_array_equal(displayed[inactive], example.reference.vertices[inactive].astype(np.float32))
            example.finish_grab()
            self.assertEqual(example.holds.count, 1)
            example.clear_holds()
            for _ in range(40):
                example.step()
            example.test_final()
            after = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
            self.assertLess(after.max(), displacement.max())
            example.reset()
            self.assertEqual(example.holds.count, 0)
            np.testing.assert_array_equal(
                example.state_0.particle_q.numpy(), example.selection.vertices.astype(np.float32)
            )

    def test_odd_substeps_match_eager_after_live_grip_changes(self):
        with _example(substeps=3) as captured, _example(substeps=3) as eager:
            if captured.device.is_cuda:
                self.assertIsNotNone(captured.graph)
            eager.graph = None
            for example in (captured, eager):
                hit = example.begin_grab((0.003, 0.004, 0.1), (0, 0, -1))
                self.assertIsNotNone(hit)
                example.move_grab(hit + (0.0002, 0, 0.0007))
            for _ in range(3):
                captured.step()
                eager.step()
                np.testing.assert_allclose(
                    captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=2.0e-8
                )
                # Float32 assembly uses atomics: independent captured/captured
                # and eager/eager runs also differ by nanometres. Compare the
                # velocity-induced displacement over one substep (40 nm), not
                # a relative tolerance on velocity components near zero.
                np.testing.assert_allclose(
                    captured.state_0.particle_qd.numpy() * captured.sim_dt,
                    eager.state_0.particle_qd.numpy() * eager.sim_dt,
                    rtol=0,
                    atol=4.0e-8,
                )
            captured.clear_holds()
            eager.clear_holds()
            captured.step()
            eager.step()
            np.testing.assert_allclose(
                captured.state_0.particle_q.numpy(), eager.state_0.particle_q.numpy(), rtol=0, atol=2.0e-8
            )
            captured.test_final()
            eager.test_final()

    def test_model_switch_restores_each_ui_callback_once(self):
        class CallbackViewer(ViewerNull):
            def __init__(self):
                self.callbacks = {"side": [], "free": []}
                super().__init__()

            def set_model(self, model):
                super().set_model(model)
                # ViewerGL clears example-owned callbacks on every model swap.
                self.callbacks = {"side": [], "free": []}

            def register_ui_callback(self, callback, position="side"):
                self.callbacks[position].append(callback)

        with _example(viewer_cls=CallbackViewer) as example:
            # Match initial registration by newton.examples.run and the GL tool.
            example.viewer.register_ui_callback(example.gui, position="side")
            example.viewer.register_ui_callback(example.screen_pick.draw_overlay, position="free")
            for selection in ([2], [], [1], [2]):
                example.select_regions(selection)
                self.assertEqual(example.viewer.callbacks["side"], [example.gui])
                self.assertEqual(example.viewer.callbacks["free"], [example.screen_pick.draw_overlay])
                self.assertIs(example.screen_pick.model, example.model)

    def test_bad_damping_rejected_before_loading(self):
        args = phantom.Example.create_parser().parse_args(["--grip-damping", "nan"])
        with self.assertRaisesRegex(ValueError, "damping"):
            phantom.Example(None, args)

    def test_map_click_selects_one_region_and_rejected_load_keeps_previous_selection(self):
        class UI:
            button_clicked = None

            def button(self, label):
                return label == self.button_clicked

            def get_item_rect_min(self):
                return SimpleNamespace(x=0, y=0)

            def get_item_rect_max(self):
                return SimpleNamespace(x=10, y=10)

            def __getattr__(self, _name):
                return lambda *_args: False

        with _example() as example:
            ui = UI()
            original_model = example.model
            with patch.object(example.region_map, "draw", return_value=1):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (1,))
            self.assertIs(example.model, original_model)
            with patch.object(example.region_map, "draw", return_value=2):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (2,))
            previous_model = example.model
            with (
                patch.object(example.region_map, "draw", return_value=1),
                patch.object(phantom, "load_mesh_regions", side_effect=ValueError("Nonmanifold cut boundary")),
            ):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, (2,))
            self.assertIs(example.model, previous_model)
            self.assertIn("Nonmanifold", example.selection_error)
            ui.button_clicked = "None"
            with patch.object(example.region_map, "draw", return_value=None):
                example.gui(ui)
            self.assertEqual(example.selected_region_ids, ())
            self.assertIsNone(example.solver)


if __name__ == "__main__":
    unittest.main()

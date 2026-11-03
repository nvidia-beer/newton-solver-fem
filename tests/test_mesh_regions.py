"""Region payload selection changes volume connectivity, not just visibility."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from newton_solver_fem import list_mesh_regions, load_mesh_regions


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "Install the examples extra for USD assets")
class MeshRegionTests(unittest.TestCase):
    def setUp(self):
        from pxr import Sdf, Usd, UsdGeom, Vt

        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "regions.usda"
        stage = Usd.Stage.CreateNew(str(self.path))
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        root = UsdGeom.Xform.Define(stage, "/Pad").GetPrim()
        stage.SetDefaultPrim(root)
        points = np.array([[0, 0, 0], [1 + 2**-40, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, -1]])
        self.points = points
        manifest = []
        for index, gids in enumerate(([0, 1, 2, 3], [0, 2, 1, 4])):
            rid = index + 1
            payload = Path(self.directory.name) / f"region_{rid}.usda"
            piece = Usd.Stage.CreateNew(str(payload))
            piece.SetDefaultPrim(UsdGeom.Xform.Define(piece, "/Region").GetPrim())
            mesh = UsdGeom.TetMesh.Define(piece, "/Region/Volume")
            mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(points[gids].astype(np.float32)))
            mesh.CreateTetVertexIndicesAttr(Vt.Vec4iArray([(0, 1, 2, 3)]))
            mesh.CreateSurfaceFaceVertexIndicesAttr(Vt.Vec3iArray([(0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3)]))
            prim = mesh.GetPrim()
            prim.CreateAttribute("newton:referencePoints", Sdf.ValueTypeNames.Double3Array).Set(
                Vt.Vec3dArray.FromNumpy(points[gids])
            )
            prim.CreateAttribute("suturebot:globalVertexIndices", Sdf.ValueTypeNames.Int64Array).Set(
                Vt.Int64Array([10 + value for value in gids])
            )
            prim.CreateAttribute("suturebot:globalTetrahedronIndices", Sdf.ValueTypeNames.Int64Array).Set(
                Vt.Int64Array([100 + index])
            )
            prim.CreateAttribute("suturebot:boundaryIsCutInterface", Sdf.ValueTypeNames.BoolArray).Set(
                Vt.BoolArray([True, False, False, False])
            )
            piece.GetRootLayer().Save()
            path = f"/Pad/Regions/region_{rid}"
            group = UsdGeom.Xform.Define(stage, path).GetPrim()
            group.GetPayloads().AddPayload(payload.name, "/Region")
            manifest.append(
                dict(
                    id=rid,
                    name=f"feature_{rid}",
                    prim_path=path,
                    color_srgb=[224, 72, 123],
                    tetrahedron_count=1,
                    vertex_count=4,
                )
            )
        root.CreateAttribute("suturebot:regionManifest", Sdf.ValueTypeNames.String).Set(json.dumps(manifest))
        stage.GetRootLayer().Save()

    def test_subset_welding_and_cut_constraints(self):
        first = load_mesh_regions(self.path, [1])
        self.assertEqual(first.indices.shape, (1, 4))
        self.assertEqual(len(first.fixed_vertex_indices), 3)
        self.assertEqual(len(first.exterior_surface_triangles), 3)
        both = load_mesh_regions(self.path, [1, 2])
        self.assertEqual(both.vertices.shape, (5, 3))
        self.assertEqual(both.indices.shape, (2, 4))
        self.assertEqual(len(both.surface_triangles), 6)
        self.assertEqual(len(both.fixed_vertex_indices), 0)
        self.assertEqual(len(both.exterior_surface_triangles), 6)
        np.testing.assert_array_equal(both.vertices, self.points)
        np.testing.assert_array_equal(both.global_vertex_indices, np.arange(10, 15))
        np.testing.assert_array_equal(both.global_tetrahedron_indices, [100, 101])
        self.assertEqual(set(both.surface_region_ids), {1, 2})

    def test_catalog_and_subset_do_not_load_unselected_payload(self):
        (Path(self.directory.name) / "region_2.usda").unlink()
        catalog = list_mesh_regions(self.path)
        self.assertEqual([region.id for region in catalog], [1, 2])
        self.assertEqual(load_mesh_regions(self.path, [1]).indices.shape, (1, 4))

    def test_activation_selects_physics_but_visibility_does_not(self):
        from pxr import Usd, UsdGeom

        stage = Usd.Stage.Open(str(self.path), Usd.Stage.LoadNone)
        stage.GetPrimAtPath("/Pad/Regions/region_2").SetActive(False)
        UsdGeom.Imageable(stage.GetPrimAtPath("/Pad/Regions/region_1")).CreateVisibilityAttr("invisible")
        stage.GetRootLayer().Save()
        before = self.path.read_bytes()
        np.testing.assert_array_equal(load_mesh_regions(self.path).tetrahedron_region_ids, [1])
        np.testing.assert_array_equal(load_mesh_regions(self.path, [2]).tetrahedron_region_ids, [2])
        self.assertEqual(before, self.path.read_bytes())

    def test_conflicting_shared_coordinates_are_rejected(self):
        from pxr import Usd

        stage = Usd.Stage.Open(str(Path(self.directory.name) / "region_2.usda"))
        attribute = stage.GetPrimAtPath("/Region/Volume").GetAttribute("newton:referencePoints")
        points = attribute.Get()
        points[0] = (0.01, 0, 0)
        attribute.Set(points)
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "conflicting reference coordinates"):
            load_mesh_regions(self.path, [1, 2])

    def test_overlap_and_corrupt_boundary_are_rejected(self):
        from pxr import Usd, Vt

        stage = Usd.Stage.Open(str(Path(self.directory.name) / "region_2.usda"))
        prim = stage.GetPrimAtPath("/Region/Volume")
        prim.GetAttribute("suturebot:globalTetrahedronIndices").Set(Vt.Int64Array([100]))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "overlapping global tetrahedron"):
            load_mesh_regions(self.path, [1, 2])
        prim.GetAttribute("surfaceFaceVertexIndices").Set(Vt.Vec3iArray([(0, 2, 1)] * 4))
        stage.GetRootLayer().Save()
        with self.assertRaisesRegex(ValueError, "authored boundary differs"):
            load_mesh_regions(self.path, [2])

    def test_empty_and_unknown_selection(self):
        mesh = load_mesh_regions(self.path, [])
        self.assertEqual(mesh.indices.shape, (0, 4))
        self.assertEqual(mesh.vertices.shape, (0, 3))
        self.assertEqual(len(mesh.fixed_vertex_indices), 0)
        with self.assertRaisesRegex(ValueError, "Unknown region IDs"):
            load_mesh_regions(self.path, [99])

    def test_vertex_and_edge_pinches_are_rejected_even_with_valid_tetrahedra(self):
        from pxr import Usd, Vt

        cases = (
            (
                "vertices have disconnected surface fans",
                [[0, 0, 0], [0, -1, 0], [-1, 0, 0], [0, 0, -1]],
                [10, 15, 16, 17],
            ),
            (
                "edges do not have exactly two incident triangles",
                [[0, 0, 0], [1 + 2**-40, 0, 0], [0, -1, 0], [0, 0, -1]],
                [10, 11, 15, 16],
            ),
        )
        for message, points, global_ids in cases:
            with self.subTest(pinch=message):
                for region in (1, 2):
                    stage = Usd.Stage.Open(str(Path(self.directory.name) / f"region_{region}.usda"))
                    prim = stage.GetPrimAtPath("/Region/Volume")
                    prim.GetAttribute("suturebot:boundaryIsCutInterface").Set(Vt.BoolArray([False] * 4))
                    if region == 2:
                        prim.GetAttribute("newton:referencePoints").Set(
                            Vt.Vec3dArray.FromNumpy(np.asarray(points, dtype=np.float64))
                        )
                        prim.GetAttribute("suturebot:globalVertexIndices").Set(Vt.Int64Array(global_ids))
                    stage.GetRootLayer().Save()
                self.assertEqual(len(load_mesh_regions(self.path, [1]).indices), 1)
                self.assertEqual(len(load_mesh_regions(self.path, [2]).indices), 1)
                with self.assertRaisesRegex(ValueError, message):
                    load_mesh_regions(self.path, [1, 2])

    def test_bundled_phantom_has_complete_volume_and_smaller_selected_payload(self):
        path = Path(__file__).parents[1] / "src/newton_solver_fem/examples/assets/phantom_pad/phantom_regions.usd"
        catalog = list_mesh_regions(path)
        self.assertEqual([region.id for region in catalog], list(range(12)))
        self.assertEqual(catalog[0].name, "base_and_lettering")
        self.assertEqual(catalog[1].name, "feature_01")
        self.assertEqual(catalog[10].name, "wound_5_slit_patch")
        self.assertEqual(sum(region.tetrahedron_count for region in catalog), 413405)
        whole = load_mesh_regions(path)
        self.assertEqual(whole.vertices.shape, (97489, 3))
        self.assertEqual(whole.indices.shape, (413405, 4))
        self.assertEqual(whole.surface_triangles.shape, (110956, 3))
        self.assertEqual(len(whole.fixed_vertex_indices), 0)
        part = load_mesh_regions(path, [9])
        self.assertEqual(part.vertices.shape, (3534, 3))
        self.assertEqual(part.indices.shape, (13909, 4))
        self.assertEqual(len(part.fixed_vertex_indices), 637)
        np.testing.assert_array_equal(part.vertices, whole.vertices[part.global_vertex_indices])
        np.testing.assert_array_equal(
            part.global_vertex_indices[part.indices], whole.indices[part.global_tetrahedron_indices]
        )
        for region in catalog:
            with self.subTest(region=region.id):
                self.assertEqual(len(load_mesh_regions(path, [region.id]).indices), region.tetrahedron_count)
        # This original pair still touches at pinched vertices; preserving
        # original ownership preserves the selector's existing rejection.
        with self.assertRaisesRegex(ValueError, "nonmanifold"):
            load_mesh_regions(path, [0, 9])
        restored_one = load_mesh_regions(path, [1, 10, 11])
        self.assertEqual(restored_one.vertices.shape, (21439, 3))
        self.assertEqual(restored_one.indices.shape, (88914, 4))
        np.testing.assert_array_equal(restored_one.vertices, whole.vertices[restored_one.global_vertex_indices])
        np.testing.assert_array_equal(
            restored_one.global_vertex_indices[restored_one.indices],
            whole.indices[restored_one.global_tetrahedron_indices],
        )
        wound_five = load_mesh_regions(path, [10])
        self.assertEqual(wound_five.indices.shape, (18790, 4))
        self.assertEqual(wound_five.vertices.shape, (4689, 3))
        self.assertGreater(wound_five.vertices[wound_five.indices].mean(axis=1)[:, 1].min(), 0.0)
        surface = whole.vertices[whole.exterior_surface_triangles]
        normals = np.cross(surface[:, 1] - surface[:, 0], surface[:, 2] - surface[:, 0])
        upward = normals[:, 2] > 0.1 * np.linalg.norm(normals, axis=1)
        centers = surface.mean(axis=1)[upward, :2]
        owners = whole.exterior_surface_region_ids[upward]
        # Source-surface landmarks: keep the slit and return the neighboring
        # long groove to original region one. The earlier straight cut included that groove.
        for x, y, expected in (
            (0.0423, 0.0174, 10),
            (0.0477, 0.0245, 10),
            (0.0530, 0.0315, 10),
            (0.0250, 0.0210, 1),
            (0.0300, 0.0280, 1),
            (0.0340, 0.0340, 1),
            (0.051050, 0.007972, 10),
            (0.045608, 0.008837, 10),
            (0.0520, 0.0470, 1),
            (0.0490, 0.0480, 1),
            (0.0100, -0.0300, 11),
            (0.0130, -0.0270, 11),
            (0.0180, -0.0100, 1),
        ):
            with self.subTest(surface_landmark=(x, y)):
                closest = np.argmin(np.linalg.norm(centers - (x, y), axis=1))
                self.assertEqual(owners[closest], expected)

    def test_dataset_labels_preserve_original_payloads_except_allowed_wound_five_split(self):
        folder = Path(__file__).parents[1] / "src/newton_solver_fem/examples/assets/phantom_pad"
        mapping = json.loads((folder / "suturebot_region_map.json").read_text())
        self.assertEqual([region["id"] for region in mapping["regions"]], list(range(1, 13)))
        self.assertEqual(
            [region["source_region_id"] for region in mapping["regions"]], [6, 2, 5, 7, 10, 9, 0, 1, 3, 4, 8, 11]
        )
        # Hashes from the original committed geometry-only payloads before
        # this task. Labels cannot silently repartition any other volume.
        original_hashes = {
            "region_000_base_and_lettering.usdc": "e1d67af37a3cb6d2c0dff2bb1b39d773f1197f7bc2db9fdd883d78267b5a48c7",
            "region_002_feature_02.usdc": "ffd338ba156b14d2391d8f48c8c903bd8084dde8ce675fa9b34a79fcecc3b013",
            "region_003_feature_03.usdc": "76c7cb1fbfeb72e61bb4402a27b5d1f10eaecb15b115efa100d1d7f0d3026ed4",
            "region_004_feature_04.usdc": "638a2728bb203b7f7ccc2c10e5d2fc17ca662d3763bc16cf65f2f6de73b5fd7a",
            "region_005_feature_05.usdc": "38824c34a68a6788146982b08192ccaf556eef5ecd6f9791ed3632d7aedc19d3",
            "region_006_feature_06.usdc": "b8ac9e3d67ca515e74a2e7495683716637911b650a7ba2c1c39918bb1de1cae3",
            "region_007_feature_07.usdc": "28763d6c606ac323169c395bfb7059e5d9d6f1f7c9c0da69dff35409bf888567",
            "region_008_feature_08.usdc": "3f11004229357330d068099eefd0e457f3bd60c0b7cd543b555348a0fc1d5fe1",
            "region_009_upper_paired_strips.usdc": "2356925ad8eed756a7a2181bc3891e8175149e8fcae05d99da97ed6013cb0040",
        }
        for name, expected in original_hashes.items():
            with self.subTest(payload=name):
                self.assertEqual(
                    hashlib.sha256((folder / "phantom_regions_payloads" / name).read_bytes()).hexdigest(), expected
                )


if __name__ == "__main__":
    unittest.main()

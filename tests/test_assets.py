"""The original Tet4/Hex8 objects use the same portable geometry interface."""

import importlib.util
import unittest

import numpy as np
from fem_fixture import ASSETS

from newton_solver_fem import load_mesh


@unittest.skipUnless(importlib.util.find_spec("pxr") is not None, "Install the examples extra for USD assets")
class AssetTests(unittest.TestCase):
    def test_original_geometry_and_small_fixtures(self):
        expected = {
            "tet_box": ("tet", 216, 750, 300),
            "tet_sphere": ("tet", 163, 320, 320),
            "tet_baymax": ("tet", 8965, 38730, 9882),
            "hex_cube": ("hex", 216, 125, 300),
            "hex_pyramid": ("hex", 150, 80, 224),
            "hex_tire": ("hex", 320, 180, 480),
            "tet_grid": ("tet", 27, 40, 48),
            "hex_grid": ("hex", 27, 8, 48),
            "tet_cantilever": ("tet", 12, 10, 20),
        }
        for name, counts in expected.items():
            with self.subTest(asset=name):
                objects = load_mesh(ASSETS / f"{name}.usda")
                self.assertEqual(len(objects), 1)
                mesh = objects[0]
                self.assertEqual(
                    (mesh.element_type, len(mesh.vertices), len(mesh.indices), len(mesh.surface_triangles)), counts
                )
                self.assertTrue(np.isfinite(mesh.vertices).all())
                self.assertEqual(len(np.unique(mesh.indices)), len(mesh.vertices))
                self.assertEqual(set(vars(mesh)), {"name", "vertices", "indices", "surface_triangles"})
        # A ring, not a renamed regular cube; original hole is preserved.
        tire = load_mesh(ASSETS / "hex_tire.usda")[0]
        self.assertGreater(np.linalg.norm(tire.vertices[:, :2], axis=1).min(), 0.08)
        cantilever = load_mesh(ASSETS / "tet_cantilever.usda")[0]
        np.testing.assert_allclose(cantilever.vertices.min(axis=0), [0.0, 0.0, 0.3], atol=2e-8)
        np.testing.assert_allclose(cantilever.vertices.max(axis=0), [0.2, 0.1, 0.4], atol=2e-8)


if __name__ == "__main__":
    unittest.main()

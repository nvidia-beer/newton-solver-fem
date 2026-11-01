# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""The selection map follows visible geometry and exposes usable click targets."""

import unittest

import numpy as np

from newton_solver_fem import MeshRegion
from newton_solver_fem.examples._region_map import RegionMap, rasterize_regions


def _rectangle(x0, y0, x1, y1, height):
    return np.array(((x0, y0, height), (x1, y0, height), (x1, y1, height), (x0, y1, height)))


class RegionMapTests(unittest.TestCase):
    def test_display_orientation_rotates_region_targets_with_the_surface(self):
        points = np.concatenate((_rectangle(0, 0, 1, 1, 0), _rectangle(0, 0.7, 0.3, 1, 0.2)))
        triangles = np.array(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)))
        regions = tuple(MeshRegion(i, f"region_{i}", f"/Pad/{i}", (1, 1, 1), 10, 8) for i in (0, 1))
        region_map = RegionMap(points, triangles, (0, 0, 1, 1), regions, resolution=64)
        # The original top view is turned 180 degrees: this feature
        # moves from the original top-left to the displayed bottom-right.
        self.assertEqual(region_map.region_at(55, 55), 1)
        self.assertEqual(region_map.region_at(8, 8), 0)
        region_map.screen_rect = (20, 30, 128, 128)
        x, y = region_map.screen_position(1)
        self.assertGreater(x, 20 + 64)
        self.assertGreater(y, 30 + 64)
        self.assertEqual(region_map.region_at((x - 20) / 2, (y - 30) / 2), 1)

    def test_visible_owner_uses_interpolated_depth_not_triangle_order(self):
        points = np.concatenate((_rectangle(0, 0, 1, 1, 0.3), _rectangle(0, 0, 1, 1, 0)))
        points[4:, 2] = points[4:, 0]
        triangles = np.array(((0, 1, 2), (0, 2, 3), (4, 5, 6), (4, 6, 7)))
        labels = np.array((0, 0, 1, 1))
        owners, depth = rasterize_regions(points, triangles, labels, resolution=64)
        reverse, _ = rasterize_regions(points, triangles[::-1], labels[::-1], resolution=64)
        np.testing.assert_array_equal(owners, reverse)
        self.assertEqual(owners[32, 14], 0)
        self.assertEqual(owners[32, 50], 1)
        self.assertAlmostEqual(float(depth[32, 14]), 1.3, places=6)
        self.assertEqual(owners[0, 0], -1)

    def test_labels_are_inside_concave_region_and_screen_picking_is_consistent(self):
        # Region 1 surrounds a hole: its geometric centroid is in region 0.
        rectangles = (
            _rectangle(0, 0, 1, 1, 0),
            _rectangle(0.1, 0.1, 0.9, 0.3, 0.1),
            _rectangle(0.1, 0.7, 0.9, 0.9, 0.1),
            _rectangle(0.1, 0.3, 0.3, 0.7, 0.1),
            _rectangle(0.7, 0.3, 0.9, 0.7, 0.1),
        )
        points = np.concatenate(rectangles)
        triangles = np.concatenate([np.array(((0, 1, 2), (0, 2, 3))) + 4 * i for i in range(5)])
        regions = tuple(MeshRegion(i, f"region_{i}", f"/Pad/{i}", (1, 1, 1), 10, 8) for i in (0, 1))
        region_map = RegionMap(points, triangles, (0, 0, 1, 1, 1, 1, 1, 1, 1, 1), regions, resolution=64)
        self.assertEqual(region_map.region_at(32, 32), 0)
        self.assertIsNone(region_map.region_at(-0.1, 0))
        self.assertIsNone(region_map.region_at(64, 30))
        self.assertIsNone(region_map.region_at(0, 0))
        with self.assertRaisesRegex(RuntimeError, "Draw the region map"):
            region_map.screen_position(1)
        region_map.screen_rect = (40, 80, 128, 128)
        for region_id in (0, 1):
            self.assertEqual(region_map.region_at(*region_map.centers[region_id]), region_id)
            x, y = region_map.screen_position(region_id)
            self.assertEqual(region_map.region_at((x - 40) / 2, (y - 80) / 2), region_id)
        inactive, active = region_map.pixels([]), region_map.pixels([1])
        changed = np.any(inactive != active, axis=-1)
        self.assertTrue(changed.any())
        self.assertTrue(np.all(region_map.region_ids[changed] == 1))


if __name__ == "__main__":
    unittest.main()

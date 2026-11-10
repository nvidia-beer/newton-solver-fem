# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Independent deforming-bound/query/conditional-capture parity for refits."""

import os
import unittest
from unittest.mock import patch

import numpy as np
import warp as wp

from newton_solver_fem._src.geometry.mesh_refit import GraphMeshRefit


@wp.func_native("""
const wp::Mesh& m=*(const wp::Mesh*)id;
if(index<m.num_tris){lower.data[index]=m.lowers[index];upper.data[index]=m.uppers[index];}
if(index==0)mean.data[0]=m.average_edge_length;
""")
def _read_bounds(
    id: wp.uint64, index: int, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], mean: wp.array[float]
): ...


@wp.kernel(enable_backward=False)
def read_bounds(id: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], mean: wp.array[float]):
    _read_bounds(id, wp.tid(), lower, upper, mean)


@wp.func_native("""
const wp::BVH& b = ((const wp::Mesh*)id)->bvh;
if(index<b.max_nodes){
    const wp::BVHPackedNodeHalf& a=b.node_lowers[index];
    const wp::BVHPackedNodeHalf& c=b.node_uppers[index];
    lower.data[index]=wp::vec3(a.x,a.y,a.z);
    upper.data[index]=wp::vec3(c.x,c.y,c.z);
    links.data[index]=wp::vec4i(a.i,c.i,a.b,b.node_parents[index]);
}
if(index==0) root.data[0]=*b.root;
""")
def _read_tree(
    id: wp.uint64,
    index: int,
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    links: wp.array[wp.vec4i],
    root: wp.array[int],
): ...


@wp.kernel(enable_backward=False)
def read_tree(
    id: wp.uint64, lower: wp.array[wp.vec3], upper: wp.array[wp.vec3], links: wp.array[wp.vec4i], root: wp.array[int]
):
    _read_tree(id, wp.tid(), lower, upper, links, root)


def tree_bounds(mesh, nodes):
    low = wp.empty(nodes, dtype=wp.vec3, device=mesh.device)
    high = wp.empty_like(low)
    links = wp.empty(nodes, dtype=wp.vec4i, device=mesh.device)
    root = wp.empty(1, dtype=int, device=mesh.device)
    wp.launch(read_tree, nodes, [mesh.id, low, high, links, root], device=mesh.device)
    low, high, links = low.numpy(), high.numpy(), links.numpy()
    pending = [int(root.numpy()[0])]
    live = []
    while pending:
        index = pending.pop()
        assert index not in live and 0 <= index < nodes, (index, links)
        live.append(index)
        if not links[index, 2]:
            pending.extend(links[index, :2])
    return np.asarray(live), low[live], high[live], links[live]


@wp.kernel(enable_backward=False)
def query(mesh: wp.uint64, points: wp.array[wp.vec3], values: wp.array[wp.vec4], faces: wp.array[int]):
    i = wp.tid()
    hit = wp.mesh_query_point_sign_normal(mesh, points[i], 1.0)
    faces[i] = hit.face if hit.result else -1
    closest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
    values[i] = wp.vec4(hit.sign, closest[0], closest[1], closest[2])


@wp.kernel(enable_backward=False)
def decrement(value: wp.array[int]):
    value[0] -= 1


def cube(n):
    positions = []
    indices = []
    lookup = {}
    planes = [
        ((1, -1, -1), (0, 2, 0), (0, 0, 2)),
        ((-1, -1, -1), (0, 0, 2), (0, 2, 0)),
        ((-1, 1, -1), (0, 0, 2), (2, 0, 0)),
        ((-1, -1, -1), (2, 0, 0), (0, 0, 2)),
        ((-1, -1, 1), (2, 0, 0), (0, 2, 0)),
        ((-1, -1, -1), (0, 2, 0), (2, 0, 0)),
    ]
    for base, u, v in planes:
        local = {}
        for i in range(n + 1):
            for j in range(n + 1):
                p = tuple(np.array(base) + np.array(u) * i / n + np.array(v) * j / n)
                if p not in lookup:
                    lookup[p] = len(positions)
                    positions.append(p)
                local[i, j] = lookup[p]
        for i in range(n):
            for j in range(n):
                a, b, c, d = local[i, j], local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]
                indices.extend(((a, b, c), (a, c, d)))
    return np.asarray(positions, np.float32), np.asarray(indices, np.int32)


def measurements(mesh, count, points):
    low = wp.empty(count, dtype=wp.vec3, device=mesh.device)
    high = wp.empty_like(low)
    mean = wp.empty(1, dtype=float, device=mesh.device)
    wp.launch(read_bounds, count, [mesh.id, low, high, mean], device=mesh.device)
    values = wp.empty(len(points), dtype=wp.vec4, device=mesh.device)
    faces = wp.empty(len(points), dtype=int, device=mesh.device)
    wp.launch(query, len(points), [mesh.id, points, values, faces], device=mesh.device)
    return low.numpy(), high.numpy(), mean.numpy(), values.numpy(), faces.numpy()


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.device = wp.get_device(os.environ.get("NEWTON_TEST_DEVICE", "cpu"))
        if not self.device.is_cuda:
            self.skipTest("CUDA-only mesh refit")

    def test_deformed_bounds_mean_closest_points_and_edge_signs(self):
        rng = np.random.default_rng(1211)
        device = os.environ.get("NEWTON_TEST_DEVICE", "cuda:0")
        for n in (1, 4, 16):
            base, triangles = cube(n)
            initial = (base * 0.01 + [0.03, -0.04, 0.006]).astype(np.float32)
            points_a = wp.array(initial, dtype=wp.vec3, device=device)
            points_b = wp.array(initial, dtype=wp.vec3, device=device)
            ids = wp.array(triangles.reshape(-1), dtype=int, device=device)
            a = wp.Mesh(points_a, ids)
            b = wp.Mesh(points_b, ids)
            refit = GraphMeshRefit(b)
            condition = wp.ones(1, dtype=int, device=device)

            def body(refit=refit, condition=condition):
                refit.refit()
                wp.launch(decrement, 1, [condition], device=device)

            refit.refit()
            with wp.ScopedCapture(device=device) as cap:
                condition.fill_(2)
                wp.capture_while(condition, body)
            for step in range(5):
                q = base.astype(float)
                q[:, 0] += 0.07 * np.sin(q[:, 1] + step * 0.2) * q[:, 2]
                q[:, 1] += 0.04 * q[:, 0] ** 2
                current = (q * 0.01 + [0.03, -0.04, 0.006]).astype(np.float32)
                points_a.assign(current)
                points_b.assign(current)
                a.refit()
                wp.capture_launch(cap.graph)
                # Random volume samples plus exact faces, edge midpoints,
                # vertices and nanometre offsets on either side of edges.
                corners = current[triangles]
                mid = (corners[:, 0] + corners[:, 1]) * 0.5
                query_points = np.concatenate(
                    (
                        rng.uniform(initial.min(0) - 0.003, initial.max(0) + 0.003, (2000, 3)),
                        current,
                        mid,
                        mid + [0, 0, 1e-9],
                        mid - [0, 0, 1e-9],
                        mid + [1e-7, 0, 0],
                        mid - [1e-7, 0, 0],
                    )
                )
                probes = wp.array(query_points, dtype=wp.vec3, device=device)
                x, y = measurements(a, len(triangles), probes), measurements(b, len(triangles), probes)
                np.testing.assert_array_equal(x[0], y[0])
                np.testing.assert_array_equal(x[1], y[1])
                np.testing.assert_allclose(x[2], y[2], rtol=2e-6, atol=0)
                np.testing.assert_array_equal(x[3], y[3])
                np.testing.assert_array_equal(x[4], y[4])
                self.assertEqual(condition.numpy()[0], 0)
                custom_tree = tree_bounds(b, refit.nodes)
                b.refit()
                native_tree = tree_bounds(b, refit.nodes)
                for expected, actual in zip(native_tree, custom_tree, strict=True):
                    np.testing.assert_array_equal(expected, actual)

    def test_one_and_two_leaf_trees_update_bounds_and_mean(self):
        for count in (1, 2):
            # Separate triangles and leaf_size=1 exercise both a single root
            # leaf and the last-child parent handoff without a large tree.
            q = np.array([[j + x, y, 0] for j in range(count) for x, y in ((0, 0), (0.5, 0), (0, 0.5))], np.float32)
            mesh = wp.Mesh(
                wp.array(q, dtype=wp.vec3, device=self.device),
                wp.array(np.arange(3 * count), dtype=int, device=self.device),
                bvh_leaf_size=1,
            )
            refit = GraphMeshRefit(mesh)
            self.assertEqual(refit.leaves, count)
            probes = wp.array([[0.1, 0.1, 0.03]], dtype=wp.vec3, device=self.device)
            previous_mean = None
            for scale in (1.0, 2.0, 0.25):
                mesh.points.assign(q * scale)
                refit.refit()
                custom = measurements(mesh, count, probes)
                custom_tree = tree_bounds(mesh, refit.nodes)
                mesh.refit()
                native = measurements(mesh, count, probes)
                native_tree = tree_bounds(mesh, refit.nodes)
                for expected, actual in zip(native, custom, strict=True):
                    np.testing.assert_array_equal(expected, actual)
                for expected, actual in zip(native_tree, custom_tree, strict=True):
                    np.testing.assert_array_equal(expected, actual)
                if previous_mean is not None:
                    self.assertNotEqual(float(custom[2][0]), previous_mean)
                previous_mean = float(custom[2][0])

    def test_winding_mesh_rejected_without_mutation(self):
        q, triangles = cube(1)
        device = os.environ.get("NEWTON_TEST_DEVICE", "cuda:0")
        mesh = wp.Mesh(
            wp.array(q, dtype=wp.vec3, device=device),
            wp.array(triangles.reshape(-1), dtype=int, device=device),
            support_winding_number=True,
        )
        with self.assertRaisesRegex(ValueError, "winding"):
            GraphMeshRefit(mesh)

    def test_unsupported_version_and_replaced_storage_are_rejected(self):
        q, triangles = cube(1)
        mesh = wp.Mesh(
            wp.array(q, dtype=wp.vec3, device=self.device),
            wp.array(triangles.reshape(-1), dtype=int, device=self.device),
        )
        with patch.object(wp, "__version__", "future"):
            with self.assertRaisesRegex(ValueError, "validated Warp"):
                GraphMeshRefit(mesh)
        refit = GraphMeshRefit(mesh)
        mesh.points = wp.clone(mesh.points)
        with self.assertRaisesRegex(ValueError, "replacing mesh storage"):
            refit.refit()


if __name__ == "__main__":
    unittest.main()

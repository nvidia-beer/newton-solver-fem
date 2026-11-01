# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Prepare finite surface grips for the public implicit MaterialHolds API."""

import numpy as np
import warp as wp

from newton_solver_fem import MaterialHolds


def triangle_parents(tetrahedra, triangles):
    """Find the unique owning tet of each outward boundary triangle."""
    faces = tetrahedra[:, ((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1))].reshape(-1, 3)
    sorted_faces = np.sort(faces.astype(np.int64), axis=1)
    dtype = np.dtype([("a", "<i8"), ("b", "<i8"), ("c", "<i8")])
    keys = np.ascontiguousarray(sorted_faces).view(dtype).reshape(-1)
    query = np.ascontiguousarray(np.sort(np.asarray(triangles, dtype=np.int64), axis=1)).view(dtype).reshape(-1)
    order = np.argsort(keys)
    indices = np.searchsorted(keys[order], query)
    if np.any(indices == len(keys)) or np.any(keys[order[indices]] != query):
        raise ValueError("The grip triangles must belong to the selected tetrahedra")
    return order[indices] // 4


class TetSurfaceHolds(MaterialHolds):
    """Area-distributed grips on an unchanged, ordinary tetrahedral boundary.

    Only a mouse press reads current nodal coordinates. Motion and the implicit
    spring/damper evaluation use MaterialHolds' persistent device arrays. The
    source field is linear within each tet, so each sampled triangle point is
    represented exactly by four barycentric shape-function weights.
    """

    def __init__(self, device, tetrahedra, triangles, fixed_vertices, **kwargs):
        super().__init__(device, **kwargs)
        self.tetrahedra = np.asarray(tetrahedra, dtype=np.int32)
        self.triangles = np.asarray(triangles, dtype=np.int32)
        self.triangle_parents = triangle_parents(self.tetrahedra, self.triangles)
        self.free_triangles = ~np.isin(self.triangles, fixed_vertices).all(axis=1)

    def begin_surface(self, triangle, barycentric, point, q):
        """Attach a finite patch around a picked exterior triangle; return success."""
        self.cancel()
        if self.count == self.capacity or not self.free_triangles[triangle]:
            return False
        positions = q.numpy()
        vertices = positions[self.triangles]
        centers = vertices.mean(axis=1)
        cross = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
        twice_area = np.linalg.norm(cross, axis=1)
        normals = cross / np.maximum(twice_area[:, None], 1.0e-30)
        eligible = self.free_triangles & (np.linalg.norm(centers - point, axis=1) < self.radius)
        eligible &= (normals @ normals[triangle]) > 0.2
        eligible[triangle] = True
        candidates = np.flatnonzero(eligible)
        if len(candidates) > self.samples_per_grip:
            areas = twice_area[candidates]
            cumulative = np.cumsum(areas)
            quantiles = (np.arange(self.samples_per_grip - 1) + 0.5) / (self.samples_per_grip - 1)
            candidates = np.unique(np.r_[triangle, candidates[np.searchsorted(cumulative, quantiles * cumulative[-1])]])
        count = len(candidates)
        weights = np.zeros((self.samples_per_grip, 4), dtype=np.float32)
        nodes = np.zeros((self.samples_per_grip, 4), dtype=np.int32)
        offsets = np.zeros((self.samples_per_grip, 3), dtype=np.float32)
        fractions = np.zeros(self.samples_per_grip, dtype=np.float32)
        parents = np.zeros(self.samples_per_grip, dtype=np.int32)
        enabled = np.zeros(self.samples_per_grip, dtype=np.int32)
        for slot, face in enumerate(candidates):
            parents[slot] = self.triangle_parents[face]
            nodes[slot] = self.tetrahedra[parents[slot]]
            coordinates = barycentric if face == triangle else (1 / 3, 1 / 3, 1 / 3)
            for vertex, value in zip(self.triangles[face], coordinates, strict=True):
                weights[slot, np.flatnonzero(nodes[slot] == vertex)[0]] = value
            offsets[slot] = weights[slot] @ positions[nodes[slot]] - point
            enabled[slot] = 1
        fractions[:count] = twice_area[candidates] / twice_area[candidates].sum()
        start = self.count * self.samples_per_grip
        for destination, values, dtype in (
            (self.nodes, nodes, int),
            (self.weights, weights, wp.vec4),
            (self.offsets, offsets, wp.vec3),
            (self.fractions, fractions, float),
            (self.parents, parents, int),
            (self.enabled, enabled, int),
        ):
            source = wp.array(values, dtype=dtype, device=self.device)
            if destination.ndim == 2:
                wp.copy(destination.flatten(), source.flatten(), dest_offset=start * 4)
            else:
                wp.copy(destination, source, dest_offset=start)
        target = wp.array(np.asarray(point, dtype=np.float32).reshape(1, 3), dtype=wp.vec3, device=self.device)
        wp.copy(self.requested, target, dest_offset=self.count)
        wp.copy(self.driven, target, dest_offset=self.count)
        self.active = self.count
        self.count += 1
        return True

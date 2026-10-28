# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Convex material fragments and finite knife patches for affine tetrahedra.

Fragments retain the original parent barycentrics. Auxiliary prism faces
preserve material continuity; only blade faces sever material connections.
"""

from dataclasses import dataclass
from math import sqrt

import numpy as np

_FACES = ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))


@dataclass(frozen=True)
class FragmentFace:
    """An outward polygon and its persistent material-interface identity."""

    indices: tuple[int, ...]
    interface: tuple[int, ...]
    kind: str


@dataclass(frozen=True)
class TetrahedralFragment:
    """Convex material region in one original tetrahedron's barycentrics."""

    parent: int
    vertices: np.ndarray
    faces: tuple[FragmentFace, ...]

    def __post_init__(self):
        vertices = np.array(self.vertices, dtype=np.float64, copy=True)
        vertices.flags.writeable = False
        object.__setattr__(self, "vertices", vertices)


def make_tetrahedral_fragments(elements: np.ndarray) -> tuple[TetrahedralFragment, ...]:
    """Create one uncut fragment per original positive tetrahedron."""
    elements = np.asarray(elements)
    if elements.ndim != 2 or elements.shape[1] != 4 or not np.issubdtype(elements.dtype, np.integer):
        raise ValueError("elements must be an integer (element_count, 4) array")
    if np.any(elements < 0) or any(len(set(row)) != 4 for row in elements):
        raise ValueError("each element must contain four distinct nonnegative node IDs")
    return tuple(
        TetrahedralFragment(
            parent,
            np.eye(4),
            tuple(FragmentFace(face, (-1, *sorted(map(int, nodes[list(face)]))), "original") for face in _FACES),
        )
        for parent, nodes in enumerate(elements)
    )


def _polygon_normal(points):
    """Use the widest triangle so a projected near edge cannot tilt the plane."""
    normal, largest = np.zeros(3), 0.0
    for a in range(len(points) - 2):
        for b in range(a + 1, len(points) - 1):
            for c in range(b + 1, len(points)):
                candidate = np.cross(points[b] - points[a], points[c] - points[b])
                area = np.linalg.norm(candidate)
                if area > largest:
                    normal, largest = candidate, area
    return normal / largest if largest else normal


def _knife_polygons(triangles, tolerance):
    """Combine the two coplanar triangles of a convex swept quad when possible."""
    remaining, result = list(range(len(triangles))), []
    while remaining:
        first = remaining.pop(0)
        triangle = triangles[first]
        polygon = triangle
        for second in remaining:
            other = triangles[second]
            shared = np.linalg.norm(triangle[:, None] - other[None], axis=2) <= tolerance
            if np.count_nonzero(shared) != 2:
                continue
            points = np.vstack((triangle, other[~shared.any(axis=0)]))
            normal = _polygon_normal(points)
            reference = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
            if normal @ reference < 0:
                normal = -normal
            # Screen projections span the whole camera depth. Their rounding
            # scale is independent of the much smaller material mesh scale.
            roundoff = 64 * np.finfo(float).eps * np.max(np.abs(points))
            if np.max(np.abs((points - points[0]) @ normal)) > max(tolerance, roundoff):
                continue
            center = points.mean(axis=0)
            edges = np.roll(points, -1, axis=0) - points
            axis = edges[np.argmax(np.linalg.norm(edges, axis=1))].copy()
            axis /= np.linalg.norm(axis)
            second_axis = np.cross(normal, axis)
            angles = np.arctan2((points - center) @ second_axis, (points - center) @ axis)
            candidate = points[np.argsort(angles)]
            turns = [
                np.cross(candidate[(i + 1) % 4] - candidate[i], candidate[(i + 2) % 4] - candidate[(i + 1) % 4])
                @ normal
                for i in range(4)
            ]
            if min(turns) > tolerance**2:
                polygon = candidate
                remaining.remove(second)
                break
        result.append(polygon)
    return result


def _polygon_overlap_2d(first, second, tolerance):
    """Small convex polygons; scalar arithmetic avoids per-edge NumPy launches."""
    subject, clip = first.tolist(), second.tolist()
    if any(
        min(point[axis] for point in subject) >= max(point[axis] for point in clip) - tolerance
        or min(point[axis] for point in clip) >= max(point[axis] for point in subject) - tolerance
        for axis in (0, 1)
    ):
        return 0.0

    def signed_area(points):
        origin = points[0]
        return sum(
            (a[0] - origin[0]) * (b[1] - origin[1]) - (a[1] - origin[1]) * (b[0] - origin[0])
            for a, b in zip(points, points[1:] + points[:1], strict=True)
        )

    if signed_area(clip) < 0:
        clip.reverse()
    for first, following in zip(clip, clip[1:] + clip[:1], strict=True):
        nx, ny = first[1] - following[1], following[0] - first[0]
        edge_tolerance = tolerance * sqrt(nx * nx + ny * ny)
        output = []
        previous = subject[-1]
        previous_distance = (previous[0] - first[0]) * nx + (previous[1] - first[1]) * ny
        if abs(previous_distance) <= edge_tolerance:
            previous_distance = 0.0
        for current in subject:
            distance = (current[0] - first[0]) * nx + (current[1] - first[1]) * ny
            if abs(distance) <= edge_tolerance:
                distance = 0.0
            if (distance >= 0.0) != (previous_distance >= 0.0):
                t = previous_distance / (previous_distance - distance)
                output.append(
                    (previous[0] + t * (current[0] - previous[0]), previous[1] + t * (current[1] - previous[1]))
                )
            if distance >= 0.0:
                output.append(current)
            previous, previous_distance = current, distance
        if len(output) < 3:
            return 0.0
        subject = output
    return abs(signed_area(subject)) / 2

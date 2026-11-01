# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Selective loading and conforming union of labelled tetrahedral USD payloads."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .mesh import _indices, _vertices


@dataclass(frozen=True)
class MeshRegion:
    """An unloaded region catalog entry; ``color`` is sRGB in [0, 1]."""

    id: int
    name: str
    path: str
    color: tuple[float, float, float]
    tetrahedron_count: int
    vertex_count: int


@dataclass(frozen=True)
class RegionMesh:
    """Selected Tet4 union in world-space metres, with Z up.

    Shared original vertex IDs are welded before the boundary is computed.
    ``fixed_vertex_indices`` identifies cuts against *unselected* regions;
    callers choose whether to constrain those nodes. No constraints, material,
    mass, or solver state are created by this geometry reader. Surface region
    IDs identify the owning tetrahedron, including newly exposed cut faces.
    """

    vertices: np.ndarray
    indices: np.ndarray
    surface_triangles: np.ndarray
    tetrahedron_region_ids: np.ndarray
    surface_region_ids: np.ndarray
    exterior_surface_triangles: np.ndarray
    exterior_surface_region_ids: np.ndarray
    fixed_vertex_indices: np.ndarray
    global_vertex_indices: np.ndarray
    global_tetrahedron_indices: np.ndarray
    regions: tuple[MeshRegion, ...]


def _open(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        from pxr import Usd
    except ModuleNotFoundError as error:
        raise ImportError("Region loading requires 'newton-solver-fem[examples]'") from error
    stage = Usd.Stage.Open(str(path), Usd.Stage.LoadNone)
    if stage is None:
        raise ValueError(f"Cannot open USD stage {path}")
    return stage


def _catalog(stage):
    root = stage.GetDefaultPrim()
    raw = root.GetAttribute("suturebot:regionManifest").Get() if root else None
    if raw is None:
        raise ValueError("USD default prim must contain suturebot:regionManifest")
    regions = tuple(
        MeshRegion(
            int(item["id"]),
            str(item["name"]),
            str(item["prim_path"]),
            tuple(float(value) / 255.0 for value in item["color_srgb"]),
            int(item["tetrahedron_count"]),
            int(item["vertex_count"]),
        )
        for item in json.loads(raw)
    )
    if len({region.id for region in regions}) != len(regions):
        raise ValueError("The region manifest contains duplicate IDs")
    for region in regions:
        if not stage.GetPrimAtPath(region.path):
            raise ValueError(f"Region {region.id} has no prim at {region.path}")
        if len(region.color) != 3 or not all(0 <= value <= 1 for value in region.color):
            raise ValueError(f"Region {region.id} has an invalid sRGB color")
    return regions


def list_mesh_regions(path: str | Path) -> tuple[MeshRegion, ...]:
    """Read the catalog without loading any region's volume payload."""
    return _catalog(_open(path))


def _ids(prim, attribute, count):
    values = np.asarray(prim.GetAttribute(attribute).Get())
    if values.shape != (count,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError(f"{prim.GetPath()}: invalid {attribute}")
    if np.any(values < 0) or len(np.unique(values)) != count:
        raise ValueError(f"{prim.GetPath()}: {attribute} must contain unique nonnegative IDs")
    return values.astype(np.int64)


def _keys(faces):
    rows = np.ascontiguousarray(np.sort(faces, axis=1), dtype=np.int64)
    return rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).reshape(-1)


def _boundary(tets):
    """Vectorized outward boundary with duplicate/nonmanifold checks."""
    if len(np.unique(np.sort(tets, axis=1), axis=0)) != len(tets):
        raise ValueError("Duplicate tetrahedra in the selected union")
    faces = tets[:, ((0, 2, 1), (0, 1, 3), (1, 2, 3), (2, 0, 3))].reshape(-1, 3)
    _, first, inverse, counts = np.unique(_keys(faces), return_index=True, return_inverse=True, return_counts=True)
    if np.any(counts > 2):
        raise ValueError("Selected tetrahedra have a nonmanifold face")
    parity = (faces[:, 0] > faces[:, 1]) ^ (faces[:, 0] > faces[:, 2]) ^ (faces[:, 1] > faces[:, 2])
    orientation_sum = np.bincount(inverse, weights=np.where(parity, -1, 1))
    if np.any(orientation_sum[counts == 2] != 0):
        raise ValueError("Selected tetrahedra have inconsistent interior face winding")
    selected = first[counts == 1]
    return faces[selected], selected // 4


def _check_boundary_manifold(boundary, selected_ids):
    """Reject open edges and disconnected incident face fans at a vertex.

    For an oriented closed triangle surface, following each corner's outgoing
    edge into its neighboring face cycles around that vertex. A manifold
    vertex has exactly one such cycle. Pointer doubling finds the minimum
    corner in each cycle without a Python traversal per triangle.
    """
    if not len(boundary):
        return
    edges = boundary[:, ((0, 1), (1, 2), (2, 0))].reshape(-1, 2)
    keys = _keys(edges)
    _, counts = np.unique(keys, return_counts=True)
    prefix = f"Region selection {list(selected_ids)} has a nonmanifold boundary"
    if np.any(counts != 2):
        raise ValueError(
            f"{prefix}: {np.count_nonzero(counts != 2)} edges do not have exactly two incident triangles. "
            "Choose a different region combination; no simulation mesh was created."
        )
    order = np.argsort(keys)
    mate = np.empty(len(edges), dtype=np.int64)
    mate[order[::2]] = order[1::2]
    mate[order[1::2]] = order[::2]
    # Corner i is the tail of edge i; the neighboring edge's head is the
    # matching corner in the adjacent triangle when winding is consistent.
    jump = (mate // 3) * 3 + (mate % 3 + 1) % 3
    corners = boundary.reshape(-1)
    if not np.array_equal(corners, corners[jump]):
        raise ValueError(f"{prefix}: inconsistent triangle winding")
    labels = np.arange(len(corners), dtype=np.int64)
    for _ in range((len(corners) - 1).bit_length()):
        labels = np.minimum(labels, labels[jump])
        jump = jump[jump]
    _, fan_counts = np.unique(corners[np.unique(labels)], return_counts=True)
    if np.any(fan_counts > 1):
        raise ValueError(
            f"{prefix}: {np.count_nonzero(fan_counts > 1)} vertices have disconnected surface fans. "
            "Choose a different region combination; no simulation mesh was created."
        )


def load_mesh_regions(path: str | Path, region_ids=None) -> RegionMesh:
    """Load only requested payloads and return one welded tetrahedral union.

    ``None`` selects the stage's active regions. Explicit IDs override active
    opinions in an in-memory session layer without modifying the asset. An
    empty iterable returns an empty mesh. Visibility does not select physics.
    Positions prefer the exact ``newton:referencePoints`` embedding; standard
    USD transforms, units and Y/Z up axes are applied. The authored regional
    boundary and cut labels are checked against the selected tetrahedra.
    A selected union with a nonmanifold boundary edge or a disconnected
    incident surface fan at a vertex raises ValueError before returning.
    """
    from pxr import Usd, UsdGeom

    stage = _open(path)
    catalog = _catalog(stage)
    if region_ids is None:
        wanted = {region.id for region in catalog if stage.GetPrimAtPath(region.path).IsActive()}
    else:
        wanted = set(region_ids)
        if not wanted.issubset({region.id for region in catalog}):
            raise ValueError(f"Unknown region IDs: {sorted(wanted - {region.id for region in catalog})}")
    selected = tuple(region for region in catalog if region.id in wanted)
    stage.SetEditTarget(stage.GetSessionLayer())
    for region in catalog:
        stage.GetPrimAtPath(region.path).SetActive(region.id in wanted)
    units = float(UsdGeom.GetStageMetersPerUnit(stage))
    up = str(UsdGeom.GetStageUpAxis(stage))
    if not np.isfinite(units) or units <= 0 or up not in ("Y", "Z"):
        raise ValueError("The USD stage must have positive units and Y or Z up")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    points, node_ids, cells, cell_ids, owners, authored, cut_labels = [], [], [], [], [], [], []
    for region in selected:
        stage.Load(region.path)
        prim = stage.GetPrimAtPath(region.path + "/Volume")
        if not prim or prim.GetTypeName() != "TetMesh":
            raise ValueError(f"Region {region.id}: missing TetMesh Volume payload")
        values = prim.GetAttribute("newton:referencePoints").Get()
        if values is None:
            values = prim.GetAttribute("points").Get()
        vertices = _vertices(values)
        matrix = np.asarray(transforms.GetLocalToWorldTransform(prim), dtype=np.float64)
        vertices = _vertices((vertices @ matrix[:3, :3] + matrix[3, :3]) * units)
        if up == "Y":
            vertices = vertices[:, [0, 2, 1]] * [1, -1, 1]
        tets = _indices(prim.GetAttribute("tetVertexIndices").Get(), 4, len(vertices)).copy()
        if len(tets) != region.tetrahedron_count or len(vertices) != region.vertex_count:
            raise ValueError(f"Region {region.id}: manifest counts do not match payload")
        edges = vertices[tets[:, 1:]] - vertices[tets[:, :1]]
        determinants = np.linalg.det(edges)
        scale = np.prod(np.linalg.norm(edges, axis=2), axis=1)
        if np.any(~np.isfinite(determinants)) or np.any(np.abs(determinants) <= 1e-12 * scale):
            raise ValueError(f"Region {region.id}: degenerate reference tetrahedron")
        inverted = determinants < 0
        tets[inverted] = tets[inverted][:, [0, 2, 1, 3]]
        gids = _ids(prim, "suturebot:globalVertexIndices", len(vertices))
        tids = _ids(prim, "suturebot:globalTetrahedronIndices", len(tets))
        boundary = _indices(prim.GetAttribute("surfaceFaceVertexIndices").Get(), 3, len(vertices))
        cuts = np.asarray(prim.GetAttribute("suturebot:boundaryIsCutInterface").Get())
        if cuts.shape != (len(boundary),) or cuts.dtype != np.bool_:
            raise ValueError(f"Region {region.id}: invalid boundary cut flags")
        expected, _ = _boundary(tets)
        if not np.array_equal(np.sort(_keys(boundary)), np.sort(_keys(expected))):
            raise ValueError(f"Region {region.id}: authored boundary differs from tetrahedral boundary")
        points.append(vertices)
        node_ids.append(gids)
        cells.append(gids[tets])
        cell_ids.append(tids)
        owners.append(np.full(len(tets), region.id, dtype=np.int32))
        authored.append(gids[boundary])
        cut_labels.append(cuts)
    if not selected:
        empty = np.empty(0, dtype=np.int32)
        faces = np.empty((0, 3), dtype=np.int32)
        return RegionMesh(
            np.empty((0, 3)),
            np.empty((0, 4), dtype=np.int32),
            faces,
            empty,
            empty,
            faces,
            empty,
            empty,
            empty.astype(np.int64),
            empty.astype(np.int64),
            selected,
        )
    gids, first, inverse = np.unique(np.concatenate(node_ids), return_index=True, return_inverse=True)
    all_points = np.concatenate(points)
    vertices = all_points[first]
    if not np.array_equal(all_points, vertices[inverse]):
        raise ValueError("Shared global vertex IDs have conflicting reference coordinates")
    tids = np.concatenate(cell_ids)
    if len(np.unique(tids)) != len(tids):
        raise ValueError("Regions contain overlapping global tetrahedron IDs")
    order = np.argsort(tids)
    tets = np.searchsorted(gids, np.concatenate(cells)[order]).astype(np.int32)
    region_ids = np.concatenate(owners)[order]
    boundary, boundary_owner = _boundary(tets)
    _check_boundary_manifold(boundary, (region.id for region in selected))
    source_keys = _keys(np.concatenate(authored))
    source_order = np.argsort(source_keys)
    unique_keys, source_first, source_counts = np.unique(source_keys, return_index=True, return_counts=True)
    source_cuts = np.concatenate(cut_labels)
    duplicate = np.repeat(source_counts, source_counts) == 2
    if np.any(source_counts > 2) or np.any(~source_cuts[source_order[duplicate]]):
        raise ValueError("Only two matching cut faces may disappear inside a selected union")
    boundary_keys = _keys(gids[boundary])
    if not np.array_equal(np.sort(boundary_keys), unique_keys[source_counts == 1]):
        raise ValueError("Regional boundaries do not cover the selected tetrahedral union")
    cuts = source_cuts[source_first[np.searchsorted(unique_keys, boundary_keys)]]
    surface_regions = region_ids[boundary_owner]
    return RegionMesh(
        vertices,
        tets,
        boundary,
        region_ids,
        surface_regions,
        boundary[~cuts],
        surface_regions[~cuts],
        np.unique(boundary[cuts]).astype(np.int32),
        gids,
        tids[order],
        selected,
    )

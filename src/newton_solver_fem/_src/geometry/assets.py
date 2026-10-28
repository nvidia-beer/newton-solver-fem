# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Load static Tet4 and Hex8 example assets from USD.

Coordinates are returned exactly as authored. Materials and pneumatic metadata are never loaded.
"""

import numpy as np


def _stage(path):
    from pxr import Usd

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Cannot open USD geometry {path}")
    return stage


def _tet(prim):
    return {
        "vertices": np.asarray(prim.GetAttribute("points").Get(), dtype=np.float32).reshape(-1, 3),
        "tets": np.asarray(prim.GetAttribute("tetVertexIndices").Get(), dtype=np.int32).reshape(-1, 4),
        "surface_triangles": np.asarray(prim.GetAttribute("surfaceFaceVertexIndices").Get(), dtype=np.int32).reshape(
            -1, 3
        ),
    }


def load_asset(path):
    """Read the first authored Tet4 geometry without material metadata."""
    stage = _stage(path)
    for prim in stage.Traverse():
        if prim.GetTypeName() == "TetMesh":
            return _tet(prim)
    raise ValueError(f"No tetrahedral geometry in {path}")


def load_hex_assembly(path):
    """Read authored Q1 Hex8 geometry and boundary triangles."""
    stage = _stage(path)
    objects = []
    for prim in stage.Traverse():
        data = prim.GetCustomDataByKey("hex_mesh")
        if prim.GetTypeName() != "Mesh" or not data:
            continue
        counts = np.asarray(prim.GetAttribute("faceVertexCounts").Get())
        if np.any(counts != 3):
            raise ValueError("Hex8 boundary must be triangulated")
        objects.append(
            {
                "name": str(prim.GetName()),
                "vertices": np.asarray(prim.GetAttribute("points").Get(), dtype=np.float32).reshape(-1, 3),
                "hex_indices": np.asarray(data["hex_vertex_indices"], dtype=np.int32).reshape(-1, 8),
                "surface_triangles": np.asarray(prim.GetAttribute("faceVertexIndices").Get(), dtype=np.int32).reshape(
                    -1, 3
                ),
            }
        )
    return objects


def detect_asset_type(path):
    """Return ``tet`` or ``hex`` from static USD geometry."""
    stage = _stage(path)
    for prim in stage.Traverse():
        if prim.GetCustomDataByKey("hex_mesh"):
            return "hex"
        if prim.GetTypeName() == "TetMesh":
            return "tet"
    raise ValueError(f"No supported FEM geometry in {path}")

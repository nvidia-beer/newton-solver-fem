# SPDX-License-Identifier: Apache-2.0
"""Sparse differentiable FEM and material cutting for Newton."""

from ._src.cutting import (
    CutConnectivityDebug,
    CutModel,
    CutSurfaceContact,
    CutVolumeGuard,
    LinearTetCut,
    MaterialHolds,
    TetrahedralTunnelGuide,
)
from ._src.geometry import HexMesh, Mesh, boundary_faces, hex_quadrature, load_mesh
from ._src.geometry.assets import detect_asset_type, load_asset, load_hex_assembly
from ._src.solvers import (
    FEMBuilder,
    LinearFEMData,
    SolverFEM,
    SolverLinearFEM,
)

__all__ = [
    "HexMesh",
    "Mesh",
    "load_mesh",
    "boundary_faces",
    "hex_quadrature",
    "SolverFEM",
    "FEMBuilder",
    "LinearFEMData",
    "SolverLinearFEM",
    "detect_asset_type",
    "load_asset",
    "load_hex_assembly",
    "CutSurfaceContact",
    "CutModel",
    "CutVolumeGuard",
    "MaterialHolds",
    "LinearTetCut",
    "TetrahedralTunnelGuide",
    "CutConnectivityDebug",
]

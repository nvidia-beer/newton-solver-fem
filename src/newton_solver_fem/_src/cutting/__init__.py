# SPDX-License-Identifier: Apache-2.0
"""Persistent material cutting, contact and holds for sparse affine FEM."""

from .contact import CutSurfaceContact
from .debug import CutConnectivityDebug
from .holds import MaterialHolds
from .model import CutModel
from .tetrahedral_cut import LinearTetCut
from .tunnel import TetrahedralTunnelGuide
from .volume_guard import CutVolumeGuard

__all__ = [
    "CutSurfaceContact",
    "CutModel",
    "CutVolumeGuard",
    "MaterialHolds",
    "LinearTetCut",
    "TetrahedralTunnelGuide",
    "CutConnectivityDebug",
]

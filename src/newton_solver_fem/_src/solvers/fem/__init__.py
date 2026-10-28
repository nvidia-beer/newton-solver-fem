# SPDX-License-Identifier: Apache-2.0
"""Sparse solid FEM with differentiable integration."""

from .builder import FEMBuilder, LinearFEMData
from .solver import SolverFEM

# Compatibility name; both imports construct the same solver.
SolverLinearFEM = SolverFEM

__all__ = ["FEMBuilder", "LinearFEMData", "SolverFEM", "SolverLinearFEM"]

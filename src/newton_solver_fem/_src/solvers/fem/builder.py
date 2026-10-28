# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Public Newton builder extension with explicit, package-owned Q1 Hex8 data."""

from dataclasses import dataclass

import newton
import numpy as np
import warp as wp


@dataclass
class LinearFEMData:
    """Hex8 reference geometry and materials stored under ``model.fem``."""

    hex_count: int
    hex_indices: wp.array
    hex_materials: wp.array
    hex_inv_J0: wp.array
    hex_det_J0_w: wp.array


class FEMBuilder(newton.ModelBuilder):
    """Newton 1.7 builder with Q1 Hex8 quadrature.

    Add particles with public Newton methods, then add connectivity with global
    indices. Mass remains the caller's responsibility.
    Finalization stores an explicit ``LinearFEMData`` object under ``model.fem``.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fem_hex_indices = []
        self.fem_hex_materials = []

    def add_builder(self, builder, xform=None, label_prefix=None):
        """Copy public Newton entities and remap package-owned Hex8 indices."""
        indices = getattr(builder, "fem_hex_indices", [])
        if indices and xform is not None:
            rotation = np.asarray(wp.transform_get_rotation(xform))
            if not np.allclose(rotation[:3], 0.0, atol=1e-7) or not np.isclose(abs(rotation[3]), 1.0):
                raise ValueError(
                    "Newton 1.7 particle replication supports translation only; rotate Hex8 rest vertices before adding them"
                )
        offset = self.particle_count
        super().add_builder(builder, xform=xform, label_prefix=label_prefix)
        if indices:
            self.fem_hex_indices.extend((np.asarray(indices) + offset).tolist())
            self.fem_hex_materials.extend([list(values) for values in builder.fem_hex_materials])

    def add_soft_mesh(self, **kwargs):
        """Add Tet4 geometry with a unique-edge spring network.

        Each unique valid tetrahedral edge receives one spring with
        stiffness ``k_mu/2`` and damping ``k_damp/2``. Stock Newton does not;
        reproduce that mechanics explicitly through its public ``add_spring``.
        Cutting constructs occupied tetrahedra separately and uses no springs.
        """
        first_tet = self.tet_count
        super().add_soft_mesh(**kwargs)
        mesh = kwargs.get("mesh")
        mu = kwargs.get("k_mu")
        damping = kwargs.get("k_damp")
        if mu is None:
            mu = mesh.k_mu if mesh is not None else self.default_tet_k_mu
        if damping is None:
            damping = mesh.k_damp if mesh is not None else self.default_tet_k_damp
        mu, damping = float(mu), float(damping)
        added = set()
        points = np.asarray(self.particle_q)
        for nodes in self.tet_indices[first_tet:]:
            a, b, c, d = nodes
            if np.linalg.det((points[[b, c, d]] - points[a]).T) <= 0.0:
                continue
            for left, right in ((a, b), (a, c), (a, d), (b, c), (b, d), (c, d)):
                key = (min(left, right), max(left, right))
                if key not in added:
                    added.add(key)
                    self.add_spring(key[0], key[1], mu * 0.5, damping * 0.5, 0.0)

    def add_soft_hex_mesh(self, hex_indices, k_mu=1.0e3, k_lambda=1.0e3, k_damp=0.0):
        indices = np.asarray(hex_indices, dtype=np.int32)
        if indices.ndim != 2 or indices.shape[1] != 8:
            raise ValueError("hex_indices must have shape (H, 8)")
        if indices.size and (indices.min() < 0 or indices.max() >= self.particle_count):
            raise ValueError("Hex8 indices must reference existing builder particles")
        materials = np.column_stack(
            [np.broadcast_to(np.asarray(v, dtype=np.float32), (len(indices),)) for v in (k_mu, k_lambda, k_damp)]
        )
        if not np.isfinite(materials).all() or np.any(materials < 0):
            raise ValueError("Hex8 materials must be finite and nonnegative")
        self.fem_hex_indices.extend(indices.tolist())
        self.fem_hex_materials.extend(materials.tolist())

    def finalize(self, *args, **kwargs):
        model = super().finalize(*args, **kwargs)
        indices = np.asarray(self.fem_hex_indices, dtype=np.int32).reshape(-1, 8)
        signs = np.asarray(
            [[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]],
            dtype=np.float64,
        )
        rest = np.asarray(self.particle_q, dtype=np.float64)
        inverses = np.zeros((len(indices), 8, 3, 3), dtype=np.float64)
        determinants = np.zeros((len(indices), 8), dtype=np.float64)
        for element, nodes in enumerate(indices):
            for point in range(8):
                xi, eta, zeta = signs[point] / np.sqrt(3.0)
                jacobian = np.zeros((3, 3), dtype=np.float64)
                for corner, (xa, ya, za) in enumerate(signs):
                    gradient = (
                        np.asarray(
                            [
                                xa * (1 + ya * eta) * (1 + za * zeta),
                                (1 + xa * xi) * ya * (1 + za * zeta),
                                (1 + xa * xi) * (1 + ya * eta) * za,
                            ]
                        )
                        * 0.125
                    )
                    jacobian += np.outer(rest[nodes[corner]], gradient)
                det = np.linalg.det(jacobian)
                if det <= 1.0e-12:
                    raise ValueError(f"Degenerate or inverted Hex8 {element}, Gauss point {point}: {det}")
                inverses[element, point] = np.linalg.inv(jacobian)
                determinants[element, point] = abs(det)
        model.fem = LinearFEMData(
            len(indices),
            wp.array(indices, dtype=int, device=model.device),
            wp.array(
                np.asarray(self.fem_hex_materials, dtype=np.float32).reshape(-1, 3), dtype=float, device=model.device
            ),
            wp.array(inverses.astype(np.float32), dtype=wp.mat33, device=model.device),
            wp.array(determinants.astype(np.float32), dtype=float, device=model.device),
        )
        return model

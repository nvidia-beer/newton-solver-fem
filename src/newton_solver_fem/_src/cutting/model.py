# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Integrate clipped material domains with SolverFEM's affine basis."""

from copy import copy

import newton
import numpy as np
import warp as wp

_ACTIVE = int(newton.ParticleFlags.ACTIVE)


@wp.kernel(enable_backward=False)
def _fragment_material(
    parents: wp.array[int],
    nodes: wp.array2d[int],
    fractions: wp.array[float],
    mass_integrals: wp.array[wp.vec4],
    count: int,
    original_elements: wp.array2d[int],
    original_poses: wp.array[wp.mat33],
    density: float,
    material: wp.vec3,
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    volume_weights: wp.array[float],
    masses: wp.array[float],
):
    fragment = wp.tid()
    if fragment >= count:
        volume_weights[fragment] = 0.0
        # Valid geometry avoids invalid gathers even in kernels that inspect
        # the reserved slots before testing their zero material coefficients.
        poses[fragment] = original_poses[0]
        for corner in range(4):
            indices[fragment, corner] = original_elements[0, corner]
        for component in range(3):
            materials[fragment, component] = 0.0
        return
    poses[fragment] = original_poses[parents[fragment]]
    volume_weights[fragment] = fractions[fragment]
    for component in range(3):
        materials[fragment, component] = fractions[fragment] * material[component]
    for corner in range(4):
        node = nodes[fragment, corner]
        indices[fragment, corner] = node
        wp.atomic_add(masses, node, density * mass_integrals[fragment][corner])


@wp.kernel(enable_backward=False)
def _active_mass(
    physical_mass: wp.array[float],
    masses: wp.array[float],
    inverse_masses: wp.array[float],
    flags: wp.array[int],
):
    node = wp.tid()
    mass = physical_mass[node]
    if mass > 0.0:
        masses[node] = mass
        inverse_masses[node] = 1.0 / mass
        flags[node] = _ACTIVE
    else:
        # Unused pool rows need an invertible matrix diagonal, but must not
        # receive gravity or count towards physical mass.
        masses[node] = 1.0
        inverse_masses[node] = 0.0
        flags[node] = 0


class CutModel:
    """Build an independent physics snapshot after a validated geometry edit.

    Four virtual parent corners represent an affine field on each clipped
    fragment. Its strain is constant, so its elastic integral is the parent
    tetrahedron's integral multiplied by the retained volume fraction. Nodal
    masses integrate the basis over the actual fragment; uniform quarter-mass
    assignment would incorrectly move its center of mass.

    Geometry and basis arrays remain device resident. ``build(capacity=...)``
    reserves a fixed tetrahedron capacity; ``update`` and ``copy_into`` preserve
    the live model's array addresses so its simulation graph remains valid.
    ``cut_fragment_count`` marks the active prefix, while ``tet_count`` is the
    fixed capacity. The base model contains only the reserved particle pool.
    """

    def __init__(
        self,
        base_model: newton.Model,
        original_positions: np.ndarray,
        original_elements: np.ndarray,
        density: float,
        young_modulus: float,
        poisson_ratio: float = 0.3,
        damping: float = 0.0,
    ):
        if any(
            (base_model.tet_count, getattr(base_model, "hex_count", 0), base_model.tri_count, base_model.spring_count)
        ):
            raise ValueError("CutModel requires a particle-only base model")
        points = np.asarray(original_positions, dtype=np.float64)
        elements = np.asarray(original_elements, dtype=np.int32)
        parameters = np.asarray((density, young_modulus, poisson_ratio, damping))
        if not np.isfinite(parameters).all() or density <= 0 or young_modulus <= 0 or damping < 0:
            raise ValueError("Material parameters must be finite with positive density and modulus")
        if not 0 < poisson_ratio < 0.5:
            raise ValueError("The Linear FEM material requires a Poisson ratio in (0, 0.5)")
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("original_positions must be finite (N, 3)")
        if (
            elements.ndim != 2
            or elements.shape[1] != 4
            or not len(elements)
            or np.any(elements < 0)
            or np.any(elements >= len(points))
        ):
            raise ValueError("original_elements must contain valid tetrahedral indices")
        tetrahedra = points[elements]
        rest_edges = (tetrahedra[:, 1:] - tetrahedra[:, :1]).transpose(0, 2, 1)
        if np.any(np.linalg.det(rest_edges) <= 0):
            raise ValueError("Original tetrahedra must have positive volume")
        self.base_model, self.device = base_model, base_model.device
        self.original_elements = wp.array(elements, dtype=int, device=self.device)
        self.original_poses = wp.array(np.linalg.inv(rest_edges), dtype=wp.mat33, device=self.device)
        self.density = float(density)
        self.material = wp.vec3(
            young_modulus / (2 * (1 + poisson_ratio)),
            young_modulus * poisson_ratio / ((1 + poisson_ratio) * (1 - 2 * poisson_ratio)),
            damping,
        )

    def _validate_fragments(self, parents, fragment_nodes, fractions, mass_integrals, count, capacity):
        if not isinstance(count, int) or count < 0 or count > parents.shape[0]:
            raise ValueError("count must select a nonnegative number of resident fragments")
        if not isinstance(capacity, int) or capacity < count:
            raise ValueError("active fragment count exceeds reserved tetrahedron capacity")
        if (
            parents.ndim != 1
            or parents.dtype != wp.int32
            or fragment_nodes.shape[1:] != (4,)
            or fragment_nodes.dtype != wp.int32
            or fractions.ndim != 1
            or fractions.dtype != wp.float32
            or mass_integrals.ndim != 1
            or mass_integrals.dtype != wp.vec4
            or min(fragment_nodes.shape[0], fractions.shape[0], mass_integrals.shape[0]) < count
        ):
            raise ValueError("Fragment arrays must have int parents/nodes, float fractions and vec4 mass integrals")
        if any(array.device != self.device for array in (parents, fragment_nodes, fractions, mass_integrals)):
            raise ValueError("All cut arrays must reside on the model device")

    def build(self, parents, fragment_nodes, fractions, mass_integrals, count, q, qd, *, capacity=None):
        """Return ``(model, state)`` with physical fragment forces and inertia.

        ``mass_integrals`` contains physical integrals of the four parent
        barycentric shape functions [m³], not normalized fractions. ``q`` and
        ``qd`` include all reserved node copies, already transferred by the
        caller's accepted connectivity edit.

        ``model.cut_physical_mass`` excludes dummy inactive diagonal masses and
        is suitable for total-mass and momentum diagnostics. Pass the resident
        ``model.cut_volume_fractions`` to the solver's ``tet_volume_weights``
        argument for volume diagnostics over the actual material. An optional
        ``capacity`` reserves inactive zero-material tetrahedra for later cuts.
        """
        capacity = count if capacity is None else capacity
        self._validate_fragments(parents, fragment_nodes, fractions, mass_integrals, count, capacity)
        for array in (q, qd):
            if array.shape != self.base_model.particle_q.shape or array.dtype != wp.vec3:
                raise ValueError("Positions and velocities must match the reserved wp.vec3 particle pool")
        if any(array.device != self.device for array in (q, qd)):
            raise ValueError("All cut arrays must reside on the model device")
        model = copy(self.base_model)
        model.tet_count = capacity
        # Required by the shared Model.control schema, unused by Linear FEM.
        model.tet_activations = wp.zeros(capacity, dtype=float, device=self.device)
        model.tet_indices = wp.empty((capacity, 4), dtype=int, device=self.device)
        model.tet_poses = wp.empty(capacity, dtype=wp.mat33, device=self.device)
        model.tet_materials = wp.empty((capacity, 3), dtype=float, device=self.device)
        model.cut_volume_fractions = wp.empty(capacity, dtype=float, device=self.device)
        model.cut_physical_mass = wp.zeros(model.particle_count, dtype=float, device=self.device)
        model.particle_mass = wp.empty_like(model.cut_physical_mass)
        model.particle_inv_mass = wp.empty_like(model.cut_physical_mass)
        model.particle_flags = wp.empty(model.particle_count, dtype=int, device=self.device)
        self.update(model, parents, fragment_nodes, fractions, mass_integrals, count)
        state = model.state()
        wp.copy(state.particle_q, q)
        wp.copy(state.particle_qd, qd)
        return model, state

    def update(self, model, parents, fragment_nodes, fractions, mass_integrals, count):
        """Replace a validated material partition without reallocating storage.

        State positions and velocities are owned by the caller and unchanged.
        Poses describe the reference geometry; subsequently call the solver's
        ``refresh_topology`` to rebuild its sparse structure. Capacity checks
        happen before any array mutation.
        """
        self._validate_fragments(parents, fragment_nodes, fractions, mass_integrals, count, model.tet_count)
        self._validate_model(model)
        model.cut_physical_mass.zero_()
        wp.launch(
            _fragment_material,
            dim=model.tet_count,
            inputs=[
                parents,
                fragment_nodes,
                fractions,
                mass_integrals,
                count,
                self.original_elements,
                self.original_poses,
                self.density,
                self.material,
                model.tet_indices,
                model.tet_poses,
                model.tet_materials,
                model.cut_volume_fractions,
                model.cut_physical_mass,
            ],
            device=self.device,
        )
        wp.launch(
            _active_mass,
            dim=model.particle_count,
            inputs=[model.cut_physical_mass, model.particle_mass, model.particle_inv_mass, model.particle_flags],
            device=self.device,
        )
        model.cut_fragment_count = count

    @staticmethod
    def _array_names():
        return (
            "tet_indices",
            "tet_poses",
            "tet_materials",
            "cut_volume_fractions",
            "cut_physical_mass",
            "particle_mass",
            "particle_inv_mass",
            "particle_flags",
        )

    def _validate_model(self, model):
        if model.device != self.device or model.particle_count != self.base_model.particle_count:
            raise ValueError("Cut model must match the reserved particle pool and device")
        for name in self._array_names():
            if not isinstance(getattr(model, name, None), wp.array):
                raise ValueError("Model must be built by CutModel before it can be updated")
        if model.tet_indices.shape != (model.tet_count, 4) or model.tet_materials.shape != (model.tet_count, 3):
            raise ValueError("Cut model tetrahedron arrays must match its reserved capacity")

    def copy_into(self, destination, source):
        """Copy a validated snapshot into live buffers, preserving all pointers.

        Both models must have the same capacities. This also supports restoring
        an owned snapshot after a rejected publication. Positions, velocities,
        and solver workspace are separate and are restored by the caller.
        """
        self._validate_model(destination)
        self._validate_model(source)
        if destination.tet_count != source.tet_count:
            raise ValueError("Cut snapshots must have matching tetrahedron capacities")
        for name in self._array_names():
            target, value = getattr(destination, name), getattr(source, name)
            if target.shape != value.shape or target.dtype != value.dtype:
                raise ValueError("Cut snapshots must have matching array layouts")
        for name in self._array_names():
            wp.copy(getattr(destination, name), getattr(source, name))
        destination.cut_fragment_count = source.cut_fragment_count

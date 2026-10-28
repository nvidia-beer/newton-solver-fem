# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Mapped surface contact for the position-only, cut-tetrahedron example.

Virtual tetrahedron corners can lie outside their remaining material. Only
the actual exterior and incision triangles participate in contact. Contact
forces are mapped back with the transpose of the surface interpolation.

The Linear FEM integrator consumes these forces explicitly. A local implicit
penalty denominator, bounded by assembled nodal contact row sums, limits the
stiffness seen by its timestep. This is discrete, frictionless contact, not a
converged contact solve or continuous collision detection.
"""

import math

import warp as wp


@wp.kernel(enable_backward=False)
def _copy_surface_topology(
    source_nodes: wp.array2d[int],
    source_weights: wp.array[wp.vec4],
    source_components: wp.array[int],
    source_kind: wp.array[int],
    source_area: wp.array[float],
    count: int,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    indices: wp.array[int],
    components: wp.array[int],
    kind: wp.array[int],
    reference_area: wp.array[float],
):
    triangle = wp.tid()
    components[triangle] = -1
    kind[triangle] = 0
    reference_area[triangle] = 0.0
    for corner in range(3):
        vertex = 3 * triangle + corner
        indices[vertex] = 0
        weights[vertex] = wp.vec4(0.0)
        for k in range(4):
            nodes[vertex, k] = 0
        if triangle < count:
            indices[vertex] = vertex
            weights[vertex] = source_weights[vertex]
            for k in range(4):
                nodes[vertex, k] = source_nodes[vertex, k]
    if triangle < count:
        components[triangle] = source_components[triangle]
        kind[triangle] = source_kind[triangle]
        reference_area[triangle] = source_area[triangle]


def _copy_topology(destination, source, triangle_count):
    count = int(triangle_count)
    if count < 0 or count != triangle_count or count > source.components.shape[0]:
        raise ValueError("triangle count must fit the source surface")
    capacity = destination.components.shape[0]
    if count > capacity:
        raise ValueError(f"Surface needs {count} triangles, exceeding reserved capacity {capacity}")
    if destination.vertex_nodes.ptr == source.vertex_nodes.ptr:
        raise ValueError("source and destination surface topology must use separate storage")
    if capacity:
        wp.launch(
            _copy_surface_topology,
            dim=capacity,
            inputs=[
                source.vertex_nodes,
                source.weights,
                source.components,
                source.kind,
                source.reference_area,
                count,
                destination.vertex_nodes,
                destination.weights,
                destination.indices,
                destination.components,
                destination.kind,
                destination.reference_area,
            ],
            device=destination.vertex_nodes.device,
        )
    destination.count.fill_(count)
    destination.error.zero_()


@wp.func
def _closest_barycentric(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3):
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = wp.dot(ab, ap), wp.dot(ac, ap)
    if d1 <= 0.0 and d2 <= 0.0:
        return wp.vec3(1.0, 0.0, 0.0)
    bp = p - b
    d3, d4 = wp.dot(ab, bp), wp.dot(ac, bp)
    if d3 >= 0.0 and d4 <= d3:
        return wp.vec3(0.0, 1.0, 0.0)
    vc = d1 * d4 - d3 * d2
    if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
        v = d1 / (d1 - d3)
        return wp.vec3(1.0 - v, v, 0.0)
    cp = p - c
    d5, d6 = wp.dot(ab, cp), wp.dot(ac, cp)
    if d6 >= 0.0 and d5 <= d6:
        return wp.vec3(0.0, 0.0, 1.0)
    vb = d5 * d2 - d1 * d6
    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
        w = d2 / (d2 - d6)
        return wp.vec3(1.0 - w, 0.0, w)
    va = d3 * d6 - d5 * d4
    if va <= 0.0 and d4 >= d3 and d5 >= d6:
        w = (d4 - d3) / (d4 - d3 + d5 - d6)
        return wp.vec3(0.0, 1.0 - w, w)
    inverse = 1.0 / (va + vb + vc)
    return wp.vec3(1.0 - (vb + vc) * inverse, vb * inverse, vc * inverse)


@wp.kernel(enable_backward=False)
def _surface_points(
    q: wp.array[wp.vec3],
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    count: wp.array[int],
    points: wp.array[wp.vec3],
):
    vertex = wp.tid()
    value = wp.vec3(0.0)
    if vertex < 3 * count[0]:
        for k in range(4):
            value += weights[vertex][k] * q[nodes[vertex, k]]
    # A viewer can retain a fixed sequential index buffer across cuts. Unused
    # triangles must collapse even after the active prefix shrinks.
    points[vertex] = value


@wp.kernel(enable_backward=False)
def _surface_bounds(
    points: wp.array[wp.vec3],
    count: wp.array[int],
    components: wp.array[int],
    lower: wp.array[wp.vec3],
    upper: wp.array[wp.vec3],
    normal: wp.array[wp.vec3],
    area: wp.array[float],
    groups: wp.array[int],
    multiple_components: wp.array[int],
):
    triangle = wp.tid()
    lower[triangle] = wp.vec3(1.0e10)
    upper[triangle] = wp.vec3(-1.0e10)
    area[triangle] = 0.0
    groups[triangle] = triangle + 1
    normal[triangle] = wp.vec3(0.0)
    if triangle < count[0]:
        a, b, c = points[3 * triangle], points[3 * triangle + 1], points[3 * triangle + 2]
        n = wp.cross(b - a, c - a)
        size = wp.length(n)
        if size > 1.0e-12:
            lower[triangle] = wp.min(wp.min(a, b), c)
            upper[triangle] = wp.max(wp.max(a, b), c)
            area[triangle] = 0.5 * size
            normal[triangle] = n / size
            groups[triangle] = 0
            if components[triangle] != components[0]:
                wp.atomic_max(multiple_components, 0, 1)


@wp.kernel(enable_backward=False)
def _floor_rows(
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    points: wp.array[wp.vec3],
    areas: wp.array[float],
    count: wp.array[int],
    height: float,
    density: float,
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
):
    row = wp.tid()
    if row < 3 * count[0] and areas[row // 3] > 0.0:
        for k in range(4):
            indices[row, k] = nodes[row, k]
            coefficients[row, k] = weights[row][k]
        normals[row] = wp.vec3(0.0, 0.0, 1.0)
        gaps[row] = points[row][2] - height
        stiffness[row] = density * areas[row // 3] / 3.0


@wp.kernel(enable_backward=False)
def _piece_rows(
    bvh: wp.uint64,
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    components: wp.array[int],
    points: wp.array[wp.vec3],
    normals: wp.array[wp.vec3],
    areas: wp.array[float],
    count: wp.array[int],
    multiple_components: wp.array[int],
    row_offset: int,
    radius: float,
    density: float,
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    contact_normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    stiffness: wp.array[float],
):
    source = wp.tid()
    if source < count[0] and multiple_components[0] != 0 and areas[source] > 0.0:
        p = (points[3 * source] + points[3 * source + 1] + points[3 * source + 2]) / 3.0
        query = wp.bvh_query_aabb(bvh, p - wp.vec3(radius), p + wp.vec3(radius))
        candidate, target = int(0), int(-1)
        best = radius * radius
        barycentric = wp.vec3(0.0)
        while wp.bvh_query_next(query, candidate):
            if candidate < count[0] and components[candidate] != components[source] and areas[candidate] > 0.0:
                a, b, c = points[3 * candidate], points[3 * candidate + 1], points[3 * candidate + 2]
                bary = _closest_barycentric(p, a, b, c)
                delta = p - (bary[0] * a + bary[1] * b + bary[2] * c)
                distance = wp.dot(delta, delta)
                if distance < best:
                    target, best, barycentric = candidate, distance, bary
        # Pick the nearest surface before testing normals. Searching only for
        # opposing faces incorrectly contacts the back side of a nearby piece.
        if target >= 0 and wp.dot(normals[source], normals[target]) < -0.1:
            row = row_offset + source
            for corner in range(3):
                for k in range(4):
                    slot = 4 * corner + k
                    indices[row, slot] = nodes[3 * source + corner, k]
                    coefficients[row, slot] = weights[3 * source + corner][k] / 3.0
                    indices[row, 12 + slot] = nodes[3 * target + corner, k]
                    coefficients[row, 12 + slot] = -barycentric[corner] * weights[3 * target + corner][k]
            normal = normals[target]
            closest = (
                barycentric[0] * points[3 * target]
                + barycentric[1] * points[3 * target + 1]
                + barycentric[2] * points[3 * target + 2]
            )
            contact_normals[row] = normal
            gaps[row] = wp.dot(p - closest, normal)
            # Both sides are sampled, so halve their quadrature contribution.
            stiffness[row] = 0.5 * density * areas[source]


@wp.kernel(enable_backward=False)
def _contact_bounds(
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    inverse_mass: wp.array[float],
    velocity: wp.array[wp.vec3],
    dt: float,
    damping_ratio: float,
    stiffness: wp.array[float],
    damping: wp.array[float],
    normal_velocity: wp.array[float],
    node_stiffness: wp.array[float],
    node_damping: wp.array[float],
):
    row = wp.tid()
    k = stiffness[row]
    damping[row] = 0.0
    if k > 0.0:
        rate, inverse, weight_sum = float(0.0), float(0.0), float(0.0)
        for i in range(24):
            index, w = int(indices[row, i]), float(coefficients[row, i])
            rate += w * wp.dot(normals[row], velocity[index])
            weight_sum += wp.abs(w)
        if wp.min(gaps[row], gaps[row] + dt * rate) >= 0.0:
            stiffness[row] = 0.0
            return
        for i in range(24):
            index, w = int(indices[row, i]), float(coefficients[row, i])
            # Repeated corner indices must be combined for effective mass.
            if w != 0.0:
                same = float(0.0)
                for j in range(24):
                    if indices[row, j] == index:
                        same += coefficients[row, j]
                inverse += w * same * inverse_mass[index]
        if inverse <= 0.0:
            stiffness[row] = 0.0
        else:
            c = 2.0 * damping_ratio * wp.sqrt(k / inverse)
            damping[row] = c
            normal_velocity[row] = rate
            # Absolute row sums include all samples sharing a basis node.
            # Independently limiting each sample would become stiffer when
            # the surface is subdivided, even with area-weighted penalties.
            for i in range(24):
                w = wp.abs(coefficients[row, i]) * weight_sum
                if w > 0.0:
                    wp.atomic_add(node_stiffness, indices[row, i], k * w)
                    wp.atomic_add(node_damping, indices[row, i], c * w)


@wp.kernel(enable_backward=False)
def _scatter_contacts(
    indices: wp.array2d[int],
    coefficients: wp.array2d[float],
    normals: wp.array[wp.vec3],
    gaps: wp.array[float],
    inverse_mass: wp.array[float],
    stiffness: wp.array[float],
    damping: wp.array[float],
    normal_velocity: wp.array[float],
    node_stiffness: wp.array[float],
    node_damping: wp.array[float],
    dt: float,
    piece_row_offset: int,
    active_rows: wp.array[int],
    force: wp.array[wp.vec3],
):
    row = wp.tid()
    if stiffness[row] > 0.0:
        stiffness_bound, damping_bound = float(0.0), float(0.0)
        for i in range(24):
            index = indices[row, i]
            if coefficients[row, i] != 0.0:
                stiffness_bound = wp.max(stiffness_bound, inverse_mass[index] * node_stiffness[index])
                damping_bound = wp.max(damping_bound, inverse_mass[index] * node_damping[index])
        rate = normal_velocity[row]
        magnitude = wp.max(0.0, -stiffness[row] * (gaps[row] + dt * rate) - damping[row] * rate)
        magnitude /= 1.0 + dt * dt * stiffness_bound + dt * damping_bound
        if magnitude > 0.0:
            wp.atomic_add(active_rows, int(row >= piece_row_offset), 1)
        value = magnitude * normals[row]
        for i in range(24):
            if coefficients[row, i] != 0.0:
                wp.atomic_add(force, indices[row, i], coefficients[row, i] * value)


class CutSurfaceContact:
    """Persistent GPU buffers for area-weighted surface contact.

    Recreate this helper when surface or particle capacity changes. The surface
    count and mappings can change in place between captured steps. ``evaluate``
    returns forces; it does not modify positions or the integrator's state.
    """

    def __init__(
        self,
        surface,
        particle_q: wp.array,
        particle_inv_mass: wp.array,
        *,
        ground_height: float = 0.0,
        ground_stiffness: float = 2.0e7,
        piece_stiffness: float = 2.0e7,
        damping_ratio: float = 0.5,
        search_radius: float = 0.03,
    ):
        for name, value in (
            ("ground_stiffness", ground_stiffness),
            ("piece_stiffness", piece_stiffness),
            ("damping_ratio", damping_ratio),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not math.isfinite(ground_height) or not math.isfinite(search_radius) or search_radius <= 0.0:
            raise ValueError("ground height must be finite and search radius finite and positive")
        self.surface, self.device = surface, particle_q.device
        self.inverse_mass = particle_inv_mass
        self.ground_height, self.ground_stiffness = float(ground_height), float(ground_stiffness)
        self.piece_stiffness, self.damping_ratio = float(piece_stiffness), float(damping_ratio)
        self.search_radius = float(search_radius)
        self.capacity = surface.components.shape[0]
        rows = 4 * self.capacity
        self.points = wp.zeros(3 * self.capacity, dtype=wp.vec3, device=self.device)
        self.normals = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.areas = wp.zeros(self.capacity, dtype=float, device=self.device)
        self.lower = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        self.upper = wp.zeros(self.capacity, dtype=wp.vec3, device=self.device)
        # Equal centroids in the reserved empty slots can form a long LBVH
        # chain. Distinct inactive sort groups keep refits bounded; queries
        # still traverse the complete tree and do not filter by these groups.
        self.groups = wp.zeros(self.capacity, dtype=int, device=self.device)
        # Recompute on device so the same graph skips pointless self queries
        # before separation and immediately enables them after a topology edit.
        self.multiple_components = wp.zeros(1, dtype=int, device=self.device)
        self.indices = wp.zeros((rows, 24), dtype=int, device=self.device)
        self.coefficients = wp.zeros((rows, 24), dtype=float, device=self.device)
        self.contact_normals = wp.zeros(rows, dtype=wp.vec3, device=self.device)
        self.gaps = wp.zeros(rows, dtype=float, device=self.device)
        self.stiffness = wp.zeros(rows, dtype=float, device=self.device)
        self.damping = wp.zeros(rows, dtype=float, device=self.device)
        self.normal_velocity = wp.zeros(rows, dtype=float, device=self.device)
        self.node_stiffness = wp.zeros(particle_q.shape[0], dtype=float, device=self.device)
        self.node_damping = wp.zeros_like(self.node_stiffness)
        # Device-side diagnostics: ground rows, then inter-piece rows.
        self.active_rows = wp.zeros(2, dtype=int, device=self.device)
        self.forces = wp.zeros_like(particle_q)
        self._update_surface(particle_q)
        self.bvh = wp.Bvh(self.lower, self.upper, groups=self.groups) if self.capacity else None

    def _update_surface(self, q):
        self.multiple_components.zero_()
        if not self.capacity:
            return
        wp.launch(
            _surface_points,
            dim=3 * self.capacity,
            inputs=[q, self.surface.vertex_nodes, self.surface.weights, self.surface.count, self.points],
            device=self.device,
        )
        wp.launch(
            _surface_bounds,
            dim=self.capacity,
            inputs=[
                self.points,
                self.surface.count,
                self.surface.components,
                self.lower,
                self.upper,
                self.normals,
                self.areas,
                self.groups,
                self.multiple_components,
            ],
            device=self.device,
        )

    @staticmethod
    def reserve_surface(source, triangle_count: int, capacity: int):
        """Copy validated surface triangles into persistent reserved storage.

        This allocates once, before graph capture. Insufficient capacity raises
        an error before changing any live topology. No device count is read.
        """
        if int(capacity) != capacity or capacity < triangle_count:
            raise ValueError("reserved surface capacity must fit its active triangles")
        surface = type(source).allocate(int(capacity), source.vertex_nodes.device)
        _copy_topology(surface, source, triangle_count)
        return surface

    def update_topology(self, source, triangle_count: int, particle_q: wp.array):
        """Publish validated triangle mappings without invalidating a graph.

        Call between graph replays, after cloning particle coordinates. Capacity
        errors leave the current surface unchanged. Particle inverse masses
        must also be updated in their original buffer by the owning model.
        """
        _copy_topology(self.surface, source, triangle_count)
        self._update_surface(particle_q)
        if self.bvh is not None:
            # A refit preserves the original leaf order. Newly active triangles
            # need a fresh spatial hierarchy, retaining its allocated storage.
            self.bvh.rebuild()

    def update_surface(self, q: wp.array) -> wp.array:
        """Update and return surface positions [m] for rendering, without contact.

        This reads the live interpolation and active triangle count. The
        returned persistent buffer is also used by :meth:`evaluate`.
        """
        self._update_surface(q)
        return self.points

    def evaluate(self, q: wp.array, qd: wp.array, dt: float, *, piece_contact: bool = True) -> wp.array:
        """Return mapped contact forces [N] without allocating or synchronizing.

        Args:
            q: Basis node positions [m].
            qd: Basis node velocities [m/s].
            dt: Integration timestep [s].
            piece_contact: Enable collisions between distinct component labels.
        """
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("contact timestep must be finite and positive")
        self.forces.zero_()
        self.active_rows.zero_()
        if not self.capacity:
            return self.forces
        self.stiffness.zero_()
        self.node_stiffness.zero_()
        self.node_damping.zero_()
        self._update_surface(q)
        if self.ground_stiffness:
            wp.launch(
                _floor_rows,
                dim=3 * self.capacity,
                inputs=[
                    self.surface.vertex_nodes,
                    self.surface.weights,
                    self.points,
                    self.areas,
                    self.surface.count,
                    self.ground_height,
                    self.ground_stiffness,
                    self.indices,
                    self.coefficients,
                    self.contact_normals,
                    self.gaps,
                    self.stiffness,
                ],
                device=self.device,
            )
        if piece_contact and self.piece_stiffness:
            self.bvh.refit()
            wp.launch(
                _piece_rows,
                dim=self.capacity,
                inputs=[
                    self.bvh.id,
                    self.surface.vertex_nodes,
                    self.surface.weights,
                    self.surface.components,
                    self.points,
                    self.normals,
                    self.areas,
                    self.surface.count,
                    self.multiple_components,
                    3 * self.capacity,
                    self.search_radius,
                    self.piece_stiffness,
                    self.indices,
                    self.coefficients,
                    self.contact_normals,
                    self.gaps,
                    self.stiffness,
                ],
                device=self.device,
            )
        wp.launch(
            _contact_bounds,
            dim=4 * self.capacity,
            inputs=[
                self.indices,
                self.coefficients,
                self.contact_normals,
                self.gaps,
                self.inverse_mass,
                qd,
                dt,
                self.damping_ratio,
                self.stiffness,
                self.damping,
                self.normal_velocity,
                self.node_stiffness,
                self.node_damping,
            ],
            device=self.device,
        )
        wp.launch(
            _scatter_contacts,
            dim=4 * self.capacity,
            inputs=[
                self.indices,
                self.coefficients,
                self.contact_normals,
                self.gaps,
                self.inverse_mass,
                self.stiffness,
                self.damping,
                self.normal_velocity,
                self.node_stiffness,
                self.node_damping,
                dt,
                3 * self.capacity,
                self.active_rows,
                self.forces,
            ],
            device=self.device,
        )
        return self.forces

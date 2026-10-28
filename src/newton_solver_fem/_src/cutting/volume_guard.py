# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bounded mass-weighted volume projection for nearly collapsed cut cells.

The constitutive stretch clamp cannot constrain integrated positions. A tiny
retained fragment can therefore invert under surface contact even though all
stresses remain finite. This inequality correction acts only below a small
positive volume ratio; it is a numerical safeguard, not an additional material
law or a converged nonlinear elasticity solve.
"""

import math

import warp as wp


@wp.func
def _longest_edge(a: wp.vec3, b: wp.vec3, c: wp.vec3):
    squared = wp.max(wp.max(wp.dot(a, a), wp.dot(b, b)), wp.dot(c, c))
    squared = wp.max(squared, wp.dot(a - b, a - b))
    squared = wp.max(squared, wp.dot(a - c, a - c))
    squared = wp.max(squared, wp.dot(b - c, b - c))
    return wp.sqrt(squared)


@wp.kernel(enable_backward=False)
def _corrections(
    q: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    fractions: wp.array[float],
    inverse_mass: wp.array[float],
    minimum: float,
    correction: wp.array[wp.vec3],
    degree: wp.array[int],
    max_degree: wp.array[int],
):
    element = wp.tid()
    if fractions[element] <= 0.0:
        return
    nodes = wp.vec4i(indices[element, 0], indices[element, 1], indices[element, 2], indices[element, 3])
    a = q[nodes[1]] - q[nodes[0]]
    b = q[nodes[2]] - q[nodes[0]]
    c = q[nodes[3]] - q[nodes[0]]
    inverse_volume = wp.determinant(poses[element])
    value = wp.dot(a, wp.cross(b, c)) * inverse_volume
    if value >= minimum:
        return
    g1 = wp.cross(b, c) * inverse_volume
    g2 = wp.cross(c, a) * inverse_volume
    g3 = wp.cross(a, b) * inverse_volume
    g0 = -g1 - g2 - g3
    gradients = wp.matrix_from_cols(g0, g1, g2, g3)
    denominator = float(0.0)
    largest_inverse_gradient = float(0.0)
    for corner in range(4):
        gradient = wp.vec3(gradients[0, corner], gradients[1, corner], gradients[2, corner])
        squared = wp.dot(gradient, gradient)
        denominator += inverse_mass[nodes[corner]] * squared
        largest_inverse_gradient = wp.max(largest_inverse_gradient, inverse_mass[nodes[corner]] * wp.sqrt(squared))
    if denominator > 0.0:
        multiplier = (minimum - value) / denominator
        rest = wp.inverse(poses[element])
        rest_a = wp.vec3(rest[0, 0], rest[1, 0], rest[2, 0])
        rest_b = wp.vec3(rest[0, 1], rest[1, 1], rest[2, 1])
        rest_c = wp.vec3(rest[0, 2], rest[1, 2], rest[2, 2])
        # Near a rank-one collapse, dividing by the tiny volume gradient can
        # create arbitrarily large corrections and destroy float32 momentum
        # cancellation. Bound this Newton step geometrically; one shared factor
        # retains the equal/opposite mass-weighted increments at all corners.
        radius = 0.25 * wp.min(_longest_edge(a, b, c), _longest_edge(rest_a, rest_b, rest_c))
        if largest_inverse_gradient > 0.0:
            multiplier = wp.min(multiplier, radius / largest_inverse_gradient)
        for corner in range(4):
            node = nodes[corner]
            gradient = wp.vec3(gradients[0, corner], gradients[1, corner], gradients[2, corner])
            wp.atomic_add(correction, node, inverse_mass[node] * multiplier * gradient)
            count = wp.atomic_add(degree, node, 1) + 1
            wp.atomic_max(max_degree, 0, count)


@wp.kernel(enable_backward=False)
def _apply(
    correction: wp.array[wp.vec3],
    max_degree: wp.array[int],
    dt: float,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    node = wp.tid()
    if max_degree[0] > 0:
        # One common relaxation factor retains the equal/opposite mass-weighted
        # corrections at shared nodes. Independent per-node averaging does not.
        delta = correction[node] / float(max_degree[0])
        q[node] += delta
        qd[node] += delta / dt


class CutVolumeGuard:
    """Keep cut cells away from zero volume with a fixed GPU work budget."""

    def __init__(self, model, *, minimum=0.05, iterations=16, components=None):
        if not math.isfinite(minimum) or not 0 < minimum < 1:
            raise ValueError("minimum must be finite and between zero and one")
        if not isinstance(iterations, int) or iterations < 1:
            raise ValueError("iterations must be a positive integer")
        self.model = model
        self.minimum = minimum
        self.iterations = iterations
        self.correction = wp.zeros(model.particle_count, dtype=wp.vec3, device=model.device)
        self.degree = wp.zeros(model.particle_count, dtype=int, device=model.device)
        self.max_degree = wp.zeros(1, dtype=int, device=model.device)
        self.components = components
        if self.components is None:
            self.components = wp.zeros(model.tet_count, dtype=int, device=model.device)
        self.node_components = wp.empty(model.particle_count, dtype=int, device=model.device)
        self.step_fraction = wp.ones(model.tet_count, dtype=float, device=model.device)
        self.mass_displacement = wp.zeros(model.tet_count, dtype=wp.vec4, device=model.device)
        self.component_fixed = wp.zeros(model.tet_count, dtype=int, device=model.device)
        self.limited_steps = wp.zeros(1, dtype=int, device=model.device)

    def apply(self, q, qd, dt, *, previous_q=None):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")

        def project():
            self.correction.zero_()
            self.degree.zero_()
            self.max_degree.zero_()
            wp.launch(
                _corrections,
                dim=self.model.tet_count,
                inputs=[
                    q,
                    self.model.tet_indices,
                    self.model.tet_poses,
                    self.model.cut_volume_fractions,
                    self.model.particle_inv_mass,
                    self.minimum,
                    self.correction,
                    self.degree,
                    self.max_degree,
                ],
                device=self.model.device,
            )
            wp.launch(
                _apply,
                dim=self.model.particle_count,
                inputs=[self.correction, self.max_degree, dt, q, qd],
                device=self.model.device,
            )

        project()

        def remaining_projections():
            for _ in range(self.iterations - 1):
                project()

        if self.model.device.is_capturing and self.iterations > 1:
            # Ordinary frames pay for one check, not sixteen projection
            # sweeps. The device condition adds no host readback or allocation.
            wp.capture_if(self.max_degree, remaining_projections)
        else:
            remaining_projections()
        if previous_q is not None:
            self.node_components.fill_(-1)
            self.step_fraction.fill_(1.0)
            self.mass_displacement.zero_()
            self.component_fixed.zero_()
            wp.launch(
                _feasible_steps,
                dim=self.model.tet_count,
                inputs=[
                    previous_q,
                    q,
                    self.model.tet_indices,
                    self.model.tet_poses,
                    self.model.cut_volume_fractions,
                    self.components,
                    self.minimum,
                    self.node_components,
                    self.step_fraction,
                    self.limited_steps,
                ],
                device=self.model.device,
            )
            wp.launch(
                _component_displacement,
                dim=self.model.particle_count,
                inputs=[
                    previous_q,
                    q,
                    self.model.particle_inv_mass,
                    self.node_components,
                    self.step_fraction,
                    self.mass_displacement,
                    self.component_fixed,
                ],
                device=self.model.device,
            )
            wp.launch(
                _limit_step,
                dim=self.model.particle_count,
                inputs=[
                    previous_q,
                    self.model.particle_inv_mass,
                    self.node_components,
                    self.step_fraction,
                    self.mass_displacement,
                    self.component_fixed,
                    dt,
                    q,
                    qd,
                ],
                device=self.model.device,
            )


@wp.func
def _polynomial(c: wp.vec4d, t: wp.float64):
    return ((c[3] * t + c[2]) * t + c[1]) * t + c[0]


@wp.func
def _minimum_on_segment(c: wp.vec4d, end: wp.float64):
    result = wp.min(c[0], _polynomial(c, end))
    # The determinant of an affine nodal motion is cubic. Check its stationary
    # points too: a positive endpoint alone can hide two orientation crossings.
    if wp.abs(c[3]) > wp.float64(1.0e-30):
        discriminant = c[2] * c[2] - wp.float64(3.0) * c[3] * c[1]
        if discriminant >= wp.float64(0.0):
            root = wp.sqrt(discriminant)
            numerator = -c[2] - root
            if c[2] < wp.float64(0.0):
                numerator = -c[2] + root
            # Recover the other root from their product. Direct subtraction
            # loses the finite root when the cubic is almost quadratic.
            if numerator != wp.float64(0.0):
                first = numerator / (wp.float64(3.0) * c[3])
                second = c[1] / numerator
                if first > wp.float64(0.0) and first < end:
                    result = wp.min(result, _polynomial(c, first))
                if second > wp.float64(0.0) and second < end:
                    result = wp.min(result, _polynomial(c, second))
    elif wp.abs(c[2]) > wp.float64(1.0e-30):
        root = -c[1] / (wp.float64(2.0) * c[2])
        if root > wp.float64(0.0) and root < end:
            result = wp.min(result, _polynomial(c, root))
    return result


@wp.kernel(enable_backward=False)
def _feasible_steps(
    previous: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    indices: wp.array2d[int],
    poses: wp.array[wp.mat33],
    fractions: wp.array[float],
    components: wp.array[int],
    minimum: float,
    node_components: wp.array[int],
    step_fraction: wp.array[float],
    limited_steps: wp.array[int],
):
    element = wp.tid()
    if fractions[element] <= 0.0:
        return
    component = components[element]
    for corner in range(4):
        wp.atomic_max(node_components, indices[element, corner], component)
    n0, n1, n2, n3 = indices[element, 0], indices[element, 1], indices[element, 2], indices[element, 3]
    origin = wp.vec3d(previous[n0])
    a, b, c = wp.vec3d(previous[n1]) - origin, wp.vec3d(previous[n2]) - origin, wp.vec3d(previous[n3]) - origin
    origin = wp.vec3d(trial[n0])
    da, db, dc = wp.vec3d(trial[n1]) - origin - a, wp.vec3d(trial[n2]) - origin - b, wp.vec3d(trial[n3]) - origin - c
    coefficients = wp.vec4d(
        wp.dot(a, wp.cross(b, c)),
        wp.dot(da, wp.cross(b, c)) + wp.dot(a, wp.cross(db, c)) + wp.dot(a, wp.cross(b, dc)),
        wp.dot(a, wp.cross(db, dc)) + wp.dot(da, wp.cross(b, dc)) + wp.dot(da, wp.cross(db, c)),
        wp.dot(da, wp.cross(db, dc)),
    ) * wp.determinant(wp.mat33d(poses[element]))
    # Never repeatedly halve an already small volume: that approaches zero
    # until a common float32 translation can flip orientation by roundoff.
    floor = wp.min(wp.float64(minimum * 0.1), coefficients[0])
    if _minimum_on_segment(coefficients, wp.float64(1.0)) < floor:
        lower, upper = wp.float64(0.0), wp.float64(1.0)
        for _ in range(24):
            middle = (lower + upper) * wp.float64(0.5)
            if _minimum_on_segment(coefficients, middle) >= floor:
                lower = middle
            else:
                upper = middle
        wp.atomic_min(step_fraction, component, float(lower * wp.float64(0.9)))
        wp.atomic_add(limited_steps, 0, 1)


@wp.kernel(enable_backward=False)
def _component_displacement(
    previous: wp.array[wp.vec3],
    trial: wp.array[wp.vec3],
    inverse_mass: wp.array[float],
    node_components: wp.array[int],
    step_fraction: wp.array[float],
    displacement: wp.array[wp.vec4],
    component_fixed: wp.array[int],
):
    node = wp.tid()
    component = node_components[node]
    if component >= 0 and step_fraction[component] < 1.0:
        if inverse_mass[node] > 0.0:
            mass = 1.0 / inverse_mass[node]
            delta = mass * (trial[node] - previous[node])
            wp.atomic_add(displacement, component, wp.vec4(delta[0], delta[1], delta[2], mass))
        else:
            wp.atomic_max(component_fixed, component, 1)


@wp.kernel(enable_backward=False)
def _limit_step(
    previous: wp.array[wp.vec3],
    inverse_mass: wp.array[float],
    node_components: wp.array[int],
    step_fraction: wp.array[float],
    displacement: wp.array[wp.vec4],
    component_fixed: wp.array[int],
    dt: float,
    q: wp.array[wp.vec3],
    qd: wp.array[wp.vec3],
):
    node = wp.tid()
    component = node_components[node]
    if component >= 0 and inverse_mass[node] > 0.0 and step_fraction[component] < 1.0:
        total = displacement[component]
        center = wp.vec3(0.0)
        if component_fixed[component] == 0 and total[3] > 0.0:
            center = wp.vec3(total[0], total[1], total[2]) / total[3]
        # Retain each disconnected piece's center-of-mass motion. A sliver
        # limits deformation of its own piece; it cannot freeze free fall.
        # An anchored piece instead exchanges momentum with its fixed support.
        delta = center + step_fraction[component] * (q[node] - previous[node] - center)
        q[node] = previous[node] + delta
        qd[node] = delta / dt

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Two-way sampled capsule contact against the actual deforming pad surface.

The samples are collision quadrature on imported USD primitives, not geometry
generation. Each rigid body's sample weights sum to one, so contact stiffness
does not grow when a curved needle uses more capsule segments. This explicit
penalty approximation does not model puncture, cutting, thread, or suturing.
"""

import newton
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def scatter_surface(
    particle_q: wp.array[wp.vec3],
    particle_qd: wp.array[wp.vec3],
    active_to_surface: wp.array[int],
    points: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
):
    i = wp.tid()
    j = active_to_surface[i]
    points[j] = particle_q[i]
    velocities[j] = particle_qd[i]


@wp.kernel(enable_backward=False)
def surface_contact_forces(
    mesh: wp.uint64,
    indices: wp.array[int],
    points: wp.array[wp.vec3],
    velocities: wp.array[wp.vec3],
    surface_to_active: wp.array[int],
    particle_free: wp.array[int],
    sample_body: wp.array[int],
    sample_point: wp.array[wp.vec3],
    sample_radius: wp.array[float],
    sample_weight: wp.array[float],
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    body_com: wp.array[wp.vec3],
    stiffness: float,
    damping: float,
    friction: float,
    margin: float,
    particle_f: wp.array[wp.vec3],
    body_f: wp.array[wp.spatial_vector],
    active_contacts: wp.array[int],
    active_body_force: wp.array[wp.vec3],
    active_body_contacts: wp.array[int],
):
    i = wp.tid()
    body = sample_body[i]
    transform = body_q[body]
    center = wp.transform_point(transform, sample_point[i])
    radius = sample_radius[i]
    hit = wp.mesh_query_point_sign_normal(mesh, center, radius + margin)
    if not hit.result:
        return
    a = indices[hit.face * 3]
    b = indices[hit.face * 3 + 1]
    c = indices[hit.face * 3 + 2]
    bary = wp.vec3(hit.u, hit.v, 1.0 - hit.u - hit.v)
    point = bary[0] * points[a] + bary[1] * points[b] + bary[2] * points[c]
    delta = center - point
    distance = wp.length(delta)
    penetration = radius - hit.sign * distance
    if penetration <= 0.0:
        return
    normal = wp.normalize(wp.cross(points[b] - points[a], points[c] - points[a]))
    if distance > 1.0e-8:
        normal = hit.sign * delta / distance
    com = wp.transform_point(transform, body_com[body])
    velocity = wp.spatial_top(body_qd[body]) + wp.cross(wp.spatial_bottom(body_qd[body]), point - com)
    velocity -= bary[0] * velocities[a] + bary[1] * velocities[b] + bary[2] * velocities[c]
    vn = wp.dot(velocity, normal)
    vt = velocity - vn * normal
    fn = stiffness * penetration - damping * wp.min(vn, 0.0)
    # A smooth velocity scale avoids a discontinuous Coulomb sign at rest.
    force = sample_weight[i] * (fn * normal - friction * fn * vt / wp.max(wp.length(vt), 0.001))
    wp.atomic_add(body_f, body, wp.spatial_vector(force, wp.cross(point - com, force)))
    vertices = wp.vec3i(a, b, c)
    free_fraction = float(0.0)
    for j in range(3):
        particle = surface_to_active[vertices[j]]
        if particle >= 0:
            wp.atomic_sub(particle_f, particle, bary[j] * force)
            if particle_free[particle] != 0 and bary[j] > 0.0:
                free_fraction += bary[j]
    # Report only the reaction corresponding to force delivered to free FEM
    # nodes. The remaining fraction is supported by fixed cut/underside nodes
    # or the inactive pad. These diagnostics do not change contact physics.
    if free_fraction > 0.0:
        wp.atomic_add(active_body_force, body, free_fraction * force)
        wp.atomic_add(active_body_contacts, body, 1)
    wp.atomic_add(active_contacts, 0, 1)


def capsule_samples(model, body_indices):
    """Return body-local quadrature points for imported sphere/capsule shapes.

    A capsule's axis is Newton-local Z. The line spacing is at most one radius,
    including both end caps; the spherical contact envelope consequently has
    a bounded scalloping error rather than treating the needle as a convex hull.
    """
    types, bodies = model.shape_type.numpy(), model.shape_body.numpy()
    scales, transforms = model.shape_scale.numpy(), model.shape_transform.numpy()
    flags = model.shape_flags.numpy()
    selected = set(body_indices)
    sample_bodies, points, radii = [], [], []
    for kind, body, scale, transform, flag in zip(types, bodies, scales, transforms, flags, strict=True):
        if body not in selected or not flag & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if kind not in (newton.GeoType.CAPSULE, newton.GeoType.SPHERE):
            continue
        radius = float(scale[0])
        half_length = float(scale[1]) if kind == newton.GeoType.CAPSULE else 0.0
        count = max(1, int(np.ceil(2 * half_length / radius)) + 1)
        for z in np.linspace(-half_length, half_length, count):
            sample_bodies.append(body)
            points.append(tuple(wp.transform_point(wp.transform(*transform), wp.vec3(0.0, 0.0, float(z)))))
            radii.append(radius)
    if not sample_bodies:
        raise ValueError("The rig USD needs capsule or sphere collision shapes on its tools and needle")
    counts = np.bincount(sample_bodies, minlength=model.body_count)
    weights = 1.0 / counts[np.asarray(sample_bodies)]
    return np.asarray(sample_bodies, dtype=np.int32), np.asarray(points), np.asarray(radii), weights


class PadContacts:
    """Own a deforming exterior BVH and conservative surface force transfer.

    ``active_body_force`` is the world-space reaction on each rigid body from
    its barycentric force contribution to free FEM particles. Per-body
    ``active_body_contacts`` counts samples with positive free-particle support;
    ``active_contacts`` also includes fixed/inactive pad contacts. All three
    diagnostics describe the most recent substep, not accumulated impulses.
    """

    def __init__(
        self,
        model,
        reference,
        selection,
        display_points,
        body_indices,
        *,
        stiffness,
        damping,
        friction,
        fixed_particle_indices=None,
    ):
        self.model = model
        self.stiffness, self.damping, self.friction = stiffness, damping, friction
        device = model.device
        self.points = display_points
        self.velocities = wp.zeros(len(reference.vertices), dtype=wp.vec3, device=device)
        self.indices = wp.array(reference.exterior_surface_triangles.reshape(-1), dtype=int, device=device)
        self.mesh = wp.Mesh(self.points, self.indices, velocities=self.velocities)
        active_to_surface = np.searchsorted(reference.global_vertex_indices, selection.global_vertex_indices)
        reverse = np.full(len(reference.vertices), -1, dtype=np.int32)
        reverse[active_to_surface] = np.arange(len(active_to_surface))
        self.active_to_surface = wp.array(active_to_surface, dtype=int, device=device)
        self.surface_to_active = wp.array(reverse, dtype=int, device=device)
        free = np.zeros(model.particle_count, dtype=np.int32)
        if model.particle_count:
            free[:] = (model.particle_mass.numpy() > 0.0) & (
                (model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE)) != 0
            )
        if fixed_particle_indices is not None:
            free[np.asarray(fixed_particle_indices, dtype=np.int32)] = 0
        self.particle_free = wp.array(free, dtype=int, device=device)
        values = capsule_samples(model, body_indices)
        self.sample_body, self.sample_point, self.sample_radius, self.sample_weight = (
            wp.array(value, dtype=dtype, device=device)
            for value, dtype in zip(values, (int, wp.vec3, float, float), strict=True)
        )
        self.active_contacts = wp.zeros(1, dtype=int, device=device)
        self.active_body_force = wp.zeros(model.body_count, dtype=wp.vec3, device=device)
        self.active_body_contacts = wp.zeros(model.body_count, dtype=int, device=device)

    def add_forces(self, state, external_forces):
        if self.model.particle_count:
            wp.launch(
                scatter_surface,
                dim=self.model.particle_count,
                inputs=[state.particle_q, state.particle_qd, self.active_to_surface, self.points, self.velocities],
                device=self.model.device,
            )
            self.mesh.refit()
        self.active_contacts.zero_()
        self.active_body_force.zero_()
        self.active_body_contacts.zero_()
        wp.launch(
            surface_contact_forces,
            dim=len(self.sample_body),
            inputs=[
                self.mesh.id,
                self.indices,
                self.points,
                self.velocities,
                self.surface_to_active,
                self.particle_free,
                self.sample_body,
                self.sample_point,
                self.sample_radius,
                self.sample_weight,
                state.body_q,
                state.body_qd,
                self.model.body_com,
                self.stiffness,
                self.damping,
                self.friction,
                0.006,
                external_forces,
                state.body_f,
                self.active_contacts,
                self.active_body_force,
                self.active_body_contacts,
            ],
            device=self.model.device,
        )

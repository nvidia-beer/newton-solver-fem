# SPDX-License-Identifier: Apache-2.0
"""Tet4 and Hex8 USD objects through one bundled, geometry-only asset loader."""

from pathlib import Path

import newton
import newton.examples
import numpy as np
import warp as wp

from newton_solver_fem import FEMBuilder, SolverLinearFEM, hex_quadrature, load_mesh

ASSETS = Path(__file__).parents[1] / "src/newton_solver_fem/examples/assets"
ORIGINAL_ASSETS = ("tet_box", "tet_sphere", "tet_baymax", "hex_cube", "hex_pyramid", "hex_tire")


def add_mesh(builder, mesh, *, vertices=None, density=100.0, material=(500.0, 1000.0, 0.2), color=(0.2, 0.65, 0.95)):
    """Apply FEM-specific mass/material choices to shared reference geometry."""
    vertices = mesh.vertices if vertices is None else vertices
    start = builder.particle_count
    indices = mesh.indices
    masses = np.zeros(len(vertices))
    if mesh.element_type == "tet":
        edges = vertices[indices[:, 1:]] - vertices[indices[:, :1]]
        volume = np.linalg.det(edges) / 6.0
        np.add.at(masses, indices.ravel(), np.repeat(density * volume / 4.0, 4))
    else:
        shape, _, weights = hex_quadrature(vertices, indices)
        np.add.at(masses, indices.ravel(), (density * weights @ shape).ravel())
    if np.any(masses <= 0):
        raise ValueError(f"{mesh.name}: every FEM particle must belong to a positive-volume element")
    for point, mass in zip(vertices, masses, strict=True):
        builder.add_particle(pos=wp.vec3(*point), vel=wp.vec3(0.0), mass=float(mass), radius=0.004)
    if mesh.element_type == "tet":
        for ids in indices + start:
            builder.add_tetrahedron(*map(int, ids), k_mu=material[0], k_lambda=material[1], k_damp=material[2])
    else:
        builder.add_soft_hex_mesh(indices + start, k_mu=material[0], k_lambda=material[1], k_damp=material[2])
    for ids in mesh.surface_triangles + start:
        builder.add_triangle(*map(int, ids), tri_ke=0.0, tri_ka=0.0, tri_kd=0.0, color=color)
    return indices + start if mesh.element_type == "hex" else np.empty((0, 8), dtype=np.int32)


class Example:
    def __init__(self, viewer, args):
        self.viewer, self.time = viewer, 0.0
        self.substeps = args.substeps
        if self.substeps < 1 or args.iterations < 1:
            raise ValueError("substeps and iterations must be positive")
        self.dt = 1.0 / (60.0 * self.substeps)
        names = ORIGINAL_ASSETS if args.original_assets else ("tet_grid", "hex_grid")
        paths = args.asset or [ASSETS / f"{name}.usda" for name in names]
        meshes = [mesh for path in paths for mesh in load_mesh(path)]
        meshes = [mesh for mesh in meshes if args.element == "showcase" or mesh.element_type == args.element]
        if not meshes:
            raise ValueError("The selected files contain no objects for this element mode")
        builder = FEMBuilder()
        hexes = []
        material = (500.0, 1000.0, 0.0 if args.original_assets else 0.2)
        columns = min(3, len(meshes))
        for index, mesh in enumerate(meshes):
            # Display layout belongs to this example. The loader preserves metres
            # and world placement; original shapes retain all vertices and cells.
            low, high = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
            scale = 0.2 / np.max(high - low)
            position = [(index % columns - (columns - 1) / 2) * 0.4, (index // columns) * 0.4, 0.03]
            vertices = (mesh.vertices - [(low[0] + high[0]) / 2, (low[1] + high[1]) / 2, low[2]]) * scale + position
            color = (0.2, 0.65, 0.95) if mesh.element_type == "tet" else (0.95, 0.55, 0.2)
            hexes.append(add_mesh(builder, mesh, vertices=vertices, material=material, color=color))
            print(
                f"{mesh.name}: {mesh.element_type}, {len(mesh.vertices)} vertices, {len(mesh.indices)} elements",
                flush=True,
            )
        indices = np.concatenate(hexes)
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.solver = SolverLinearFEM(
            self.model,
            dt=self.dt,
            linear_solver_maxiter=args.iterations,
            solver_type="cg",
            preconditioner_type="block_diag",
            ground_plane=(0.0, 0.0, 1.0, 0.0),
            ground_ke=1.0e3,
            ground_kd=5.0,
            self_contact_ke=0.0,
        )
        self.state_0, self.state_1 = self.model.state(), self.model.state()
        self._tet_indices = self.model.tet_indices.numpy()
        self._hex_indices = indices
        self.viewer.set_model(self.model)
        self.viewer.set_camera(wp.vec3(1.1, -1.6, 1.0), pitch=-22, yaw=115)

    def step(self):
        for _ in range(self.substeps):
            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.time += self.dt

    def render(self):
        self.viewer.begin_frame(self.time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()

    def test_post_step(self):
        q = self.state_0.particle_q.numpy()
        if not np.isfinite(q).all() or q[:, 2].min() < -0.02:
            raise AssertionError("Invalid FEM/contact state")
        if len(self._tet_indices):
            points = q[self._tet_indices]
            if np.any(np.linalg.det(points[:, 1:] - points[:, :1]) <= 0):
                raise AssertionError("A tetrahedron inverted during the FEM rollout")
        if len(self._hex_indices):
            hex_quadrature(q, self._hex_indices)

    def test_final(self):
        self.test_post_step()


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--element", choices=("tet", "hex", "showcase"), default="showcase")
    parser.add_argument(
        "--asset", action="append", type=Path, help="USD path; repeat for multiple files (Tet4 or Hex8)."
    )
    parser.add_argument(
        "--original-assets", action="store_true", help="Load the six original USD shapes at full topology."
    )
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=128, help="Maximum checked CG iterations per step.")
    viewer, args = newton.examples.init(parser)
    newton.examples.run(Example(viewer, args), args)

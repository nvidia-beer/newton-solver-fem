# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Disconnected small pieces must participate in a bounded Krylov solve."""

import os
import unittest
from types import SimpleNamespace

import numpy as np
import warp as wp
from test_reference_streaming import _ORIGINAL_NODES, _SPLIT, _deformed, _dense, _edit, _model, _solver
from warp.optim.linear import cg
from warp.sparse import bsr_set_from_triplets, bsr_zeros

from newton_solver_fem._src.solvers.fem.component_scaling import ComponentScaling


class TestLinearFEMComponentScaling(unittest.TestCase):
    device = os.environ.get("NEWTON_TEST_DEVICE", "cpu")

    def system(self):
        rng = np.random.default_rng(702)
        blocks = []
        for scale, largest in ((1.0, 4.0), (1e-20, 100.0)):
            basis, _ = np.linalg.qr(rng.normal(size=(12, 12)))
            blocks.append(scale * (basis @ np.diag(np.geomspace(1, largest, 12)) @ basis.T))
        dense = np.zeros((24, 24), dtype=np.float32)
        dense[:12, :12], dense[12:, 12:] = blocks
        rows, columns, values = [], [], []
        for row in range(8):
            for column in range(8):
                value = dense[3 * row : 3 * row + 3, 3 * column : 3 * column + 3]
                if np.any(value):
                    rows.append(row)
                    columns.append(column)
                    values.append(value)
        matrix = bsr_zeros(8, 8, wp.mat33, device=self.device, nnz_capacity=len(rows))
        bsr_set_from_triplets(
            matrix,
            wp.array(rows, dtype=int, device=self.device),
            wp.array(columns, dtype=int, device=self.device),
            wp.array(values, dtype=wp.mat33, device=self.device),
        )
        mass = np.repeat((0.25, 0.25e-20), 4).astype(np.float32)
        labels = wp.array(np.repeat((0, 1), 4), dtype=int, device=self.device)
        model = SimpleNamespace(
            particle_count=8, particle_mass=wp.array(mass, device=self.device), device=matrix.device
        )
        scaling = ComponentScaling(model, matrix, labels, 3)
        inverse = np.asarray([np.linalg.inv(dense[3 * i : 3 * i + 3, 3 * i : 3 * i + 3]) for i in range(8)])
        inverse = wp.array(inverse, dtype=wp.mat33, device=self.device)
        exact = rng.normal(size=(8, 3)).astype(np.float32)
        rhs = wp.array((dense @ exact.ravel()).reshape(-1, 3), dtype=wp.vec3, device=self.device)
        return model, matrix, scaling, inverse, rhs, exact

    def test_equation_scaling_preserves_dense_solution_and_symmetry(self):
        model, matrix, scaling, inverse, rhs, exact = self.system()
        mass = model.particle_mass.numpy().copy()
        scaling.prepare(matrix, inverse, rhs)
        physical = _dense(matrix)
        scaled = _dense(scaling.matrix)
        factors = np.repeat(scaling.scale.numpy(), 3)
        np.testing.assert_allclose(scaled, physical * factors[:, None], rtol=1e-7, atol=0)
        np.testing.assert_allclose(scaled, scaled.T, rtol=0, atol=2e-6)
        np.testing.assert_allclose(
            np.linalg.solve(scaled, scaling.rhs.numpy().ravel()), exact.ravel(), rtol=1e-5, atol=1e-5
        )
        np.testing.assert_array_equal(model.particle_mass.numpy(), mass)
        np.testing.assert_allclose(scaling.inverse.numpy(), inverse.numpy() / factors[::3, None, None], rtol=1e-7)

    def test_fixed_iteration_pcg_resolves_tiny_disconnected_block(self):
        _, matrix, scaling, inverse, rhs, exact = self.system()
        scaling.prepare(matrix, inverse, rhs)
        result = wp.zeros_like(rhs)
        cg(
            scaling.operator,
            scaling.rhs,
            result,
            M=scaling.preconditioner,
            maxiter=30,
            tol=1e-3,
            atol=0,
            check_every=0,
            use_cuda_graph=False,
        )
        np.testing.assert_allclose(result.numpy(), exact, rtol=0, atol=0.03)

    def test_mass_and_label_updates_reuse_buffers(self):
        model, matrix, scaling, inverse, rhs, _ = self.system()
        pointers = (scaling.scale.ptr, scaling.total_mass.ptr, scaling.matrix.values.ptr, scaling.inverse.ptr)
        scaling.labels.assign(np.asarray([0] * 4 + [-1, 2, 2, 2], dtype=np.int32))
        model.particle_mass.assign(np.asarray([0.5] * 4 + [1.0, 2e-22, 3e-22, 5e-22], dtype=np.float32))
        scaling.refresh()
        scaling.prepare(matrix, inverse, rhs)
        np.testing.assert_allclose(scaling.scale.numpy(), [0.5] * 4 + [1.0] + [1e21] * 3, rtol=1e-7)
        self.assertEqual(
            pointers, (scaling.scale.ptr, scaling.total_mass.ptr, scaling.matrix.values.ptr, scaling.inverse.ptr)
        )

    def test_capture_uses_updated_component_masses_and_rhs(self):
        if not wp.get_device(self.device).is_cuda:
            self.skipTest("CUDA is unavailable")
        model, matrix, scaling, inverse, rhs, exact = self.system()
        scaling.prepare(matrix, inverse, rhs)
        result = wp.zeros_like(rhs)
        solve = cg(
            scaling.operator,
            scaling.rhs,
            result,
            M=scaling.preconditioner,
            maxiter=30,
            tol=1e-3,
            atol=0,
            check_every=0,
            use_cuda_graph=False,
            run=False,
        )
        solve()
        with wp.ScopedCapture(device=self.device) as capture:
            scaling.prepare(matrix, inverse, rhs)
            result.zero_()
            solve()
        original = rhs.numpy().copy()
        for factor in (0.5, 1.0, 2.0):
            model.particle_mass.assign(model.particle_mass.numpy() * factor)
            scaling.refresh()
            rhs.assign(original * factor)
            wp.capture_launch(capture.graph)
            np.testing.assert_allclose(result.numpy(), exact * factor, rtol=0, atol=0.06)

    def test_topology_refresh_retains_scaled_connectivity_storage(self):
        for streaming in (False, True):
            with self.subTest(streaming=streaming):
                model, weights = _model(self.device)
                solver = _solver(model, weights, streaming=streaming)
                labels = wp.array(np.asarray([0] * 5 + [-1] * 10), dtype=int, device=self.device)
                solver.prepare_component_scaling(labels, 6)
                current, following = model.state(), model.state()
                deformed = _deformed(model)
                current.particle_q.assign(deformed)
                solver.refresh_tangent(deformed)
                solver.step(current, following, model.control(), None, solver.dt)
                scaled = solver._component_scaling
                pointers = (scaled.matrix.offsets.ptr, scaled.matrix.columns.ptr, scaled.matrix.values.ptr)
                _edit(model, weights, _SPLIT)
                labels.assign(np.asarray([0] * 5 + [1] * 4 + [-1] * 6, dtype=np.int32))
                solver.refresh_component_scaling()
                solver.refresh_topology(deformed)
                solver.step(current, following, model.control(), None, solver.dt)
                raw = solver.A_bsr if not streaming else solver._streaming.compact_matrix
                self.assertEqual(scaled.matrix.offsets.ptr, raw.offsets.ptr)
                self.assertEqual(scaled.matrix.columns.ptr, raw.columns.ptr)
                self.assertEqual(
                    pointers, (scaled.matrix.offsets.ptr, scaled.matrix.columns.ptr, scaled.matrix.values.ptr)
                )
                factors = np.repeat(scaled.scale.numpy(), 3)
                np.testing.assert_allclose(_dense(scaled.matrix), _dense(raw) * factors[:, None], rtol=1e-7, atol=0)

    def test_sparse_preparation_after_component_scaling_preserves_bound_matrix(self):
        model, weights = _model(self.device)
        solver = _solver(model, weights)
        labels = wp.array(np.asarray([0] * 5 + [-1] * 10), dtype=int, device=self.device)
        solver.prepare_component_scaling(labels, 6)
        matrix, scaling = solver.A_bsr, solver._component_scaling
        pointers = matrix.offsets.ptr, matrix.columns.ptr, matrix.values.ptr

        with self.assertRaisesRegex(RuntimeError, "before component scaling"):
            solver.prepare_tetrahedral_updates(_ORIGINAL_NODES)

        self.assertIsNone(solver._streaming)
        self.assertIs(solver.A_bsr, matrix)
        self.assertIs(solver._component_scaling, scaling)
        self.assertEqual(pointers, (matrix.offsets.ptr, matrix.columns.ptr, matrix.values.ptr))
        self.assertEqual(scaling.matrix.offsets.ptr, matrix.offsets.ptr)
        self.assertEqual(scaling.matrix.columns.ptr, matrix.columns.ptr)


if __name__ == "__main__":
    unittest.main()

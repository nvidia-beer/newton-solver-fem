# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent tetrahedral assembly matches complete reconstruction after cuts."""

import os
import unittest

import newton
import numpy as np
import warp as wp

from newton_solver_fem import SolverLinearFEM
from newton_solver_fem._src.solvers.fem.implicit_predictor import subtract_elastic_predictor

_ORIGINAL_NODES = 5
_CAPACITY = 6
_CORNERS = np.asarray(((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (0.2, 0.3, -1)), dtype=np.float32)
_INITIAL = ((0, (0, 1, 2, 3), 1.0), (1, (0, 2, 1, 4), 1.0))
_SPLIT = ((1, (0, 2, 1, 4), 1.0), (0, (5, 6, 7, 8), 0.6), (0, (0, 1, 2, 3), 0.4))


def _edit(model, weights, records):
    indices = np.zeros((_CAPACITY, 4), dtype=np.int32)
    poses = np.tile(np.eye(3), (_CAPACITY, 1, 1))
    materials = np.zeros((_CAPACITY, 3))
    fractions = np.zeros(_CAPACITY)
    masses = np.zeros(model.particle_count)
    rest = np.tile(_CORNERS, (3, 1))
    for slot, (family, connectivity, fraction) in enumerate(records):
        nodes = list(connectivity)
        edges = (rest[nodes[1:]] - rest[nodes[0]]).T
        if np.linalg.det(edges) < 0:
            nodes[1], nodes[2] = nodes[2], nodes[1]
            edges = (rest[nodes[1:]] - rest[nodes[0]]).T
        indices[slot] = nodes
        poses[slot] = np.linalg.inv(edges)
        materials[slot] = fraction * np.asarray((1000.0 + 300.0 * family, 1500.0 + 200.0 * family, 0.0))
        fractions[slot] = fraction
        masses[nodes] += fraction * 200.0 * np.linalg.det(edges) / 24.0
    model.tet_indices.assign(indices)
    model.tet_poses.assign(poses)
    model.tet_materials.assign(materials)
    weights.assign(fractions)
    model.particle_mass.assign(np.where(masses > 0, masses, 1.0))
    model.particle_inv_mass.assign(np.divide(1.0, masses, out=np.zeros_like(masses), where=masses > 0))
    model.particle_flags.assign((masses > 0).astype(np.int32))


def _model(device, records=_INITIAL):
    builder = newton.ModelBuilder(gravity=(0.0, 0.0, -9.81))
    for point in np.tile(_CORNERS, (3, 1)):
        builder.add_particle(point, vel=(0, 0, 0), mass=1.0)
    model = builder.finalize(device=device)
    model.tet_count = _CAPACITY
    model.tet_activations = wp.zeros(_CAPACITY, dtype=float, device=device)
    model.tet_indices = wp.zeros((_CAPACITY, 4), dtype=int, device=device)
    model.tet_poses = wp.array(np.tile(np.eye(3), (_CAPACITY, 1, 1)), dtype=wp.mat33, device=device)
    model.tet_materials = wp.zeros((_CAPACITY, 3), dtype=float, device=device)
    weights = wp.zeros(_CAPACITY, dtype=float, device=device)
    _edit(model, weights, records)
    return model, weights


def _solver(model, weights, *, streaming=False):
    solver = SolverLinearFEM(
        model,
        tet_volume_weights=weights,
        dt=1.0 / 120,
        solver_type="cg",
        preconditioner_type="block_diag",
        linear_solver_maxiter=30,
        self_contact_ke=0.0,
    )
    if streaming:
        solver.prepare_tetrahedral_updates(_ORIGINAL_NODES)
    return solver


def _deformed(model):
    positions = model.particle_q.numpy().copy()
    positions[:, 0] += 0.17 * positions[:, 2]
    positions[:, 1] -= 0.08 * positions[:, 0]
    positions[:, 2] *= 0.91
    return wp.array(positions, dtype=wp.vec3, device=model.device)


def _dense(matrix):
    offsets, columns, blocks = matrix.offsets.numpy(), matrix.columns.numpy(), matrix.values.numpy()
    result = np.zeros((matrix.nrow * 3, matrix.ncol * 3))
    for row in range(matrix.nrow):
        for slot in range(offsets[row], offsets[row + 1]):
            column = columns[slot]
            result[3 * row : 3 * row + 3, 3 * column : 3 * column + 3] += blocks[slot]
    return result


def _sources(device, entries):
    values = np.full(_CAPACITY, -1, dtype=np.int32)
    values[: len(entries)] = entries
    return wp.array(values, dtype=int, device=device)


def _matrix_pointers(matrix):
    return tuple(array.ptr for array in (matrix.offsets, matrix.columns, matrix.values))


def _captured_pointers(solver):
    return (
        *_matrix_pointers(solver.A_bsr),
        *_matrix_pointers(solver._streaming.compact_matrix),
        solver._pc_inv_diag.ptr,
        solver.dv.ptr,
    )


class TestLinearFEMStreaming(unittest.TestCase):
    def _check_matrix(self, solver, positions, records):
        model, weights = _model(solver.model.device, records)
        reference = _solver(model, weights)
        reference.refresh_tangent(positions)
        expected = _dense(reference.A_bsr)
        np.testing.assert_allclose(_dense(solver.A_bsr), expected, rtol=3e-5, atol=3e-6)
        np.testing.assert_allclose(solver._pc_inv_diag.numpy(), reference._pc_inv_diag.numpy(), rtol=3e-5, atol=3e-6)
        random = np.random.default_rng(725)
        x = wp.array(random.normal(size=(model.particle_count, 3)), dtype=wp.vec3, device=model.device)
        y_values = random.normal(size=(model.particle_count, 3)).astype(np.float32)
        y = wp.array(y_values, dtype=wp.vec3, device=model.device)
        z = wp.empty_like(y)
        solver._streaming.operator.matvec(x, y, z, 0.7, -0.3)
        expected_product = (0.7 * expected @ x.numpy().ravel()).reshape((-1, 3)) - 0.3 * y_values
        np.testing.assert_allclose(z.numpy(), expected_product, rtol=3e-5, atol=4e-6)
        solver._streaming.operator.matvec(x, y, y, 0.7, -0.3)
        np.testing.assert_allclose(y.numpy(), expected_product, rtol=3e-5, atol=4e-6)

        rhs = wp.array(y_values, dtype=wp.vec3, device=model.device)
        reference_rhs = wp.clone(rhs)
        solver._streaming.subtract_predictor(x, rhs)
        wp.launch(
            subtract_elastic_predictor,
            dim=model.particle_count,
            inputs=[reference.A_bsr.offsets, reference.A_bsr.columns, reference.A_bsr.values, x, reference_rhs],
            device=model.device,
        )
        np.testing.assert_allclose(rhs.numpy(), reference_rhs.numpy(), rtol=3e-5, atol=4e-6)

    def _check_split_and_removal(self, device):
        model, weights = _model(device)
        solver = _solver(model, weights, streaming=True)
        positions = _deformed(model)
        solver.refresh_tangent(positions)
        self._check_matrix(solver, positions, _INITIAL)

        _edit(model, weights, _SPLIT)
        solver.refresh_topology(positions, element_sources=_sources(device, (1, 0, 0)))
        self._check_matrix(solver, positions, _SPLIT)
        self.assertEqual(solver._streaming.last_update["dirty_elements"], 2)
        self.assertEqual(solver._streaming.last_update["reused_elements"], 1)
        self.assertEqual(solver._streaming.last_update["active_elements"], 3)

        # The removed child shares three original nodes with the retained tet.
        # Its contribution must vanish without erasing the retained stiffness.
        retained = _SPLIT[:2]
        _edit(model, weights, retained)
        solver.refresh_topology(positions)
        self._check_matrix(solver, positions, retained)
        self.assertEqual(solver._streaming.last_update["dirty_elements"], 0)
        self.assertEqual(solver._streaming.last_update["removed_elements"], 1)

        _edit(model, weights, ())
        solver.refresh_topology(positions)
        self._check_matrix(solver, positions, ())
        self.assertEqual(solver._streaming.last_update["active_elements"], 0)
        state, output = model.state(), model.state()
        solver.step(state, output, model.control(), None, solver.dt)
        np.testing.assert_array_equal(output.particle_q.numpy(), state.particle_q.numpy())
        np.testing.assert_array_equal(output.particle_qd.numpy(), 0.0)

    def test_split_shuffled_lineage_shared_edge_removal_cpu(self):
        self._check_split_and_removal("cpu")

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_DEVICE", "cpu").startswith("cuda") and wp.is_cuda_available(),
        "Set NEWTON_TEST_DEVICE=cuda:0 for CUDA",
    )
    def test_split_shuffled_lineage_shared_edge_removal_cuda(self):
        self._check_split_and_removal("cuda:0")

    def test_moved_unchanged_tet_refreshes_current_tangent(self):
        model, weights = _model("cpu")
        solver = _solver(model, weights, streaming=True)
        positions = _deformed(model)
        solver.refresh_tangent(positions)
        moved = positions.numpy()
        moved[4] += (0.15, -0.05, 0.09)
        positions.assign(moved)
        solver.refresh_topology(positions)
        self._check_matrix(solver, positions, _INITIAL)
        self.assertEqual(solver._streaming.last_update["dirty_elements"], 1)
        self.assertEqual(solver._streaming.last_update["reused_elements"], 1)
        self.assertEqual(solver._streaming.last_update["touched_rows"], 0)
        self.assertEqual(solver._streaming.last_update["topology_inserted_elements"], 0)
        self.assertEqual(solver._streaming.last_update["topology_removed_elements"], 0)

    def test_legacy_rebuild_preserves_prepared_storage(self):
        model, weights = _model("cpu")
        solver = _solver(model, weights, streaming=True)
        pointers = _matrix_pointers(solver.A_bsr)
        columns = solver.A_bsr.columns.numpy().copy()
        positions = _deformed(model)
        solver._rebuild_element_blocks(positions, solver.dt)
        self.assertEqual(pointers, _matrix_pointers(solver.A_bsr))
        np.testing.assert_array_equal(solver.A_bsr.columns.numpy(), columns)
        self._check_matrix(solver, positions, _INITIAL)

    def test_reject_unreserved_edge_without_mutating_assembly(self):
        model, weights = _model("cpu")
        solver = _solver(model, weights, streaming=True)
        positions = _deformed(model)
        solver.refresh_tangent(positions)
        arrays = (solver.A_bsr.values, solver._pc_inv_diag, solver._streaming.stiffness_values)
        saved = [array.numpy().copy() for array in arrays]
        # Original nodes 3 and 4 never shared an original tetrahedron.
        _edit(model, weights, ((0, (0, 1, 3, 4), 1.0),))
        with self.assertRaises(ValueError):
            solver.refresh_topology(positions)
        for array, expected in zip(arrays, saved, strict=True):
            np.testing.assert_array_equal(array.numpy(), expected)
        _edit(model, weights, _INITIAL)
        solver.refresh_topology(positions)
        self._check_matrix(solver, positions, _INITIAL)
        self.assertEqual(solver._streaming.last_update["dirty_elements"], 0)

    def test_invalid_source_rejected_before_assembly_changes(self):
        model, weights = _model("cpu")
        solver = _solver(model, weights, streaming=True)
        positions = _deformed(model)
        solver.refresh_tangent(positions)
        expected = _dense(solver.A_bsr)
        with self.assertRaises(ValueError):
            solver.refresh_topology(positions, element_sources=_sources("cpu", (_CAPACITY, 1)))
        np.testing.assert_array_equal(_dense(solver.A_bsr), expected)

    @unittest.skipUnless(
        os.environ.get("NEWTON_TEST_DEVICE", "cpu").startswith("cuda") and wp.is_cuda_available(),
        "Set NEWTON_TEST_DEVICE=cuda:0 for CUDA",
    )
    def test_original_capture_replays_after_multiple_edits(self):
        device = "cuda:0"
        model, weights = _model(device)
        solver = _solver(model, weights, streaming=True)
        state, output = model.state(), model.state()
        wp.copy(state.particle_q, _deformed(model))
        state.particle_qd.assign(state.particle_q.numpy() * (0.1, -0.05, 0.02))
        control = model.control()
        solver.refresh_tangent(state.particle_q)
        solver.step(state, output, control, None, solver.dt)
        pointers = _captured_pointers(solver)
        with wp.ScopedCapture(device=device) as capture:
            solver.refresh_tangent(state.particle_q)
            solver.step(state, output, control, None, solver.dt)
        for records, sources in ((_SPLIT, (1, 0, 0)), (_SPLIT[:2], (0, 1))):
            _edit(model, weights, records)
            solver.refresh_topology(state.particle_q, element_sources=_sources(device, sources))
            wp.capture_launch(capture.graph)
            reference_model, reference_weights = _model(device, records)
            reference = _solver(reference_model, reference_weights)
            expected = reference_model.state()
            reference.refresh_tangent(state.particle_q)
            reference.step(state, expected, reference_model.control(), None, reference.dt)
            self.assertEqual(pointers, _captured_pointers(solver))
            np.testing.assert_allclose(output.particle_q.numpy(), expected.particle_q.numpy(), rtol=2e-5, atol=2e-6)
            np.testing.assert_allclose(output.particle_qd.numpy(), expected.particle_qd.numpy(), rtol=1e-4, atol=2e-5)


if __name__ == "__main__":
    unittest.main()

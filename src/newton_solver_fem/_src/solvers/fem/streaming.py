# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent sparse assembly for cuts that duplicate tetrahedral basis nodes.

The candidate matrix contains the original adjacency lifted to reserved node
copies. Numerical assembly and matrix products visit only live entries. Element
lineage permits subtraction and addition of changed contributions without
sorting the global matrix after a cut.
"""

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator
from warp.sparse import bsr_mv, bsr_set_from_triplets, bsr_zeros

from .kernels import inverse_diagonal_block


@wp.func
def _find_slot(row: int, column: int, offsets: wp.array[int], columns: wp.array[int]):
    first, end = int(offsets[row]), int(offsets[row + 1])
    while first < end:
        middle = (first + end) // 2
        if columns[middle] < column:
            first = middle + 1
        else:
            end = middle
    if first < offsets[row + 1] and columns[first] == column:
        return first
    return int(-1)


@wp.kernel(enable_backward=False)
def _stage_elements(
    nodes: wp.array2d[int],
    poses: wp.array[wp.mat33],
    materials: wp.array2d[float],
    positions: wp.array[wp.vec3],
    sources: wp.array[int],
    old_nodes: wp.array2d[int],
    old_poses: wp.array[wp.mat33],
    old_materials: wp.array2d[float],
    old_positions: wp.array[wp.vec3],
    old_active: wp.array[int],
    old_slots: wp.array[int],
    original_node_count: int,
    offsets: wp.array[int],
    columns: wp.array[int],
    slots: wp.array[int],
    active: wp.array[int],
    dirty: wp.array[int],
    reuse: wp.array[int],
    kept: wp.array[int],
    topology_kept: wp.array[int],
    retained_sources: wp.array[int],
    counters: wp.array[int],
):
    element = wp.tid()
    active[element] = 0
    dirty[element] = 0
    reuse[element] = -1
    retained_sources[element] = -1
    for block in range(16):
        slots[16 * element + block] = -1
    enabled = materials[element, 0] != 0.0 or materials[element, 1] != 0.0 or materials[element, 2] != 0.0
    if not enabled:
        return
    source = sources[element]
    if source < -1 or source >= old_active.shape[0]:
        wp.atomic_max(counters, 0, 1)
        return
    for corner in range(4):
        node = nodes[element, corner]
        if node < 0 or node >= positions.shape[0]:
            wp.atomic_max(counters, 0, 2)
            return
        for earlier in range(corner):
            if node % original_node_count == nodes[element, earlier] % original_node_count:
                wp.atomic_max(counters, 0, 4)
                return

    same = source >= 0
    if same:
        same = old_active[source] != 0
    if same:
        for corner in range(4):
            if nodes[element, corner] != old_nodes[source, corner]:
                same = False
    if same:
        for block in range(16):
            slots[16 * element + block] = old_slots[16 * source + block]
        if wp.atomic_cas(topology_kept, source, 0, 1) == 0:
            retained_sources[element] = source
    else:
        for row in range(4):
            for column in range(4):
                slot = _find_slot(nodes[element, row], nodes[element, column], offsets, columns)
                if slot < 0:
                    wp.atomic_max(counters, 0, 3)
                    return
                slots[16 * element + 4 * row + column] = slot
    if retained_sources[element] < 0:
        wp.atomic_add(counters, 6, 1)
        same = False
    if same:
        for corner in range(4):
            node = nodes[element, corner]
            for axis in range(3):
                if positions[node][axis] != old_positions[node][axis]:
                    same = False
        for row in range(3):
            if materials[element, row] != old_materials[source, row]:
                same = False
            for column in range(3):
                if poses[element][row, column] != old_poses[source][row, column]:
                    same = False
    if same:
        # A source may have several children. Its old contribution can be
        # retained exactly once, even if two children have identical data.
        same = wp.atomic_cas(kept, source, 0, 1) == 0
    active[element] = 1
    wp.atomic_add(counters, 4, 1)
    if same:
        reuse[element] = source
        wp.atomic_add(counters, 3, 1)
    else:
        dirty[element] = 1
        wp.atomic_add(counters, 1, 1)


@wp.kernel(enable_backward=False)
def _remove_elements(
    active: wp.array[int],
    kept: wp.array[int],
    topology_kept: wp.array[int],
    nodes: wp.array2d[int],
    slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    stiffness: wp.array[wp.mat33],
    references: wp.array[int],
    touched: wp.array[int],
    counters: wp.array[int],
    full_refresh: bool,
):
    block = wp.tid()
    element = block // 16
    if active[element] == 0 or kept[element] != 0:
        return
    if block % 16 == 0:
        wp.atomic_add(counters, 2, 1)
        if topology_kept[element] == 0:
            wp.atomic_add(counters, 7, 1)
    slot = slots[block]
    if not full_refresh:
        wp.atomic_sub(stiffness, slot, blocks[block])
    if topology_kept[element] == 0:
        wp.atomic_sub(references, slot, 1)
        if block % 4 == 0:
            wp.atomic_max(touched, nodes[element, (block % 16) // 4], 1)


@wp.kernel(enable_backward=False)
def _reuse_blocks(
    reuse: wp.array[int],
    active: wp.array[int],
    nodes: wp.array2d[int],
    old_blocks: wp.array[wp.mat33],
    block_offset: int,
    rows: wp.array[int],
    columns: wp.array[int],
    blocks: wp.array[wp.mat33],
):
    element = wp.tid()
    source = reuse[element]
    if active[element] == 0 or source >= 0:
        for row in range(4):
            for column in range(4):
                block = 16 * element + 4 * row + column
                value = wp.mat33(0.0)
                i, j = int(-1), int(-1)
                if source >= 0:
                    value = old_blocks[16 * source + 4 * row + column]
                    i = int(nodes[element, row])
                    j = int(nodes[element, column])
                rows[block_offset + block] = i
                columns[block_offset + block] = j
                blocks[block_offset + block] = value


@wp.kernel(enable_backward=False)
def _add_elements(
    dirty: wp.array[int],
    retained_sources: wp.array[int],
    nodes: wp.array2d[int],
    slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    block_offset: int,
    stiffness: wp.array[wp.mat33],
    references: wp.array[int],
    touched: wp.array[int],
    full_refresh: bool,
):
    block = wp.tid()
    element = block // 16
    if dirty[element] == 0:
        return
    slot = slots[block]
    if not full_refresh:
        wp.atomic_add(stiffness, slot, blocks[block_offset + block])
    if retained_sources[element] < 0:
        wp.atomic_add(references, slot, 1)
        if block % 4 == 0:
            wp.atomic_max(touched, nodes[element, (block % 16) // 4], 1)


@wp.kernel(enable_backward=False)
def _compact_rows(
    touched: wp.array[int],
    offsets: wp.array[int],
    columns: wp.array[int],
    references: wp.array[int],
    stiffness: wp.array[wp.mat33],
    values: wp.array[wp.mat33],
    active_slots: wp.array[int],
    active_counts: wp.array[int],
    counters: wp.array[int],
):
    row = wp.tid()
    if touched[row] == 0:
        return
    wp.atomic_add(counters, 5, 1)
    count = int(0)
    for slot in range(offsets[row], offsets[row + 1]):
        if references[slot] == 0:
            # Exact clearing removes subtraction roundoff when the last
            # incident element disappears, including a disconnected diagonal.
            stiffness[slot] = wp.mat33(0.0)
            values[slot] = wp.mat33(0.0)
        if references[slot] > 0 or columns[slot] == row:
            active_slots[offsets[row] + count] = slot
            count += 1
    active_counts[row] = count


@wp.kernel(enable_backward=False)
def _clear_active(compact_offsets: wp.array[int], compact_to_candidate: wp.array[int], stiffness: wp.array[wp.mat33]):
    slot = wp.tid()
    if slot < compact_offsets[compact_offsets.shape[0] - 1]:
        stiffness[compact_to_candidate[slot]] = wp.mat33(0.0)


@wp.kernel(enable_backward=False)
def _assemble_elements(
    active: wp.array[int],
    slots: wp.array[int],
    blocks: wp.array[wp.mat33],
    block_offset: int,
    stiffness: wp.array[wp.mat33],
):
    index = wp.tid()
    element = index // 16
    if active[element] != 0:
        wp.atomic_add(stiffness, slots[index], blocks[block_offset + index])


@wp.kernel(enable_backward=False)
def _scan_chunks(counts: wp.array[int], offsets: wp.array[int], totals: wp.array[int]):
    chunk = wp.tid()
    total = int(0)
    for row in range(256 * chunk, wp.min(256 * (chunk + 1), counts.shape[0])):
        offsets[row] = total
        total += counts[row]
    totals[chunk] = total


@wp.kernel(enable_backward=False)
def _scan_chunk_totals(totals: wp.array[int], prefix: wp.array[int], offsets: wp.array[int]):
    total = int(0)
    for chunk in range(totals.shape[0]):
        prefix[chunk] = total
        total += totals[chunk]
    offsets[offsets.shape[0] - 1] = total


@wp.kernel(enable_backward=False)
def _add_chunk_prefix(prefix: wp.array[int], offsets: wp.array[int]):
    row = wp.tid()
    offsets[row] += prefix[row // 256]


@wp.kernel(enable_backward=False)
def _map_compact_slots(
    offsets: wp.array[int],
    columns: wp.array[int],
    active_slots: wp.array[int],
    active_counts: wp.array[int],
    compact_offsets: wp.array[int],
    compact_columns: wp.array[int],
    compact_to_candidate: wp.array[int],
    compact_rows: wp.array[int],
):
    row = wp.tid()
    for local in range(active_counts[row]):
        slot = active_slots[offsets[row] + local]
        compact_slot = compact_offsets[row] + local
        compact_columns[compact_slot] = columns[slot]
        compact_to_candidate[compact_slot] = slot
        compact_rows[compact_slot] = row


@wp.kernel(enable_backward=False)
def _restore_values(
    stiffness: wp.array[wp.mat33],
    masses: wp.array[float],
    values: wp.array[wp.mat33],
    compact_offsets: wp.array[int],
    compact_columns: wp.array[int],
    compact_to_candidate: wp.array[int],
    compact_rows: wp.array[int],
    compact_values: wp.array[wp.mat33],
):
    compact_slot = wp.tid()
    if compact_slot < compact_offsets[compact_offsets.shape[0] - 1]:
        slot = compact_to_candidate[compact_slot]
        row = compact_rows[compact_slot]
        value = stiffness[slot]
        if compact_columns[compact_slot] == row:
            value += wp.diag(wp.vec3(masses[row]))
        values[slot] = value
        compact_values[compact_slot] = value


@wp.kernel(enable_backward=False)
def _gather_values(
    values: wp.array[wp.mat33],
    compact_offsets: wp.array[int],
    compact_to_candidate: wp.array[int],
    compact_values: wp.array[wp.mat33],
):
    slot = wp.tid()
    if slot < compact_offsets[compact_offsets.shape[0] - 1]:
        compact_values[slot] = values[compact_to_candidate[slot]]


@wp.kernel(enable_backward=False)
def _subtract_predictor(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[wp.mat33],
    velocity: wp.array[wp.vec3],
    rhs: wp.array[wp.vec3],
):
    row = wp.tid()
    correction = wp.vec3(0.0)
    for slot in range(offsets[row], offsets[row + 1]):
        correction += values[slot] * (velocity[columns[slot]] - velocity[row])
    rhs[row] -= correction


@wp.kernel(enable_backward=False)
def _invert_diagonal(
    diagonal_slots: wp.array[int],
    values: wp.array[wp.mat33],
    inverse: wp.array[wp.mat33],
):
    row = wp.tid()
    inverse[row] = inverse_diagonal_block(values[diagonal_slots[row]])


class StreamingTetAssembly:
    """Reserved sparse graph and local assembly for duplicated Tet4 nodes.

    This internal path requires tetrahedra only and a block-Jacobi
    preconditioner. The node pool consists of equal-size copies of the original
    nodes. Every accepted edge must belong to the original tetrahedral graph.
    Geometry, materials, and state are validated by the caller before entering
    this assembly layer; candidate sparse edges and source IDs are checked here
    before changing any accepted matrix data.
    """

    def __init__(self, solver, original_node_count: int):
        self.solver, self.model = solver, solver.model
        model = self.model
        count = int(original_node_count)
        if count < 1 or model.particle_count % count:
            raise ValueError("The particle pool must contain whole copies of the original node set")
        if model.tet_count < 1 or any((solver.hex_count, model.spring_count, model.tri_count)):
            raise ValueError("Streaming assembly requires tetrahedral elements only")
        if solver.preconditioner_type != "block_diag":
            raise ValueError("Streaming assembly requires the block_diag preconditioner")
        self.device = model.device
        self.node_count, self.element_count = model.particle_count, model.tet_count
        self.original_node_count = count
        self.copies = model.particle_count // count
        nodes = model.tet_indices.numpy()
        materials = model.tet_materials.numpy()
        adjacency = [set() for _ in range(count)]
        for element in nodes[np.any(materials != 0.0, axis=1)]:
            if np.any(element < 0) or np.any(element >= self.node_count):
                raise ValueError("Tetrahedral node indices are outside the reserved particle pool")
            original = element % count
            if len(set(original.tolist())) != 4:
                raise ValueError("Each tetrahedron must have four distinct original basis nodes")
            for node in original:
                adjacency[node].update(int(other) for other in original if other != node)

        row_columns = []
        offsets = [0]
        diagonal = []
        for row in range(self.node_count):
            neighbours = sorted(adjacency[row % count])
            columns = np.asarray(
                sorted([row, *(node + copy * count for copy in range(self.copies) for node in neighbours)]),
                dtype=np.int32,
            )
            diagonal.append(offsets[-1] + int(np.searchsorted(columns, row)))
            row_columns.append(columns)
            offsets.append(offsets[-1] + len(columns))
        columns = np.concatenate(row_columns)
        rows = np.repeat(np.arange(self.node_count, dtype=np.int32), np.diff(offsets))
        matrix = bsr_zeros(self.node_count, self.node_count, wp.mat33, device=self.device, nnz_capacity=len(columns))
        bsr_set_from_triplets(
            matrix,
            wp.array(rows, dtype=int, device=self.device),
            wp.array(columns, dtype=int, device=self.device),
            wp.zeros(len(columns), dtype=wp.mat33, device=self.device),
            prune_numerical_zeros=False,
        )
        self.matrix = matrix
        self.offsets, self.columns = matrix.offsets, matrix.columns
        self.diagonal_slots = wp.array(diagonal, dtype=int, device=self.device)
        self.stiffness_values = wp.zeros_like(matrix.values)
        self.references = wp.zeros(matrix.values.size, dtype=int, device=self.device)
        self.active_slots = wp.empty(matrix.values.size, dtype=int, device=self.device)
        self.active_counts = wp.zeros(self.node_count, dtype=int, device=self.device)
        self.touched = wp.ones(self.node_count, dtype=int, device=self.device)
        self.counters = wp.zeros(8, dtype=int, device=self.device)
        self.old_nodes = wp.zeros_like(model.tet_indices)
        self.old_poses = wp.zeros_like(model.tet_poses)
        self.old_materials = wp.zeros_like(model.tet_materials)
        self.linearization_positions = wp.zeros_like(model.particle_q)
        self.old_active = wp.zeros(self.element_count, dtype=int, device=self.device)
        self.old_slots = wp.empty(16 * self.element_count, dtype=int, device=self.device)
        self.cached_blocks = wp.zeros(16 * self.element_count, dtype=wp.mat33, device=self.device)
        self.active = wp.zeros_like(self.old_active)
        self.dirty = wp.zeros_like(self.old_active)
        self.reuse = wp.empty_like(self.old_active)
        self.kept = wp.zeros_like(self.old_active)
        self.topology_kept = wp.zeros_like(self.old_active)
        self.retained_sources = wp.empty_like(self.old_active)
        self.slots = wp.empty_like(self.old_slots)
        self.identity_sources = wp.array(np.arange(self.element_count, dtype=np.int32), device=self.device)
        self.last_update = {}
        # Krylov products use contiguous numerical rows. The larger candidate
        # graph is only the stable address space for local assembly and holds.
        # This private scratch matrix uses device row offsets, not host nnz.
        # Only forward bsr_mv(tile_size=-1) may consume it; generic BSR algebra
        # can depend on host nnz and must use solver.A_bsr instead.
        self.compact_matrix = bsr_zeros(
            self.node_count,
            self.node_count,
            wp.mat33,
            device=self.device,
            nnz_capacity=self.node_count + 16 * self.element_count,
        )
        self.compact_capacity = self.node_count + 16 * self.element_count
        self.compact_to_candidate = wp.empty(self.compact_capacity, dtype=int, device=self.device)
        self.compact_rows = wp.empty_like(self.compact_to_candidate)
        self.scan_totals = wp.empty((self.node_count + 255) // 256, dtype=int, device=self.device)
        self.scan_prefix = wp.empty_like(self.scan_totals)

        def matvec(x, y, z, alpha, beta):
            if beta != 0.0 and y.ptr != z.ptr:
                wp.copy(z, y)
            bsr_mv(self.compact_matrix, x, z, alpha=alpha, beta=beta, tile_size=-1)

        self.operator = LinearOperator((self.node_count, self.node_count), wp.vec3, self.device, matvec)
        # Every row, including unused pool copies, initially needs its mass
        # diagonal in the compact active list.
        self._compact()
        self.update(model.particle_q, solver.dt)
        solver.A_bsr = matrix

    def _launch(self, kernel, dim, inputs):
        wp.launch(kernel, dim=dim, inputs=inputs, device=self.device)

    def _compact(self):
        self._launch(
            _compact_rows,
            self.node_count,
            [
                self.touched,
                self.offsets,
                self.columns,
                self.references,
                self.stiffness_values,
                self.matrix.values,
                self.active_slots,
                self.active_counts,
                self.counters,
            ],
        )
        self._launch(
            _scan_chunks, self.scan_totals.size, [self.active_counts, self.compact_matrix.offsets, self.scan_totals]
        )
        self._launch(_scan_chunk_totals, 1, [self.scan_totals, self.scan_prefix, self.compact_matrix.offsets])
        self._launch(_add_chunk_prefix, self.node_count, [self.scan_prefix, self.compact_matrix.offsets])
        self._launch(
            _map_compact_slots,
            self.node_count,
            [
                self.offsets,
                self.columns,
                self.active_slots,
                self.active_counts,
                self.compact_matrix.offsets,
                self.compact_matrix.columns,
                self.compact_to_candidate,
                self.compact_rows,
            ],
        )

    def _restore_matrix(self):
        self._launch(
            _restore_values,
            self.compact_capacity,
            [
                self.stiffness_values,
                self.model.particle_mass,
                self.matrix.values,
                self.compact_matrix.offsets,
                self.compact_matrix.columns,
                self.compact_to_candidate,
                self.compact_rows,
                self.compact_matrix.values,
            ],
        )

    def _save_blocks(self):
        wp.copy(
            self.cached_blocks,
            self.solver.bsr_values,
            src_offset=self.solver._tet_bsr_offset,
            count=16 * self.element_count,
        )

    def refresh(self, particle_q, dt):
        """Reassemble changing tangents without sorting or touching unused entries."""
        self.solver._tet_all_dirty.fill_(1)
        self.solver._write_element_blocks(particle_q, dt)
        self._launch(
            _clear_active,
            self.compact_capacity,
            [self.compact_matrix.offsets, self.compact_to_candidate, self.stiffness_values],
        )
        self._launch(
            _assemble_elements,
            16 * self.element_count,
            [
                self.old_active,
                self.old_slots,
                self.solver.bsr_values,
                self.solver._tet_bsr_offset,
                self.stiffness_values,
            ],
        )
        self._restore_matrix()
        wp.copy(self.linearization_positions, particle_q)

    def update(self, particle_q, dt, sources=None):
        """Publish exact local element deltas after checking candidate adjacency."""
        if self.device.is_capturing:
            raise RuntimeError("Topology validation must run between CUDA graph launches")
        sources = self.identity_sources if sources is None else sources
        if sources.dtype != wp.int32 or sources.ndim != 1 or sources.size < self.element_count:
            raise ValueError("Element sources must be a capacity-sized int32 array")
        if sources.device != self.device:
            raise ValueError("Element sources must reside on the solver device")
        self.kept.zero_()
        self.topology_kept.zero_()
        self.counters.zero_()
        self._launch(
            _stage_elements,
            self.element_count,
            [
                self.model.tet_indices,
                self.model.tet_poses,
                self.model.tet_materials,
                particle_q,
                sources,
                self.old_nodes,
                self.old_poses,
                self.old_materials,
                self.linearization_positions,
                self.old_active,
                self.old_slots,
                self.original_node_count,
                self.offsets,
                self.columns,
                self.slots,
                self.active,
                self.dirty,
                self.reuse,
                self.kept,
                self.topology_kept,
                self.retained_sources,
                self.counters,
            ],
        )
        staged = self.counters.numpy()
        errors = {
            1: "Element source is outside the previous element capacity",
            2: "Tetrahedral node is outside the reserved particle pool",
            3: "Cut introduces an edge outside the reserved original-node adjacency",
            4: "A tetrahedron must have four distinct original basis nodes",
        }
        if staged[0]:
            raise ValueError(errors.get(int(staged[0]), "Invalid sparse topology update"))

        # When most tangents have changed, one clean accumulation costs less
        # than subtracting and adding every numerical block. The sparse graph
        # still changes only where connectivity changed.
        full_refresh = int(staged[1]) > int(staged[4]) // 2
        if not full_refresh or int(staged[3]) != 0:
            self._save_blocks()
        self.touched.zero_()
        self._launch(
            _remove_elements,
            16 * self.element_count,
            [
                self.old_active,
                self.kept,
                self.topology_kept,
                self.old_nodes,
                self.old_slots,
                self.cached_blocks,
                self.stiffness_values,
                self.references,
                self.touched,
                self.counters,
                full_refresh,
            ],
        )
        self._launch(
            _reuse_blocks,
            self.element_count,
            [
                self.reuse,
                self.active,
                self.model.tet_indices,
                self.cached_blocks,
                self.solver._tet_bsr_offset,
                self.solver.bsr_rows,
                self.solver.bsr_cols,
                self.solver.bsr_values,
            ],
        )
        wp.copy(self.solver._tet_all_dirty, self.dirty)
        self.solver._write_element_blocks(particle_q, dt)
        self._launch(
            _add_elements,
            16 * self.element_count,
            [
                self.dirty,
                self.retained_sources,
                self.model.tet_indices,
                self.slots,
                self.solver.bsr_values,
                self.solver._tet_bsr_offset,
                self.stiffness_values,
                self.references,
                self.touched,
                full_refresh,
            ],
        )
        self._compact()
        if full_refresh:
            self._launch(
                _clear_active,
                self.compact_capacity,
                [self.compact_matrix.offsets, self.compact_to_candidate, self.stiffness_values],
            )
            self._launch(
                _assemble_elements,
                16 * self.element_count,
                [self.active, self.slots, self.solver.bsr_values, self.solver._tet_bsr_offset, self.stiffness_values],
            )
        self._restore_matrix()
        wp.copy(self.linearization_positions, particle_q)
        for target, source in (
            (self.old_nodes, self.model.tet_indices),
            (self.old_poses, self.model.tet_poses),
            (self.old_materials, self.model.tet_materials),
            (self.old_active, self.active),
            (self.old_slots, self.slots),
        ):
            wp.copy(target, source)
        result = self.counters.numpy()
        self.last_update = {
            "dirty_elements": int(result[1]),
            "removed_elements": int(result[2]),
            "reused_elements": int(result[3]),
            "active_elements": int(result[4]),
            "touched_rows": int(result[5]),
            "topology_inserted_elements": int(result[6]),
            "topology_removed_elements": int(result[7]),
            "full_numeric_refresh": full_refresh,
        }
        return self.last_update

    def subtract_predictor(self, velocity, rhs):
        matrix = self.compact_matrix
        self._launch(
            _subtract_predictor, self.node_count, [matrix.offsets, matrix.columns, matrix.values, velocity, rhs]
        )

    def refresh_preconditioner(self, inverse_diagonal):
        self._launch(
            _invert_diagonal,
            self.node_count,
            [
                self.diagonal_slots,
                self.matrix.values,
                inverse_diagonal,
            ],
        )
        self._launch(
            _gather_values,
            self.compact_capacity,
            [self.matrix.values, self.compact_matrix.offsets, self.compact_to_candidate, self.compact_matrix.values],
        )

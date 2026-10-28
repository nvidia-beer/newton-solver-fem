# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Capturable material connectivity and deterministic parent-basis node copies."""

import numpy as np
import warp as wp

from .fragment_connectivity import FragmentConnectivity


@wp.func
def _root(parents: wp.array[int], index: int):
    current = index
    while parents[current] != current:
        current = parents[current]
    return current


@wp.func
def _join(parents: wp.array[int], first: int, second: int):
    a, b = first, second
    while True:
        a, b = _root(parents, a), _root(parents, b)
        if a == b:
            break
        lower, upper = wp.min(a, b), wp.max(a, b)
        if wp.atomic_cas(parents, upper, upper, lower) == upper:
            break


@wp.kernel(enable_backward=False)
def _validate(
    count: wp.array[int],
    edge_count: wp.array[int],
    overflow: wp.array[int],
    fragment_capacity: int,
    edge_capacity: int,
    error: wp.array[int],
):
    if (
        count[0] < 0
        or count[0] > fragment_capacity
        or edge_count[0] < 0
        or edge_count[0] > edge_capacity
        or overflow[0] != 0
    ):
        error[0] = 3


@wp.kernel(enable_backward=False)
def _initialize(
    parent: wp.array[int],
    incoming: wp.array2d[int],
    count: wp.array[int],
    element_count: int,
    pool_count: int,
    basis: wp.array[int],
    regions: wp.array[int],
    material: wp.array[int],
    inherited: wp.array2d[int],
    group_source: wp.array[int],
    assigned: wp.array[int],
    used: wp.array[int],
    output: wp.array2d[int],
    error: wp.array[int],
):
    fragment, corner = wp.tid()
    if fragment >= count[0] or error[0] != 0:
        return
    index = 4 * fragment + corner
    basis[index] = index
    group_source[index] = -1
    assigned[index] = -1
    output[fragment, corner] = -1
    source = incoming[fragment, corner]
    inherited[fragment, corner] = source
    if source < 0 or source >= pool_count or parent[fragment] < 0 or parent[fragment] >= element_count:
        wp.atomic_max(error, 0, 1)
    else:
        wp.atomic_max(used, source, 1)
    if corner == 0:
        regions[fragment] = fragment
        material[fragment] = fragment


@wp.kernel(enable_backward=False)
def _join_edges(
    parent: wp.array[int],
    elements: wp.array2d[int],
    inherited: wp.array2d[int],
    count: wp.array[int],
    edges: wp.array2d[int],
    edge_count: wp.array[int],
    basis: wp.array[int],
    regions: wp.array[int],
    material: wp.array[int],
    error: wp.array[int],
):
    edge = wp.tid()
    if edge >= edge_count[0] or error[0] != 0:
        return
    first, second = edges[edge, 0], edges[edge, 1]
    if first < 0 or first >= count[0] or second < 0 or second >= count[0]:
        wp.atomic_max(error, 0, 1)
        return
    a, b = parent[first], parent[second]
    _join(material, first, second)
    if a == b:
        _join(regions, first, second)
    for i in range(4):
        for j in range(4):
            if elements[a, i] == elements[b, j]:
                if inherited[first, i] != inherited[second, j]:
                    wp.atomic_max(error, 0, 1)
                else:
                    _join(basis, 4 * first + i, 4 * second + j)


@wp.kernel(enable_backward=False)
def _compress(
    count: wp.array[int],
    basis: wp.array[int],
    regions: wp.array[int],
    material: wp.array[int],
    inherited: wp.array2d[int],
    group_source: wp.array[int],
    group_size: wp.array[int],
    error: wp.array[int],
):
    fragment, corner = wp.tid()
    if fragment >= count[0] or error[0] != 0:
        return
    index = 4 * fragment + corner
    root = _root(basis, index)
    basis[index] = root
    wp.atomic_add(group_size, root, 1)
    source = inherited[fragment, corner]
    previous = wp.atomic_cas(group_source, root, -1, source)
    if previous != -1 and previous != source:
        wp.atomic_max(error, 0, 1)
    if corner == 0:
        regions[fragment] = _root(regions, fragment)
        material[fragment] = _root(material, fragment)


@wp.kernel(enable_backward=False)
def _register_groups(
    count: wp.array[int],
    basis: wp.array[int],
    group_source: wp.array[int],
    group_count: wp.array[int],
    groups: wp.array2d[int],
    error: wp.array[int],
):
    root = wp.tid()
    if root >= 4 * count[0] or error[0] != 0 or basis[root] != root:
        return
    source = group_source[root]
    index = wp.atomic_add(group_count, source, 1)
    if index >= groups.shape[1]:
        # Every disconnected group needs a distinct copy of this original
        # node. More groups than the entire copy pool cannot be accepted.
        wp.atomic_max(error, 0, 2)
    else:
        groups[source, index] = root


@wp.kernel(enable_backward=False)
def _allocate_copies(
    node_count: int,
    max_copies: int,
    group_count: wp.array[int],
    groups: wp.array2d[int],
    group_size: wp.array[int],
    used: wp.array[int],
    assigned: wp.array[int],
    clone_sources: wp.array2d[int],
    clone_count: wp.array[int],
    error: wp.array[int],
):
    original = wp.tid()
    if error[0] != 0:
        return
    # Distinct threads own distinct original-node families. Within each family,
    # source IDs and (-group size, root ID) follow the CPU reference's order.
    for copy in range(max_copies):
        source = original + copy * node_count
        for rank in range(group_count[source]):
            selected, size = int(-1), int(-1)
            for group in range(group_count[source]):
                root = groups[source, group]
                current = group_size[root]
                if assigned[root] == -1 and (current > size or (current == size and root < selected)):
                    selected, size = root, current
            if selected < 0:
                wp.atomic_max(error, 0, 1)
                return
            target = source
            if rank > 0:
                target = int(-1)
                for candidate in range(max_copies):
                    node = original + candidate * node_count
                    if used[node] == 0 and target == -1:
                        target = node
                if target < 0:
                    wp.atomic_max(error, 0, 2)
                    return
                used[target] = 1
                clone = wp.atomic_add(clone_count, 0, 1)
                clone_sources[clone, 0] = target
                clone_sources[clone, 1] = source
            assigned[selected] = target


@wp.kernel(enable_backward=False)
def _write_nodes(
    count: wp.array[int], basis: wp.array[int], assigned: wp.array[int], output: wp.array2d[int], error: wp.array[int]
):
    fragment, corner = wp.tid()
    if fragment < count[0] and error[0] == 0:
        output[fragment, corner] = assigned[basis[4 * fragment + corner]]


class FragmentConnectivityGPU:
    """Resolve adjacency and allocate exact FN field copies entirely on device.

    The adjacency producer supplies compact positive-area fragment pairs and a
    device count. ``launch`` has no readback or allocation and can be inserted
    directly into the producer's CUDA capture. Outputs retain fixed addresses.
    Error codes reject the entire proposed edit: 1 invalid/reconnecting fields,
    2 exhausted node-copy capacity, 3 exceeded geometry/edge capacity.
    Material coordinates are copied later by the edit publisher from the compact
    ``clone_sources`` pairs. No averaging or penalty weld is introduced.
    """

    def __init__(self, elements, node_count, *, fragment_capacity=2048, edge_capacity=65536, max_copies=4, device=None):
        topology = elements.numpy() if isinstance(elements, wp.array) else np.asarray(elements)
        if topology.ndim != 2 or topology.shape[1] != 4 or not np.issubdtype(topology.dtype, np.integer):
            raise ValueError("elements must have integer shape (parent_count, 4)")
        if (
            min(node_count, fragment_capacity, edge_capacity, max_copies) < 1
            or np.any(topology < 0)
            or np.any(topology >= node_count)
        ):
            raise ValueError("positive capacities and valid original node IDs are required")
        self.device = wp.get_device(device)
        self.node_count, self.max_copies = int(node_count), int(max_copies)
        self.fragment_capacity, self.edge_capacity = int(fragment_capacity), int(edge_capacity)
        pool_count = node_count * max_copies
        self.elements = wp.array(topology, dtype=int, device=self.device)

        def empty(shape):
            return wp.empty(shape, dtype=int, device=self.device)

        self.basis, self.group_source, self.group_size, self.assigned = (empty(4 * fragment_capacity) for _ in range(4))
        self.parent_regions, self.material_components = empty(fragment_capacity), empty(fragment_capacity)
        self.inherited, self.nodes = empty((fragment_capacity, 4)), empty((fragment_capacity, 4))
        self.used, self.group_count = empty(pool_count), empty(pool_count)
        self.groups = empty((pool_count, max_copies))
        self.clone_sources = empty((pool_count, 2))
        self.clone_count, self.error = empty(1), empty(1)
        self._no_overflow = wp.zeros(1, dtype=int, device=self.device)
        self._inputs = None
        wp.load_module(module=__name__, device=self.device, block_dim=256)

    def launch(self, parents, inherited_nodes, count, edges, edge_count, overflow=None):
        """Queue fixed-capacity connectivity kernels; inspect ``error`` before commit."""
        overflow = self._no_overflow if overflow is None else overflow
        self._inputs = parents, inherited_nodes, count, edges, edge_count, overflow
        self.error.zero_()
        self.clone_count.zero_()
        self.group_size.zero_()
        self.group_count.zero_()
        self.used.zero_()

        def run(kernel, dim, inputs):
            wp.launch(kernel, dim, inputs=inputs, device=self.device)

        run(_validate, 1, [count, edge_count, overflow, self.fragment_capacity, self.edge_capacity, self.error])
        run(
            _initialize,
            (self.fragment_capacity, 4),
            [
                parents,
                inherited_nodes,
                count,
                len(self.elements),
                self.node_count * self.max_copies,
                self.basis,
                self.parent_regions,
                self.material_components,
                self.inherited,
                self.group_source,
                self.assigned,
                self.used,
                self.nodes,
                self.error,
            ],
        )
        run(
            _join_edges,
            self.edge_capacity,
            [
                parents,
                self.elements,
                self.inherited,
                count,
                edges,
                edge_count,
                self.basis,
                self.parent_regions,
                self.material_components,
                self.error,
            ],
        )
        run(
            _compress,
            (self.fragment_capacity, 4),
            [
                count,
                self.basis,
                self.parent_regions,
                self.material_components,
                self.inherited,
                self.group_source,
                self.group_size,
                self.error,
            ],
        )
        run(
            _register_groups,
            4 * self.fragment_capacity,
            [count, self.basis, self.group_source, self.group_count, self.groups, self.error],
        )
        run(
            _allocate_copies,
            self.node_count,
            [
                self.node_count,
                self.max_copies,
                self.group_count,
                self.groups,
                self.group_size,
                self.used,
                self.assigned,
                self.clone_sources,
                self.clone_count,
                self.error,
            ],
        )
        run(_write_nodes, (self.fragment_capacity, 4), [count, self.basis, self.assigned, self.nodes, self.error])

    def result(self):
        """Read an owned CPU reference result outside capture (diagnostics/tests)."""
        if self._inputs is None:
            raise RuntimeError("Launch connectivity before reading a result")
        error = int(self.error.numpy()[0])
        if error:
            reasons = {
                1: "invalid adjacency would reconnect separated fields",
                2: "node-copy capacity exhausted",
                3: "geometry or adjacency capacity exceeded",
            }
            raise ValueError(reasons[error])
        count = int(self._inputs[2].numpy()[0])
        clones = int(self.clone_count.numpy()[0])
        return FragmentConnectivity(
            self.nodes.numpy()[:count].copy(),
            self.clone_sources.numpy()[:clones].copy(),
            self.parent_regions.numpy()[:count].copy(),
            self.material_components.numpy()[:count].copy(),
        )

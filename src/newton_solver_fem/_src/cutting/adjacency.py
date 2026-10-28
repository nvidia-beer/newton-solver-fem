# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reuse accepted fragment connections outside parents touched by a cut."""

import numpy as np
import warp as wp

from .fragment_adjacency import FragmentAdjacency
from .fragment_device import FragmentArrays


@wp.kernel(enable_backward=False)
def _map_unchanged(
    geometry: FragmentArrays,
    changed: wp.array[int],
    accepted_parents: wp.array[int],
    accepted_count: wp.array[int],
    old_to_new: wp.array[int],
    error: wp.array[int],
):
    fragment = wp.tid()
    if geometry.error[0] != 0 or geometry.count[0] > geometry.capacity:
        wp.atomic_max(error, 0, 3)
        return
    if fragment >= geometry.count[0] or error[0] != 0:
        return
    parent = geometry.parent[fragment]
    if parent < 0 or parent >= changed.shape[0]:
        wp.atomic_max(error, 0, 3)
        return
    if changed[parent] == 0:
        source = geometry.source[fragment]
        if source < 0 or source >= accepted_count[0]:
            wp.atomic_max(error, 0, 3)
            return
        if accepted_parents[source] != parent:
            wp.atomic_max(error, 0, 3)
            return
        # Untouched fragments have a one-to-one lineage, even if compaction
        # shuffled their slots. Duplicate sources would reuse incorrect edges.
        if wp.atomic_cas(old_to_new, source, -1, fragment) != -1:
            wp.atomic_max(error, 0, 3)


@wp.kernel(enable_backward=False)
def _validate_unchanged(
    changed: wp.array[int],
    accepted_parents: wp.array[int],
    accepted_count: wp.array[int],
    old_to_new: wp.array[int],
    error: wp.array[int],
):
    old = wp.tid()
    if old < accepted_count[0] and error[0] == 0:
        if changed[accepted_parents[old]] == 0 and old_to_new[old] < 0:
            wp.atomic_max(error, 0, 3)


@wp.kernel(enable_backward=False)
def _reuse_edges(
    accepted_edges: wp.array2d[int],
    accepted_count: wp.array[int],
    old_to_new: wp.array[int],
    edges: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
):
    edge = wp.tid()
    if edge >= accepted_count[0] or error[0] != 0:
        return
    first = old_to_new[accepted_edges[edge, 0]]
    second = old_to_new[accepted_edges[edge, 1]]
    if first >= 0 and second >= 0:
        destination = wp.atomic_add(count, 0, 1)
        if destination >= edges.shape[0]:
            wp.atomic_max(error, 0, 1)
        else:
            edges[destination, 0] = first
            edges[destination, 1] = second


@wp.kernel(enable_backward=False)
def _stage_metadata(
    geometry: FragmentArrays,
    parents: wp.array[int],
    count: wp.array[int],
    error: wp.array[int],
):
    index = wp.tid()
    if index < wp.min(geometry.count[0], geometry.capacity):
        parents[index] = geometry.parent[index]
    if index == 0:
        count[0] = geometry.count[0]
        error[0] = geometry.error[0]


@wp.kernel(enable_backward=False)
def _commit(
    staged_parents: wp.array[int],
    staged_count: wp.array[int],
    staged_error: wp.array[int],
    edges: wp.array2d[int],
    count: wp.array[int],
    error: wp.array[int],
    accepted_edges: wp.array2d[int],
    accepted_edge_count: wp.array[int],
    accepted_parents: wp.array[int],
    accepted_fragment_count: wp.array[int],
):
    index = wp.tid()
    if (
        error[0] != 0
        or staged_error[0] != 0
        or count[0] < 0
        or count[0] > accepted_edges.shape[0]
        or staged_count[0] < 0
        or staged_count[0] > accepted_parents.shape[0]
    ):
        return
    if index < count[0]:
        accepted_edges[index, 0] = edges[index, 0]
        accepted_edges[index, 1] = edges[index, 1]
    if index < staged_count[0]:
        accepted_parents[index] = staged_parents[index]
    if index == 0:
        accepted_edge_count[0] = count[0]
        accepted_fragment_count[0] = staged_count[0]


class CutAdjacency:
    """Recompute changed-parent interfaces and remap all other accepted edges.

    The geometry editor must mark every modified parent and preserve
    ``geometry.source`` as the previous fragment slot for unchanged parents.
    Original interfaces have at most two incident parent tetrahedra. Auxiliary
    interfaces are local to one parent, and cut faces never connect material.

    Initialize once before graph capture. :meth:`propose` and :meth:`commit`
    only launch kernels on persistent buffers. A proposal does not change the
    accepted cache; call ``commit`` after the complete geometry/mechanics edit
    succeeds. Results use the wrapped adjacency's existing edge/count/overflow
    arrays, so downstream connectivity launches keep the same addresses.
    """

    def __init__(self, adjacency: FragmentAdjacency, interface_parents):
        parents = np.asarray(interface_parents, dtype=np.int32)
        if parents.ndim != 2 or parents.shape[1] != 2 or np.any(parents < -1):
            raise ValueError("Original interface parents must have two columns, with -1 for an exterior side")
        self.adjacency = adjacency
        self.device = adjacency.device
        self.interface_parents = wp.array(parents, dtype=int, device=self.device)
        self.accepted_edges = wp.empty_like(adjacency.edges)
        self.accepted_edge_count = wp.zeros(1, dtype=int, device=self.device)
        self.accepted_fragment_count = wp.zeros(1, dtype=int, device=self.device)
        self.accepted_parents = self.old_to_new = None
        self._staged_parents = None
        self._staged_count = wp.zeros(1, dtype=int, device=self.device)
        self._staged_error = wp.zeros(1, dtype=int, device=self.device)
        self._maximum_parent = int(parents.max()) if parents.size else -1

    def initialize(self, geometry, rest, elements):
        """Build and retain the initial complete adjacency before graph capture."""
        if self.device.is_capturing:
            raise RuntimeError("Initialize cut adjacency before graph capture")
        if self.accepted_parents is not None:
            raise RuntimeError("Cut adjacency is already initialized")
        if self._maximum_parent >= elements.shape[0]:
            raise ValueError("The original interface map contains an out-of-range parent")
        self.accepted_parents = wp.empty_like(geometry.parent)
        self.old_to_new = wp.empty_like(geometry.parent)
        self._staged_parents = wp.empty_like(geometry.parent)
        self.adjacency.pack_device_geometry(geometry, rest, elements)
        self.adjacency.launch()
        if int(self.adjacency.overflow.numpy()[0]):
            raise ValueError("Initial fragment adjacency exceeds its capacity")
        self._stage(geometry)
        self.commit()

    def _stage(self, geometry):
        # Different captured line/sphere graphs may use different geometry
        # buffers. Record the chosen metadata on the device during each replay.
        wp.launch(
            _stage_metadata,
            geometry.capacity,
            inputs=[geometry.arrays, self._staged_parents, self._staged_count, self._staged_error],
            device=self.device,
        )

    def propose(self, geometry, changed, rest, elements):
        """Stage local connections plus unchanged accepted edges, without reads."""
        if self.accepted_parents is None:
            raise RuntimeError("Initialize cut adjacency before proposing an edit")
        if geometry.parent.shape != self.accepted_parents.shape:
            raise ValueError("Cut adjacency requires the original reserved fragment capacity")
        self._stage(geometry)
        self.adjacency.pack_device_geometry(
            geometry,
            rest,
            elements,
            active_parents=changed,
            interface_parents=self.interface_parents,
        )
        self.adjacency.launch()
        self.old_to_new.fill_(-1)
        wp.launch(
            _map_unchanged,
            geometry.capacity,
            inputs=[
                geometry.arrays,
                changed,
                self.accepted_parents,
                self.accepted_fragment_count,
                self.old_to_new,
                self.adjacency.overflow,
            ],
            device=self.device,
        )
        wp.launch(
            _validate_unchanged,
            geometry.capacity,
            inputs=[
                changed,
                self.accepted_parents,
                self.accepted_fragment_count,
                self.old_to_new,
                self.adjacency.overflow,
            ],
            device=self.device,
        )
        wp.launch(
            _reuse_edges,
            self.adjacency.edge_capacity,
            inputs=[
                self.accepted_edges,
                self.accepted_edge_count,
                self.old_to_new,
                self.adjacency.edges,
                self.adjacency.edge_count,
                self.adjacency.overflow,
            ],
            device=self.device,
        )

    def commit(self):
        """Retain valid staged edges after the caller accepts the complete edit."""
        if self._staged_parents is None:
            raise RuntimeError("No fragment adjacency has been prepared")
        wp.launch(
            _commit,
            max(self.adjacency.edge_capacity, self.accepted_parents.shape[0]),
            inputs=[
                self._staged_parents,
                self._staged_count,
                self._staged_error,
                self.adjacency.edges,
                self.adjacency.edge_count,
                self.adjacency.overflow,
                self.accepted_edges,
                self.accepted_edge_count,
                self.accepted_parents,
                self.accepted_fragment_count,
            ],
            device=self.device,
        )

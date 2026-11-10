# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Allocation-free CUDA mesh refit for the installed Warp Mesh/BVH layout.

Triangle bounds and bottom-up BVH updates follow Warp's native algorithms.
The edge-length mean uses deterministic double-precision chunk reductions of
identical float32 edge sums. Only its reduction roundoff differs from CUB scan.
No stale bounds, normals or average edge length are retained. Winding-number
meshes are explicitly unsupported because they need extra solid-angle refits.
The native structs and refit algorithm are version guarded to Warp 1.17.0.
This helper does not replace Mesh.refit or alter installed Warp.
"""

import warp as wp


@wp.func_native("""
const wp::Mesh& mesh = *(const wp::Mesh*)id;
info.data[0] = mesh.num_tris;
info.data[1] = mesh.bvh.num_leaf_nodes;
info.data[2] = mesh.bvh.max_nodes;
info.data[3] = (mesh.solid_angle_props != nullptr);
""")
def _metadata(id: wp.uint64, info: wp.array[int]): ...


@wp.kernel(enable_backward=False)
def metadata(id: wp.uint64, info: wp.array[int]):
    _metadata(id, info)


@wp.func_native("""
wp::Mesh& mesh = *(wp::Mesh*)id;
if (index < mesh.num_tris) {
    int i=mesh.indices.data[3*index];
    int j=mesh.indices.data[3*index+1];
    int k=mesh.indices.data[3*index+2];
    wp::vec3 p=mesh.points.data[i], q=mesh.points.data[j], r=mesh.points.data[k];
    mesh.lowers[index] = wp::min(wp::min(p,q),r);
    mesh.uppers[index] = wp::max(wp::max(p,q),r);
    edges.data[index] = wp::length(p-q)+wp::length(p-r)+wp::length(q-r);
}
if (index < mesh.bvh.max_nodes) mesh.bvh.node_counts[index] = 0;
""")
def _triangle(id: wp.uint64, index: int, edges: wp.array[float]): ...


@wp.kernel(enable_backward=False)
def triangle_data(id: wp.uint64, edges: wp.array[float]):
    _triangle(id, wp.tid(), edges)


@wp.kernel(enable_backward=False)
def reduce_edges(edges: wp.array[float], partial: wp.array[wp.float64]):
    chunk = wp.tid()
    total = wp.float64(0)
    start = chunk * 256
    for index in range(start, wp.min(start + 256, edges.shape[0])):
        total += wp.float64(edges[index])
    partial[chunk] = total


@wp.func_native("""
wp::Mesh& mesh = *(wp::Mesh*)id;
mesh.average_edge_length = static_cast<float>(total / static_cast<double>(3 * mesh.num_tris));
""")
def _set_mean(id: wp.uint64, total: wp.float64): ...


@wp.kernel(enable_backward=False)
def set_mean(id: wp.uint64, partial: wp.array[wp.float64]):
    total = wp.float64(0)
    for i in range(partial.shape[0]):
        total += partial[i]
    _set_mean(id, total)


# The loop below is copied from Warp 1.17 native/bvh.cu bvh_refit_kernel.
# Retain its threadfence and last-child atomic handoff exactly.
@wp.func_native("""
#if defined(__CUDA_ARCH__)
wp::Mesh& mesh = *(wp::Mesh*)id;
wp::BVH& bvh = mesh.bvh;
const int n = bvh.num_leaf_nodes;
const int* parents=bvh.node_parents;
int* child_count=bvh.node_counts;
const int* primitive_indices=bvh.primitive_indices;
wp::BVHPackedNodeHalf* node_lowers=bvh.node_lowers;
wp::BVHPackedNodeHalf* node_uppers=bvh.node_uppers;
const wp::vec3* item_lowers=bvh.item_lowers;
const wp::vec3* item_uppers=bvh.item_uppers;
using wp::BVHPackedNodeHalf;
using wp::bounds3;
using wp::vec3;
using wp::min;
using wp::max;
    if (index < n) {
        bool leaf = node_lowers[index].b;
        int parent = parents[index];

        if (leaf) {
            BVHPackedNodeHalf& lower = node_lowers[index];
            BVHPackedNodeHalf& upper = node_uppers[index];
            // update the leaf node

            // Single-node trees store the root leaf without a parent.
            if (parent == -1 || !node_lowers[parent].b) {
                const int start = lower.i;
                const int end = upper.i;

                bounds3 bound;
                for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                    const int primitive = primitive_indices[primitive_counter];
                    bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                }
                (vec3&)lower = bound.lower;
                (vec3&)upper = bound.upper;
            }
        } else {
            // only keep leaf threads
            return;
        }

        // update hierarchy
        for (;;) {
            parent = parents[index];
            // reached root
            if (parent == -1)
                return;

            // ensure all writes are visible
            __threadfence();

            int finished = atomicAdd(&child_count[parent], 1);

            // if we have are the last thread (such that the parent node is now complete)
            // then update its bounds and move onto the next parent in the hierarchy
            if (finished == 1) {
                BVHPackedNodeHalf& parent_lower = node_lowers[parent];
                BVHPackedNodeHalf& parent_upper = node_uppers[parent];
                if (parent_lower.b)
                // a packed leaf node can still be a parent in LBVH, we need to recompute its bounds
                // since we've lost its left and right child node index in the muting process
                {
                    // update the leaf node
                    int parent_parent = parents[parent];

                    // only need to compute bound when this is a valid leaf node
                    if (parent_parent == -1 || !node_lowers[parent_parent].b) {
                        const int start = parent_lower.i;
                        const int end = parent_upper.i;
                        bounds3 bound;
                        for (int primitive_counter = start; primitive_counter < end; primitive_counter++) {
                            const int primitive = primitive_indices[primitive_counter];
                            bound.add_bounds(item_lowers[primitive], item_uppers[primitive]);
                        }

                        (vec3&)parent_lower = bound.lower;
                        (vec3&)parent_upper = bound.upper;
                    }
                } else {
                    const int left_child = parent_lower.i;
                    const int right_child = parent_upper.i;

                    vec3 left_lower = (vec3&)(node_lowers[left_child]);
                    vec3 left_upper = (vec3&)(node_uppers[left_child]);
                    vec3 right_lower = (vec3&)(node_lowers[right_child]);
                    vec3 right_upper = (vec3&)(node_uppers[right_child]);

                    // union of child bounds
                    vec3 lower = min(left_lower, right_lower);
                    vec3 upper = max(left_upper, right_upper);

                    // write new BVH nodes
                    (vec3&)parent_lower = lower;
                    (vec3&)parent_upper = upper;
                }
                // move onto processing the parent
                index = parent;
            } else {
                // parent not ready (we are the first child), terminate thread
                break;
            }
        }
    }
#endif
""")
def _refit_leaf(id: wp.uint64, index: int): ...


@wp.kernel(enable_backward=False)
def refit_leaves(id: wp.uint64):
    _refit_leaf(id, wp.tid())


class GraphMeshRefit:
    """Persistent scratch for one fixed-topology, non-winding CUDA mesh.

    Construct outside capture, then call :meth:`refit` after changing positions
    in place. Construction reads four metadata integers once; refitting only
    launches kernels. Rebuild this helper if mesh storage or topology changes.
    The caller must supply finite positions and valid indices, as for Mesh.refit.
    """

    def __init__(self, mesh):
        if wp.__version__ != "1.17.0":
            raise ValueError("GraphMeshRefit supports only the validated Warp 1.17.0 native Mesh/BVH layout")
        if not mesh.device.is_cuda:
            raise ValueError("GraphMeshRefit requires a CUDA mesh")
        self.mesh = mesh
        self.device = mesh.device
        info = wp.empty(4, dtype=int, device=self.device)
        wp.launch(metadata, 1, [mesh.id, info], device=self.device)
        count, leaves, nodes, winding = map(int, info.numpy())
        if winding:
            raise ValueError("GraphMeshRefit does not support winding-number meshes")
        if count <= 0 or leaves <= 0 or nodes <= 0:
            raise ValueError("GraphMeshRefit requires a nonempty valid BVH")
        self.count, self.leaves, self.nodes = count, leaves, nodes
        self.edges = wp.empty(count, dtype=float, device=self.device)
        self.partial = wp.empty((count + 255) // 256, dtype=wp.float64, device=self.device)
        self.points, self.indices = mesh.points, mesh.indices

    def refit(self):
        if self.mesh.points is not self.points or self.mesh.indices is not self.indices:
            raise ValueError("Rebuild GraphMeshRefit after replacing mesh storage")
        wp.launch(triangle_data, max(self.count, self.nodes), [self.mesh.id, self.edges], device=self.device)
        wp.launch(reduce_edges, self.partial.size, [self.edges, self.partial], device=self.device)
        wp.launch(set_mean, 1, [self.mesh.id, self.partial], device=self.device)
        wp.launch(refit_leaves, self.leaves, [self.mesh.id], device=self.device)

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in visualization of the material samples attached by each surface grip."""

import warp as wp


@wp.kernel(enable_backward=False)
def _reset_components(ranges: wp.array2d[int]):
    grip = wp.tid()
    ranges[grip, 0] = 2147483647
    ranges[grip, 1] = -1


@wp.kernel(enable_backward=False)
def _component_ranges(
    fragment_nodes: wp.array2d[int],
    components: wp.array[int],
    fragment_count: int,
    anchor_nodes: wp.array2d[int],
    enabled: wp.array[int],
    samples_per_grip: int,
    ranges: wp.array2d[int],
):
    anchor = wp.tid()
    if enabled[anchor] == 0:
        return
    lower, upper = int(2147483647), int(-1)
    for fragment in range(fragment_count):
        component = components[fragment]
        if component < 0:
            continue
        same = bool(True)
        for corner in range(4):
            if fragment_nodes[fragment, corner] != anchor_nodes[anchor, corner]:
                same = False
        if same:
            lower, upper = wp.min(lower, component), wp.max(upper, component)
    grip = anchor // samples_per_grip
    wp.atomic_min(ranges, grip, 0, lower)
    wp.atomic_max(ranges, grip, 1, upper)


@wp.kernel(enable_backward=False)
def _hold_lines(
    q: wp.array[wp.vec3],
    nodes: wp.array2d[int],
    weights: wp.array[wp.vec4],
    enabled: wp.array[int],
    offsets: wp.array[wp.vec3],
    driven: wp.array[wp.vec3],
    samples_per_grip: int,
    starts: wp.array[wp.vec3],
    ends: wp.array[wp.vec3],
):
    anchor = wp.tid()
    point, target = wp.vec3(0.0), wp.vec3(0.0)
    if enabled[anchor] != 0:
        for corner in range(4):
            point += weights[anchor][corner] * q[nodes[anchor, corner]]
        target = driven[anchor // samples_per_grip] + offsets[anchor]
    starts[anchor] = point
    ends[anchor] = target


class HoldDebug:
    """Draw each enabled material anchor to its world target on the GPU.

    Call ``rebuild`` after a cut or attachment change, only while diagnostics
    are enabled. It downloads two component labels per grip, never positions.
    ``render`` uses persistent device buffers and performs no host readback.
    """

    def __init__(self, holds, device, *, name="cut_debug/holds"):
        self.holds, self.device, self.name = holds, wp.get_device(device), str(name)
        if self.device != holds.device:
            raise ValueError("Hold diagnostics must use the attachments' device")
        self.starts = wp.zeros(holds.anchor_capacity, dtype=wp.vec3, device=self.device)
        self.ends = wp.zeros_like(self.starts)
        self.colors = wp.full(holds.anchor_capacity, wp.vec3(0.2, 1.0, 0.3), dtype=wp.vec3, device=self.device)
        self._ranges = wp.zeros((holds.capacity, 2), dtype=int, device=self.device)
        self.spanning_grips = ()
        self._visible = False

    def rebuild(self, fragment_nodes, components, fragment_count):
        """Identify grips attached to multiple material components.

        ``fragment_count`` selects the active prefix of reserved topology
        arrays. Matching all four basis nodes avoids confusing neighbouring
        material samples that merely share a tetrahedral corner.
        """
        if (
            fragment_nodes.dtype != wp.int32
            or fragment_nodes.ndim != 2
            or fragment_nodes.shape[1] != 4
            or components.dtype != wp.int32
            or components.ndim != 1
            or fragment_nodes.device != self.device
            or components.device != self.device
            or not isinstance(fragment_count, int)
            or not 0 <= fragment_count <= min(fragment_nodes.shape[0], components.shape[0])
        ):
            raise ValueError("Hold diagnostics require a valid active prefix of device fragment nodes and labels")
        holds = self.holds
        if not holds.count:
            self.spanning_grips = ()
            return
        wp.launch(_reset_components, dim=holds.capacity, inputs=[self._ranges], device=self.device)
        wp.launch(
            _component_ranges,
            dim=holds.anchor_capacity,
            inputs=[
                fragment_nodes,
                components,
                fragment_count,
                holds.nodes,
                holds.enabled,
                holds.samples_per_grip,
                self._ranges,
            ],
            device=self.device,
        )
        self.spanning_grips = tuple(
            index for index, (lower, upper) in enumerate(self._ranges.numpy()[: holds.count]) if lower < upper
        )

    def render(self, viewer, q, *, hidden=False):
        """Draw green attachment lines, updating their current endpoints [m]."""
        if hidden:
            self.hide(viewer)
            return
        holds = self.holds
        wp.launch(
            _hold_lines,
            dim=holds.anchor_capacity,
            inputs=[
                q,
                holds.nodes,
                holds.weights,
                holds.enabled,
                holds.offsets,
                holds.driven,
                holds.samples_per_grip,
                self.starts,
                self.ends,
            ],
            device=self.device,
        )
        viewer.log_lines(self.name, self.starts, self.ends, self.colors)
        self._visible = True

    def hide(self, viewer):
        """Hide previously submitted lines once, with no device kernel launches."""
        if self._visible:
            viewer.log_lines(self.name, self.starts, self.ends, self.colors, hidden=True)
            self._visible = False

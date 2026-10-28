# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Opt-in controls and revision caching for cut connectivity inspection."""

from newton_solver_fem import CutConnectivityDebug


class CutDebugView:
    """Refresh topology only on demand; interpolate debug lines on the device."""

    def __init__(self, viewer, device, topology, *, kind):
        self.viewer, self.topology, self.kind = viewer, topology, kind
        self.connections = CutConnectivityDebug(device=device)
        self._revision = None
        self._hide_skin = True

    @property
    def enabled(self):
        return self.connections.enabled

    @property
    def hide_skin(self):
        return self.enabled and self._hide_skin

    def invalidate(self):
        self._revision = None

    def render(self, q, revision):
        if self.enabled and revision != self._revision:
            geometry, components, nodes = self.topology()
            self.connections.rebuild(geometry, components, basis_nodes=nodes)
            self._revision = revision
        self.connections.render(self.viewer, q, kind=self.kind)

    def gui(self, ui):
        changed, self.connections.enabled = ui.checkbox("Debug element connections", self.enabled)
        if changed:
            self.invalidate()
        if not self.enabled:
            return
        _, self._hide_skin = ui.checkbox("Hide skin", self._hide_skin)
        _, self.connections.show_wireframe = ui.checkbox("Show material element edges", self.connections.show_wireframe)
        _, self.connections.show_connections = ui.checkbox(
            "Show retained connections", self.connections.show_connections
        )
        if ui.button("Refresh connection diagnostics"):
            self.invalidate()
        debug = self.connections
        ui.text_wrapped(f"Material pieces: {debug.component_count} / fragments: {debug.fragment_count}")
        ui.text_wrapped(f"Shared nodes across pieces: {debug.shared_node_count} / red links: {debug.bridge_count}")
        ui.text_wrapped(f"Cut-face connections still retained: {debug.incision_connection_count}")
        ui.text_wrapped("Orange: opposing cut faces still sharing nodes. Red: shared nodes across separate pieces.")
        ui.text_wrapped("Yellow: other retained connections. Element edge colors identify material pieces.")
        ui.text_wrapped("Contact and world holds are separate forces.")

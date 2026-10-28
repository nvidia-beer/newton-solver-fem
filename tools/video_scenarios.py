"""Script public example interactions for representative FEM recordings."""

import numpy as np

_CUT_STROKES = {
    30: (0, -0.30, (-1.0, -0.05), (1.0, 1.0)),
    75: (0, 0.30, (-1.0, -0.05), (1.0, 1.0)),
    120: (0, -0.15, (-1.0, -0.05), (1.0, 1.0)),
    165: (0, 0.15, (-1.0, -0.05), (1.0, 1.0)),
    210: (0, 0.0, (-1.0, -0.05), (1.0, 1.0)),
    255: (2, 0.22, (-1.0, -1.0), (1.0, 1.0)),
}


def configure(name, example):
    if name == "linear_fem":
        example.viewer.set_camera(pos=(4.2, 0.0, 2.2), pitch=-15.0, yaw=180.0)
    elif name == "linear_fem_baymax_cut":
        example.viewer.set_camera(pos=(0.0, -1.6, 0.7), pitch=-13.0, yaw=90.0)
    elif name == "diffsim_bear":
        example.viewer.set_camera(pos=(13.0, -10.0, 8.0), pitch=-17.0, yaw=142.5)


def finite_incision(axis, coordinate, lower, upper):
    """Two finite triangles spanning a rectangular scalpel stroke."""
    corners = np.zeros((4, 3), dtype=np.float64)
    corners[:, axis] = coordinate
    remaining = [index for index in range(3) if index != axis]
    corners[:, remaining] = (lower, (upper[0], lower[1]), upper, (lower[0], upper[1]))
    return corners[[[0, 1, 2], [0, 2, 3]]]


def before_frame(name, example, frame):
    if name != "linear_fem_baymax_cut":
        return
    if frame in _CUT_STROKES:
        if not example.cut(finite_incision(*_CUT_STROKES[frame])):
            raise RuntimeError(f"The recorded incision was not accepted: {example.cut_error}")
        pieces = len(np.unique(example.cutter.material_components.numpy()[: example.fragment_count]))
        print(f"Recorded incision {example.cut_revision}: {pieces} material pieces", flush=True)
    if frame == 330:
        # This is the same control as "Debug element connections" in the viewer.
        # Its default Hide skin option reveals the material edges, colored by piece.
        example.connection_debug.connections.enabled = True
        example.connection_debug.invalidate()
        example.viewer.set_camera(pos=(0.0, -1.25, 1.25), pitch=-40.0, yaw=90.0)
    if frame == 331:
        debug = example.connection_debug.connections
        if debug.component_count < 2 or debug.shared_node_count:
            raise RuntimeError("Expected separate cut pieces with no shared basis nodes")
        print(f"Debug view: {debug.component_count} pieces, {debug.shared_node_count} shared nodes", flush=True)


def frame_label(name, example):
    if name == "linear_fem_baymax_cut":
        debug = example.connection_debug.connections
        if debug.enabled:
            return f"Debug view | {debug.component_count} material pieces | skin hidden"
        return f"Finite slices: {example.cut_revision}/{len(_CUT_STROKES)} | surface view"
    if name == "diffsim_bear":
        losses = example.problem.loss_history
        if losses:
            return f"Live inference after Adam | objective: {losses[0]:.2f} to {losses[-1]:.2f}"
        return "Live controller inference"
    return "Tet4 and Hex8 | gravity and floor contact"

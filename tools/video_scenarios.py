"""Script public example interactions for representative FEM recordings."""

import json

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
    elif name == "fem_phantom_regions":
        example.recording_events = []
        example.recording_drag = None
    elif name == "fem_suturebot":
        example.recording_seen_frames = set()
        example.recording_view_changes = []
    if name in ("fem_phantom_regions", "fem_suturebot"):
        if name == "fem_suturebot" or getattr(example.screen_pick, "window", None) is not None:
            # Use the ordinary example panel, with unrelated viewer sections
            # collapsed so its clickable pad map remains visible in the clip.
            def collapse_viewer_sections(ui):
                storage = ui.get_state_storage()
                for title in ("Model Information", "Visualization", "Rendering Options"):
                    storage.set_int(ui.get_id(title), 0)

            example.viewer.register_ui_callback(collapse_viewer_sections, position="panel")


def _screen_point(example, point):
    """Project a physical attachment target to the same window used for mouse input."""
    camera = example.viewer.camera
    view = camera.get_view_matrix().reshape((4, 4), order="F")
    projection = camera.get_projection_matrix().reshape((4, 4), order="F")
    clip = projection @ view @ np.append(point, 1.0)
    if clip[3] <= 0 or np.any(np.abs(clip[:3]) > clip[3]):
        raise RuntimeError("The recorded phantom grab is outside the camera view")
    return (clip[:2] / clip[3] + 1.0) * np.asarray(example.screen_pick.window.get_size()) / 2.0


def _phantom_grab_point(example, region_id):
    selection = example.selection
    faces = selection.exterior_surface_triangles
    fixed = np.zeros(len(selection.vertices), dtype=bool)
    fixed[example.fixed_vertex_indices] = True
    candidates = (selection.exterior_surface_region_ids == region_id) & ~np.any(fixed[faces], axis=1)
    corners = selection.vertices[faces]
    centers = corners.mean(axis=1)
    normals = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    candidates &= normals @ np.asarray(example.view_direction) < 0.0
    if not np.any(candidates):
        raise RuntimeError(f"Region {region_id} has no visible, unconstrained exterior face to grab")
    candidates &= centers[:, 2] >= np.quantile(centers[candidates, 2], 0.65)
    available = np.flatnonzero(candidates)
    middle = np.median(centers[available], axis=0)
    return centers[available[np.argmin(np.linalg.norm(centers[available] - middle, axis=1))]]


def _phantom_begin_drag(example, region_id, frame):
    point = _phantom_grab_point(example, region_id)
    origin = np.asarray(example.view_position)
    direction = point - origin
    direction /= np.linalg.norm(direction)
    tool = example.screen_pick
    if tool.window is not None:
        pixel = _screen_point(example, point)
        tool.on_mouse_press(*pixel, 4, 0)
        if not tool.active:
            raise RuntimeError("The recorded right mouse press did not grab the active FEM surface")
        hit = tool.target.copy()
    else:
        # The profiler has no window. Replay the same ray and physical target
        # through the public callbacks used by ScreenPickTool.
        hit = example.begin_grab(origin, direction)
        pixel = None
        if hit is None:
            raise RuntimeError("The recorded ray did not grab the active FEM surface")
    example.recording_drag = {"point": np.asarray(hit), "pixel": pixel, "frame": frame}
    example.recording_events.append({"frame": frame, "action": "begin_right_drag", "region": region_id})


def _phantom_pointer(example, region_id):
    tool = example.screen_pick
    gui = getattr(example.viewer, "gui", None)
    io = getattr(getattr(gui, "ui", None), "io", None)
    if tool.window is not None and io is not None:
        pixel = _screen_point(example, _phantom_grab_point(example, region_id))
        size = np.asarray(tool.window.get_size())
        framebuffer = np.asarray(tool.window.get_framebuffer_size())
        pixel = pixel * framebuffer / size
        io.add_mouse_pos_event(float(pixel[0]), float(framebuffer[1] - pixel[1]))


def _phantom_move_drag(example, fraction):
    drag = example.recording_drag
    smooth = fraction * fraction * (3.0 - 2.0 * fraction)
    point = drag["point"] + 0.003 * smooth * np.asarray(example.view_up)
    tool = example.screen_pick
    if tool.window is not None:
        pixel = _screen_point(example, point)
        delta = pixel - drag["pixel"]
        tool.on_mouse_drag(*pixel, *delta, 4, 0)
        drag["pixel"] = pixel
    else:
        example.move_grab(point)


def _phantom_finish_drag(example, frame):
    tool = example.screen_pick
    if tool.window is not None:
        tool.on_mouse_release(*example.recording_drag["pixel"], 4, 0)
    else:
        example.finish_grab()
    displacement = np.linalg.norm(example.state_0.particle_q.numpy() - example.selection.vertices, axis=1)
    maximum = float(displacement.max())
    if maximum < 1.0e-5:
        raise RuntimeError("The recorded spring grab did not produce measurable FEM deformation")
    example.test_final()
    example.recording_events.append({"frame": frame, "action": "release_to_hold", "maximum_displacement_m": maximum})
    example.recording_drag = None


def _phantom_frame(example, frame):
    if frame == 0:
        example.test_final()
        example.select_regions((4,))
        example.recording_events.append(
            {
                "frame": frame,
                "action": "select_regions",
                "regions": list(example.selected_region_ids),
                "tetrahedra": len(example.selection.indices),
            }
        )
    # Each selection uses the same map input as the user. Mouse movement,
    # press and release span separate frames so ImGui processes the full click.
    # The null profiler replays the identical selection changes without a UI.
    gui = getattr(example.viewer, "gui", None)
    io = getattr(getattr(gui, "ui", None), "io", None)
    clicks = ((170, 6, (6,)), (330, None, ()), (360, 4, (4,)))
    for clicked_frame, region_id, expected in clicks:
        if frame == clicked_frame - 2 and io is not None:
            pixel = (
                example.region_none_button_position
                if region_id is None
                else example.region_map.screen_position(region_id)
            )
            io.add_mouse_pos_event(*pixel)
        elif frame == clicked_frame:
            example.test_final()
            if io is not None:
                io.add_mouse_button_event(0, True)
            else:
                example.select_regions(expected)
        elif frame == clicked_frame + 1 and io is not None:
            io.add_mouse_button_event(0, False)
        elif frame == clicked_frame + 3:
            if tuple(example.selected_region_ids) != expected:
                raise RuntimeError(
                    f"Pad map click on region {region_id} selected {example.selected_region_ids}; expected {expected}"
                )
            example.recording_events.append(
                {
                    "frame": clicked_frame,
                    "action": (
                        "select_regions"
                        if io is None
                        else "click_none_button"
                        if region_id is None
                        else "click_region_map"
                    ),
                    "clicked_region": region_id,
                    "regions": list(example.selected_region_ids),
                    "tetrahedra": len(example.selection.indices),
                }
            )
    for start, region in ((20, 4), (190, 6), (390, 4)):
        if frame == start - 2:
            _phantom_pointer(example, region)
        elif frame == start:
            _phantom_begin_drag(example, region, frame)
        elif start < frame <= start + 70:
            _phantom_move_drag(example, (frame - start) / 70)
        elif frame == start + 71:
            _phantom_finish_drag(example, frame)
    if frame in (130, 300, 500):
        example.clear_holds()
        example.recording_events.append({"frame": frame, "action": "clear_holds"})


def finite_incision(axis, coordinate, lower, upper):
    """Two finite triangles spanning a rectangular scalpel stroke."""
    corners = np.zeros((4, 3), dtype=np.float64)
    corners[:, axis] = coordinate
    remaining = [index for index in range(3) if index != axis]
    corners[:, remaining] = (lower, (upper[0], lower[1]), upper, (lower[0], upper[1]))
    return corners[[[0, 1, 2], [0, 2, 3]]]


def before_frame(name, example, frame):
    if name == "fem_suturebot":
        example.recording_seen_frames.add(example.frame_index)
        if frame == 120:
            example.frame_phantom()
            example.recording_view_changes.append({"capture_frame": frame, "view": "phantom"})
        elif frame == 180 and "point_labels" in example.recording.manifest:
            labels = example.recording.manifest["point_labels"]
            report = json.loads(example.recording.asset(labels["report"]).read_text())
            center = np.mean([point["position_world_m"] for point in report["points"]], axis=0)
            # Keep the physical-size markers legible after video compression.
            example.viewer.set_camera(pos=center + np.array([-0.06, 0.06, 0.08]))
            example.viewer.camera.look_at(center)
            example.recording_view_changes.append({"capture_frame": frame, "view": "surface point labels"})
        elif frame == 850:
            example.frame_phantom()
            example.recording_view_changes.append({"capture_frame": frame, "view": "phantom"})
        elif frame == 1000:
            example.reset_camera()
            example.recording_view_changes.append({"capture_frame": frame, "view": "complete rig"})
        if frame % 60 == 0:
            example.test_final()
        return
    if name == "fem_phantom_regions":
        _phantom_frame(example, frame)
        return
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
    if name == "fem_suturebot":
        return f"{example.recording.manifest['sample_id']} | frame {example.frame_index} | {example.recording.time_s[example.frame_index]:.2f} s | four camera overlays"
    if name == "fem_phantom_regions":
        regions = ", ".join(str(value) for value in example.selected_region_ids) or "none"
        action = "Right-drag: physical spring grab" if example.recording_drag else "Click pad map | fixed inactive pad"
        return f"Active: {regions} | {len(example.selection.indices):,} tets | {action}"
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


def recording_metadata(name, example):
    if name == "fem_suturebot":
        example.test_final()
        example.recording_seen_frames.add(example.frame_index)
        if example.overlay is None:
            raise RuntimeError("The recording must include all four model/video overlays")
        if not all(count > 0 for count in example.overlay.coverage.values()):
            raise RuntimeError("The recording must render geometry in every camera view")
        return {
            "sample_id": example.recording.manifest["sample_id"],
            "source_episode": example.recording.manifest["source_episode"],
            "source_frame_count": example.recording.frame_count,
            "observed_source_frames": len(example.recording_seen_frames),
            "supported_source_frames": [example.recording.first_valid_frame, example.recording.last_valid_frame],
            "alignment": example.recording.manifest["alignment"],
            "final_overlay_coverage_pixels": example.overlay.coverage,
            "motion": "measured Cartesian/joint kinematic replay",
            "region_id": example.recording.manifest["region_id"],
            "point_labels": example.recording.manifest.get("point_labels"),
            "view_changes": example.recording_view_changes,
        }
    if name == "fem_phantom_regions":
        return {"drag_distance_m": 0.003, "events": example.recording_events}
    return None

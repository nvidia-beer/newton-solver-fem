# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Right-drag a material point and leave a world-space attachment on release."""

from collections.abc import Callable

import numpy as np


def screen_ray(camera, pixel, window_size, framebuffer_size):
    """Return the world ray through a bottom-left window pixel, including DPI scaling."""
    window = np.asarray(window_size, dtype=np.float64)
    framebuffer = np.asarray(framebuffer_size, dtype=np.float64)
    point = np.asarray(pixel, dtype=np.float64)
    if (
        window.shape != (2,)
        or framebuffer.shape != (2,)
        or point.shape != (2,)
        or not np.isfinite(np.concatenate((window, framebuffer, point))).all()
        or np.any(window <= 0)
        or np.any(framebuffer <= 0)
    ):
        raise ValueError("A finite pixel and positive viewport dimensions are required")
    x, y = point * framebuffer / window
    origin, direction = camera.get_world_ray(float(x), float(y))
    origin, direction = np.asarray(origin, dtype=np.float64), np.asarray(direction, dtype=np.float64)
    if (
        origin.shape != (3,)
        or direction.shape != (3,)
        or not np.isfinite(origin).all()
        or not np.isfinite(direction).all()
        or np.linalg.norm(direction) == 0
    ):
        raise ValueError("The camera must provide a finite, nonzero world ray")
    return origin, direction / np.linalg.norm(direction)


def intersect_drag_plane(origin, direction, point, normal):
    """Intersect a camera ray with the fixed plane through the picked point [m]."""
    denominator = float(np.dot(direction, normal))
    if not np.isfinite(denominator) or abs(denominator) < 1.0e-8:
        raise ValueError("The view ray is parallel to the dragging plane")
    distance = float(np.dot(np.asarray(point) - origin, normal)) / denominator
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError("The dragging plane must be in front of the camera")
    return np.asarray(origin) + distance * direction


class ScreenPickTool:
    """Control transient and persistent attachments without reading simulation state.

    ``begin`` raycasts the current surface once, reserves a transient attachment,
    and returns its world position [m], or ``None`` on a miss. ``move`` updates its
    target [m]. ``finish`` makes it persistent; ``cancel`` removes only that
    transient attachment. A backend can therefore keep captured array addresses
    unchanged throughout a drag. Only the initial pick needs a small GPU readback.

    ``remove(index)`` releases a committed attachment on middle-click. It returns
    whether removal succeeded and compacts its storage by moving the last live
    attachment into the removed slot; the marker list mirrors that swap.

    Input handlers outlive same-model resets. Call :meth:`reset` when clearing the
    backend attachments to clear the matching markers as well.
    """

    def __init__(
        self,
        viewer,
        model,
        *,
        begin: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
        move: Callable[[np.ndarray], None],
        finish: Callable[[], None],
        cancel: Callable[[], None],
        remove: Callable[[int], bool] | None = None,
        can_start: Callable[[], bool] | None = None,
    ):
        self.viewer, self.model = viewer, model
        self.window = getattr(getattr(viewer, "renderer", None), "window", None)
        self._begin, self._move, self._finish, self._cancel = begin, move, finish, cancel
        self._remove = remove
        self._can_start = can_start
        self.target = self._plane_point = self._plane_normal = self._offset = None
        self._middle_press = None
        self.pins = []
        self._right, self._middle, self._escape, self._handled = 4, 2, 65307, True
        if self.window is not None:
            import pyglet  # noqa: PLC0415

            self._right, self._escape = pyglet.window.mouse.RIGHT, pyglet.window.key.ESCAPE
            self._middle = pyglet.window.mouse.MIDDLE
            self._handled = pyglet.event.EVENT_HANDLED
            self.window.push_handlers(
                on_mouse_press=self.on_mouse_press,
                on_mouse_drag=self.on_mouse_drag,
                on_mouse_release=self.on_mouse_release,
                on_key_press=self.on_key_press,
                on_deactivate=self.on_deactivate,
                on_resize=self.on_resize,
            )
            viewer.register_ui_callback(self.draw_overlay, position="free")

    @property
    def active(self):
        return self.target is not None

    def _owns_scene(self):
        return getattr(self.viewer, "model", None) is self.model

    def _pointer(self, x, y):
        gui = getattr(self.viewer, "gui", None)
        io = getattr(getattr(gui, "ui", None), "io", None)
        if io is not None:
            width, height = self.window.get_size()
            fb_width, fb_height = self.window.get_framebuffer_size()
            if width > 0 and height > 0:
                io.add_mouse_pos_event(x * fb_width / width, (height - y) * fb_height / height)

    def _ray(self, x, y):
        return screen_ray(self.viewer.camera, (x, y), self.window.get_size(), self.window.get_framebuffer_size())

    def _clear_drag(self):
        self.target = self._plane_point = self._plane_normal = self._offset = None

    def cancel(self):
        """Remove the active attachment, keeping all previously released pins."""
        if self.active:
            self._cancel()
        self._clear_drag()
        self._middle_press = None

    def reset(self):
        """Cancel the current drag and clear markers after resetting backend pins."""
        self.cancel()
        self.pins.clear()

    def on_mouse_press(self, x, y, button, modifiers):
        if not self._owns_scene() or button not in (self._right, self._middle):
            return None
        gui = getattr(self.viewer, "gui", None)
        if gui is not None and gui.is_mouse_capturing():
            return None
        if button == self._middle:
            self._middle_press = None
            if (
                self._remove is not None
                and not self.active
                and not modifiers
                and (self._can_start is None or self._can_start())
            ):
                index = self._pin_at(x, y)
                if index is not None:
                    self._middle_press = (index, np.asarray((x, y)), False)
            # Let the camera receive its normal press/release pair. Only a
            # stationary click on a marker removes a hold; dragging still orbits.
            return None
        if self._can_start is not None and not self._can_start():
            return self._handled
        self.cancel()
        try:
            origin, direction = self._ray(x, y)
        except ValueError:
            return None
        hit = self._begin(origin, direction)
        self._pointer(x, y)
        if hit is None:
            return self._handled
        hit = np.asarray(hit, dtype=np.float64)
        normal = np.asarray(self.viewer.camera.get_front(), dtype=np.float64)
        if hit.shape != (3,) or not np.isfinite(hit).all():
            self._cancel()
            raise ValueError("The picking backend must return a finite world point or None")
        try:
            intersection = intersect_drag_plane(origin, direction, hit, normal)
        except ValueError:
            self._cancel()
            return self._handled
        self.target = self._plane_point = hit.copy()
        self._plane_normal = normal.copy()
        # Preserve the grab offset if a backend chooses a nearby material node.
        self._offset = hit - intersection
        return self._handled

    def _update_target(self, x, y):
        try:
            origin, direction = self._ray(x, y)
            target = intersect_drag_plane(origin, direction, self._plane_point, self._plane_normal) + self._offset
        except ValueError:
            return
        self._move(target)
        self.target = target
        self._pointer(x, y)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons & self._middle and self._middle_press is not None:
            index, start, dragged = self._middle_press
            self._middle_press = (index, start, dragged or np.linalg.norm(np.asarray((x, y)) - start) > 6.0)
        if not self.active or not buttons & self._right:
            return None
        if not self._owns_scene():
            self.cancel()
            return None
        self._update_target(x, y)
        return self._handled

    def on_mouse_release(self, x, y, button, modifiers):
        if button == self._middle:
            pending, self._middle_press = self._middle_press, None
            if (
                pending is not None
                and self._owns_scene()
                and not self.active
                and not modifiers
                and (self._can_start is None or self._can_start())
            ):
                index, start, dragged = pending
                if not dragged and np.linalg.norm(np.asarray((x, y)) - start) <= 6.0 and index < len(self.pins):
                    if self._remove(index):
                        # Match the backend's compaction while keeping captured
                        # array addresses and the remaining marker IDs stable.
                        self.pins[index] = self.pins[-1]
                        self.pins.pop()
            return None
        if not self.active or button != self._right:
            return None
        if not self._owns_scene():
            self.cancel()
            return None
        width, height = self.window.get_size()
        if not 0 <= x <= width or not 0 <= y <= height:
            self.cancel()
            return self._handled
        self._update_target(x, y)
        self._finish()
        self.pins.append(self.target.copy())
        self._clear_drag()
        return self._handled

    def on_key_press(self, symbol, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if symbol == self._escape:
            self._middle_press = None
            if self.active:
                self.cancel()
                return self._handled
        return None

    def on_deactivate(self):
        self.cancel()

    def on_resize(self, width, height):
        self.cancel()

    def _clip_coordinates(self, points):
        homogeneous = np.column_stack((points, np.ones(len(points))))
        camera = self.viewer.camera
        view = camera.get_view_matrix().reshape((4, 4), order="F")
        projection = camera.get_projection_matrix().reshape((4, 4), order="F")
        return homogeneous @ view.T @ projection.T

    def _pin_at(self, x, y):
        if not self.pins:
            return None
        clip = self._clip_coordinates(self.pins)
        visible = (clip[:, 3] > 0.0) & np.all(np.abs(clip[:, :3]) <= clip[:, 3, None], axis=1)
        indices = np.flatnonzero(visible)
        if not len(indices):
            return None
        clip = clip[indices]
        pixels = (clip[:, :2] / clip[:, 3, None] + 1.0) * np.asarray(self.window.get_size()) / 2.0
        distances = np.linalg.norm(pixels - (x, y), axis=1)
        nearest = int(np.argmin(distances))
        return int(indices[nearest]) if distances[nearest] <= 12.0 else None

    def draw_overlay(self, imgui):
        if not self._owns_scene() or (not self.pins and not self.active):
            return
        points = self.pins + ([self.target] if self.active else [])
        clip = self._clip_coordinates(points)
        viewport = imgui.get_main_viewport()
        size = np.asarray(self.window.get_framebuffer_size())
        draw = imgui.get_foreground_draw_list()
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.05, 0.05, 0.05, 0.95))
        fixed = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.2, 1.0, 0.45, 1.0))
        active = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.8, 0.15, 1.0))
        for index, point in enumerate(clip):
            if point[3] <= 0.0 or np.any(np.abs(point[:3]) > point[3]):
                continue
            pixel = (point[:2] / point[3] + 1.0) * size / 2.0
            pixel[1] = size[1] - pixel[1]
            pixel += (viewport.pos.x, viewport.pos.y)
            center = imgui.ImVec2(*pixel)
            color = active if self.active and index == len(points) - 1 else fixed
            draw.add_circle_filled(center, 6.0, outline)
            draw.add_circle_filled(center, 4.0, color)
            draw.add_circle(center, 8.0, color, 0, 1.5)

    @staticmethod
    def gui(imgui):
        imgui.text_wrapped("Right-drag the body to lift it. Release to leave a spring hold at the green marker.")
        imgui.text_wrapped("Middle-click a green marker to release it. Middle-drag still moves the camera.")
        imgui.text_wrapped("Esc cancels the current grab. Clear holds releases all supports.")

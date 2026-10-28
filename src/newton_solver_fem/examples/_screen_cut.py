# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Release a screen-space line to submit one finite, view-projected incision."""

from collections import deque
from collections.abc import Callable

import numpy as np


def project_screen_segment(camera, start, end, window_size, framebuffer_size):
    """Project a window-pixel segment through the camera's visible depth [m].

    Window and ray coordinates both have a bottom-left origin. Intersecting the
    endpoint rays with common depth planes preserves the finite screen segment
    at every depth, including under perspective and high-DPI scaling.
    """
    window = np.asarray(window_size, dtype=np.float64)
    framebuffer = np.asarray(framebuffer_size, dtype=np.float64)
    pixels = np.asarray((start, end), dtype=np.float64)
    near, far = float(camera.near), float(camera.far)
    if (
        window.shape != (2,)
        or framebuffer.shape != (2,)
        or pixels.shape != (2, 2)
        or not np.isfinite(np.concatenate((window, framebuffer, pixels.ravel(), (near, far)))).all()
        or np.any(window <= 0)
        or np.any(framebuffer <= 0)
        or not 0 < near < far
    ):
        raise ValueError("Valid viewport dimensions, endpoints and camera clipping depths are required")
    # Pyglet's vector arithmetic requires Python scalars, not NumPy scalars.
    rays = [camera.get_world_ray(float(x), float(y)) for x, y in pixels * (framebuffer / window)]
    origins = np.asarray([ray[0] for ray in rays], dtype=np.float64)
    directions = np.asarray([ray[1] for ray in rays], dtype=np.float64)
    front = np.asarray(camera.get_front(), dtype=np.float64)
    front /= np.linalg.norm(front)
    depth_rate = directions @ front
    if not np.isfinite(origins).all() or not np.isfinite(depth_rate).all() or np.any(depth_rate <= 0):
        raise ValueError("The screen segment must project in front of the camera")
    at_near = origins + directions * (near / depth_rate[:, None])
    at_far = origins + directions * (far / depth_rate[:, None])
    quad = np.asarray((at_near[0], at_near[1], at_far[1], at_far[0]))
    return quad[[[0, 1, 2], [0, 2, 3]]]


class ScreenCutTool:
    """Own scene left drags, preview their line, and queue cuts only on release.

    The helper registers window handlers ahead of the camera's handlers. GUI
    presses and other mouse buttons keep their normal behavior. It submits
    only input geometry; simulated coordinates never return to the CPU.
    """

    def __init__(self, viewer, model, *, minimum_drag_pixels=6.0, can_start: Callable[[], bool] | None = None):
        self.viewer, self.model = viewer, model
        self.window = getattr(getattr(viewer, "renderer", None), "window", None)
        self.minimum_drag_pixels = float(minimum_drag_pixels)
        self._can_start = can_start
        self.start = self.end = None
        self._pending = deque()
        self._left, self._escape, self._handled = 1, 65307, True
        if self.window is not None:
            import pyglet  # noqa: PLC0415

            self._left, self._escape = pyglet.window.mouse.LEFT, pyglet.window.key.ESCAPE
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
    def pending(self):
        return bool(self._pending)

    def consume(self):
        """Return one released stroke, or no triangles while drawing/idle."""
        return self._pending.popleft() if self._pending else np.empty((0, 3, 3), dtype=np.float64)

    def cancel(self):
        self.start = self.end = None
        self._pending.clear()

    def _owns_scene(self):
        # Example switching clears the model before constructing the next one;
        # an old input helper must never intercept that next example's events.
        return getattr(self.viewer, "model", None) is self.model

    def _pointer(self, x, y):
        gui = getattr(self.viewer, "gui", None)
        io = getattr(getattr(gui, "ui", None), "io", None)
        if io is not None:
            width, height = self.window.get_size()
            fb_width, fb_height = self.window.get_framebuffer_size()
            if width > 0 and height > 0:
                # Consumed drags bypass ImGui's input handler. Keep its hover
                # position current without giving it this scene-owned click.
                io.add_mouse_pos_event(x * fb_width / width, (height - y) * fb_height / height)

    def on_mouse_press(self, x, y, button, modifiers):
        if not self._owns_scene() or button != self._left:
            return None
        gui = getattr(self.viewer, "gui", None)
        if gui is not None and gui.is_mouse_capturing():
            return None
        if self._can_start is not None and not self._can_start():
            return self._handled
        self.start = self.end = np.asarray((x, y), dtype=np.float64)
        self._pointer(x, y)
        return self._handled

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if self.start is None or not buttons & self._left:
            return None
        if not self._owns_scene():
            self.cancel()
            return None
        self.end = np.asarray((x, y), dtype=np.float64)
        self._pointer(x, y)
        return self._handled

    def on_mouse_release(self, x, y, button, modifiers):
        if self.start is None or button != self._left:
            return None
        if not self._owns_scene():
            self.cancel()
            return None
        start, end = self.start, np.asarray((x, y), dtype=np.float64)
        self.start = self.end = None
        self._pointer(x, y)
        size = self.window.get_size()
        if 0 <= x <= size[0] and 0 <= y <= size[1] and np.linalg.norm(end - start) >= self.minimum_drag_pixels:
            self._pending.append(
                project_screen_segment(self.viewer.camera, start, end, size, self.window.get_framebuffer_size())
            )
        return self._handled

    def on_key_press(self, symbol, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if symbol == self._escape:
            # Reserve Escape even while idle: key repeats after cancellation
            # must not fall through to the viewer's Escape-to-quit handler.
            self.cancel()
            return self._handled
        return None

    def on_deactivate(self):
        self.cancel()

    def on_resize(self, width, height):
        self.cancel()

    def draw_overlay(self, imgui):
        if self.start is None or not self._owns_scene():
            return
        width, height = self.window.get_size()
        if width <= 0 or height <= 0:
            return
        viewport = imgui.get_main_viewport()
        scale = np.asarray(self.window.get_framebuffer_size()) / (width, height)
        points = np.asarray((self.start, self.end)) * scale
        points[:, 1] = height * scale[1] - points[:, 1]
        points += (viewport.pos.x, viewport.pos.y)
        start, end = (imgui.ImVec2(*point) for point in points)
        draw = imgui.get_foreground_draw_list()
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.05, 0.05, 0.05, 0.9))
        orange = imgui.color_convert_float4_to_u32(imgui.ImVec4(1.0, 0.45, 0.05, 1.0))
        draw.add_line(start, end, outline, 5.0)
        draw.add_line(start, end, orange, 2.0)
        draw.add_circle_filled(start, 4.0, orange)
        draw.add_circle_filled(end, 4.0, orange)

    @staticmethod
    def gui(imgui):
        imgui.text_wrapped("Hold left mouse and draw a line across the body. Release to cut through its depth.")
        imgui.text_wrapped("Esc cancels the line. Middle mouse orbits the camera. Cuts also apply while paused.")

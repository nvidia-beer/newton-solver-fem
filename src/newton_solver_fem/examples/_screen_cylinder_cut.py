# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Click the visible surface to request one complete cylindrical through-hole."""

from collections import deque
from collections.abc import Callable

import numpy as np

from ._screen_pick import intersect_drag_plane, screen_ray


class ScreenCylinderCutTool:
    """Preview a blue circle and submit a fixed camera ray on each left click.

    ``begin(origin, direction)`` returns the first visible surface point [m], or
    ``None``. The hit depth sets the cylinder's world radius from an eight-pixel
    framebuffer diameter. The caller consumes ``(origin, direction, radius)``
    and removes material through the object's full depth along that ray.

    A press submits one request; dragging or holding never repeats it. Release
    clears the marker without losing an unconsumed click. Cancellation clears
    both markers and requests. Hover only previews the next click.
    """

    def __init__(
        self,
        viewer,
        model,
        *,
        begin: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
        can_start: Callable[[], bool] | None = None,
        enabled: bool = False,
        diameter_pixels: float = 8.0,
    ):
        self.viewer, self.model = viewer, model
        self.window = getattr(getattr(viewer, "renderer", None), "window", None)
        self._begin, self._can_start = begin, can_start
        self.diameter_pixels = float(diameter_pixels)
        if not np.isfinite(self.diameter_pixels) or self.diameter_pixels <= 0.0:
            raise ValueError("The cylinder's pixel diameter must be finite and positive")
        self.center = self.origin = self.direction = self._pixel = None
        self.radius = 0.0
        self._pending = deque()
        self._pressed = False
        self._enabled = bool(enabled)
        self._left, self._escape, self._handled = 1, 65307, True
        if self.window is not None:
            import pyglet  # noqa: PLC0415

            self._left, self._escape = pyglet.window.mouse.LEFT, pyglet.window.key.ESCAPE
            self._handled = pyglet.event.EVENT_HANDLED
            self.window.push_handlers(
                on_mouse_motion=self.on_mouse_motion,
                on_mouse_press=self.on_mouse_press,
                on_mouse_drag=self.on_mouse_drag,
                on_mouse_release=self.on_mouse_release,
                on_key_press=self.on_key_press,
                on_deactivate=self.on_deactivate,
                on_resize=self.on_resize,
            )
            viewer.register_ui_callback(self.draw_overlay, position="free")

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)
        if not self._enabled:
            self.cancel()

    @property
    def active(self):
        return self._pressed and self.center is not None

    @property
    def pending(self):
        return bool(self._pending)

    def _owns_scene(self):
        return getattr(self.viewer, "model", None) is self.model

    def _clear_marker(self):
        self.center = self.origin = self.direction = self._pixel = None
        self.radius = 0.0

    def cancel(self):
        self._clear_marker()
        self._pending.clear()
        self._pressed = False

    def consume(self):
        """Return one clicked ray and radius [m], or ``None`` when idle."""
        if not self._owns_scene() or not self.enabled:
            self.cancel()
            return None
        return self._pending.popleft() if self._pending else None

    def _gui_capturing(self):
        gui = getattr(self.viewer, "gui", None)
        return gui is not None and gui.is_mouse_capturing()

    def _pointer(self, x, y):
        gui = getattr(self.viewer, "gui", None)
        io = getattr(getattr(gui, "ui", None), "io", None)
        if io is not None:
            width, height = self.window.get_size()
            fb_width, fb_height = self.window.get_framebuffer_size()
            if width > 0 and height > 0:
                io.add_mouse_pos_event(x * fb_width / width, (height - y) * fb_height / height)

    def _pick(self, x, y):
        self._clear_marker()
        window_size = self.window.get_size()
        framebuffer_size = self.window.get_framebuffer_size()
        if not 0 <= x <= window_size[0] or not 0 <= y <= window_size[1]:
            return
        try:
            origin, direction = screen_ray(self.viewer.camera, (x, y), window_size, framebuffer_size)
        except ValueError:
            return
        hit = self._begin(origin, direction)
        if hit is None:
            return
        hit = np.asarray(hit, dtype=np.float64)
        if hit.shape != (3,) or not np.isfinite(hit).all():
            self.cancel()
            raise ValueError("The cylinder picking backend must return a finite world point or None")
        if np.dot(hit - origin, direction) <= 0.0:
            return
        # Size the cross-section at the picked entry depth. The offset ray
        # intersects a disk perpendicular to the cylinder's fixed camera ray.
        offset_x = 0.5 * self.diameter_pixels * window_size[0] / framebuffer_size[0]
        offset_origin, offset_direction = screen_ray(
            self.viewer.camera, (x + offset_x, y), window_size, framebuffer_size
        )
        try:
            edge = intersect_drag_plane(offset_origin, offset_direction, hit, direction)
        except ValueError:
            return
        radius = float(np.linalg.norm(edge - hit))
        if not np.isfinite(radius) or radius <= 0.0:
            return
        self.origin, self.direction, self.center = origin.copy(), direction.copy(), hit.copy()
        self.radius = radius
        self._pixel = np.asarray((x, y), dtype=np.float64)

    def on_mouse_motion(self, x, y, dx, dy):
        if not self._owns_scene():
            self.cancel()
            return None
        if not self.enabled or self._pressed:
            return None
        if self._gui_capturing() or (self._can_start is not None and not self._can_start()):
            self._clear_marker()
            return None
        self._pick(x, y)
        return None

    def on_mouse_press(self, x, y, button, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if not self.enabled or button != self._left or self._gui_capturing():
            return None
        if self._pressed:
            return self._handled
        self._pressed = True
        self._pointer(x, y)
        if self._can_start is not None and not self._can_start():
            self._clear_marker()
            return self._handled
        self._pick(x, y)
        if self.center is not None:
            self._pending.append((self.origin.copy(), self.direction.copy(), self.radius))
        return self._handled

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if not self._pressed and not buttons & self._left:
            self._clear_marker()
            return None
        if not self._pressed or not buttons & self._left:
            return None
        self._pointer(x, y)
        return self._handled

    def on_mouse_release(self, x, y, button, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if not self._pressed or button != self._left:
            return None
        self._pointer(x, y)
        self._pressed = False
        self._clear_marker()
        return self._handled

    def on_key_press(self, symbol, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if self.enabled and symbol == self._escape:
            other_gesture = self._can_start is not None and not self._can_start()
            self.cancel()
            # Let a lower attachment handler cancel its active grab. The line
            # helper below it still reserves Escape when every tool is idle.
            return None if other_gesture else self._handled
        return None

    def on_deactivate(self):
        self.cancel()

    def on_resize(self, width, height):
        self.cancel()

    def draw_overlay(self, imgui):
        if not self._owns_scene():
            self.cancel()
            return
        if self._pixel is None or not self.enabled:
            return
        if not self._pressed and self._gui_capturing():
            self._clear_marker()
            return
        size = np.asarray(self.window.get_size())
        framebuffer = np.asarray(self.window.get_framebuffer_size())
        if np.any(size <= 0) or np.any(framebuffer <= 0):
            return
        pixel = self._pixel * framebuffer / size
        pixel[1] = framebuffer[1] - pixel[1]
        viewport = imgui.get_main_viewport()
        pixel += (viewport.pos.x, viewport.pos.y)
        center = imgui.ImVec2(*pixel)
        draw = imgui.get_foreground_draw_list()
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.05, 0.05, 0.05, 0.95))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.15, 0.5, 1.0, 1.0))
        draw.add_circle(center, 0.5 * self.diameter_pixels, outline, 32, 4.0)
        draw.add_circle(center, 0.5 * self.diameter_pixels, blue, 32, 2.0)

    def gui(self, imgui):
        imgui.text_wrapped("Click the body once to remove a blue cylindrical hole through its full depth.")
        imgui.text_wrapped("The click fixes the direction and diameter. Holding does not repeat the cut. Esc cancels.")
        if self.center is not None:
            imgui.text(f"Hole diameter: {2000.0 * self.radius:.2f} mm")

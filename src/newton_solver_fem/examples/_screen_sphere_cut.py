# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Hold left mouse to advance a spherical burr along the initially picked ray."""

from collections.abc import Callable

import numpy as np

from ._screen_pick import screen_ray


def _project_sphere_in_view(center, radius, projection, framebuffer):
    """Return the circle enclosing a perspective sphere silhouette in pixels."""
    depth = -center[2]
    if depth <= radius:
        return None
    denominator = depth * depth - radius * radius
    scale = np.diag(projection)[:2] * framebuffer / 2.0
    # A sphere projects to an ellipse away from the optical axis. Its largest
    # semiaxis gives a round preview that still encloses the actual tool.
    pixel = framebuffer / 2.0 + scale * center[:2] * depth / denominator
    scaled_center = scale * center[:2]
    shape = (
        radius * radius / denominator * (np.diag(scale * scale) + np.outer(scaled_center, scaled_center) / denominator)
    )
    return pixel, float(np.sqrt(np.linalg.eigvalsh(shape)[-1]))


def project_sphere_marker(camera, center, radius, framebuffer_size):
    """Project the current sphere [m] to a round, depth-scaled pixel outline."""
    view = camera.get_view_matrix().reshape((4, 4), order="F")
    projection = camera.get_projection_matrix().reshape((4, 4), order="F")
    point = view @ np.append(center, 1.0)
    clip = projection @ point
    if clip[3] <= 0.0 or abs(clip[2]) > clip[3]:
        return None
    return _project_sphere_in_view(point[:3], radius, projection, np.asarray(framebuffer_size))


class ScreenSphereCutTool:
    """Advance a fixed-radius burr only while the scene owns a left-button hold.

    ``begin(origin, direction)`` raycasts the visible surface and returns a world
    point [m], or ``None``. The sphere starts tangent to that point, on the camera
    side. Its initial diameter is eight framebuffer pixels: four times the
    orange incision line width. Its world radius and initial ray remain fixed
    throughout the hold, including when the pointer moves.

    Call :meth:`advance` once per elapsed interval. It returns the swept segment
    and radius [m], without queuing any work. On release, ``finish`` can publish
    the already displayed segment synchronously, without advancing the sphere.
    Cancellation discards the gesture without calling ``finish``. The caller
    controls the clock and material removal.
    """

    def __init__(
        self,
        viewer,
        model,
        *,
        begin: Callable[[np.ndarray, np.ndarray], np.ndarray | None],
        finish: Callable[[], None] | None = None,
        can_start: Callable[[], bool] | None = None,
        speed: float = 1.0,
        enabled: bool = False,
        diameter_pixels: float = 8.0,
    ):
        self.viewer, self.model = viewer, model
        self.window = getattr(getattr(viewer, "renderer", None), "window", None)
        self._begin, self._can_start = begin, can_start
        self._finish = finish
        self.center = self.direction = None
        self.radius = 0.0
        self.gesture_id = 0
        self._pressed = False
        self._enabled = bool(enabled)
        self.speed = speed
        self.diameter_pixels = float(diameter_pixels)
        if not np.isfinite(self.diameter_pixels) or self.diameter_pixels <= 0:
            raise ValueError("The sphere's pixel diameter must be finite and positive")
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
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = bool(value)
        if not self._enabled:
            self.cancel()

    @property
    def speed(self):
        """Advance rate in sphere diameters per second."""
        return self._speed

    @speed.setter
    def speed(self, value):
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("The sphere speed must be finite and nonnegative")
        self._speed = value

    @property
    def active(self):
        return self.center is not None

    def _owns_scene(self):
        return getattr(self.viewer, "model", None) is self.model

    def cancel(self):
        self.center = self.direction = None
        self.radius = 0.0
        self._pressed = False

    def _pointer(self, x, y):
        gui = getattr(self.viewer, "gui", None)
        io = getattr(getattr(gui, "ui", None), "io", None)
        if io is not None:
            width, height = self.window.get_size()
            fb_width, fb_height = self.window.get_framebuffer_size()
            if width > 0 and height > 0:
                io.add_mouse_pos_event(x * fb_width / width, (height - y) * fb_height / height)

    def _start_sphere(self, hit, direction):
        camera = self.viewer.camera
        view = camera.get_view_matrix().reshape((4, 4), order="F")
        projection = camera.get_projection_matrix().reshape((4, 4), order="F")
        hit_view = (view @ np.append(hit, 1.0))[:3]
        direction_view = view[:3, :3] @ direction
        depth, rate = -hit_view[2], -direction_view[2]
        if depth <= 0.0 or rate <= 0.0:
            return
        framebuffer = np.asarray(self.window.get_framebuffer_size())
        # Solve the screen size once. Off-axis perspective and backing away
        # from the hit by one radius both affect the initial apparent diameter.
        lower, upper = 0.0, 0.999 * depth / (1.0 + rate)
        for _ in range(32):
            radius = 0.5 * (lower + upper)
            _, pixels = _project_sphere_in_view(hit_view - radius * direction_view, radius, projection, framebuffer)
            if 2.0 * pixels < self.diameter_pixels:
                lower = radius
            else:
                upper = radius
        self.radius = 0.5 * (lower + upper)
        self.center = hit - self.radius * direction
        self.direction = direction.copy()

    def on_mouse_press(self, x, y, button, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if not self.enabled or button != self._left:
            return None
        gui = getattr(self.viewer, "gui", None)
        if gui is not None and gui.is_mouse_capturing():
            return None
        self.cancel()
        self._pressed = True
        self._pointer(x, y)
        if self._can_start is not None and not self._can_start():
            return self._handled
        try:
            origin, direction = screen_ray(
                self.viewer.camera, (x, y), self.window.get_size(), self.window.get_framebuffer_size()
            )
        except ValueError:
            return self._handled
        hit = self._begin(origin, direction)
        if hit is not None:
            hit = np.asarray(hit, dtype=np.float64)
            if hit.shape != (3,) or not np.isfinite(hit).all():
                self.cancel()
                raise ValueError("The sphere picking backend must return a finite world point or None")
            if np.dot(hit - origin, direction) > 0.0:
                self._start_sphere(hit, direction)
                if self.active:
                    self.gesture_id += 1
        return self._handled

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if not self._owns_scene():
            self.cancel()
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
        try:
            if self.active and self._finish is not None:
                self._finish()
        finally:
            self.cancel()
        return self._handled

    def advance(self, dt: float):
        """Return the held sphere's next swept segment, using elapsed time [s]."""
        if not self._owns_scene() or not self.enabled:
            self.cancel()
            return None
        if not self.active:
            return None
        dt = float(dt)
        if not np.isfinite(dt):
            raise ValueError("Elapsed sphere motion time must be finite")
        if dt <= 0.0 or self.speed == 0.0:
            return None
        start = self.center.copy()
        self.center = start + self.direction * (2.0 * self.radius * self.speed * dt)
        return start, self.center.copy(), self.radius

    def on_key_press(self, symbol, modifiers):
        if not self._owns_scene():
            self.cancel()
            return None
        if self.enabled and symbol == self._escape:
            other_gesture = self._can_start is not None and not self._can_start()
            self.cancel()
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
        if not self.active:
            return
        size = np.asarray(self.window.get_framebuffer_size())
        projected = project_sphere_marker(self.viewer.camera, self.center, self.radius, size)
        if projected is None:
            return
        pixel, radius = projected
        viewport = imgui.get_main_viewport()
        pixel[1] = size[1] - pixel[1]
        pixel += (viewport.pos.x, viewport.pos.y)
        center = imgui.ImVec2(*pixel)
        draw = imgui.get_foreground_draw_list()
        outline = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.05, 0.05, 0.05, 0.95))
        blue = imgui.color_convert_float4_to_u32(imgui.ImVec4(0.15, 0.5, 1.0, 1.0))
        draw.add_circle(center, radius, outline, 32, 4.0)
        draw.add_circle(center, radius, blue, 32, 2.0)

    def gui(self, imgui):
        changed, speed = imgui.slider_float("Advance (diameters/s)", self.speed, 0.0, 10.0, "%.2f")
        if changed:
            self.speed = speed
        imgui.text_wrapped(
            "Hold left mouse on the body to advance the blue sphere into it. Release to stop removing material."
        )
        imgui.text_wrapped("The direction stays fixed during a hold. Esc cancels the tool.")
        imgui.text_wrapped(
            "The tunnel updates at tetrahedral boundaries. Release completes the current partial segment."
        )
        if self.active:
            imgui.text(f"Tool diameter: {2000.0 * self.radius:.2f} mm")

# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Cached, depth-tested top view for selecting regions of a surface mesh."""

import numpy as np


def rasterize_regions(vertices, triangles, region_ids, resolution=512):
    """Return region IDs and relative heights of the highest surface at each pixel.

    Barycentric depths, rather than triangle-average sorting, handle overlapping
    surfaces. ID -1 denotes background. Only initialization visits the triangles.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int32)
    labels = np.asarray(region_ids, dtype=np.int64)
    if len(triangles) != len(labels) or np.any(labels < 0) or np.any(labels >= 65535):
        raise ValueError("Each triangle needs a region ID in [0, 65534]")
    span = np.ptp(vertices[:, :2], axis=0)
    if resolution < 8 or np.any(span <= 0):
        raise ValueError("The region map needs a nonzero XY extent and resolution >= 8")
    scale = (resolution - 4) / span.max()
    width, height = np.ceil(span * scale).astype(int) + 4
    projected = np.empty_like(vertices)
    projected[:, 0] = (vertices[:, 0] - vertices[:, 0].min()) * scale + 2
    projected[:, 1] = (vertices[:, 1].max() - vertices[:, 1]) * scale + 2
    projected[:, 2] = vertices[:, 2] - vertices[:, 2].min() + 1.0
    corners = projected[triangles]
    lower = np.maximum(np.floor(corners[:, :, :2].min(axis=1)).astype(int), 0)
    upper = np.minimum(np.ceil(corners[:, :, :2].max(axis=1)).astype(int), (width, height))
    dimensions = upper - lower
    a, b = corners[:, 1, :2] - corners[:, 0, :2], corners[:, 2, :2] - corners[:, 0, :2]
    determinant = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    usable = np.flatnonzero((np.abs(determinant) > 1.0e-12) & np.all(dimensions > 0, axis=1))
    z_buffer = np.zeros(width * height, dtype=np.uint64)
    # The positive float32 depth in the high bits sorts monotonically. The low
    # bits identify its owner and give deterministic ownership to exact ties.
    for start in range(0, len(usable), 1024):
        batch = usable[start : start + 1024]
        counts = np.prod(dimensions[batch], axis=1)
        face = np.repeat(batch, counts)
        offset = np.arange(counts.sum()) - np.repeat(np.cumsum(counts) - counts, counts)
        x = lower[face, 0] + offset % dimensions[face, 0]
        y = lower[face, 1] + offset // dimensions[face, 0]
        dx, dy = x + 0.5 - corners[face, 0, 0], y + 0.5 - corners[face, 0, 1]
        u = (dx * b[face, 1] - dy * b[face, 0]) / determinant[face]
        v = (a[face, 0] * dy - a[face, 1] * dx) / determinant[face]
        inside = (u >= -1.0e-9) & (v >= -1.0e-9) & (u + v <= 1 + 1.0e-9)
        face, u, v = face[inside], u[inside], v[inside]
        z = (
            corners[face, 0, 2]
            + u * (corners[face, 1, 2] - corners[face, 0, 2])
            + v * (corners[face, 2, 2] - corners[face, 0, 2])
        ).astype(np.float32)
        key = (z.view(np.uint32).astype(np.uint64) << np.uint64(16)) | (labels[face] + 1).astype(np.uint64)
        np.maximum.at(z_buffer, y[inside] * width + x[inside], key)
    ids = ((z_buffer & np.uint64(65535)).astype(np.int32) - 1).reshape(height, width)
    heights = (z_buffer >> np.uint64(16)).astype(np.uint32).view(np.float32).reshape(height, width)
    return ids, heights


def _interior_label(mask):
    """Choose a deep interior pixel, including for concave or disconnected regions."""
    y, x = np.nonzero(mask)
    center = (x.mean(), y.mean())
    inner = mask.copy()
    while np.any(inner):
        deepest = inner
        inner = np.zeros_like(inner)
        inner[1:-1, 1:-1] = (
            deepest[1:-1, 1:-1] & deepest[:-2, 1:-1] & deepest[2:, 1:-1] & deepest[1:-1, :-2] & deepest[1:-1, 2:]
        )
    y, x = np.nonzero(deepest)
    closest = np.argmin((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return float(x[closest]) + 0.5, float(y[closest]) + 0.5


class RegionMap:
    """One cached image with region hit testing, labels and selection highlighting.

    OpenGL is needed only by ``draw``. ``screen_position`` exposes a reliable
    interior point after drawing, for real mouse interaction in demonstrations.
    """

    def __init__(self, vertices, triangles, region_ids, regions, resolution=512, *, anchors=None):
        self.region_ids, heights = rasterize_regions(vertices, triangles, region_ids, resolution)
        # Present the authored pad with its 3-D Med lettering upright. Rotate
        # both ownership and relief before deriving labels and mouse targets.
        self.region_ids = self.region_ids[::-1, ::-1].copy()
        heights = heights[::-1, ::-1].copy()
        pixel_size = np.ptp(np.asarray(vertices)[:, :2], axis=0).max() / (resolution - 4)
        dy, dx = np.gradient(heights, pixel_size)
        normal_length = np.sqrt(dx * dx + dy * dy + 1)
        diffuse = np.clip((0.4 * dx + 0.5 * dy + 0.8) / normal_length, 0, 1)
        self._shade = (0.68 + 0.32 * diffuse)[..., None]
        self.regions = {region.id: region for region in regions}
        self.centers = {}
        for region_id in self.regions:
            mask = self.region_ids == region_id
            if not mask.any():
                raise ValueError(f"Region {region_id} has no visible area in the top view")
            self.centers[region_id] = _interior_label(mask)
            if anchors is not None and region_id in anchors:
                # Project a dataset label location through the existing map
                # orientation, then keep its click target on its original owner.
                anchor = anchors[region_id]
                scale = (resolution - 4) / np.ptp(np.asarray(vertices)[:, :2], axis=0).max()
                target_x = mask.shape[1] - ((anchor[0] - np.min(vertices[:, 0])) * scale + 2)
                target_y = mask.shape[0] - ((np.max(vertices[:, 1]) - anchor[1]) * scale + 2)
                y, x = np.nonzero(mask)
                nearest = np.argmin((x + 0.5 - target_x) ** 2 + (y + 0.5 - target_y) ** 2)
                self.centers[region_id] = float(x[nearest]) + 0.5, float(y[nearest]) + 0.5
        self.boundary = np.zeros_like(self.region_ids, dtype=bool)
        self.boundary[1:] |= self.region_ids[1:] != self.region_ids[:-1]
        self.boundary[:, 1:] |= self.region_ids[:, 1:] != self.region_ids[:, :-1]
        self.screen_rect = None
        self.hovered_region = None
        self._texture = None
        self._texture_state = None

    def region_at(self, x, y):
        """Return the visible region at map pixel coordinates, or None outside."""
        height, width = self.region_ids.shape
        if not 0 <= x < width or not 0 <= y < height:
            return None
        region = int(self.region_ids[int(y), int(x)])
        return region if region >= 0 else None

    def screen_position(self, region_id):
        """Return an interior point in absolute ImGui screen coordinates."""
        if self.screen_rect is None:
            raise RuntimeError("Draw the region map before requesting screen positions")
        x, y, width, height = self.screen_rect
        px, py = self.centers[region_id]
        return x + px * width / self.region_ids.shape[1], y + py * height / self.region_ids.shape[0]

    def pixels(self, selected_ids, hovered=None):
        """Return the neutral/pink map image; active regions have a stronger fill."""
        image = np.empty((*self.region_ids.shape, 4), dtype=np.uint8)
        image[:] = (31, 34, 40, 255)
        image[self.region_ids >= 0] = (180, 159, 161, 255)
        image[np.isin(self.region_ids, tuple(selected_ids))] = (241, 137, 160, 255)
        if hovered is not None:
            mask = self.region_ids == hovered
            image[mask, :3] = np.minimum(image[mask, :3].astype(int) + 22, 255)
        image[..., :3] = (image[..., :3] * self._shade).astype(np.uint8)
        image[self.boundary] = (78, 54, 62, 255)
        return image

    def _update_texture(self, selected_ids, hovered):
        from OpenGL import GL

        state = tuple(sorted(selected_ids)), hovered
        if state == self._texture_state:
            return
        pixels = self.pixels(selected_ids, hovered)
        previous = GL.glGetIntegerv(GL.GL_TEXTURE_BINDING_2D)
        if self._texture is None:
            self._texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA,
            pixels.shape[1],
            pixels.shape[0],
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, previous)
        self._texture_state = state

    def draw(self, ui, selected_ids):
        """Draw the map in the current window and return a clicked region ID."""
        origin = ui.get_cursor_screen_pos()
        width = min(320.0, max(80.0, ui.get_content_region_avail().x))
        height = width * self.region_ids.shape[0] / self.region_ids.shape[1]
        self.screen_rect = origin.x, origin.y, width, height
        ui.invisible_button("##phantom_region_map", ui.ImVec2(width, height))
        hovered = None
        if ui.is_item_hovered():
            mouse = ui.get_mouse_pos()
            hovered = self.region_at(
                (mouse.x - origin.x) * self.region_ids.shape[1] / width,
                (mouse.y - origin.y) * self.region_ids.shape[0] / height,
            )
        self.hovered_region = hovered
        clicked = hovered if ui.is_item_clicked(0) else None
        self._update_texture(selected_ids, hovered)
        draw = ui.get_window_draw_list()
        draw.add_image(ui.ImTextureRef(self._texture), origin, ui.ImVec2(origin.x + width, origin.y + height))
        for region_id in self.regions:
            px, py = self.screen_position(region_id)
            text = str(region_id)
            extent = ui.calc_text_size(text)
            active = region_id in selected_ids
            color = (0.48, 0.12, 0.23, 1.0) if active else (0.2, 0.2, 0.23, 0.95)
            draw.add_circle_filled(
                ui.ImVec2(px, py), max(extent.x, extent.y) * 0.6 + 1, ui.color_convert_float4_to_u32(ui.ImVec4(*color))
            )
            draw.add_text(ui.ImVec2(px - extent.x / 2, py - extent.y / 2), 0xFFFFFFFF, text)
        if hovered is not None:
            region = self.regions[hovered]
            name = region.name.replace("_", " ").capitalize()
            status = "active FEM" if hovered in selected_ids else "fixed; click to simulate this region"
            ui.set_tooltip(f"{hovered}: {name}\n{region.tetrahedron_count:,} tetrahedra\n{status}")
        return clicked

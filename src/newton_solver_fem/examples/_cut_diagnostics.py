# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recognize explicit material-reserve limits without masking invalid cuts."""

import ast


def capacity_resource(error, fragment_capacity):
    """Return the exhausted reserve, or None for any other diagnostic."""
    marker = "linear tetrahedron edit rejected: "
    message = str(error)
    if marker not in message:
        return None
    try:
        diagnostics = ast.literal_eval(message.split(marker, 1)[1])
    except (SyntaxError, ValueError):
        return None
    if not isinstance(diagnostics, dict):
        return None
    if any(diagnostics.get(name) != 0 for name in ("integration_error", "surface_error")):
        return None
    geometry, connectivity = diagnostics.get("geometry_error"), diagnostics.get("connectivity_error")
    if geometry == 0 and connectivity == 2:
        return "node_copies"
    count = diagnostics.get("fragment_count")
    if geometry == 4 and connectivity == 3 and isinstance(count, int) and count > fragment_capacity:
        return "fragment_capacity"
    return None


def rejection_message(error, fragment_capacity):
    """Keep the precise diagnostic after an actionable reserve-limit message."""
    resource = capacity_resource(error, fragment_capacity)
    if resource is None:
        return f"Cut rejected: {error}"
    label = "Material node" if resource == "node_copies" else "Material fragment"
    return f"Cut rejected: {label} reserve exhausted; restart with a larger --{resource}. {error}"

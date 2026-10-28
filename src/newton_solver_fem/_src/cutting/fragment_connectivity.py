# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Published basis-node connectivity for one transactional material edit."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FragmentConnectivity:
    """Basis-node map and exact state-copy instructions for one geometry edit."""

    nodes: np.ndarray
    clone_sources: np.ndarray
    parent_regions: np.ndarray
    material_components: np.ndarray

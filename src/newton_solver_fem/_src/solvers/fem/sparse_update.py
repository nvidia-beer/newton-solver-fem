# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""In-place numerical assembly for a fixed block-sparse matrix structure."""

import warp as wp


@wp.kernel(enable_backward=False)
def map_triplet_blocks(
    rows: wp.array[int],
    columns: wp.array[int],
    offsets: wp.array[int],
    matrix_columns: wp.array[int],
    slots: wp.array[int],
):
    index = wp.tid()
    row, column = rows[index], columns[index]
    if row < 0 or column < 0:
        slots[index] = -1
        return
    first, last = int(offsets[row]), int(offsets[row + 1])
    while first < last:
        middle = (first + last) // 2
        if matrix_columns[middle] < column:
            first = middle + 1
        else:
            last = middle
    # The matrix was built from these exact triplets without zero pruning.
    slots[index] = first


@wp.kernel(enable_backward=False)
def accumulate_triplet_blocks(
    slots: wp.array[int],
    triplets: wp.array[wp.mat33f],
    values: wp.array[wp.mat33f],
):
    index = wp.tid()
    slot = slots[index]
    if slot < 0:
        return
    block = triplets[index]
    # Padded element slots share valid placeholder indices. Adding their zero
    # blocks would serialize thousands of atomics at those same matrix entries.
    # Compare components exactly: small stiffness values and NaNs must survive.
    if (
        block[0, 0] == 0.0
        and block[0, 1] == 0.0
        and block[0, 2] == 0.0
        and block[1, 0] == 0.0
        and block[1, 1] == 0.0
        and block[1, 2] == 0.0
        and block[2, 0] == 0.0
        and block[2, 1] == 0.0
        and block[2, 2] == 0.0
    ):
        return
    wp.atomic_add(values, slot, block)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The projector's spatial downsampling of the DeepSeek-V4.1 image encoder: each `ratio x ratio` patch
neighbourhood becomes one row, channel-first (stock's `DeepseekV41Aligner`: right / bottom zero padding, then
`F.unfold`)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class VisionPatchMerge(OpWrapper):
    """`x [n_h * n_w, C]` (row-major patch grid) -> `[ceil(n_h / r) * ceil(n_w / r), C * r * r]`: output row
    `(i, j)` holds, channel by channel, the `r x r` patches `(i * r + a, j * r + b)` in (a, b) row-major order, zero
    where the grid is padded. Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a 36 x 36 grid, ratio 3 (no padding)", spec=dict(n_h=36, n_w=36, c=1024, r=3)),
        Cell(why="a 25 x 31 grid: both edges padded", spec=dict(n_h=25, n_w=31, c=1024, r=3)),
    )

    def __call__(self, x: torch.Tensor, n_h: int, n_w: int, ratio: int) -> torch.Tensor:
        grid = x.reshape(n_h, n_w, -1).permute(2, 0, 1)
        grid = F.pad(grid, (0, -n_w % ratio, 0, -n_h % ratio))
        return F.unfold(grid.unsqueeze(0), ratio, stride=ratio).squeeze(0).transpose(0, 1)

    def reference(self, x: torch.Tensor, n_h: int, n_w: int, ratio: int) -> torch.Tensor:
        c = x.shape[-1]
        rows, cols = -(-n_h // ratio), -(-n_w // ratio)
        padded = x.new_zeros(rows * ratio, cols * ratio, c)
        padded[:n_h, :n_w] = x.reshape(n_h, n_w, c)
        blocks = padded.reshape(rows, ratio, cols, ratio, c).permute(0, 2, 4, 1, 3)  # [i, j, channel, a, b]
        return blocks.reshape(rows * cols, c * ratio * ratio)

    def is_valid(self, x: torch.Tensor, n_h: int, n_w: int, ratio: int) -> None:
        assert x.dim() == 2 and x.shape[0] == n_h * n_w and n_h > 0 and n_w > 0 and ratio > 0


vision_patch_merge = VisionPatchMerge()

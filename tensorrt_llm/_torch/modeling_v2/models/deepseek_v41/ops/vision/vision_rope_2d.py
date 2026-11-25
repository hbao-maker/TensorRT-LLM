# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""2-D rotary position of the DeepSeek-V4.1 image encoder: half of each head's rotary frequencies follow the patch
row, half the patch column (stock's `_get_vision_cos_sin` and `_apply_vision_rotary`)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def vision_rope_tables(n_h: int, n_w: int, dim: int, theta: float, device) -> tuple[torch.Tensor, torch.Tensor]:
    """`(cos, sin)` `[n_h * n_w, 1, dim]` fp32 of a row-major `n_h x n_w` patch grid, for heads rotating `2 * dim`
    channels: frequency `i` of `dim / 2` multiplies the row index, frequency `i` again the column index (stock's
    `_get_vision_cos_sin`, the same torch ops). A setup helper: computed once per image, shared by every block."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim))
    height = torch.arange(n_h, device=device).view(n_h, 1).expand(n_h, n_w)
    width = torch.arange(n_w, device=device).view(1, n_w).expand(n_h, n_w)
    positions = torch.stack((height, width), dim=-1).reshape(-1, 2, 1).float()
    frequencies = (positions * inv_freq).flatten(1).unsqueeze(1)
    return frequencies.cos(), frequencies.sin()


class VisionRope2d(OpWrapper):
    """`x [N, heads, 2 * dim]` -> the rotated rows in `x`'s dtype: with `x = (a, b)` split in halves along the head,
    `(a * cos - b * sin, b * cos + a * sin)` in fp32 (`cos`, `sin` from `vision_rope_tables`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a 36 x 36 patch grid, 16 heads of 64", spec=dict(n_h=36, n_w=36, heads=16, head_dim=64)),
        Cell(why="a non-square grid (rows != columns)", spec=dict(n_h=24, n_w=45, heads=16, head_dim=64)),
    )

    def __call__(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        first, second = x.float().chunk(2, dim=-1)
        return torch.cat((first * cos - second * sin, second * cos + first * sin), dim=-1).to(x.dtype)

    def reference(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """In float64: rotate each (channel i, channel i + dim) pair by its angle."""
        half = x.shape[-1] // 2
        a, b = x[..., :half].double(), x[..., half:].double()
        c, s = cos.double(), sin.double()
        return torch.cat((a * c - b * s, b * c + a * s), dim=-1).to(x.dtype)

    def is_valid(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> None:
        assert x.dim() == 3 and x.shape[-1] % 2 == 0
        assert cos.shape == sin.shape == (x.shape[0], 1, x.shape[-1] // 2), "one angle per (patch, frequency)"
        assert cos.dtype == sin.dtype == torch.float32


vision_rope_2d = VisionRope2d()

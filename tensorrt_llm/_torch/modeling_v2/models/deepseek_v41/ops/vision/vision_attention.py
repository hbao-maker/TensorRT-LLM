# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bidirectional self-attention over one image's patches (stock's `_DeepseekV41VisionAttention`: torch's
`scaled_dot_product_attention`, no mask, the default `1 / sqrt(head_dim)` scale)."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class VisionAttention(OpWrapper):
    """`q, k, v [N, heads, d]` (one image, every patch attends to every patch) -> `[N, heads * d]` in their dtype.

    The same SDPA call stock makes, on the same `[1, heads, N, d]` views, so the backend torch picks is stock's."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a 36 x 36 grid (1296 patches), 16 heads of 64", spec=dict(n=1296, heads=16, head_dim=64)),
        Cell(why="the largest image the processor plans: 1024 tokens of 3 x 3 patches", spec=dict(n=9216, heads=16, head_dim=64)),
        Cell(why="a tiny grid (fewer patches than a tile)", spec=dict(n=9, heads=16, head_dim=64)),
    )

    def __call__(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        out = F.scaled_dot_product_attention(
            q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0), v.transpose(0, 1).unsqueeze(0)
        )
        return out.squeeze(0).transpose(0, 1).reshape(q.shape[0], -1)

    def reference(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Softmax attention per head in float32, rounded once."""
        scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) * q.shape[-1] ** -0.5
        out = torch.einsum("hqk,khd->qhd", scores.softmax(-1), v.float())
        return out.reshape(q.shape[0], -1).to(q.dtype)

    def is_valid(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
        assert q.shape == k.shape == v.shape and q.dim() == 3
        assert q.dtype == k.dtype == v.dtype


vision_attention = VisionAttention()

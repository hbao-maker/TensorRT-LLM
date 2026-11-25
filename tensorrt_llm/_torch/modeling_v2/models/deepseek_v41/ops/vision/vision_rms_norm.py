# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMSNorm of the DeepSeek-V4.1 image encoder: fp32 statistics and an fp32 weight, rounded once to the input's dtype
(stock's `_DeepseekV41VisionRMSNorm`, which keeps its weight fp32 when the encoder is cast to bf16)."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class VisionRmsNorm(OpWrapper):
    """`x [m, d]` -> `(weight * (x / sqrt(mean(x^2) + eps))).to(x.dtype)`, every step in fp32 (`weight` fp32 `[d]`).
    The same torch ops in the same order as stock, so the rows are stock's bit for bit."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a vision block's pre-norm: 1296 patches of width 1024", spec=dict(m=1296, d=1024)),
        Cell(why="the encoder's final norm on a 3 x 3 grid", spec=dict(m=9, d=1024)),
    )

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        value = x.float()
        value = value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)
        return (weight * value).to(x.dtype)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """In float64, rounded once."""
        value = x.double()
        value = value / torch.sqrt(value.square().mean(-1, keepdim=True) + eps)
        return (weight.double() * value).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        assert weight.dtype == torch.float32 and weight.shape == (x.shape[-1],), "stock keeps the norm weight fp32"
        assert eps > 0


vision_rms_norm = VisionRmsNorm()

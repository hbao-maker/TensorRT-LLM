# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dense bf16 projections of the DeepSeek-V4.1 image encoder and its projector: torch's cuBLAS linear, the GEMM
stock's `nn.Linear` layers run (`DeepseekV41VisionModel`, `DeepseekV41Aligner`), so the encoder's rows match stock's."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul


class VisionLinear(OpWrapper):
    """`x [m, k] @ weight[n, k].T (+ bias [n])` -> `[m, n]` in `x`'s dtype, fp32-accumulated (`F.linear`).

    The image encoder's projections run once per image at prefill, on a few thousand patch rows; their weights are
    bf16 in the checkpoint and stay bf16 (stock casts the encoder to the text embedding's dtype)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="patch embedding: 3 x 14 x 14 pixels -> 1024, with bias", spec=dict(m=1296, k=588, n=1024, bias=True)),
        Cell(why="fused q | k | v of a vision block, with bias", spec=dict(m=1296, k=1024, n=3072, bias=True)),
        Cell(why="vision MLP gate | up, no bias", spec=dict(m=1296, k=1024, n=5632, bias=False)),
        Cell(why="vision MLP down, no bias", spec=dict(m=1296, k=2816, n=1024, bias=False)),
        Cell(why="projector: 3 x 3 patch neighbourhood (9216) -> text width 5120", spec=dict(m=144, k=9216, n=5120, bias=True)),
        Cell(why="projector second layer at the text width; a 1024-token image", spec=dict(m=1024, k=5120, n=5120, bias=True)),
    )
    note = "Bit-equal to stock's `nn.Linear` of the same weights (the same torch call); the test gates against fp32."

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        return F.linear(x, weight, bias)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        with true_fp32_matmul():
            out = x.float() @ weight.float().t()
        if bias is not None:
            out = out + bias.float()
        return out.to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None) -> None:
        assert x.dtype == weight.dtype == torch.bfloat16, "the encoder runs bf16 (stock casts it to the text dtype)"
        assert x.shape[-1] == weight.shape[1]
        assert bias is None or (bias.shape == (weight.shape[0],) and bias.dtype == weight.dtype)


vision_linear = VisionLinear()

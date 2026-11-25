# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SwiGLU with the checkpoint's training clamps over a fused `[gate | up]` row, in one launch (the engine's Triton kernel)."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .clamped_swiglu import clamped_swiglu


class ClampedSwigluFused(OpWrapper):
    """`silu(gate) * up` of `x = [gate | up]` `[M, 2I]` bf16 -> bf16 `[M, I]`, computed in fp32 in one launch.

    With `limit > 0` the up half is clamped to `[-limit, limit]` and the gate half to `<= limit` (`clamped_swiglu`'s
    clamps). Rows may be strided; columns must have unit stride. Replaces the half-dozen torch launches of
    `clamped_swiglu` plus the cast back to bf16 on the shared expert's path.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: six rows of the shared expert's 512-wide slice, limit 10", spec=dict(m=6, inter=512, limit=10.0)),
        Cell(why="tep8 verify of four requests: 24 rows", spec=dict(m=24, inter=512, limit=10.0)),
        Cell(why="a prefill chunk of the whole shared expert", spec=dict(m=4096, inter=2048, limit=10.0)),
        Cell(why="limit 0 means unclamped", spec=dict(m=8, inter=512, limit=0.0)),
    )
    note = "fp32 math, bf16 out: one bf16 rounding away from `clamped_swiglu(gate, up, limit).to(bfloat16)`."

    def __call__(self, x: torch.Tensor, limit: float) -> torch.Tensor:
        return torch.ops.trtllm.silu_and_mul(x, swiglu_limit=limit if limit > 0 else None)

    def reference(self, x: torch.Tensor, limit: float) -> torch.Tensor:
        """`clamped_swiglu.reference` of the two halves, rounded to bf16."""
        gate, up = x.chunk(2, dim=-1)
        return clamped_swiglu.reference(gate, up, limit).to(torch.bfloat16)

    def is_valid(self, x: torch.Tensor, limit: float) -> None:
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.stride(-1) == 1, "x is bf16 [M, 2I] with unit column stride"
        assert x.shape[1] % 2 == 0
        assert limit >= 0, "a negative limit is treated as no clamp"


clamped_swiglu_fused = ClampedSwigluFused()

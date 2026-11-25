# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The mixes of a Hyper-Connections boundary alone (pre, post, Sinkhorn comb), for a side stream."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.hc_mixes import (
    hc_mixes,
)


class HcCoefficients(OpWrapper):
    """`residual [n, hc, d]` -> `(pre [n, hc], post [n, hc], comb [n, hc, hc])`, fp32.

    The coefficient half of a split boundary: the engine's mixer GEMM + sum of squares and `mhc_big_fuse`
    (sigmoid, softmax, Sinkhorn), with the collapse and the layer-input write pointed at a tiny stand-in so no
    residual is read twice. Equal to the plain-torch `hc_mixes` entry. Nothing here depends on the
    boundary's collapse, so the target runs it on a side stream while the sub-layer computes.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="decode rows 6", spec=dict(n=6, d=5120)),
        Cell(why="decode rows 24", spec=dict(n=24, d=5120)),
        Cell(why="decode rows 64: the largest split boundary", spec=dict(n=64, d=5120)),
    )
    note = """
    The product is the engine's (exact fp32 on CUDA cores, or its DeepGEMM TF32 split-K tactic when the
    autotuner picked it), so the band is the `hc_pre_mapping` one. The stand-in buffers are the engine's
    process-wide scratch, grown only outside a CUDA-graph capture: a capture must be preceded by an eager call
    of at least as many rows.
    """

    def __call__(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        from tensorrt_llm._torch.modules.mhc.hyper_connection import (
            _coefficient_scratch_pre,
            _coefficient_scratch_residual,
        )
        from tensorrt_llm._torch.modules.mhc.mhc_cuda import mhc_pre_mapping_fused

        n, _, d = residual.shape
        ones = _coefficient_scratch_pre(n, hc_mult, residual.device)
        standin = _coefficient_scratch_residual(n, hc_mult, residual.device)
        post, comb, _, pre = mhc_pre_mapping_fused(
            residual.view(n, hc_mult * d),
            hc_fn,
            residual,
            hc_mult,
            hc_scale,
            hc_base,
            d,
            rms_eps,
            hc_eps,
            hc_eps,
            post_mult,
            sinkhorn_iters,
            pre_mix_ext=ones,
            bigfuse_residual=standin,
        )
        return pre.view(n, hc_mult), post.view(n, hc_mult), comb.view(n, hc_mult, hc_mult)

    def reference(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert post_mult == 2.0, "hc_mixes fixes the post scale at 2"
        return hc_mixes.reference(residual, hc_fn, hc_scale, hc_base, hc_mult, sinkhorn_iters, rms_eps, hc_eps)

    def is_valid(
        self,
        residual: torch.Tensor,
        hc_fn: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        hc_mult: int,
        sinkhorn_iters: int,
        rms_eps: float,
        hc_eps: float,
        post_mult: float,
    ) -> None:
        assert residual.dim() == 3 and residual.shape[1] == hc_mult == 4, "residual is [n, 4, d]"
        n, _, d = residual.shape
        assert 0 < n, "the kernels do not take an empty grid"
        assert residual.dtype == torch.bfloat16 and residual.is_contiguous() and residual.is_cuda
        assert d % 8 == 0 and residual.data_ptr() % 16 == 0
        mix = (2 + hc_mult) * hc_mult
        assert hc_fn.shape == (mix, hc_mult * d) and hc_fn.dtype == torch.float32 and hc_fn.is_contiguous()
        assert hc_scale.shape == (3,) and hc_scale.dtype == torch.float32
        assert hc_base.shape == (mix,) and hc_base.dtype == torch.float32
        assert sinkhorn_iters >= 1 and post_mult == 2.0


hc_coefficients = HcCoefficients()

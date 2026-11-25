# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the clamped_swiglu_fused entry: the shared expert's clamped SwiGLU over a fused [gate | up] row.

Band: fp32 math rounded once to bf16, against the float64 reference rounded to bf16 => at most one bf16 ulp
(rtol 2**-7), atol 1e-5 for results near zero.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.activation.clamped_swiglu_fused import (  # noqa: E501
    clamped_swiglu_fused,
)


@pytest.mark.parametrize("cell", clamped_swiglu_fused.CELLS, ids=[c.why[:44] for c in clamped_swiglu_fused.CELLS])
def test_certified_cells(cell, certifying):
    certifying(clamped_swiglu_fused)
    s = cell.spec
    gen = gen_for(cell)
    x = (torch.randn(s["m"], 2 * s["inter"], generator=gen, device=DEV) * 12).to(torch.bfloat16)  # reaches the clamp
    got, ref = clamped_swiglu_fused(x, s["limit"]), clamped_swiglu_fused.reference(x, s["limit"])
    assert got.dtype == torch.bfloat16 and got.shape == (s["m"], s["inter"])
    torch.testing.assert_close(got.float(), ref.float(), rtol=2**-7, atol=1e-5)
    if s["limit"] > 0:
        gate, up = x.float().chunk(2, dim=-1)
        assert (gate > s["limit"]).any() and (up.abs() > s["limit"]).any()


def test_strided_rows():
    """Rows of a wider buffer (a split GEMM output) are read in place."""
    gen = torch.Generator(device=DEV).manual_seed(0)
    wide = (torch.randn(6, 1536, generator=gen, device=DEV) * 12).to(torch.bfloat16)
    x = wide[:, :1024]
    with validating(clamped_swiglu_fused):
        got = clamped_swiglu_fused(x, 10.0)
    torch.testing.assert_close(got.float(), clamped_swiglu_fused.reference(x.contiguous(), 10.0).float(), rtol=2**-7, atol=1e-5)


def test_under_cuda_graph():
    gen = torch.Generator(device=DEV).manual_seed(1)
    x = (torch.randn(6, 1024, generator=gen, device=DEV) * 12).to(torch.bfloat16)
    clamped_swiglu_fused(x, 10.0)  # compile outside the capture
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = clamped_swiglu_fused(x, 10.0)
    x.copy_((torch.randn(6, 1024, generator=gen, device=DEV) * 12).to(torch.bfloat16))
    g.replay()
    torch.testing.assert_close(out.float(), clamped_swiglu_fused.reference(x, 10.0).float(), rtol=2**-7, atol=1e-5)


def test_rejections():
    x = torch.zeros(2, 8, dtype=torch.bfloat16, device=DEV)
    clamped_swiglu_fused.is_valid(x, 0.0)
    with pytest.raises(AssertionError, match="negative"):
        clamped_swiglu_fused.is_valid(x, -1.0)
    with pytest.raises(AssertionError):
        clamped_swiglu_fused.is_valid(x.float(), 1.0)
    with pytest.raises(AssertionError):
        clamped_swiglu_fused.is_valid(torch.zeros(2, 7, dtype=torch.bfloat16, device=DEV), 1.0)

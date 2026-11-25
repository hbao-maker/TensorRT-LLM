# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's own attention entries: `csa2_bf16_lowm_proj` (bf16 projections of decode-sized steps on the
low-M kernel, as the engine's CSA2 `bf16_projection`) and `csa2_consumer_schedule` (the candidate consumers'
sparse-logits schedule once per step: bit for bit the one `consume_candidates` derives per layer).

Band: both paths accumulate the bf16 operands in fp32 and round once to bf16, so against the float64 product rounded
to bf16 they differ by at most one bf16 step of an element (element_ulp 2 in units of the row max's top-of-binade
spacing) and well under one ulp RMS; the scaled form multiplies that bf16 value in fp32 exactly as the reference.
"""

import contextlib
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.attention.csa2_bf16_lowm_proj import (  # noqa: E402,E501
    csa2_bf16_lowm_proj,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.attention.csa2_consumer_schedule import (  # noqa: E402,E501
    csa2_consumer_schedule,
)

from tensorrt_llm._torch.modules.linear import flashinfer_mxfp8_decode_graph_capture  # noqa: E402

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)


@pytest.mark.parametrize("cell", csa2_bf16_lowm_proj.CELLS, ids=[c.why[:48] for c in csa2_bf16_lowm_proj.CELLS])
def test_bf16_lowm_proj_cells(cell, certifying):
    certifying(csa2_bf16_lowm_proj)
    s, gen = cell.spec, gen_for(cell)
    x = torch.randn(s["m"], s["k"], generator=gen, device=DEV).bfloat16()
    w = (torch.randn(s["n"], s["k"], generator=gen, device=DEV) * 0.02).bfloat16()
    scale = 128**-0.5 * 32**-0.5 if s["scale"] else None
    kw = {"out_dtype": s["out_dtype"]} if "out_dtype" in s else {}
    with flashinfer_mxfp8_decode_graph_capture() if s.get("graph") else contextlib.nullcontext():  # a decode-graph step
        got = csa2_bf16_lowm_proj(x, w, scale, **kw)
        fp32_form = csa2_bf16_lowm_proj(x, w, scale) if kw else None
    ref = csa2_bf16_lowm_proj.reference(x, w, scale, **kw)
    assert got.shape == ref.shape and got.dtype == (kw.get("out_dtype", torch.float32) if scale else torch.bfloat16)
    assert_within_ulp(got.float(), ref.float(), element_ulp=2.0, rms_ulp=0.5, ulp=2**-8)  # bf16 steps, also for the fp32 form
    if kw:  # one launch into bf16: the fp32 form cast to bf16, bit for bit
        assert torch.equal(got, fp32_form.to(torch.bfloat16))


def test_bf16_lowm_proj_low_m_only_in_a_decode_graph():
    """24 rows: the low-M kernel under the decode-graph flag, cuBLAS outside it (the engine's `bf16_projection` gate);
    eager equals the cuBLAS product exactly."""
    from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

    gen = torch.Generator(device=DEV).manual_seed(7)
    x = torch.randn(24, 5120, generator=gen, device=DEV).bfloat16()
    w = (torch.randn(1024, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
    assert torch.equal(csa2_bf16_lowm_proj(x, w), cublas_mm(x, w.t()))
    with flashinfer_mxfp8_decode_graph_capture():
        lowm = csa2_bf16_lowm_proj(x, w)
    assert_within_ulp(lowm.float(), csa2_bf16_lowm_proj.reference(x, w).float(), element_ulp=2.0, rms_ulp=0.5, ulp=2**-8)


def test_bf16_lowm_proj_cuda_graph():
    gen = torch.Generator(device=DEV).manual_seed(3)
    x = torch.randn(6, 5120, generator=gen, device=DEV).bfloat16()
    w = (torch.randn(32, 5120, generator=gen, device=DEV) * 0.02).bfloat16()
    csa2_bf16_lowm_proj(x, w, 0.5)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_bf16_lowm_proj(x, w, 0.5)
    x.copy_(torch.randn(6, 5120, generator=gen, device=DEV))
    g.replay()
    assert torch.equal(out, csa2_bf16_lowm_proj(x, w, 0.5))


def test_bf16_lowm_proj_rejections():
    x = torch.zeros(6, 64, dtype=torch.bfloat16, device=DEV)
    csa2_bf16_lowm_proj.is_valid(x, torch.zeros(8, 64, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(AssertionError):
        csa2_bf16_lowm_proj.is_valid(x, torch.zeros(8, 64, device=DEV))
    with pytest.raises(AssertionError):
        csa2_bf16_lowm_proj.is_valid(x, torch.zeros(64, 8, dtype=torch.bfloat16, device=DEV).t())


@pytest.mark.parametrize("cell", csa2_consumer_schedule.CELLS, ids=[c.why[:48] for c in csa2_consumer_schedule.CELLS])
def test_consumer_schedule_cells(cell, certifying):
    certifying(csa2_consumer_schedule)
    s, gen = cell.spec, gen_for(cell)
    n, nb, pages = s["n"], s["blocks"], s["pages"]
    blocks = torch.randperm(4 * pages * n, generator=gen, device=DEV)[: n * pages].view(n, pages).to(torch.int32).contiguous()
    counts = torch.randint(1, nb + 1, (n,), generator=gen, device=DEV, dtype=torch.int32)
    ids = torch.randint(0, pages * 8, (n, nb), generator=gen, device=DEV, dtype=torch.int32)
    got = csa2_consumer_schedule(counts, blocks, ids)
    ref = csa2_consumer_schedule.reference(counts, blocks, ids)
    # The schedule buffer is DeepGEMM's (not every word is written: two calls need not be bitwise equal). What it must
    # give -- the consumer's picks, bit for bit those of the per-layer schedule -- is gated with the real indexer
    # inputs in test_modeling_v2_tep8_indexer_options.py; here the call itself and its geometry.
    assert got.shape == ref.shape and got.dtype == ref.dtype and got.device == ref.device


from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.attention.csa2_q_norm import (  # noqa: E402,E501
    csa2_q_norm,
)


@pytest.mark.parametrize("cell", csa2_q_norm.CELLS, ids=[c.why[:48] for c in csa2_q_norm.CELLS])
def test_q_norm_cells(cell, certifying):
    """The normalized latent within one bf16 rounding of the reference, bit for bit stock's FULL-layer norm
    (`rms_norm_mxfp8(quantize=False)`)."""
    certifying(csa2_q_norm)
    s, gen = cell.spec, gen_for(cell)
    x = (torch.randn(s["n"], s["k"], generator=gen, device=DEV) * 3).bfloat16()
    w = (1 + 0.1 * torch.randn(s["k"], generator=gen, device=DEV)).bfloat16()
    with validating(csa2_q_norm):
        got = csa2_q_norm(x, w, 1e-6)
    ref = csa2_q_norm.reference(x, w, 1e-6)
    assert_within_ulp(got.float(), ref.float(), element_ulp=2.0, rms_ulp=0.5, ulp=2**-8)
    from tensorrt_llm._torch.attention.backends.sparse.csa2.projection import rms_norm_mxfp8

    assert torch.equal(got, rms_norm_mxfp8(x, w, 1e-6, quantize=False)[0])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target's attention-TP path's decode-path Engram entries: `engram_ll_allgather` (peer-memory all-gather of the head / WKV
shards), `engram_gate_fused` (the engine's one-launch gate over the WKV output). (The look-back hash is the
family's `engram_lookback_hash`, tested with the family's ops.)

Bands. The gather moves bits: exact. The gate's fp32 reductions over d against float64 with one bf16 cast: one bf16
rounding step of the row's largest element, 0.2 ulp RMS (the band of the family's `engram_gate`, which computes the
same function); and the fused gate must agree with the family's plain-torch `engram_gate` on the split key / value to
the same band, since the target switches between the two by step size.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))  # _validating
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_gate_fused import (  # noqa: E402,E501
    engram_gate_fused,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__default.attention_tp.ops.engram.engram_ll_allgather import (  # noqa: E402,E501
    engram_ll_allgather,
    ll_allgather_takes,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.residual.engram_gate import (  # noqa: E402
    engram_gate,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)
EPS = 1e-6


def _one_rank_gather():
    """A one-rank group of the engine's gather: its own receive buffer is the only peer (same kernel and protocol)."""
    from tensorrt_llm._torch.distributed.ll_allgather import _DEFAULT_MAX_BYTES, LowLatencyAllGather

    buf = torch.zeros(LowLatencyAllGather.buffer_bytes(1, _DEFAULT_MAX_BYTES), dtype=torch.uint8, device=DEV)
    return LowLatencyAllGather(0, 1, [buf.data_ptr()], _DEFAULT_MAX_BYTES, keepalive=buf)


@pytest.fixture(scope="module")
def gather():
    return _one_rank_gather()


@pytest.mark.parametrize("cell", engram_ll_allgather.CELLS, ids=[c.why[:48] for c in engram_ll_allgather.CELLS])
def test_ll_allgather_cells(cell, gather, certifying):
    certifying(engram_ll_allgather)
    s = cell.spec
    gen = gen_for(cell)
    for _ in range(3):  # the epoch advances every launch: repeated gathers must keep landing
        x = torch.randn(s["rows"], s["width"], generator=gen, device=DEV).bfloat16()
        got = engram_ll_allgather(gather, x)
        assert torch.equal(got, engram_ll_allgather.reference(gather, x))


def test_ll_allgather_cuda_graph(gather):
    x = torch.zeros(6, 3200, dtype=torch.bfloat16, device=DEV)
    engram_ll_allgather(gather, x)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = engram_ll_allgather(gather, x)
    for seed in range(3):
        x.copy_(torch.randn(6, 3200, generator=torch.Generator(device=DEV).manual_seed(seed), device=DEV))
        g.replay()
        assert torch.equal(out, x)


def test_ll_allgather_rejections(gather):
    assert not ll_allgather_takes(gather, 64, 3200, torch.bfloat16)  # 400 KiB > 256 KiB: the target keeps NCCL
    assert not ll_allgather_takes(None, 6, 3200, torch.bfloat16)
    with pytest.raises(AssertionError, match="larger than the peer buffers"):
        engram_ll_allgather.is_valid(gather, torch.zeros(64, 3200, dtype=torch.bfloat16, device=DEV))
    with pytest.raises(AssertionError, match="contiguous"):
        engram_ll_allgather.is_valid(gather, torch.zeros(3200, 6, dtype=torch.bfloat16, device=DEV).t())


def _gate_inputs(spec, cell):
    n, hc, d = spec["n"], spec["hc"], spec["d"]
    gen = gen_for(cell)
    h = torch.randn(n, hc, d, generator=gen, device=DEV).bfloat16()
    kv = torch.randn(n, (hc + 1) * d, generator=gen, device=DEV).bfloat16()
    qw = (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16().float()  # bf16 values, as the checkpoint's
    kw = (torch.rand(hc, d, generator=gen, device=DEV) + 0.5).bfloat16().float()
    return h, kv, qw, kw, (qw * kw).contiguous()


@pytest.mark.parametrize("cell", engram_gate_fused.CELLS, ids=[c.why[:48] for c in engram_gate_fused.CELLS])
def test_gate_fused_cells(cell, certifying):
    certifying(engram_gate_fused)
    s = cell.spec
    h, kv, qw, kw, prod = _gate_inputs(s, cell)
    got = engram_gate_fused(h, kv, qw, kw, prod, EPS)
    assert got.dtype == h.dtype and got.shape == h.shape
    assert_within_ulp(got, engram_gate_fused.reference(h, kv, qw, kw, prod, EPS), element_ulp=2.0, rms_ulp=0.2)
    # the path it replaces at decode sizes: the family's plain-torch gate on the split key / value
    hc, d = s["hc"], s["d"]
    old = engram_gate(h, kv[:, : hc * d].reshape(-1, hc, d), kv[:, hc * d :], qw, kw, EPS, None)
    assert_within_ulp(got, old, element_ulp=2.0, rms_ulp=0.2)


def test_gate_fused_rejections():
    h = torch.zeros(2, 4, 64, dtype=torch.bfloat16, device=DEV)
    w = torch.ones(4, 64, device=DEV)
    kv = torch.zeros(2, 5 * 64, dtype=torch.bfloat16, device=DEV)
    engram_gate_fused.is_valid(h, kv, w, w, w, EPS)
    with pytest.raises(AssertionError):
        engram_gate_fused.is_valid(h, kv[:, :256], w, w, w, EPS)
    with pytest.raises(AssertionError):
        engram_gate_fused.is_valid(h, kv, w, w, w.double(), EPS)

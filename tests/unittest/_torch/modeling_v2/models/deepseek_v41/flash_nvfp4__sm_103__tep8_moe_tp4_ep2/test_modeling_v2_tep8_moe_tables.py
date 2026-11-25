# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's decode-sized routed experts on routing tables (`ops/moe/dsv4_route_tables.py`).

`dsv4_route_nvfp4_quant_tables` must route and quantize bit for bit as the family's `dsv4_route_nvfp4_quant` (the same
launch, with the tables written on the side). `fp4_moe_runner_tables` must give bit for bit the outputs of the
catalog runner routing itself on the same ids and weights: finalized rows equal, and unfinalized the FC2 rows read
through each one's permutation (the table's head on the tables) equal. Both runs use the same tactic, so any
difference is a wrong table, not rounding.

`dsv4_route_tables` (rows the FFN boundary quantized) must give the same ids, weights and tables, and the boundary's
NVFP4 copy (`fp4_quantize(h, g, 16, False, False)`, the family boundary's quantizer) the same rows and scales as the
routing launch's own quantization.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # _v41_util
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # _validating
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "moe"))  # _fp4_moe_test_utils
from _v41_util import DEV, gen_for  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.ops.moe.dsv4_route_tables import (  # noqa: E402,E501
    dsv4_route_nvfp4_quant_tables,
    dsv4_route_tables,
    fp4_moe_runner_tables,
    table_permutation,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (  # noqa: E402
    dsv4_route_nvfp4_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)
H, TOPK, SCALE = 5120, 6, 2.5
TILES = [8, 16, 32]


def _route_inputs(n, experts, gen):
    x = (torch.randn(n, H, generator=gen, device=DEV) * 2).bfloat16()
    logits = torch.randn(n, experts, generator=gen, device=DEV) * 3
    bias = torch.randn(experts, generator=gen, device=DEV) * 0.5
    g = (448.0 * 6.0 / x.float().abs().max()).reshape(1).to(DEV)
    return logits, bias, x, g


@pytest.mark.parametrize(
    "cell", dsv4_route_nvfp4_quant_tables.CELLS, ids=[c.why[:48] for c in dsv4_route_nvfp4_quant_tables.CELLS]
)
def test_route_tables_cells(cell, certifying):
    certifying(dsv4_route_nvfp4_quant_tables)
    s = cell.spec
    logits, bias, x, g = _route_inputs(s["n"], s["experts"], gen_for(cell))
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES)
    ids0, w0, xq0, xsf0 = dsv4_route_nvfp4_quant(logits, bias, x, g, TOPK, SCALE)
    for a, b in ((ids, ids0), (w, w0), (xq, xq0), (xsf, xsf0)):
        assert torch.equal(a, b)
    assert tables.dtype == torch.int32 and tables.shape[:2] == (len(TILES), 5)
    # and the routing itself is the entry's reference (the family test's gate, here on the picks only)
    rids = dsv4_route_nvfp4_quant_tables.reference(logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES)[0]
    assert torch.equal(ids.long().sort(-1).values, rids.long().sort(-1).values)


@pytest.mark.parametrize("cell", dsv4_route_tables.CELLS, ids=[c.why[:48] for c in dsv4_route_tables.CELLS])
def test_route_only_tables_cells(cell, certifying):
    certifying(dsv4_route_tables)
    s = cell.spec
    logits, bias, x, g = _route_inputs(s["n"], s["experts"], gen_for(cell))
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, s["first"], s["count"], TILES)
    ids2, w2, tables2 = dsv4_route_tables(logits, bias, TOPK, SCALE, H, s["first"], s["count"], TILES)
    for a, b in ((ids, ids2), (w, w2)):
        assert torch.equal(a, b)
    assert tables2.dtype == torch.int32 and tables2.shape == tables.shape  # the runner test proves them
    rids = dsv4_route_tables.reference(logits, bias, TOPK, SCALE, H, s["first"], s["count"], TILES)[0]
    assert torch.equal(ids2.long().sort(-1).values, rids.long().sort(-1).values)
    # the FFN boundary's NVFP4 copy is the routing launch's quantization bit for bit
    fq, fsf = torch.ops.trtllm.fp4_quantize(x, g, 16, False, False)
    assert torch.equal(fq.view(torch.uint8), xq.view(torch.uint8))
    assert torch.equal(fsf.reshape(-1).view(torch.uint8), xsf.reshape(-1).view(torch.uint8))


_EXPERTS, _LOCAL, _OFF, _INTER = 256, 128, 128, 128


@pytest.fixture(scope="module")
def layer():
    from _fp4_moe_test_utils import _build, _scalars

    args, _ = _build(_EXPERTS, H, _INTER, seed=8)
    win = {k: args[k][_OFF : _OFF + _LOCAL].contiguous() for k in ("gemm1_weights", "gemm1_weights_scale",
                                                                    "gemm2_weights", "gemm2_weights_scale")}
    return win, _scalars


def _rows_combined(rows, w, perm):
    """sum_k w[t, k] * rows[perm[t, k]] in fp32 (-1: another rank's expert, skipped)."""
    ok = perm >= 0
    picked = rows.float()[perm.clamp(min=0).long()] * (w.float() * ok).unsqueeze(-1)
    return picked.sum(1)


# 2: the routing the target tells the runner (the engine's DeepSeek-V4 MoE: DeepSeekV3, n_group = topk_group = 1, its
# routed scaling); 1: Renormalize. The ids and weights are the caller's either way; the runner's own routing (the
# reference here) runs that type's kernel.
@pytest.mark.parametrize("routing", [2, 1])
@pytest.mark.parametrize("cell", fp4_moe_runner_tables.CELLS, ids=[c.why[:48] for c in fp4_moe_runner_tables.CELLS])
def test_runner_tables_cells(cell, layer, certifying, routing):
    certifying(fp4_moe_runner_tables)
    win, scalars = layer
    n, finalize = cell.spec["n"], cell.spec["finalize"]
    logits, bias, x, g = _route_inputs(n, _EXPERTS, gen_for(cell))
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, _OFF, _LOCAL, TILES)
    s1, sg, s2 = scalars(_LOCAL, float(g), 1.0 / 32.0)
    args = (xq, xsf.reshape(-1).view(torch.float8_e4m3fn), win["gemm1_weights"], win["gemm1_weights_scale"], None,
            win["gemm2_weights"], win["gemm2_weights_scale"], s1, sg, s2, _EXPERTS, TOPK, _INTER, _OFF, _LOCAL,
            routing, finalize, 0, w, ids, tables, TILES)  # fmt: skip
    from tensorrt_llm._torch.autotuner import autotune

    kw = dict(n_group=1, topk_group=1, routed_scaling_factor=SCALE) if routing == 2 else {}
    with autotune():  # the tables serve tuned tactics only (an untuned run routes in the runner), as in the engine
        fp4_moe_runner_tables(*args, **kw)
    got = fp4_moe_runner_tables(*args, **kw)
    ref = fp4_moe_runner_tables.reference(*args, **kw)
    if finalize:
        assert torch.equal(got[0], ref[0])
        return
    perm, used = table_permutation(got, tables, TILES, n, TOPK)
    assert used is not None, "the runner's tactic did not run on the tables: nothing was proved"
    ref_perm, ref_used = table_permutation(ref, tables, TILES, n, TOPK)
    assert ref_used is None
    assert torch.equal(_rows_combined(got[0], w, perm), _rows_combined(ref[0], w, ref_perm))


def test_runner_on_route_only_tables(layer):
    """The runner on `dsv4_route_tables`' tables with the boundary's NVFP4 rows (`fp4_quantize`) gives bit for bit the
    expert rows of the runner on `dsv4_route_nvfp4_quant_tables`' tables and rows (same tactic, on the tables)."""
    from tensorrt_llm._torch.autotuner import autotune

    win, scalars = layer
    n = 6
    logits, bias, x, g = _route_inputs(n, _EXPERTS, torch.Generator(device=DEV).manual_seed(11))
    s1, sg, s2 = scalars(_LOCAL, float(g), 1.0 / 32.0)
    ids, w, xq, xsf, tables = dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, _OFF, _LOCAL, TILES)
    ids2, w2, tables2 = dsv4_route_tables(logits, bias, TOPK, SCALE, H, _OFF, _LOCAL, TILES)
    fq, fsf = torch.ops.trtllm.fp4_quantize(x, g, 16, False, False)

    def run(q, sf, w_, ids_, t):
        args = (q, sf.reshape(-1).view(torch.float8_e4m3fn), win["gemm1_weights"], win["gemm1_weights_scale"], None,
                win["gemm2_weights"], win["gemm2_weights_scale"], s1, sg, s2, _EXPERTS, TOPK, _INTER, _OFF, _LOCAL,
                2, False, 0, w_, ids_, t, TILES)  # fmt: skip
        kw = dict(n_group=1, topk_group=1, routed_scaling_factor=SCALE)
        with autotune():
            fp4_moe_runner_tables(*args, **kw)
        out = fp4_moe_runner_tables(*args, **kw)
        perm, used = table_permutation(out, t, TILES, n, TOPK)
        assert used is not None, "the runner's tactic did not run on the tables: nothing was proved"
        return _rows_combined(out[0], w_, perm)

    assert torch.equal(run(fq, fsf, w2, ids2, tables2), run(xq, xsf, w, ids, tables))


def test_rejections():
    logits, bias, x, g = _route_inputs(6, 384, torch.Generator(device=DEV).manual_seed(1))
    dsv4_route_nvfp4_quant_tables.is_valid(logits, bias, x, g, TOPK, SCALE, 192, 192, TILES)
    big = _route_inputs(9, 384, torch.Generator(device=DEV).manual_seed(1))
    with pytest.raises(AssertionError, match="at most 8 rows"):
        dsv4_route_nvfp4_quant_tables.is_valid(*big, TOPK, SCALE, 192, 192, TILES)
    with pytest.raises(AssertionError):
        dsv4_route_nvfp4_quant_tables.is_valid(logits, bias, x, g, TOPK, SCALE, 192, 192, [64])


@pytest.mark.parametrize("n", range(1, 9))
def test_every_tile_table_row_is_16_byte_aligned(n):
    """An unfinalized run's permutation is row 0 of its tile's table (`table_permutation`), which the next boundary
    kernel reads as a 16-byte aligned tensor: every row of every tile's table starts 16 bytes aligned, for every row
    count the launch takes and both launches (with and without the quantization)."""
    logits, bias, x, g = _route_inputs(n, 384, torch.Generator(device=DEV).manual_seed(n))
    for tables in (dsv4_route_nvfp4_quant_tables(logits, bias, x, g, TOPK, SCALE, 192, 192, TILES)[4],
                   dsv4_route_tables(logits, bias, TOPK, SCALE, H, 192, 192, TILES)[2]):
        assert tables.is_contiguous() and tables.shape[:2] == (len(TILES), 5)
        for t in range(len(TILES)):
            for r in range(5):
                assert tables[t, r].data_ptr() % 16 == 0, (n, t, r, tables.shape)
        perm = tables[len(TILES) - 1, 0][: n * TOPK].view(n, TOPK)
        assert perm.data_ptr() % 16 == 0

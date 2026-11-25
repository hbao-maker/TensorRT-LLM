# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_staged_context entry: stock's native FP8 context path of a < 64-head rank (E4M3 staging of every
row a query reads, trtllm-gen context FMHA) on the packed pools, against the plain-torch reference.

The query and the staged rows are E4M3 at the forward's staging scale and the FMHA's probabilities E4M3, so the gate is
FP8-level: relative RMS below 6e-2 against the fp32 attention over the dequantized rows (E4M3 Q, K and probabilities:
three roundings of ~2.5% each, measured 2.5-4%). The window rows of a query are stock's (`csa2_swa_slots`: the step's own rows are
published in the pool first) and its compressed selections map through the owner's page table inside stock's staging
kernel (`main_mapping`, `csa2_main_mapping`). The generation entry (`csa2_staged_generation`, the same backend's generation
tile past the direct decode's 16 rows: trtllm-gen MLA generation FMHA) is gated the same way, and its staging is
deterministic across repeats and identical between an eager tile and a graph-reserved one; a further test checks that mapping (`csa2_main_rows`, stock's
`_map_global_slots`, which the reference uses) against a direct page-table walk.
"""

import pytest
import torch
from _v41_util import DEV
from test_modeling_v2_csa2_sparse_decode import WIN, _build

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_staged_context import (  # noqa: E501
    csa2_main_mapping,
    csa2_main_rows,
    csa2_staged_context,
    csa2_staged_context_supported,
    csa2_staged_generation,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_swa_slots import (  # noqa: E501
    csa2_swa_slots,
    csa2_swa_table_descriptors,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not csa2_staged_context_supported(8, "cuda"), reason="stock's native FP8 path (SM100f)"
)


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().mean().sqrt() / b.square().mean().sqrt()).item()


def _case(spec):
    hist = spec.get("hist", (0,) * len(spec["lengths"]))
    ratio = 2 if spec["topk"] else 0
    c = _build(list(zip(hist, spec["lengths"])), ratio, spec["topk"], gen=True, heads=spec["heads"])
    nreq = len(spec["lengths"])
    desc = csa2_swa_table_descriptors(c["swa_table"].unsqueeze(0))
    reads, _, _ = csa2_swa_slots(c["pos"], c["rid"], c["swa_table"].unsqueeze(0), desc,
                                 torch.zeros(1, dtype=torch.int32, device=DEV), torch.zeros(nreq, dtype=torch.int64, device=DEV),
                                 WIN, WIN)  # fmt: skip
    main_logical = mapping = None
    if ratio:
        main_logical = c["idx"]
        page = 128 // ratio
        mapping = csa2_main_mapping(c["main_table"], c["rid"], (c["pos"].long() + 1) // ratio, ratio,
                                    c["main_table"].shape[1] * page)
    starts = list(hist)
    args = (c["q"], c["sink"], c["swa_pool"], reads[0], c["main_pool"] if ratio else None, main_logical, mapping,
            list(spec["lengths"]), starts, spec["topk"])  # fmt: skip
    return c, args


@pytest.mark.parametrize("cell", csa2_staged_context.CELLS, ids=[c.why[:44] for c in csa2_staged_context.CELLS])
def test_certified_cells(cell, certifying):
    certifying(csa2_staged_context)
    c, args = _case(cell.spec)
    n, heads = c["q"].shape[:2]
    ref = csa2_staged_context.reference(*args)
    with validating(csa2_staged_context):
        got = csa2_staged_context(*args)
    assert got.dtype == torch.bfloat16 and got.shape == (n, heads, 512)
    rel = _rel(got, ref)
    print(f"staged context {cell.why}: vs reference {rel:.4f}")
    assert rel < 6e-2, rel
    again = csa2_staged_context(*args)
    assert torch.equal(again, got), "a repeated forward over the same rows is the same launch sequence"


def test_main_rows_walk_the_page_table():
    c, args = _case(dict(heads=8, lengths=(37, 90), topk=512, hist=(300, 1500)))
    logical = args[5]
    rows = csa2_main_rows(logical, args[6])
    page = 64
    expect = torch.full_like(rows, -1)
    vis = (c["pos"].long() + 1) // 2
    for t in range(rows.shape[0]):
        r = int(c["rid"][t])
        for j in range(logical.shape[1]):
            p = int(logical[t, j])
            if 0 <= p < int(vis[t]):
                expect[t, j] = int(c["main_table"][r, p // page]) * page + p % page
    assert torch.equal(rows, expect)


def _gen_case(spec):
    hist = spec.get("hist", (0,) * len(spec["rows"]))
    ratio = 2 if spec["topk"] else 0
    c = _build(list(zip(hist, spec["rows"])), ratio, spec["topk"], gen=True, heads=spec["heads"])
    nreq = len(spec["rows"])
    desc = csa2_swa_table_descriptors(c["swa_table"].unsqueeze(0))
    reads, _, _ = csa2_swa_slots(c["pos"], c["rid"], c["swa_table"].unsqueeze(0), desc,
                                 torch.zeros(1, dtype=torch.int32, device=DEV), torch.zeros(nreq, dtype=torch.int64, device=DEV),
                                 WIN, WIN)  # fmt: skip
    main_logical = mapping = None
    if ratio:
        main_logical = c["idx"]
        mapping = csa2_main_mapping(c["main_table"], c["rid"], (c["pos"].long() + 1) // ratio, ratio,
                                    c["main_table"].shape[1] * (128 // ratio))
    return c, (c["q"], c["sink"], c["swa_pool"], reads[0], c["main_pool"] if ratio else None, main_logical, mapping,
               spec["topk"])


@pytest.mark.parametrize("cell", csa2_staged_generation.CELLS, ids=[c.why[:44] for c in csa2_staged_generation.CELLS])
def test_generation_certified_cells(cell, certifying):
    certifying(csa2_staged_generation)
    c, args = _gen_case(cell.spec)
    n, heads = c["q"].shape[:2]
    ref = csa2_staged_generation.reference(*args)
    with validating(csa2_staged_generation):
        got = csa2_staged_generation(*args)
    assert got.dtype == torch.bfloat16 and got.shape == (n, heads, 512)
    rel = _rel(got, ref)
    print(f"staged generation {cell.why}: vs reference {rel:.4f}")
    assert rel < 6e-2, rel
    assert torch.equal(csa2_staged_generation(*args), got), "a repeated forward over the same rows is the same launches"
    # a CUDA-graph step's tile (reserved for the graph; its warm-up is this eager call) attends alike
    assert torch.equal(csa2_staged_generation(*args, graph=True), got)


def test_generation_under_a_cuda_graph():
    """The generation tile warmed up eagerly as a graph step, then captured and replayed with new query values: the
    replay equals an eager forward of the same values (the engine captures generation steps this way)."""
    c, args = _gen_case(dict(heads=8, rows=(4,) * 16, topk=512, hist=(3000,) * 16))
    q = args[0].clone()
    rest = args[1:]
    csa2_staged_generation(q, *rest, graph=True)  # warm-up (sizes the native workspace)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = csa2_staged_generation(q, *rest, graph=True)
    q.copy_(args[0] * 0.5)
    g.replay()
    torch.cuda.synchronize()
    want = csa2_staged_generation(args[0] * 0.5, *rest)
    assert _rel(out, want) < 1e-6

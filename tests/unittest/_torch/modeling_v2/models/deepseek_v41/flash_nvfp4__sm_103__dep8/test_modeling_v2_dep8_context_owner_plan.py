# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep8 attention's compressed groups of a context chunk (`_context_owner_plan`): request `r` of the step's `segs`
has `t1 - t0` rows ending at position `span`, as the engine's prepare uploads its query lengths and KV ends; every
owner's GLOBAL write slots, first positions and (ratio 2) compression batch then equal the engine's PyTorch form
(`CSA2TrtllmMetadata._refresh_owner_slots`) bit for bit, and its batch has the step's rows as output rows."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="imports the target (CUDA build)")
att = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep8.attention")
DEV = "cuda"


@pytest.mark.parametrize("rows, ends", [((5,), (5,)), ((3, 7, 1, 20), (3, 1008, 65, 57)), ((1, 130, 37), (2048, 259, 37))])
def test_context_owner_plan_is_the_engines(rows, ends):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    gen = torch.Generator(device=DEV).manual_seed(sum(rows))
    sources, ratios = (2, 3, 4), {2: 2, 3: 2, 4: 1}
    requests, n, cols = len(rows), sum(rows), 64
    tables = {o: torch.randperm(requests * cols, generator=gen, device=DEV).view(requests, cols).int().contiguous()
              for o in sources}
    for t in tables.values():
        t[torch.rand(t.shape, generator=gen, device=DEV) < 0.05] = -1
    segs, t0 = [], 0
    for r, (k, e) in enumerate(zip(rows, ends)):
        segs.append((t0, t0 + k, r, e))
        t0 += k
    ctx = NS(pos=torch.zeros(n, dtype=torch.int32, device=DEV), segs=segs, paged=False, nreq=requests,
             cache={("t", o, att.GLOBAL): tables[o] for o in sources})
    plan = att._owner_plan(ctx, sources, ratios)
    assert plan is ctx.cache["owners"] and att._owner_plan(ctx, sources, ratios) is plan
    lengths = torch.tensor(rows, dtype=torch.int32, device=DEV)
    e = torch.tensor(ends, dtype=torch.int32, device=DEV)
    assert torch.equal(plan["ends"], e) and torch.equal(plan["lengths"], lengths)
    assert plan["cu_seq"].tolist() == [0, *torch.tensor(rows).cumsum(0).tolist()]
    assert plan["max_outputs"] == max(1, -(-max(rows) // 2))
    starts = e - lengths
    for o in sources:
        r = ratios[o]
        got = plan["owners"][o]
        slots, first = torch.empty(n, dtype=torch.int64, device=DEV), torch.empty(n, dtype=torch.int32, device=DEV)
        batch = tuple(torch.empty(k, dtype=torch.int32, device=DEV) for k in (requests, requests, requests + 1)) if r == 2 else (None,) * 3
        CSA2TrtllmMetadata._refresh_owner_slots(starts, lengths, torch.zeros_like(starts), r, 128 // r, tables[o], slots,
                                                first, *batch)
        assert torch.equal(got[1], slots) and torch.equal(got[2], first), o
        if r == 2:
            assert all(torch.equal(a, b) for a, b in zip(got[5:], batch)), o

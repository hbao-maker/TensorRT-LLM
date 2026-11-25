# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the csa2_indexer_candidates entries, driven on the logits the indexer scores.

`publish_candidates` (block maxima -> top blocks -> expanded positions) is gated on its position set
against `csa2_candidate_blocks` over the same logits: at least 97% of the reference's kept positions
are in the op's set (blocks with equal maxima may be chosen differently). `consume_candidates` (sparse
FP4 logits over those positions -> top-k) is gated on the score band of `check_consumer_picks`: the
kernel returns bf16 scores and takes bf16 head weights.
"""

import pytest
import torch
from _v41_util import DEV, IS_SM103, check_consumer_picks, request_index_pools

__extra_import_path__ = ["../../../.."]
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_candidates import (  # noqa: E501
    consume_candidates,
    publish_candidates,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_indexer_decode import (  # noqa: E501
    _logits,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_pack_rows import (  # noqa: E501
    csa2_pack_rows,
    csa2_unpack_rows,
)

pytestmark = pytest.mark.skipif(
    not IS_SM103, reason="the sparse FP4 MQA-logits kernel is certified on sm_103 only"
)
HEADS, TOPK = 32, 64


def _qdq(q):
    return csa2_unpack_rows(csa2_pack_rows(q.reshape(-1, 128), "index"), "index", q.dtype).view_as(q)


def _setup(spec):
    """Pool, rows and the layer's logits of one cell: `nreq` requests of `rows` trailing positions
    (generation) or one request's `n`-row chunk after `hist` tokens (context)."""
    ratio = spec["ratio"]
    if "nreq" in spec:
        lens = [2500 + 101 * i for i in range(spec["nreq"])]
        pp = [[ln - spec["rows"] + j for j in range(spec["rows"])] for ln in lens]
    else:
        lens = [spec["hist"] + spec["n"]]
        pp = [list(range(spec["hist"], spec["hist"] + spec["n"]))]
    pool, table = request_index_pools(ratio, lens, seed=5)
    pos = torch.tensor([p for ps in pp for p in ps], device=DEV, dtype=torch.int32)
    rid = torch.tensor([r for r, ps in enumerate(pp) for _ in ps], device=DEV)
    q = _qdq(torch.randn(len(pos), HEADS, 128, device=DEV).bfloat16())
    w = torch.rand(len(pos), HEADS, device=DEV) * 0.1
    clen = ((pos.long() + 1) // ratio).clamp(min=0).to(torch.int32)
    span = max(lens) // ratio + 8
    width = min(table.shape[1] * (128 // ratio), -(-span // 8) * 8)
    logits = _logits(q, w, clen.clamp(min=1), pool, table, rid, ratio, width)
    return dict(pool=pool, table=table, pos=pos, rid=rid, q=q, w=w, clen=clen, width=width, logits=logits)


@pytest.mark.parametrize("cell", publish_candidates.CELLS, ids=[c.why[:44] for c in publish_candidates.CELLS])
def test_publish_certified_cells(cell, certifying):
    certifying(publish_candidates)
    s = cell.spec
    c = _setup(s)
    args = (c["logits"], c["table"], c["rid"], s["ratio"], c["clen"], s["block"], s["blocks"])
    positions, ids, counts = publish_candidates(*args)
    mask = publish_candidates.reference(*args)
    n, width = mask.shape
    assert positions.dtype == ids.dtype == counts.dtype == torch.int32
    assert positions.shape[0] == n and ids.shape == (n, positions.shape[1] // 8) and counts.shape == (n, 1)
    # ascending, -1 padded
    live = positions >= 0
    assert bool((live[:, :-1] | ~live[:, 1:]).all()), "valid positions come first"
    kept = torch.zeros(n, width + 1, dtype=torch.bool, device=DEV)
    kept.scatter_(1, torch.where(live, positions.long(), width), torch.ones_like(live))
    kept = kept[:, :width]
    assert (mask & kept).sum() / mask.sum() > 0.97
    assert bool((counts >= 0).all() and (counts <= positions.shape[1]).all())


@pytest.mark.parametrize("cell", consume_candidates.CELLS, ids=[c.why[:44] for c in consume_candidates.CELLS])
def test_consume_certified_cells(cell, certifying):
    certifying(consume_candidates)
    s = cell.spec
    c = _setup(s)
    cand = publish_candidates(c["logits"], c["table"], c["rid"], s["ratio"], c["clen"], s["block"], s["blocks"])
    q2 = _qdq(torch.randn_like(c["q"].float()).bfloat16())
    args = (q2, c["w"], c["rid"], c["pool"], c["table"], s["ratio"], TOPK, cand, c["clen"])
    got = consume_candidates(*args)
    ref = consume_candidates.reference(*args)
    assert got.dtype == torch.int32 and got.shape == ref.shape
    with true_fp32_matmul():
        for picks in (got, ref):
            check_consumer_picks(
                picks, q2, c["w"], c["rid"], c["pool"], c["table"], s["ratio"], TOPK, cand[0], c["clen"], c["width"]
            )


def test_rejections():
    c = _setup(publish_candidates.CELLS[0].spec)
    ratio = publish_candidates.CELLS[0].spec["ratio"]
    args = (c["logits"], c["table"], c["rid"], ratio, c["clen"], 8, 40)
    publish_candidates.is_valid(*args)
    with pytest.raises(AssertionError, match="8 keys"):
        publish_candidates.is_valid(c["logits"], c["table"], c["rid"], ratio, c["clen"], 16, 40)
    with pytest.raises(AssertionError):
        publish_candidates.is_valid(c["logits"], c["table"], c["rid"], ratio, c["clen"].long(), 8, 40)
    cand = publish_candidates(*args)
    consume_candidates.is_valid(c["q"], c["w"], c["rid"], c["pool"], c["table"], ratio, TOPK, cand, c["clen"])
    with pytest.raises(AssertionError, match="triple"):
        consume_candidates.is_valid(c["q"], c["w"], c["rid"], c["pool"], c["table"], ratio, TOPK, cand[:2], c["clen"])

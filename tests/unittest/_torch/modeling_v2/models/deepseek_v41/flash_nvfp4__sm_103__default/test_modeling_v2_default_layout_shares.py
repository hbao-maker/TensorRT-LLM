# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The default target reads every share from the mapping: at each layout routing admits for it, the module it builds
for a rank holds exactly that rank's piece (attention heads and output groups, the routed experts' window and
intermediate slice, the shared expert's slice), and the pieces of all ranks tile the checkpoint's tensors. Built on CPU
with the checkpoint's shapes; no weights."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the modules create CUDA events")

_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
_CFG = SimpleNamespace(
    hidden_size=5120, head_dim=512, qk_rope_head_dim=64, num_attention_heads=64, q_lora_rank=1280, o_lora_rank=1024,
    o_groups=8, sliding_window=128, compress_ratios=[0, 0] + [2] * 18 + [1] * 20, kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36], candidate_source_layer_id=20, candidate_block_size=8,
    candidate_topk_blocks=2048, index_topk=512, index_n_heads=32, index_head_dim=128, rms_norm_eps=1e-20,
    n_routed_experts=384, num_experts_per_tok=6, moe_intermediate_size=2304, swiglu_limit=10.0,
)  # fmt: skip
# routing's `_DEFAULT_LAYOUTS`: (world, attention DP, moe tp, moe ep)
_LAYOUTS = [(4, True, 1, 4), (6, True, 1, 6), (16, True, 1, 16), (4, False, 1, 4), (4, False, 4, 1), (4, False, 2, 2),
            (8, False, 1, 8), (8, False, 8, 1), (8, False, 2, 4)]  # fmt: skip


def _mapping(world, adp, moe_tp, moe_ep, rank):
    return Mapping(world_size=world, rank=rank, tp_size=world, enable_attention_dp=adp, moe_tp_size=moe_tp,
                   moe_ep_size=moe_ep)  # fmt: skip


def test_the_layouts_are_routings():
    routing = importlib.import_module(f"{_PKG}.routing")
    assert set(_LAYOUTS) == set(routing._DEFAULT_LAYOUTS)


@pytest.mark.parametrize("layout", _LAYOUTS, ids=lambda v: f"w{v[0]}-{'adp' if v[1] else 'tp'}-moe{v[2]}x{v[3]}")
def test_each_rank_holds_its_piece(layout):
    world, adp, moe_tp, moe_ep = layout
    core = importlib.import_module(f"{_PKG}.flash_nvfp4__sm_103__default.attention_{'dp' if adp else 'tp'}.core")
    attn_tp = 1 if adp else world
    experts, inters, heads = set(), {}, []
    for rank in range(world):
        m = _mapping(world, adp, moe_tp, moe_ep, rank)
        a = core._Attention(_CFG, 24, m)
        assert (a.hl, a.gl) == (64 // attn_tp, 8 // attn_tp)
        assert tuple(a.wq_b.shape) == (a.hl * 512, 1280) and tuple(a.wo_b.shape) == (5120, a.gl * 1024)
        heads.append(a.tp_rank)
        moe = core._Moe(_CFG, m)
        assert moe.local == 384 // moe_ep and moe.offset == m.moe_ep_rank * moe.local
        assert moe.tp == moe_tp and moe.inter_t >= 2304 // moe_tp and moe.inter_t % 32 == 0
        # stock's TRTLLM-Gen storage width: 32-aligned unless the 128-row scale layout needs 128 (288 -> 384)
        assert moe.inter_t == {1: 2304, 2: 1152, 4: 576, 8: 384}[moe_tp]
        experts.add((moe.offset, moe.tp_rank))
        inters[moe.tp_rank] = moe.inter_t
        if not adp:  # the shared expert's intermediate width is split over the attention-TP group
            se = core._SharedExpert(5120, 2304, 10.0, world)
            assert se.part == 2304 // world and se.part_p % 128 == 0
    # every (expert window, intermediate slice) is held by exactly world / (moe_tp * moe_ep) = 1 rank
    assert len(experts) == world == moe_tp * moe_ep
    assert sorted(heads) == ([0] * world if adp else list(range(world)))

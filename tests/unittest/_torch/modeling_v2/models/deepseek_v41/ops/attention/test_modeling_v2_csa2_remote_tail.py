# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated remote-tail replay (`bounded_replay_on_generation`): the destination's decoder SWA floors entry against
its reference, and the setup helpers against stock's contract (`DeepseekV41Model.__init__` / `_resolve_replay_policy` /
`_make_bounded_replay_plan`, `configs/deepseek_v41.py::disagg_context_decoder_skipping_enabled`): the same errors where
stock raises, the same refusals where stock runs full prefill, no compaction in a remote-tail step. Host only."""

from types import SimpleNamespace as NS

import pytest
import torch

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (  # noqa: E501
    csa2_replay_plan,
    csa2_replay_policy,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_remote_tail import (  # noqa: E501
    csa2_context_decoder_skipping,
    csa2_enter_context_only,
    csa2_remote_tail_floors,
    csa2_remote_tail_mode,
    csa2_remote_tail_policy,
    csa2_remote_tail_refusal,
)


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    for name in ("TRTLLM_DISAGG_ROLE", "TRTLLM_V41_DECODER_BOUNDED_REPLAY", "TRTLLM_V41_BOUNDED_REPLAY_TOKENS"):
        monkeypatch.delenv(name, raising=False)


@pytest.mark.parametrize("cell", csa2_remote_tail_floors.CELLS, ids=[c.why[:44] for c in csa2_remote_tail_floors.CELLS])
def test_cells(cell, certifying):
    certifying(csa2_remote_tail_floors)
    s = cell.spec
    args = (list(s["starts"]), s["nctx"], s["nreq"], None if s["prior"] is None else list(s["prior"]))
    got = csa2_remote_tail_floors(*args)
    assert got.dtype == torch.int64 and got.tolist() == csa2_remote_tail_floors.reference(*args)


def test_floors_refuse_a_start_per_context_mismatch():
    with pytest.raises(ValueError, match="remote-tail starts must match the context rows"):
        csa2_remote_tail_floors([10, 20], 1, 3)


def _cfg(**kw):
    base = dict(kv_source_layer_ids=[2, 8, 14, 20], compress_ratios=[0, 0] + [2] * 18 + [1] * 20,
                engram_layer_ids=[1, 14], sliding_window=128, num_hidden_layers=40)
    base.update(kw)
    return NS(**base)


def _dspark(embedded=True):
    return NS(spec_dec_mode=NS(is_dspark=lambda: True), draft_is_embedded_in_target=embedded,
              target_layer_ids=[1, 20, 39])


def _mc(adp=True, switch=True, spec=None):
    return NS(mapping=NS(enable_attention_dp=adp, pp_size=1, cp_size=1), spec_config=spec,
              extra_attrs={"bounded_replay_on_generation": switch})


def test_switch_off_changes_nothing():
    assert csa2_replay_policy(_cfg(), _mc(switch=False), captures=True) == (20, 140)
    assert csa2_remote_tail_policy(_mc(switch=False), 20) == (False, False)
    assert csa2_remote_tail_refusal(_mc(switch=False)) is None


def test_attention_dp_needs_an_explicit_role(monkeypatch):
    """Stock's ADP checks: no role (or a context role with decoder replay off) refuses the replay, and the model then
    raises stock's error; the generation role and the context-only worker replay from layer 20."""
    assert csa2_replay_policy(_cfg(), _mc(), captures=True) == (None, 0)
    with pytest.raises(ValueError, match="requires supported decoder bounded replay"):
        csa2_remote_tail_policy(_mc(), None)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    assert csa2_replay_policy(_cfg(), _mc(), captures=True) == (20, 140)
    assert csa2_remote_tail_policy(_mc(), 20) == (True, False)
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    assert csa2_context_decoder_skipping(True) and not csa2_context_decoder_skipping(False)
    assert csa2_replay_policy(_cfg(), _mc(), captures=False) == (20, 140)
    assert csa2_remote_tail_policy(_mc(), 20) == (True, True)
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "0")
    assert not csa2_context_decoder_skipping(True)
    assert csa2_replay_policy(_cfg(), _mc(), captures=False) == (None, 0)
    with pytest.raises(ValueError, match="requires supported decoder bounded replay"):
        csa2_remote_tail_policy(_mc(), None)


def test_without_attention_dp_the_role_is_free():
    assert csa2_replay_policy(_cfg(), _mc(adp=False), captures=True) == (20, 140)
    assert csa2_remote_tail_policy(_mc(adp=False), 20) == (True, False)


def test_speculation(monkeypatch):
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "generation")
    assert csa2_replay_policy(_cfg(), _mc(spec=_dspark()), captures=True) == (20, 140)
    assert "embedded DSpark" in csa2_remote_tail_refusal(_mc(spec=_dspark(embedded=False)))
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    with pytest.raises(ValueError, match="requires speculative decoding to be disabled"):
        csa2_remote_tail_policy(_mc(spec=_dspark()), 20)


def test_mode_and_context_only_entry():
    md = NS(csa2_remote_tail_mode=None)
    csa2_enter_context_only(md)
    assert csa2_remote_tail_mode(md) == "source"
    with pytest.raises(ValueError, match="cannot execute a remote-tail destination"):
        csa2_enter_context_only(NS(csa2_remote_tail_mode="destination"))
    with pytest.raises(ValueError, match="unknown CSA2 remote-tail mode"):
        csa2_remote_tail_mode(NS(csa2_remote_tail_mode="sideways"))
    assert csa2_remote_tail_mode(NS()) is None


def _plan_metadata(mode, adp):
    seq = torch.tensor([600, 4], dtype=torch.int32)
    return NS(is_cuda_graph=False, mapping=NS(enable_attention_dp=adp) if adp else None, padded_num_tokens=None,
              num_tokens=604, num_contexts=1, num_chunked_ctx_requests=0, seq_lens=seq, seq_lens_cuda=seq,
              num_seqs=2, kv_cache_manager=NS(enable_block_reuse=False),
              kv_cache_params=NS(num_cached_tokens_per_seq=[0, 100]), decoder_context_ends=(600, 0),
              prompt_lens=[600, 1000], csa2_remote_tail_mode=mode)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the plan's rows go to the device")
def test_no_compaction_in_a_remote_tail_step():
    """Stock's `_make_bounded_replay_plan`: a source or destination step runs its rows as they are, and a remote-tail
    model under attention DP never compacts (every rank stops or continues at the same layer); otherwise as before."""
    model = NS(decoder_replay_split=20, decoder_replay_window=140, ced_kv_precompute=True,
               disagg_remote_tail_replay=True)
    for mode in ("source", "destination"):
        assert csa2_replay_plan(model, _plan_metadata(mode, adp=False), False, None) is None
    assert csa2_replay_plan(model, _plan_metadata(None, adp=True), False, None) is None
    plan = csa2_replay_plan(model, _plan_metadata(None, adp=False), False, None)
    assert plan is not None and plan.replay_seq_lens.tolist() == [140, 4]

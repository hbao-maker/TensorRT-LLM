# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep8 target as the generation side of disaggregated remote-tail replay (`bounded_replay_on_generation`), as
stock's `DeepseekV41Model._enter_remote_tail_boundary` / `enter_remote_tail_decoder` run it: a destination step runs
every layer over every row; at the boundary (layer 20) the owner publishes its compressed rows from the resolved stream
and from there each context's SWA reads start at its `remote_tail_start` (generation rows keep theirs). A source step
stops at the boundary with the collapse, after publishing from the collapse normalized by the owner's input norm.
`_run_layers` around stub layers, host stand-ins for the kernels; the floors are uploaded, so a device is needed."""

import dataclasses
import importlib
from types import SimpleNamespace as NS
from typing import Optional

import pytest
import torch
from torch import nn

TARGET = "flash_nvfp4__sm_103__dep8"
modeling = importlib.import_module(f"tensorrt_llm._torch.modeling_v2.models.deepseek_v41.{TARGET}.modeling")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the floors are uploaded to the device")

SPLIT, LAYERS, HC, DIM = 2, 4, 2, 8
PER_ROW_FLOORS = True  # dep8 / dep32 keep one floor per step row, tep8 one per request row


@dataclasses.dataclass
class _Ctx:
    rid: torch.Tensor
    nreq: int
    paged: bool = False
    published: bool = False
    floor: Optional[torch.Tensor] = None


class _StubLayer:
    def __init__(self, idx, log):
        self.idx, self.log = idx, log
        self.attn_norm, self.norm_eps = torch.full((DIM,), 3.0), 1e-6
        self.attn = NS(publish_global=lambda h, ctx: log.append(("publish_source", idx, h.clone())))

    def publish_global(self, residual, pre_mix, ctx):
        self.log.append(("publish", self.idx, residual.clone(), pre_mix.clone()))

    def __call__(self, st, ctx, *args, **kwargs):
        self.log.append(("run", self.idx, st.residual.shape[0], ctx))
        x = st.residual + (st.x_prev[:, None, :] if st.deferred else 0)  # this layer's boundary applies the last update
        # the next state is deferred: this layer's FFN output (ones) is applied by the next boundary
        return modeling._HcState(x, st.pre_mix, post=torch.ones(x.shape[0], HC), comb=None,
                                 x_prev=torch.ones(x.shape[0], DIM))


def _core(log):
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core.layers = [_StubLayer(i, log) for i in range(LAYERS)]
    core.decoder_replay_split, core.hc = SPLIT, HC
    core.engram_hash = NS(token_map=None, mults=None, primes=None, pad_id=0, lookback=lambda *a: None)
    # the Engram look-back rows (`req_slot` / `hist` read them)
    core.engram_slots = NS(req_slot=torch.zeros(4, dtype=torch.int32), hist=None)
    core.embed_tokens = None
    core._image_mask_everywhere = False  # a text-only core (no attention-DP TRTLLM experts gathering image rows)
    core._engram_exchange_launch = lambda ids, md: None
    core._engram_precompute = lambda ids, md: None
    core._close = lambda st: ("closed", st)
    return core


@pytest.fixture(autouse=True)
def _host_ops(monkeypatch):
    monkeypatch.setattr(modeling, "token_embedding", lambda table, ids: torch.ones(ids.shape[0], DIM))
    monkeypatch.setattr(modeling, "engram_lookback_hash", lambda *a: None)
    monkeypatch.setattr(modeling, "csa2_step_replay_plan", lambda *a, **k: None)
    monkeypatch.setattr(modeling, "hc_post_mapping", lambda x, residual, post, comb: residual + x[:, None, :])
    monkeypatch.setattr(modeling, "hc_collapse", lambda x, pre: (x.float() * pre[..., None]).sum(1).to(x.dtype))
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", lambda h, w, eps: h * w)


def _run(mode, starts=(), nctx=0, lens=(3, 1)):
    log = []
    n = sum(lens)
    rid = torch.repeat_interleave(torch.arange(len(lens)), torch.tensor(lens))
    step = NS(positions=torch.arange(n, dtype=torch.int32), n=n, tok_req=rid)
    ctx = _Ctx(rid=rid, nreq=len(lens))
    md = NS(csa2_remote_tail_mode=mode, csa2_remote_tail_starts=list(starts), num_contexts=nctx,
            all_rank_num_tokens=[n])
    out = _core(log)._run_layers(step, ctx, md, torch.zeros(n, dtype=torch.int32))
    return out, log


def test_destination_runs_every_layer_with_the_tail_floors():
    out, log = _run("destination", starts=(1024,), nctx=1)
    assert out[0] == "closed"
    runs = [e for e in log if e[0] == "run"]
    assert [(e[1], e[2]) for e in runs] == [(i, 4) for i in range(LAYERS)]  # every layer, every row
    (pub,) = [e for e in log if e[0] == "publish"]
    assert pub[1] == SPLIT and torch.equal(pub[2], torch.full((4, HC, DIM), 1.0 + SPLIT))  # the resolved stream
    assert not any(e[0] == "publish_source" for e in log)
    for e in runs:
        ctx = e[3]
        if e[1] < SPLIT:
            assert not ctx.published and ctx.floor is None
        else:
            assert ctx.published
            want = [1024, 1024, 1024, 0] if PER_ROW_FLOORS else [1024, 0]
            assert ctx.floor.cpu().tolist() == want


def test_source_stops_at_the_boundary_with_the_collapse():
    out, log = _run("source", starts=(0,), nctx=1)
    assert [e[1] for e in log if e[0] == "run"] == list(range(SPLIT))
    collapsed = torch.full((4, DIM), 1.0 + SPLIT)  # pre_mix (1, 0): the first copy
    (pub,) = [e for e in log if e[0].startswith("publish")]
    assert pub[0] == "publish_source" and pub[1] == SPLIT and torch.equal(pub[2], collapsed * 3.0)
    assert torch.equal(out, collapsed)


def test_without_a_mode_nothing_changes():
    out, log = _run(None)
    assert out[0] == "closed" and not any(e[0].startswith("publish") for e in log)
    assert all(not e[3].published and e[3].floor is None for e in log if e[0] == "run")

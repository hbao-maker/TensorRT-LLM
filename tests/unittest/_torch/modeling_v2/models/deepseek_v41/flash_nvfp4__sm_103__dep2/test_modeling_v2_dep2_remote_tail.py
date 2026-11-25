# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2 target as the context side of disaggregated remote-tail replay (`bounded_replay_on_generation`), as stock's
`DeepseekV41ForCausalLM` / `DeepseekV41Model` run it:

* a source step runs the encoder layers, publishes the boundary owner's (layer 20's) compressed rows from the boundary
  state collapsed with its pre-mix and normalized by the owner's input norm, and returns that collapse; no later layer
  runs (`_run_layers` around stub layers, host only);
* a context-only worker forces every batch to be a source and refuses a destination;
* its decoder weights, final norm and head are never built (`__pp_init__`) nor read (`weights.load`), the boundary keeps
  its input norm and GLOBAL KV / index-key projections, the cache follows the published boundary, and the sampler gets
  zero logits.
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling")
weights_mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.weights")

SPLIT, LAYERS, HC, DIM = 2, 4, 2, 8


class _StubLayer:
    def __init__(self, idx, log):
        self.idx, self.log = idx, log
        self.attn_norm = torch.full((DIM,), 3.0)
        self.attn = NS(publish_global=lambda h, ctx: log.append(("publish", idx, h.clone(), ctx)))

    def __call__(self, x, pre_mix, ctx, attn_metadata, engram_ids, counts, enc_counts, pending, defer=True, image=None):
        self.log.append(("run", self.idx, x.shape[0]))
        return x + 1, pre_mix, None


def _core(log):
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core.layers = [_StubLayer(i, log) for i in range(LAYERS)]
    core.decoder_replay_split, core.hc, core.norm_eps = SPLIT, HC, 1e-6
    core.engram_hash = NS(lookback=lambda *a: None)
    # the Engram look-back rows (`req_slot` / `hist` read them)
    core.engram_slots = NS(req_slot=torch.zeros(4, dtype=torch.int32), hist=None)
    core.embed_tokens = None
    return core


@pytest.fixture(autouse=True)
def _host_ops(monkeypatch):
    """Plain-torch stand-ins for the embedding, the collapse and the norm (their entries are tested on their own)."""
    monkeypatch.setattr(modeling, "token_embedding", lambda table, ids: torch.ones(ids.shape[0], DIM))
    monkeypatch.setattr(modeling, "hc_collapse", lambda x, pre: (x.float() * pre[..., None]).sum(1).to(x.dtype))
    monkeypatch.setattr(modeling, "flashinfer_rmsnorm", lambda h, w, eps: h * w)


def _run(n, mode, paged=False):
    log = []
    step = NS(positions=torch.arange(n), n=n, tok_req=torch.zeros(n, dtype=torch.int64))
    ctx = NS(paged=paged)
    md = NS(decoder_replay_plan=None, all_rank_num_tokens=[n, 3], csa2_remote_tail_mode=mode)
    out = _core(log)._run_layers(step, ctx, md, torch.zeros(n, dtype=torch.int32))
    return out, log, ctx


def test_source_stops_at_the_boundary_with_the_collapse():
    out, log, ctx = _run(5, "source")
    assert [e[:3] for e in log[:SPLIT]] == [("run", i, 5) for i in range(SPLIT)]
    (kind, idx, h, got_ctx), = log[SPLIT:]
    # the stream entering layer 2 is the embedding copies plus one per layer; pre_mix is (1, 0)
    collapsed = torch.full((5, DIM), 1.0 + SPLIT)
    assert (kind, idx, got_ctx) == ("publish", SPLIT, ctx)
    assert torch.equal(h, collapsed * 3.0)  # normalized by the owner's input norm (stand-in: scaled)
    assert torch.equal(out, collapsed)  # the collapse, not normalized, is the output


def test_a_generation_only_source_step_publishes_nothing():
    out, log, _ = _run(1, "source", paged=True)
    assert [e[0] for e in log] == ["run"] * SPLIT and out.shape == (1, DIM)


def test_no_source_runs_every_layer(monkeypatch):
    monkeypatch.setattr(modeling, "hc_pre", lambda x, pre: (x.float() * pre[..., None]).sum(1).to(x.dtype))
    monkeypatch.setattr(modeling.ModelingV2Core, "norm", torch.ones(DIM), raising=False)
    _, log, _ = _run(3, None)
    assert [e[:2] for e in log] == [("run", i) for i in range(LAYERS)]


def test_destination_is_refused():
    with pytest.raises(ValueError, match="no remote-tail destination"):
        _run(3, "destination")


def test_context_only_forward_forces_a_source():
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core.disagg_context_only, core._recovery_logged = True, False
    seen = []
    core._forward = lambda md, *a, **k: seen.append(md.csa2_remote_tail_mode) or "out"
    md = NS(csa2_remote_tail_mode=None, seq_lens=[4])
    assert core.forward(md, input_ids=torch.zeros(4, dtype=torch.int32)) == "out" and seen == ["source"]
    with pytest.raises(ValueError, match="cannot execute a remote-tail destination"):
        core.forward(NS(csa2_remote_tail_mode="destination", seq_lens=[4]), input_ids=torch.zeros(4))


def _param(*shape):
    return nn.Parameter(torch.zeros(*shape), requires_grad=False)


class _Attn(nn.Module):
    def __init__(self):
        super().__init__()
        for name in ("attn_sink", "q_norm", "kv_norm", "wq_a", "wq_a_s", "wq_b", "wq_b_s", "wkv", "wkv_s", "wo_a",
                     "wo_a_scale", "wo_b", "wo_b_s", "c_norm", "c_wkv", "i_wk", "i_knorm", "i_wq_b", "i_wq_b_s",
                     "i_wproj"):
            setattr(self, name, _param(2))
        self.ratio, self.is_owner = 1, True


class _FullLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = _Attn()
        for name in ("attn_norm", "ffn_norm", "hc_attn_fn", "hc_attn_base", "hc_attn_scale", "hc_ffn_fn",
                     "hc_ffn_base", "hc_ffn_scale"):
            setattr(self, name, _param(2))
        self.moe, self.shared = nn.Linear(2, 2), nn.Linear(2, 2)
        self.engram = None


def _shell(context_only):
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core.layers = nn.ModuleList([_FullLayer() for _ in range(LAYERS)])
    core.norm = _param(DIM)
    core.decoder_replay_split, core.disagg_context_only = SPLIT, context_only
    core.cfg = NS(vocab_size=11, num_hidden_layers=LAYERS)
    core.model_config = NS(extra_attrs={"csa2_context_swa_layer_limit": 99})
    shell = object.__new__(modeling.ModelingV2DeepseekV41FlashNvfp4Sm103Dep2)
    nn.Module.__init__(shell)
    shell.model, shell.model_config = core, NS(mapping=NS(has_pp=lambda: False))
    shell.lm_head = nn.Embedding(11, DIM)
    shell.logits_processor = "engine"
    return shell


def test_context_only_worker_keeps_the_prefix_and_the_boundary_projections():
    shell = _shell(True)
    shell.__pp_init__()
    core = shell.model
    names = {n for n, _ in shell.named_parameters()}
    prefix = {f"model.layers.{i}.{n}" for i in range(SPLIT) for n, _ in _FullLayer().named_parameters()}
    boundary = {f"model.layers.{SPLIT}.{n}" for n in ("attn_norm", "attn.c_norm", "attn.c_wkv", "attn.i_wk",
                                                      "attn.i_knorm")}
    assert names == prefix | boundary
    assert all(isinstance(core.layers[i], modeling._OmittedContextDecoder) for i in range(SPLIT + 1, LAYERS))
    assert core.norm is None and isinstance(shell.lm_head, modeling._OmittedContextDecoder)
    assert shell.lm_head.num_embeddings == 11 and shell.epilogue == [shell.lm_head]
    assert core.model_config.extra_attrs["csa2_context_swa_layer_limit"] == SPLIT
    md = NS(seq_lens_cuda=torch.tensor([3, 1]))
    logits = shell.logits_processor.forward(torch.ones(4, DIM), shell.lm_head, md)
    assert logits.shape == (2, 11) and logits.dtype == torch.float32 and not logits.any()
    with pytest.raises(ValueError, match="does not support context logits"):
        shell.logits_processor.forward(torch.ones(4, DIM), shell.lm_head, md, True)
    with pytest.raises(RuntimeError, match="unavailable on a context-only worker"):
        core.layers[LAYERS - 1]()


def test_a_full_worker_is_untouched():
    shell = _shell(False)
    before = {n for n, _ in shell.named_parameters()}
    shell.__pp_init__()
    assert {n for n, _ in shell.named_parameters()} == before
    assert "csa2_context_swa_layer_limit" not in shell.model.model_config.extra_attrs  # a stale boundary is dropped
    assert shell.logits_processor == "engine"


class _Slice:
    """A lazy safetensors slice as the engine's context-only loader hands it: materialized by indexing."""

    def __init__(self, t):
        self.t = t

    def get_shape(self):
        return list(self.t.shape)

    def __getitem__(self, idx):
        if idx == () and self.t.dim():
            raise IndexError("a non-scalar slice takes [:]")
        if idx != () and not self.t.dim():
            raise IndexError("slice() cannot be applied to a 0-dim tensor")
        return self.t[idx]


def test_lazy_slices_are_materialized():
    view = weights_mod._Materialized({"a": _Slice(torch.ones(2, 3)), "s": _Slice(torch.tensor(5.0)),
                                      "layers.1.engram.embed.weight": _Slice(torch.ones(4))})
    assert torch.equal(view["a"], torch.ones(2, 3)) and float(view["s"]) == 5.0
    assert isinstance(view["layers.1.engram.embed.weight"], _Slice)  # the table reads only its row range


class _Recording(dict):
    def __init__(self, *a):
        super().__init__(*a)
        self.read, self.consumed = set(), []

    def __getitem__(self, key):
        self.read.add(key)
        return super().__getitem__(key)

    def mark_consumed(self, prefix):
        self.consumed.append(prefix)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the loader places weights on the device")
def test_context_only_weights_read_only_the_retained_tensors():
    """With the boundary at layer 0 (so no full layer has to be built): the embedding, layer 0's input norm and GLOBAL
    projections are read; the boundary's other tensors, the later layer, the final norm and the head are not."""
    dev = torch.device("cuda")
    attn = nn.Module()
    for name, shape in (("c_norm", (4,)), ("c_wkv", (4, DIM)), ("i_wk", (2, 4)), ("i_knorm", (2,))):
        setattr(attn, name, nn.Parameter(torch.zeros(*shape, dtype=torch.bfloat16, device=dev), requires_grad=False))
    attn.ratio = 1
    boundary = NS(attn=attn, attn_norm=nn.Parameter(torch.zeros(DIM, dtype=torch.bfloat16, device=dev)))
    core = NS(cfg=None, decoder_replay_split=0, disagg_context_only=True, embed_tokens=nn.Embedding(3, DIM).to(dev),
              norm=None, layers=[boundary, modeling._OmittedContextDecoder()])
    full = lambda *s: torch.ones(*s, dtype=torch.bfloat16)  # noqa: E731
    p = "layers.0."
    keep = {"embed.weight": full(3, DIM), p + "attn_norm.weight": full(DIM), p + "attn.compressor.norm.weight": full(4),
            p + "attn.compressor.wkv.weight": _Slice(full(4, DIM)), p + "attn.indexer.wk.weight": full(2, 4),
            p + "attn.indexer.k_norm.weight": full(2)}
    decoys = {"norm.weight": full(DIM), "head.weight": full(3, DIM), p + "attn.wq_a.weight": full(2),
              p + "ffn.gate.weight": full(2), p + "ffn_norm.weight": full(DIM), "layers.1.attn_norm.weight": full(DIM)}
    weights = _Recording({**keep, **decoys})
    weights_mod.load(core, weights)
    assert weights.read == set(keep) and weights.consumed == ["layers.0", "layers.1"]
    assert bool((boundary.attn_norm == 1).all()) and bool((attn.c_wkv == 1).all()) and bool((attn.i_knorm == 1).all())

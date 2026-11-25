# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight routing: deepseek-v4.1-flash-nvfp4 / sm_103 / dep2.

Every rank is handed the whole checkpoint dict; this file says which checkpoint tensor fills which
parameter and what is done to it on the way in.

* attention, norms, HC parameters, router, embedding, head: copied (the head sliced per TP rank);
* shared experts: block-32 FP8 with UE8M0 scales, requantized to 128x128 FP8 tiles (stock's requantization);
* engram projection: block-32 FP8 with UE8M0 scales, dequantized to bf16 (and kept as E4M3 for the packed path);
* routed experts: the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,weight_scale_2,input_scale}` (NVFP4, group 16),
  this rank's experts, in the CuTe-DSL grouped GEMMs' layout (`cute_dsl_moe_layout`);
* CSA2 attention: the module reads HF-style names, so block scales are renamed
  `<proj>.scale` -> `<proj>.weight_scale_inv`, and the compressor's value and gate projections are
  fused (ratio 2: `[wkv; wgate]`; ratio 1 has no gate and keeps `wkv`);
* engram tables: FP8 bytes plus UE8M0 scales, this rank's row range only (see `_Engram.load`).

A context-only worker (`bounded_replay_on_generation` on the context role, see `modeling.py`) reads no tensor of the
omitted decoder layers, of the boundary layer beyond its input norm and GLOBAL KV / index-key projections, nor the final
norm and the head: the engine's lazy safetensors loader then never touches their bytes.

The checkpoint also carries a vision tower and the `mtp.*` DSpark stages; neither is read here (the
draft stages are loaded by the engine's draft-model loader, the vision tower by the family's image model,
`vision/image_model.py`). Serving images, the router's vision bias `ffn.gate.bias_vl` is loaded too.
"""

import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_moe_nvfp4 import cute_dsl_moe_layout
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import requantize_block128
from tensorrt_llm.logger import logger

_BLOCK = 32  # FP8 block edge of the checkpoint's dense weights

SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K
_TILE_K = 128  # the MoE runner's intermediate width must be a multiple of this


def _pad_rows(t: torch.Tensor, rows: int) -> torch.Tensor:
    return t if t.shape[0] == rows else torch.cat([t, t.new_zeros(rows - t.shape[0], t.shape[1])])


def _pad_cols(t: torch.Tensor, cols: int) -> torch.Tensor:
    return t if t.shape[1] == cols else torch.cat([t, t.new_zeros(t.shape[0], cols - t.shape[1])], 1)


def padded_inter(inter: int, tp: int) -> int:
    """This rank's intermediate width, rounded up to the runner's tile (the pad is zero weights and
    zero scales, so it contributes exactly nothing)."""
    per = inter // tp
    assert per * tp == inter, (inter, tp)
    return -(-per // _TILE_K) * _TILE_K


def load_experts(moe, weights: dict, prefix: str, device) -> None:
    """Fill one layer's `_Moe` from the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,...}`.

    w1 = gate, w3 = up, w2 = down. This rank keeps experts `[offset, offset + local)` and the
    `tp_rank`-th `1/tp` slice of the intermediate width, in the CuTe-DSL grouped GEMMs' layout.
    """
    cfg, inter_t = moe.cfg, moe.inter_t
    e_total, tp, tr = cfg.n_routed_experts, moe.tp, moe.tp_rank
    per = cfg.moe_intermediate_size // tp
    lo, hi = tr * per, (tr + 1) * per

    def scalar(e: int, w: str, name: str) -> torch.Tensor:
        return weights[f"{prefix}{e}.{w}.{name}"].reshape(()).float().to(device)

    isc1 = torch.stack([torch.maximum(scalar(e, "w1", "input_scale"), scalar(e, "w3", "input_scale"))
                        for e in range(e_total)])
    isc2 = torch.stack([scalar(e, "w2", "input_scale") for e in range(e_total)])
    ws1 = torch.stack([scalar(e, "w1", "weight_scale_2") for e in range(e_total)])
    ws3 = torch.stack([scalar(e, "w3", "weight_scale_2") for e in range(e_total)])
    ws2 = torch.stack([scalar(e, "w2", "weight_scale_2") for e in range(e_total)])
    i1 = isc1.max()  # one activation scale for every expert: the runner quantizes the batch once
    win = slice(moe.offset, moe.offset + moe.local)
    gate_alpha = (i1 * ws1[win]).contiguous()
    moe.o1_gate.copy_(gate_alpha)
    moe.g1.copy_((1.0 / i1).reshape(1))
    # stock's CuteDSL scalars: one FC2 input scale for every expert (the largest), the limit in real units
    moe.fc2_global.copy_((1.0 / isc2.max()).reshape(1))
    moe.fc2_alpha.copy_(isc2.max() * ws2[win])

    mism = 0
    for local in range(moe.local):
        e = moe.offset + local
        q = f"{prefix}{e}."
        up_w = weights[q + "w3.weight"].to(device).view(torch.uint8)[lo:hi]
        gate_w = weights[q + "w1.weight"].to(device).view(torch.uint8)[lo:hi]
        up_s = weights[q + "w3.weight_scale"].to(device)[lo:hi]
        gate_s = weights[q + "w1.weight_scale"].to(device)[lo:hi]
        if not torch.equal(ws1[e], ws3[e]):
            # One FC1 weight scalar serves both halves: fold the difference into the up scales.
            up_s = (up_s.float() * (ws3[e] / ws1[e])).to(torch.float8_e4m3fn)
            mism += 1
        down_w = weights[q + "w2.weight"].to(device).view(torch.uint8)[:, lo // 2 : hi // 2]
        down_s = weights[q + "w2.weight_scale"].to(device)[:, lo // SF_BLOCK : hi // SF_BLOCK]
        # pad the intermediate width to the tile
        up_w, gate_w = _pad_rows(up_w, inter_t), _pad_rows(gate_w, inter_t)
        up_s = _pad_rows(up_s.view(torch.uint8), inter_t)
        gate_s = _pad_rows(gate_s.view(torch.uint8), inter_t)
        down_w = _pad_cols(down_w, inter_t // 2)
        down_s = _pad_cols(down_s.view(torch.uint8), inter_t // SF_BLOCK)

        f1w, f1s, f2w, f2s = cute_dsl_moe_layout(up_w, up_s, gate_w, gate_s, down_w, down_s)
        moe.fc1_w[local].copy_(f1w)
        moe.fc1_s[local].copy_(f1s.view(torch.float8_e4m3fn))
        moe.fc2_w[local].copy_(f2w)
        moe.fc2_s[local].copy_(f2s.view(torch.float8_e4m3fn))
    if mism:
        logger.info(f"modeling_v2 deepseek_v41: {prefix}: folded w3/w1 weight_scale_2 into {mism} experts")



def _dequant_block32(w: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    """FP8 [R, C] weight with a [ceil(R/32), ceil(C/32)] UE8M0 scale -> bf16 [R, C]."""
    r, c = w.shape
    sf = s.float().repeat_interleave(_BLOCK, 0)[:r].repeat_interleave(_BLOCK, 1)[:, :c]
    return (w.float() * sf).to(torch.bfloat16)




def _release(weights, prefix: str) -> None:
    """Drop the checkpoint tensors under `prefix` from the engine's loading dict (`ConsumableWeightsDict.mark_consumed`)
    once their layer is loaded, as stock's module loading does. The loader hands each rank host copies of the tensors
    it reads, and the dict keeps them until the whole load ends; released layer by layer, a rank's load-time host
    peak is one layer's tensors instead of every layer's (the experts dominate: a dep2 c7000 ctx boot peaked at
    ~300 GiB RSS per rank against stock's ~150, and two dep2 servers share a node). A plain dict (tests) is left
    alone."""
    consume = getattr(weights, "mark_consumed", None)
    if consume is not None:
        consume(prefix)


class _Materialized:
    """A context-only worker's checkpoint dict: the engine hands it lazy safetensors slices (`HfCheckpointLoader`'s
    context-only lazy loading, so omitted tensors are never read); a read materializes the whole tensor (`[()]` for a
    scalar, which a slice cannot take), as the eager dict would hold it. The engram tables stay lazy: their loader reads
    only this rank's row range."""

    def __init__(self, weights):
        self.weights = weights

    def __getitem__(self, key):
        v = self.weights[key]
        if isinstance(v, torch.Tensor) or ".engram.embed." in key:
            return v
        return v[()] if len(v.get_shape()) == 0 else v[:]

    def __contains__(self, key):
        return key in self.weights

    def mark_consumed(self, prefix):
        _release(self.weights, prefix)


def load(core, weights: dict) -> None:
    """Fill the trunk from the checkpoint dict (every rank is handed the whole of it)."""
    dev = torch.device("cuda")
    cfg = core.cfg
    split = core.decoder_replay_split if getattr(core, "disagg_context_only", False) else None
    if split is not None:
        weights = _Materialized(weights)
    core.embed_tokens.weight.data.copy_(weights["embed.weight"])
    if split is None:
        core.norm.data.copy_(weights["norm.weight"])
    for i, layer in enumerate(core.layers):
        p = f"layers.{i}."
        if split is not None and i >= split:  # context-only: the boundary's GLOBAL projections, nothing after it
            if i == split:
                layer.attn_norm.data.copy_(weights[p + "attn_norm.weight"])
                load_global_projections(layer.attn, weights, p + "attn.", dev)
            _release(weights, f"layers.{i}")
            continue
        for name in ("attn_norm", "ffn_norm"):
            getattr(layer, name).data.copy_(weights[p + name + ".weight"])
        for n in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                getattr(layer, f"hc_{n}_{part}").data.copy_(weights[p + f"hc_{n}_{part}"])
        layer.moe.gate_weight.data.copy_(weights[p + "ffn.gate.weight"])
        layer.moe.gate_bias.data.copy_(weights[p + "ffn.gate.bias"])
        if layer.moe.gate_bias_vl is not None:  # serving images: the image rows' selection bias (stock's `bias_vl`)
            layer.moe.gate_bias_vl.data.copy_(weights[p + "ffn.gate.bias_vl"])
        # shared expert: block-32 FP8 -> 128x128 FP8 tiles (stock's requantization)
        sh = {
            w: _dequant_block32(
                weights[p + f"ffn.shared_experts.{w}.weight"].to(dev),
                weights[p + f"ffn.shared_experts.{w}.scale"].to(dev),
            )
            for w in ("w1", "w2", "w3")
        }
        for (wq, sq), src in ((("w13", "s13"), torch.cat([sh["w1"], sh["w3"]], 0)), (("w2", "s2"), sh["w2"])):
            q, sc = requantize_block128(src)
            getattr(layer.shared, wq).data.copy_(q)
            getattr(layer.shared, sq).data.copy_(sc)
        load_experts(layer.moe, weights, p + "ffn.experts.", dev)
        load_attention(layer.attn, weights, p + "attn.", dev)
        if layer.engram is not None:
            layer.engram.load(weights, p + "engram.", dev, f"{core.checkpoint_dir}:{p}engram")
        _release(weights, f"layers.{i}")
        if i % 8 == 0:
            logger.info(f"modeling_v2 deepseek_v41 dep2: loaded layer {i}")


from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E402
    mxfp8_row_scales,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (  # noqa: E402
    requantize_block128 as _requantize_block128,
)


def load_mxfp8(a, name: str, weights: dict, key: str, device, rows=None, cols=None) -> None:
    """Copy the E4M3 weight `key` and its per-row 1x32 UE8M0 scales into `a.<name>` / `a.<name>_s`.

    `rows` / `cols` are this rank's `(first, end)` slice of the weight (tensor parallelism). The weight bytes are the
    checkpoint's; only the scale metadata is expanded from 32x32 blocks to rows (exactly).
    """
    w = weights[key].to(device)
    assert w.dtype == torch.float8_e4m3fn, f"{key}: the attention projections are E4M3 in the checkpoint"
    scale = mxfp8_row_scales(weights[key[: -len("weight")] + "scale"].to(device), tuple(w.shape))
    if rows is not None:
        w, scale = w[rows[0] : rows[1]], scale[rows[0] : rows[1]]
    if cols is not None:
        w, scale = w[:, cols[0] : cols[1]], scale[:, cols[0] // 32 : cols[1] // 32]
    getattr(a, name).data.copy_(w)
    getattr(a, name + "_s").data.copy_(scale)


def load_o_a(a, weights: dict, key: str, device, rows) -> None:
    """The grouped `wo_a` rows `rows = (first, end)` of this rank as E4M3 with 128x128 power-of-two block scales.

    The checkpoint stores it with 32x32 UE8M0 scales; the engine's O-A kernels read 128x128 blocks, so (as the engine's
    loader does) the bf16 values are requantized to power-of-two block multipliers.
    """
    w = weights[key].to(device)
    assert w.dtype == torch.float8_e4m3fn
    s = weights[key[: -len("weight")] + "scale"].to(device).float().repeat_interleave(32, 0)[: w.shape[0]]
    dense = (w.float() * s.repeat_interleave(32, 1)[:, : w.shape[1]])[rows[0] : rows[1]].to(torch.bfloat16)
    q, scale = _requantize_block128(dense)
    a.wo_a.data.copy_(q)
    a.wo_a_scale.data.copy_(scale)


def _dense(weights: dict, key: str, device) -> torch.Tensor:
    """A dense checkpoint weight as bf16: block-32 FP8 with a `.scale` sibling is dequantized."""
    w = weights[key].to(device)
    if w.dtype == torch.float8_e4m3fn:
        return _dequant_block32(w, weights[key[: -len("weight")] + "scale"].to(device))
    return w.to(torch.bfloat16)


def load_attention(a, weights: dict, p: str, device) -> None:
    """Fill one layer's `_Attention` from `layers.N.attn.*`; under TP this rank takes its heads and groups."""
    t = a.tp_rank
    h0, h1 = t * a.hl, (t + 1) * a.hl
    g0, g1 = t * a.gl, (t + 1) * a.gl
    cp = lambda dst, src: dst.data.copy_(src)  # noqa: E731
    cp(a.attn_sink, weights[p + "attn_sink"].to(device).float()[h0:h1])
    cp(a.q_norm, weights[p + "q_norm.weight"].to(device))
    cp(a.kv_norm, weights[p + "kv_norm.weight"].to(device))
    load_mxfp8(a, "wq_a", weights, p + "wq_a.weight", device)
    load_mxfp8(a, "wq_b", weights, p + "wq_b.weight", device, rows=(h0 * a.hd, h1 * a.hd))
    load_mxfp8(a, "wkv", weights, p + "wkv.weight", device)
    load_o_a(a, weights, p + "wo_a.weight", device, (g0 * a.o_rank, g1 * a.o_rank))
    load_mxfp8(a, "wo_b", weights, p + "wo_b.weight", device, cols=(g0 * a.o_rank, g1 * a.o_rank))
    if a.is_owner:
        load_global_projections(a, weights, p, device)
    if a.is_index_source:
        load_mxfp8(a, "i_wq_b", weights, p + "indexer.wq_b.weight", device)
        cp(a.i_wproj, weights[p + "indexer.weights_proj.weight"].to(device))


def load_global_projections(a, weights: dict, p: str, device) -> None:
    """A KV owner's GLOBAL KV (compressor) and index-key projections from `layers.N.attn.*`."""
    cp = lambda dst, src: dst.data.copy_(src)  # noqa: E731
    cp(a.c_norm, weights[p + "compressor.norm.weight"].to(device))
    cp(a.c_wkv, weights[p + "compressor.wkv.weight"].to(device).to(a.c_wkv.dtype))
    if a.ratio == 2:
        cp(a.c_wgate, weights[p + "compressor.wgate.weight"].to(device).float())
    cp(a.i_wk, weights[p + "indexer.wk.weight"].to(device))
    cp(a.i_knorm, weights[p + "indexer.k_norm.weight"].to(device))


def load_head(shell, weights: dict) -> None:
    """The LM head: column-parallel under TP, so this rank takes its slice of the vocabulary rows."""
    head = weights["head.weight"]
    rows = shell.lm_head.weight.shape[0]
    if rows != head.shape[0]:
        start = shell.model.mapping.tp_rank * rows
        head = head[start : start + rows]
    shell.lm_head.weight.data.copy_(head)

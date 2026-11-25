# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight routing: deepseek-v4.1-flash-nvfp4 / sm_103 / default, attention-DP path.

Every rank is handed the whole checkpoint dict; this file says which checkpoint tensor fills which
parameter and what is done to it on the way in.

* attention, norms, HC parameters, router, embedding, head: copied (the head sliced per TP rank);
* shared experts: as stock, block-32 FP8 dequantized and requantized to 128x128 power-of-two tiles, the tiles' UE8M0
  exponents expanded to one per row per 32 (gate and up rows fused); into the MegaMoE kernel's layout instead when
  `megamoe_fuse_shared_expert` is on;
* the engram projection: block-32 FP8 with UE8M0 scales;
* routed experts: handed to the engine's MoE loader in its `{e}.{w1,w2,w3}.{weight,weight_scale,
  weight_scale_2,input_scale}` layout (NVFP4, group 16), which takes this rank's experts itself; with
  `moe_config.backend: MEGAMOE_CUTEDSL` they are packed per local slot instead (`load_megamoe_experts`);
* CSA2 attention: the module reads HF-style names, so block scales are renamed
  `<proj>.scale` -> `<proj>.weight_scale_inv`, and the compressor's value and gate projections are
  fused (ratio 2: `[wkv; wgate]`; ratio 1 has no gate and keeps `wkv`);
* engram tables: FP8 bytes plus UE8M0 scales, this rank's row range only (see `_Engram.load`).

The checkpoint also carries a vision tower and the `mtp.*` DSpark stages; neither is read here (the
draft stages are loaded by the engine's draft-model loader).
"""

import torch

from tensorrt_llm.logger import logger

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_pack_experts import (
    megamoe_expert_scalars,
    megamoe_pack_experts,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.megamoe_shared_expert import (
    megamoe_pack_shared_expert,
)

_BLOCK = 32  # FP8 block edge of the checkpoint's dense weights
_REQUANT_BLOCK = 128  # the engine's requantized tile edge (shared expert, grouped wo_a)

SF_BLOCK = 16  # NVFP4: one e4m3 scale per 16 elements along K
_TILE_K = 128  # the MoE runner's intermediate width must be a multiple of this


def _block32_perm(rows: int, device) -> torch.Tensor:
    """Gather index of the 32-row block shuffle: inside each aligned block of 32 rows, source
    row `4u + v` (0<=u<8, 0<=v<4) lands at destination row `8v + u`."""
    assert rows % 32 == 0, rows
    src = torch.arange(32)
    dst = (src % 4) * 8 + src // 4
    idx = torch.empty(32, dtype=torch.long)
    idx[dst] = src
    blocks = rows // 32
    return (idx.repeat(blocks) + torch.arange(blocks).repeat_interleave(32) * 32).to(device)


def _interleave_perm(rows: int, device) -> torch.Tensor:
    """Gather index interleaving the `[up | gate]` halves of a `2*I` row stack into up0, gate0, up1, ..."""
    p = torch.empty(rows, dtype=torch.long)
    p[0::2] = torch.arange(0, rows // 2)
    p[1::2] = torch.arange(rows // 2, rows)
    return p.to(device)


def _fc1_perm(rows: int, device) -> torch.Tensor:
    return _interleave_perm(rows, device)[_block32_perm(rows, device)]


def _swizzle(x: torch.Tensor) -> torch.Tensor:
    """128x4 block-scale swizzle of a `[M, C]` uint8 scale matrix (`M % 128 == 0`, `C % 4 == 0`)."""
    return torch.ops.trtllm.block_scale_interleave(x.contiguous())


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
    `tp_rank`-th `1/tp` slice of the intermediate width, shuffled and swizzled for the runner.
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
    moe.o1.copy_(gate_alpha / isc2[win])
    moe.o2.copy_(isc2[win] * ws2[win])
    moe.g1.copy_((1.0 / i1).reshape(1))
    # The runner clamps in accumulator units (before the FC1 scalar is applied), so the real-valued
    # swiglu limit is divided by each expert's FC1 gate scalar. Measured against a torch reference.
    moe.clamp.copy_(cfg.swiglu_limit / gate_alpha)

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

        w = torch.cat([up_w, gate_w], 0)
        moe.fc1_w[local].copy_(torch.index_select(w, 0, _fc1_perm(w.shape[0], device)))
        s = torch.index_select(torch.cat([up_s, gate_s], 0), 0, _fc1_perm(2 * inter_t, device))
        moe.fc1_s[local].copy_(_swizzle(s).reshape(s.shape).view(torch.float8_e4m3fn))
        moe.fc2_w[local].copy_(torch.index_select(down_w, 0, _block32_perm(down_w.shape[0], device)))
        d = torch.index_select(down_s, 0, _block32_perm(down_s.shape[0], device))
        moe.fc2_s[local].copy_(_swizzle(d).reshape(d.shape).view(torch.float8_e4m3fn))
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
    peak is one layer's tensors instead of every layer's (the experts dominate: dep8 c7000 measured +44 GiB anonymous
    memory per rank at the end of the load, 4 ranks per node, on top of the pinned engram tables and the checkpoint
    read-ahead). A plain dict (tests) is left alone."""
    consume = getattr(weights, "mark_consumed", None)
    if consume is not None:
        consume(prefix)


def load(core, weights: dict) -> None:
    """Fill the trunk from the checkpoint dict (every rank is handed the whole of it)."""
    dev = torch.device("cuda")
    core.embed_tokens.weight.data.copy_(weights["embed.weight"])
    core.norm.data.copy_(weights["norm.weight"])
    for i, layer in enumerate(core.layers):
        p = f"layers.{i}."
        for name in ("attn_norm", "ffn_norm"):
            getattr(layer, name).data.copy_(weights[p + name + ".weight"])
        for n in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                getattr(layer, f"hc_{n}_{part}").data.copy_(weights[p + f"hc_{n}_{part}"])
        layer.moe.gate_weight.data.copy_(weights[p + "ffn.gate.weight"])
        layer.moe.gate_bias.data.copy_(weights[p + "ffn.gate.bias"])
        if layer.shared is not None:  # shared expert: stock's requantized FP8 (absent when fused into MegaMoE)
            load_shared_expert(layer.shared, weights, p + "ffn.shared_experts.", dev)
        if core.mega_cfg is None:
            load_experts(layer.moe, weights, p + "ffn.experts.", dev)
        else:
            load_megamoe_experts(layer.moe, weights, p + "ffn.experts.", dev)
            if core.mega_cfg.fused_shared:
                load_megamoe_shared(layer.moe, weights, p + "ffn.", dev)
        load_attention(layer.attn, weights, p + "attn.", dev)
        if layer.engram is not None:
            layer.engram.load(weights, p + "engram.", dev, f"{core.checkpoint_dir}:{p}engram")
        _release(weights, f"layers.{i}")
        if i % 8 == 0:
            logger.info(f"modeling_v2 deepseek_v41 default (attention DP): loaded layer {i}")


from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_mxfp8_linear import (  # noqa: E402
    mxfp8_row_scales,
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


from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_o_projection import (  # noqa: E402
    requantize_block128 as _requantize_block128,
)


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
        cp(a.c_norm, weights[p + "compressor.norm.weight"].to(device))
        cp(a.c_wkv, weights[p + "compressor.wkv.weight"].to(device).to(a.c_wkv.dtype))
        if a.ratio == 2:
            cp(a.c_wgate, weights[p + "compressor.wgate.weight"].to(device).float())
        cp(a.i_wk, weights[p + "indexer.wk.weight"].to(device))
        cp(a.i_knorm, weights[p + "indexer.k_norm.weight"].to(device))
    if a.is_index_source:
        load_mxfp8(a, "i_wq_b", weights, p + "indexer.wq_b.weight", device)
        cp(a.i_wproj, weights[p + "indexer.weights_proj.weight"].to(device))


def load_head(shell, weights: dict) -> None:
    """The LM head: column-parallel under TP, so this rank takes its slice of the vocabulary rows."""
    head = weights["head.weight"]
    rows = shell.lm_head.weight.shape[0]
    if rows != head.shape[0]:
        start = shell.model.mapping.tp_rank * rows
        head = head[start : start + rows]
    shell.lm_head.weight.data.copy_(head)


# ---- MegaMoE (`moe_config.backend: MEGAMOE_CUTEDSL`): the routed experts and the shared expert in the kernel's layout ----


def load_megamoe_experts(moe, weights: dict, prefix: str, device) -> None:
    """Fill one layer's `_MegaMoe` from the checkpoint's `{e}.{w1,w2,w3}.{weight,weight_scale,...}`.

    This rank's slots hold the experts `moe.assignments[rank * S : (rank + 1) * S]` (a copy slot reads the same
    checkpoint bytes as its twin). `w1` = gate, `w3` = up, `w2` = down are stacked per slot and handed to
    `megamoe_pack_experts`; the scalars follow `megamoe_expert_scalars`: ONE FC1 activation scale for the layer,
    the maximum of every expert's `w1` and `w3` `input_scale` (the activation is quantized once, before dispatch),
    and the per-slot ones of the expert the slot holds. A `weight_scale_2` that differs between `w1` and `w3` is
    folded into the up block scales (the kernel has one FC1 alpha per slot).
    """
    mc = moe.mc
    slots = mc.local_slots
    mine = moe.assignments[mc.ep_rank * slots : (mc.ep_rank + 1) * slots]
    hidden, inter = mc.hidden, mc.inter

    def scalar(e: int, w: str, name: str) -> torch.Tensor:
        return weights[f"{prefix}{e}.{w}.{name}"].reshape(()).float().to(device)

    isc1 = torch.stack(
        [
            torch.maximum(scalar(e, "w1", "input_scale"), scalar(e, "w3", "input_scale"))
            for e in range(mc.num_experts)
        ]
    )

    def per_slot(w: str, name: str) -> torch.Tensor:
        return torch.stack([scalar(e, w, name) for e in mine])

    scalars = megamoe_expert_scalars(
        isc1.max(),
        per_slot("w2", "input_scale"),
        per_slot("w1", "weight_scale_2"),
        per_slot("w3", "weight_scale_2"),
        per_slot("w2", "weight_scale_2"),
    )
    u8 = torch.uint8
    w1 = torch.empty(slots, inter, hidden // 2, dtype=u8, device=device)
    w3 = torch.empty_like(w1)
    w2 = torch.empty(slots, hidden, inter // 2, dtype=u8, device=device)
    w1_sf = torch.empty(slots, inter, hidden // SF_BLOCK, dtype=u8, device=device)
    w3_sf = torch.empty_like(w1_sf)
    w2_sf = torch.empty(slots, hidden, inter // SF_BLOCK, dtype=u8, device=device)
    for s, e in enumerate(mine):
        q = f"{prefix}{e}."
        for dst, key in (
            (w1, "w1.weight"), (w3, "w3.weight"), (w2, "w2.weight"),
            (w1_sf, "w1.weight_scale"), (w3_sf, "w3.weight_scale"), (w2_sf, "w2.weight_scale"),
        ):  # fmt: skip
            dst[s].copy_(weights[q + key].view(u8))
    fc1, fc1_sf, fc2, fc2_sf = megamoe_pack_experts(w1, w1_sf, w3, w3_sf, w2, w2_sf, scalars["w3_scale_fold"])
    moe.set_experts(fc1, fc1_sf, fc2, fc2_sf, scalars, device)


def load_megamoe_shared(moe, weights: dict, prefix: str, device) -> None:
    """The shared expert (`layers.N.ffn.shared_experts.{w1,w3,w2}.{weight,scale}`: block-32 FP8, UE8M0 scales) as the
    kernel's extra tiles. As the engine's loader does for these stems (`_V41_REQUANT_STEM_TAILS`), each projection is
    first requantized to 128x128 power-of-two tiles (block-32 dequantized to bf16, then `requantize_block128`), so the
    kernel reads stock's FP8 bytes and scales (`build_shared_expert_weights` of the requantized projections)."""
    got = {}
    for w in ("w1", "w3", "w2"):
        scale = weights[f"{prefix}shared_experts.{w}.scale"].to(device)
        if scale.element_size() == 1:  # UE8M0 in any 1-byte dtype: the byte `e` means 2**(e-127)
            scale = torch.exp2(scale.view(torch.uint8).float() - 127.0)
        dense = _dequant_block32(weights[f"{prefix}shared_experts.{w}.weight"].to(device), scale)
        q, scale = _requantize_block128(dense)
        got[w] = (q, ((scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8))  # UE8M0 exponent bytes
    pack = megamoe_pack_shared_expert(
        got["w1"][0], got["w1"][1], got["w3"][0], got["w3"][1], got["w2"][0], got["w2"][1],
        moe.cfg.swiglu_limit, (_REQUANT_BLOCK, _REQUANT_BLOCK),
    )  # fmt: skip
    moe.set_shared(pack)


def load_shared_expert(se, weights: dict, prefix: str, device) -> None:
    """`se.w13` / `se.s13` = w1 rows then w3 rows, `se.w2` / `se.s2`, with stock's bytes: each projection's checkpoint
    32x32 blocks dequantized to bf16 and requantized to 128x128 power-of-two tiles (stock's `_requantize_dense_stem`),
    E4M3 with the tiles' FP32 multipliers (stock's `weight_scale_block128`). The whole width: under attention DP stock
    runs the shared expert replicated (its shared TP size is 1)."""
    inter, t = se.part, _REQUANT_BLOCK
    for half, w in enumerate(("w1", "w3", "w2")):
        src = weights[prefix + f"{w}.weight"].to(device)
        assert src.dtype == torch.float8_e4m3fn, f"{prefix}{w}: the shared expert is E4M3 in the checkpoint"
        # dequantized on the GPU, requantized on the host (no load-time device temporaries left in the cache)
        q, tiles = _requantize_block128(_dequant_block32(src, weights[prefix + f"{w}.scale"].to(device)).cpu())
        q, tiles = q.to(device), tiles.to(device)
        if w == "w2":
            se.w2.data.copy_(q)
            se.s2.data.copy_(tiles)
        else:
            se.w13.data[half * inter : (half + 1) * inter].copy_(q)
            se.s13.data[half * inter // t : (half + 1) * inter // t].copy_(tiles)

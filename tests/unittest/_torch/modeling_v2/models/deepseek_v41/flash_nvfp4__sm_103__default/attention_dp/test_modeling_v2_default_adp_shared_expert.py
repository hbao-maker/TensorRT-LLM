# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP path's shared expert, as stock runs it under attention DP (replicated, its `GatedMLP` on FP8 block
scales): `weights.load_shared_expert` gives it stock's bytes (the checkpoint's block-32 E4M3 dequantized and requantized
to 128x128 power-of-two tiles with their FP32 multipliers), and its forward is `shared_expert_fp8_deep_gemm` of exactly
those weights (stock's small-M / DeepGEMM choice per row count). Checked bit for bit against the family op on the same
weights, and against an fp32 reference over the requantized weights at decode and context row counts (the
band of two fp8 1x128 activation roundings: 5%)."""

import importlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.shared_expert_fp8 import (
    shared_expert_fp8_deep_gemm,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)
_PATH = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__default.attention_dp"
DEV = "cuda"
H, I, LIMIT = 5120, 2304, 10.0


def _checkpoint(gen):
    """Block-32 E4M3 projections with UE8M0 (power-of-two) block scales, as the checkpoint stores the shared expert."""
    out = {}
    for w, shape in (("w1", (I, H)), ("w3", (I, H)), ("w2", (H, I))):
        q = (torch.randn(shape, generator=gen, device=DEV) * 64).clamp(-448, 448).to(torch.float8_e4m3fn)
        s = torch.pow(2.0, torch.randint(-14, -10, (shape[0] // 32, shape[1] // 32), generator=gen, device=DEV).float())
        out[f"p.{w}.weight"], out[f"p.{w}.scale"] = q, s.to(torch.float8_e8m0fnu)
    return out


@pytest.mark.parametrize("rows", [4, 64, 512])
def test_shared_expert_is_the_ffn_of_stocks_requantized_weights(rows):
    core = importlib.import_module(f"{_PATH}.core")
    wts = importlib.import_module(f"{_PATH}.weights")
    gen = torch.Generator(device=DEV).manual_seed(rows)
    ckpt = _checkpoint(gen)
    se = core._SharedExpert(H, I, LIMIT).to(DEV)
    wts.load_shared_expert(se, ckpt, "p.", DEV)
    se.post_load()

    # the loaded bytes are exactly stock's requantization of the checkpoint (`requantize_block128` of the block-32
    # dequantized projection: E4M3 and the 128x128 tiles' FP32 multipliers)
    for half, w in enumerate(("w1", "w3")):
        q, tiles = wts._requantize_block128(wts._dequant_block32(ckpt[f"p.{w}.weight"], ckpt[f"p.{w}.scale"]).cpu())
        assert torch.equal(se.w13[half * I : (half + 1) * I].view(torch.uint8).cpu(), q.view(torch.uint8))
        assert torch.equal(se.s13[half * I // 128 : (half + 1) * I // 128].cpu(), tiles)
    q, tiles = wts._requantize_block128(wts._dequant_block32(ckpt["p.w2.weight"], ckpt["p.w2.scale"]).cpu())
    assert torch.equal(se.w2.view(torch.uint8).cpu(), q.view(torch.uint8)) and torch.equal(se.s2.cpu(), tiles)
    x = (torch.randn(rows, H, generator=gen, device=DEV) * 0.5).to(torch.bfloat16)
    out = se(x)
    assert out.shape == (rows, H)
    # the family op (stock's GatedMLP path) on the same weights, bit for bit
    want = shared_expert_fp8_deep_gemm(x, se.w13, se.sf13, se.s13, se.w2, se.sf2, se.s2, LIMIT)
    assert torch.equal(out, want)
    ref = shared_expert_fp8_deep_gemm.reference(x, se.w13, se.sf13, se.s13, se.w2, se.sf2, se.s2, LIMIT).float()
    # two fp8 1x128 activation roundings (x before w13, the intermediate before w2) against exact activations
    assert ((out.float() - ref).norm() / ref.norm()).item() < 5e-2
    assert se(x[:0]).shape == (0, H)

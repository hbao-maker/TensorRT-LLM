# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's image encoder against stock's: the same weights give the same image rows, bit for bit.

Stock's modules (`DeepseekV41VisionModel`, `DeepseekV41Aligner`) and its span assembly
(`DeepseekV41ForConditionalGeneration.encode_multimodal_inputs`) are the reference here, at the checkpoint's widths
(vision 1024 x 16 heads, intermediate 2816, patch 14, ratio 3, text 5120) with random weights; fewer blocks than the
checkpoint's 32 keep the test short (every block is the same code)."""

import pytest
import torch
from _v41_util import DEV

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.vision.encoder import (
    DeepseekV41ImageEncoder,
    image_weight_specs,
)
from tensorrt_llm._torch.configs.deepseek_v41 import DeepseekV41VisionConfig
from tensorrt_llm._torch.models.deepseek_v41_vision import DeepseekV41Aligner, DeepseekV41VisionModel

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)

_TEXT = 5120


def _stock_and_weights(layers: int):
    torch.manual_seed(1234)
    vc = DeepseekV41VisionConfig(num_hidden_layers=layers)
    vision = DeepseekV41VisionModel(vc).to(DEV, torch.bfloat16)  # the norms stay fp32 (stock's `_apply`)
    aligner = DeepseekV41Aligner(vc, _TEXT).to(DEV, torch.bfloat16)
    for module in (vision, aligner):
        for p in module.parameters():
            p.data.copy_(torch.randn_like(p, dtype=torch.float32) * (0.5 if p.dim() == 1 else p.shape[-1] ** -0.5))
    weights = {f"vision.{k}": v for k, v in vision.state_dict().items()}
    weights.update({f"aligner.{k}": v for k, v in aligner.state_dict().items()})
    weights.update({k: torch.randn(_TEXT, device=DEV).bfloat16() for k in ("image_start", "image_newline", "image_end")})
    return vc, vision, aligner, weights


def _stock_span(vision, aligner, weights, pixels, h, w):
    """Stock's `encode_multimodal_inputs` for one image (the separators from `weights`)."""
    features = aligner(vision(pixels, h, w), h, w)
    rows, cols = -(-h // 3), -(-w // 3)
    features = features.reshape(rows, cols, _TEXT)
    newline = weights["image_newline"].to(features.dtype).expand(rows, 1, -1)
    return torch.cat((weights["image_start"].unsqueeze(0), torch.cat((features, newline), dim=1).flatten(0, 1),
                      weights["image_end"].unsqueeze(0)))


@pytest.mark.parametrize("grid", [(36, 36), (25, 31), (3, 3)], ids=["36x36", "25x31-padded", "3x3"])
def test_image_rows_match_stock_bit_for_bit(grid):
    h, w = grid
    vc, vision, aligner, weights = _stock_and_weights(layers=2)
    ours = DeepseekV41ImageEncoder(vc, _TEXT, torch.bfloat16).to(DEV)
    ours.load(weights, torch.device(DEV))
    pixels = torch.randn(h * w, 3 * 14 * 14, device=DEV).bfloat16()
    rows, cols = -(-h // 3), -(-w // 3)
    with torch.inference_mode():
        ref = _stock_span(vision, aligner, weights, pixels, h, w)
        got = ours(pixels, h, w, rows, cols, rows * (cols + 1) + 2)
    assert got.shape == ref.shape == (rows * (cols + 1) + 2, _TEXT)
    assert torch.equal(got, ref)


def test_specs_cover_the_checkpoint_keys():
    """The encoder loads exactly the tensors stock loads: `vision.*`, `aligner.*` and the three separators (266 for
    the checkpoint's 32 blocks), the norms in fp32."""
    vc, vision, aligner, weights = _stock_and_weights(layers=2)
    specs = image_weight_specs(vc, _TEXT, torch.bfloat16)
    assert set(specs) == set(weights)
    assert all(specs[k][0] == tuple(v.shape) and specs[k][1] == v.dtype for k, v in weights.items())
    assert len(image_weight_specs(DeepseekV41VisionConfig(), _TEXT, torch.bfloat16)) == 266


def test_load_refuses_missing_and_misshapen_weights():
    vc, _, _, weights = _stock_and_weights(layers=1)
    ours = DeepseekV41ImageEncoder(vc, _TEXT).to(DEV)
    with pytest.raises(ValueError, match="Missing"):
        ours.load({k: v for k, v in weights.items() if k != "image_end"}, torch.device(DEV))
    bad = dict(weights, image_end=torch.zeros(7, device=DEV))
    with pytest.raises(ValueError, match="shape"):
        ours.load(bad, torch.device(DEV))


def test_grid_must_match_the_placeholder_span():
    vc, _, _, weights = _stock_and_weights(layers=1)
    ours = DeepseekV41ImageEncoder(vc, _TEXT).to(DEV)
    ours.load(weights, torch.device(DEV))
    pixels = torch.randn(9, 3 * 14 * 14, device=DEV).bfloat16()
    with pytest.raises(ValueError, match="placeholder span"):
        ours(pixels, 3, 3, 1, 1, 5)

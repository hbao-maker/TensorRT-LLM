# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DeepSeek-V4.1 image encoder, shared by the family's targets: the vision transformer, the projector to the text
width and the image-span separators, built from the family's `ops/vision` entries.

It computes what stock's `DeepseekV41VisionModel`, `DeepseekV41Aligner` and
`DeepseekV41ForConditionalGeneration.encode_multimodal_inputs` compute, with the same torch calls in the same order,
so an image's rows are stock's bit for bit. The checkpoint keys are stock's: `vision.*`, `aligner.*` and the three
separator embeddings `image_start`, `image_newline`, `image_end`.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.image_span import image_span
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_attention import (
    vision_attention,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_linear import vision_linear
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_patch_merge import (
    vision_patch_merge,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_rms_norm import (
    vision_rms_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_rope_2d import (
    vision_rope_2d,
    vision_rope_tables,
)

_SEPARATORS = ("image_start", "image_newline", "image_end")


def _field(config, name: str, default=None):
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)


def image_weight_specs(vision_config, text_hidden: int, dtype: torch.dtype) -> Dict[str, Tuple[tuple, torch.dtype]]:
    """Checkpoint key -> (shape, dtype) of every tensor the encoder loads. The norms stay fp32 (stock keeps them fp32
    when it casts the encoder to the text embedding's dtype); everything else is `dtype`."""
    hidden, inter = _field(vision_config, "hidden_size"), _field(vision_config, "intermediate_size")
    patch, ratio = _field(vision_config, "patch_size"), _field(vision_config, "downsample_ratio")
    f32 = torch.float32
    specs = {
        "vision.patch_embed.proj.weight": ((hidden, 3 * patch * patch), dtype),
        "vision.patch_embed.proj.bias": ((hidden,), dtype),
    }
    for i in range(_field(vision_config, "num_hidden_layers")):
        p = f"vision.blocks.{i}."
        specs.update({
            p + "norm1.weight": ((hidden,), f32),
            p + "attn.wqkv.weight": ((3 * hidden, hidden), dtype),
            p + "attn.wqkv.bias": ((3 * hidden,), dtype),
            p + "attn.wo.weight": ((hidden, hidden), dtype),
            p + "attn.wo.bias": ((hidden,), dtype),
            p + "norm2.weight": ((hidden,), f32),
            p + "mlp.w1.weight": ((2 * inter, hidden), dtype),
            p + "mlp.w2.weight": ((hidden, inter), dtype),
        })
    specs.update({
        "vision.norm.weight": ((hidden,), f32),
        "aligner.w1.weight": ((text_hidden, hidden * ratio * ratio), dtype),
        "aligner.w1.bias": ((text_hidden,), dtype),
        "aligner.w2.weight": ((text_hidden, text_hidden), dtype),
        "aligner.w2.bias": ((text_hidden,), dtype),
    })
    specs.update({name: ((text_hidden,), dtype) for name in _SEPARATORS})
    return specs


class DeepseekV41ImageEncoder(nn.Module):
    """Pixels of one image -> its `rows * (columns + 1) + 2` rows of the token stream, at the text width.

    `vision_config`: the checkpoint's `vision_config`; `text_hidden`: the text model's width; `dtype`: the text
    embedding's dtype (bf16). Holds no state but its weights."""

    def __init__(self, vision_config, text_hidden: int, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.hidden = _field(vision_config, "hidden_size")
        self.heads = _field(vision_config, "num_attention_heads")
        if self.heads <= 0 or self.hidden % self.heads or (self.hidden // self.heads) % 4:
            raise ValueError("the vision head dimension must be a positive multiple of four (2-D rotary halves)")
        self.head_dim = self.hidden // self.heads
        self.layers = _field(vision_config, "num_hidden_layers")
        self.theta = float(_field(vision_config, "rope_theta"))
        self.eps = float(_field(vision_config, "rms_norm_eps", 1e-6))
        self.ratio = _field(vision_config, "downsample_ratio")
        self.text_hidden, self.dtype = text_hidden, dtype
        self.specs = image_weight_specs(vision_config, text_hidden, dtype)
        # The checkpoint keys contain dots, which module attribute names cannot: the tensors live in this dict and
        # are registered as buffers under the key with dots turned into "__" (so `.to()` / `state_dict` see them).
        self.w: Dict[str, torch.Tensor] = {}
        for key, (shape, dt) in self.specs.items():
            self._set(key, torch.empty(shape, dtype=dt))

    def _set(self, key: str, value: torch.Tensor) -> None:
        self.register_buffer(key.replace(".", "__"), value, persistent=False)
        self.w[key] = getattr(self, key.replace(".", "__"))

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        self.w = {key: getattr(self, key.replace(".", "__")) for key in self.specs}
        return out

    def load(self, weights, device) -> None:
        """Every key of `image_weight_specs` from the checkpoint's tensors (stock's checks: present, same shape)."""
        for key, (shape, dt) in self.specs.items():
            if key not in weights:
                raise ValueError(f"Missing DeepSeek-V4.1 image weight: {key}")
            value = weights[key]
            if not isinstance(value, torch.Tensor):
                value = value[:]
            if tuple(value.shape) != tuple(shape):
                raise ValueError(f"DeepSeek-V4.1 image weight shape mismatch: {key} {tuple(value.shape)} != {shape}")
            self._set(key, value.to(device=device, dtype=dt))

    def vision(self, pixels: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """`[n_h * n_w, 3 * P * P]` (or `[n_h * n_w, 3, P, P]`) patches -> `[n_h * n_w, hidden]` (stock's
        `DeepseekV41VisionModel.forward`)."""
        if n_h <= 0 or n_w <= 0 or pixels.ndim < 2 or pixels.shape[0] != n_h * n_w:
            raise ValueError("Vision patches must match a positive n_h by n_w grid")
        w, n = self.w, pixels.shape[0]
        x = vision_linear(pixels.flatten(1), w["vision.patch_embed.proj.weight"], w["vision.patch_embed.proj.bias"])
        cos, sin = vision_rope_tables(n_h, n_w, self.head_dim // 2, self.theta, pixels.device)
        for i in range(self.layers):
            p = f"vision.blocks.{i}."
            y = vision_rms_norm(x, w[p + "norm1.weight"], self.eps)
            q, k, v = (part.reshape(n, self.heads, self.head_dim)
                       for part in vision_linear(y, w[p + "attn.wqkv.weight"], w[p + "attn.wqkv.bias"]).chunk(3, dim=-1))
            attended = vision_attention(vision_rope_2d(q, cos, sin), vision_rope_2d(k, cos, sin), v)
            x = x + vision_linear(attended, w[p + "attn.wo.weight"], w[p + "attn.wo.bias"])
            gate, up = vision_linear(vision_rms_norm(x, w[p + "norm2.weight"], self.eps), w[p + "mlp.w1.weight"]).chunk(2, dim=-1)
            x = x + vision_linear(F.silu(gate) * up, w[p + "mlp.w2.weight"])
        return vision_rms_norm(x, w["vision.norm.weight"], self.eps)

    def project(self, features: torch.Tensor, n_h: int, n_w: int) -> torch.Tensor:
        """`[n_h * n_w, hidden]` -> `[ceil(n_h / r) * ceil(n_w / r), text_hidden]` (stock's `DeepseekV41Aligner`)."""
        if n_h <= 0 or n_w <= 0 or features.ndim != 2 or features.shape[0] != n_h * n_w:
            raise ValueError("Aligner features must match a positive n_h by n_w grid")
        w = self.w
        merged = vision_patch_merge(features, n_h, n_w, self.ratio)
        hidden = F.gelu(vision_linear(merged, w["aligner.w1.weight"], w["aligner.w1.bias"]))
        return vision_linear(hidden, w["aligner.w2.weight"], w["aligner.w2.bias"])

    def forward(self, pixels: torch.Tensor, n_h: int, n_w: int, rows: int, columns: int, length: int) -> torch.Tensor:
        """One image -> its span of `length` rows: start, each projected row then a newline, end."""
        features = self.project(self.vision(pixels, n_h, n_w), n_h, n_w)
        if features.shape[0] != rows * columns or length != rows * (columns + 1) + 2:
            raise ValueError("DeepSeek-V4.1 image grid does not match its placeholder span")
        w = self.w
        return image_span(features, rows, columns, w["image_start"], w["image_newline"], w["image_end"])

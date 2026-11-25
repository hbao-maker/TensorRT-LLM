# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4 routing of a step that carries image rows: each row picks its experts with its own selection bias, the
checkpoint's vision bias (`ffn.gate.bias_vl`) for an image row and the text bias for the others; the weights are the
unbiased scores either way (stock's `DeepSeekV4MoeRoutingMethod.apply_with_aux` with the image mask as routing aux).

The fused routing launches (`dsv4_route`, `dsv4_route_nvfp4_quant`) take one bias for every row. These entries run the
text launch and the vision launch over the same logits and keep, per row, the launch whose bias that row selects. Only
steps with image rows run them; a text step keeps its single launch."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .dsv4_route_nvfp4_quant import _sum_partials, dsv4_route, dsv4_route_nvfp4_quant


def _route_rows(logits, text_bias, image_bias, image, top_k: int, scale: float):
    """fp32 `(ids [n, top_k], weights [n, top_k])` of the per-row bias selection, in plain torch."""
    if hasattr(logits, "parts"):
        logits = _sum_partials(logits.parts)
    scores = torch.sqrt(F.softplus(logits.float()))
    bias = torch.where(image[:, None], image_bias.float(), text_bias.float())
    ids = torch.topk(scores + bias, top_k, dim=-1)[1]
    w = torch.gather(scores, 1, ids)
    w = w / (torch.sum(w, dim=-1, keepdim=True) + 1e-20)
    return ids.to(torch.int32), w * scale


_CELLS = (
    Cell(why="an image prompt's first chunk: 300 rows, 280 of them image", spec=dict(n=300, image=(10, 290))),
    Cell(why="a chunk of only image rows", spec=dict(n=256, image=(0, 256))),
    Cell(why="no image row (the text bias everywhere)", spec=dict(n=64, image=(0, 0))),
)
_NOTE = "Experts picked from exactly tied scores may come out in a different order than torch.topk's (the test compares sets)."


class Dsv4RouteImage(OpWrapper):
    """`dsv4_route` with a per-row bias: `(ids int32 [n, top_k], weights [n, top_k])` from fp32 `logits [n, E]`, the
    text `selection_bias [E]`, the `image_bias [E]` and `image` bool `[n]` (True: the row is an image token)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _CELLS
    note = _NOTE

    def __call__(self, logits, selection_bias, image_bias, image, top_k: int, routed_scaling_factor: float, hidden: int,
                 weights_dtype: torch.dtype = torch.bfloat16):
        ids, w = dsv4_route(logits, selection_bias, top_k, routed_scaling_factor, hidden, weights_dtype)
        ids_i, w_i = dsv4_route(logits, image_bias, top_k, routed_scaling_factor, hidden, weights_dtype)
        rows = image[:, None]
        return torch.where(rows, ids_i, ids), torch.where(rows, w_i, w)

    def reference(self, logits, selection_bias, image_bias, image, top_k: int, routed_scaling_factor: float,
                  hidden: int, weights_dtype: torch.dtype = torch.bfloat16):
        ids, w = _route_rows(logits, selection_bias, image_bias, image, top_k, routed_scaling_factor)
        return ids, w.to(weights_dtype)

    def is_valid(self, logits, selection_bias, image_bias, image, top_k: int, routed_scaling_factor: float,
                 hidden: int, weights_dtype: torch.dtype = torch.bfloat16) -> None:
        dsv4_route.is_valid(logits, selection_bias, top_k, routed_scaling_factor, hidden, weights_dtype)
        assert image_bias.dtype == torch.float32 and image_bias.shape == selection_bias.shape
        assert image.dtype == torch.bool and image.shape == (logits.shape[0],)


class Dsv4RouteNvfp4QuantImage(OpWrapper):
    """`dsv4_route_nvfp4_quant` with a per-row bias: `(ids, weights, x_fp4, x_sf)` as that entry returns them, the
    routing as `dsv4_route_image`. The input is quantized once (the text launch); the vision launch only routes."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = _CELLS
    note = _NOTE

    def __call__(self, logits, selection_bias, image_bias, image, x, global_scale, top_k: int,
                 routed_scaling_factor: float, weights_dtype: torch.dtype = torch.bfloat16):
        ids, w, xq, xsf = dsv4_route_nvfp4_quant(
            logits, selection_bias, x, global_scale, top_k, routed_scaling_factor, weights_dtype
        )
        ids_i, w_i = dsv4_route(logits, image_bias, top_k, routed_scaling_factor, x.shape[1], weights_dtype)
        rows = image[:, None]
        return torch.where(rows, ids_i, ids), torch.where(rows, w_i, w), xq, xsf

    def reference(self, logits, selection_bias, image_bias, image, x, global_scale, top_k: int,
                  routed_scaling_factor: float, weights_dtype: torch.dtype = torch.bfloat16):
        ids, w = _route_rows(logits, selection_bias, image_bias, image, top_k, routed_scaling_factor)
        _, _, xq, sf = dsv4_route_nvfp4_quant.reference(logits, selection_bias, x, global_scale, top_k,
                                                        routed_scaling_factor)
        return ids, w.to(weights_dtype), xq, sf

    def is_valid(self, logits, selection_bias, image_bias, image, x, global_scale, top_k: int,
                 routed_scaling_factor: float, weights_dtype: torch.dtype = torch.bfloat16) -> None:
        dsv4_route_nvfp4_quant.is_valid(logits, selection_bias, x, global_scale, top_k, routed_scaling_factor,
                                        weights_dtype)
        assert not hasattr(logits, "parts"), "the vision launch reads whole logits (no K-chunk partials)"
        assert image_bias.dtype == torch.float32 and image_bias.shape == selection_bias.shape
        assert image.dtype == torch.bool and image.shape == (x.shape[0],)


class ImageRowsSelect(OpWrapper):
    """In place: `ids[r], weights[r] = image_ids[r], image_weights[r]` for every image row `r` (`image` bool `[n]`) of
    routing already written, e.g. into a MegaMoE bucket's staged buffers by the text launch, from the vision launch's
    copy. -> `(ids, weights)`. Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a MegaMoE bucket's slots (int64 on sm_103) and fp32 weights, 300 live rows", spec=dict(n=300, top_k=6)),
        Cell(why="no live row", spec=dict(n=0, top_k=6)),
    )

    def __call__(self, ids, weights, image_ids, image_weights, image):
        rows = image[:, None]
        ids.copy_(torch.where(rows, image_ids, ids))
        weights.copy_(torch.where(rows, image_weights, weights))
        return ids, weights

    def reference(self, ids, weights, image_ids, image_weights, image):
        ids, weights = ids.clone(), weights.clone()
        ids[image] = image_ids[image]
        weights[image] = image_weights[image]
        return ids, weights

    def is_valid(self, ids, weights, image_ids, image_weights, image) -> None:
        assert ids.shape == weights.shape == image_ids.shape == image_weights.shape
        assert ids.dtype == image_ids.dtype and weights.dtype == image_weights.dtype
        assert image.dtype == torch.bool and image.shape == (ids.shape[0],)


dsv4_route_image = Dsv4RouteImage()
dsv4_route_nvfp4_quant_image = Dsv4RouteNvfp4QuantImage()
image_rows_select = ImageRowsSelect()

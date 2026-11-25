# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image positions receive no Engram delta (stock's `DeepseekV41Engram.forward(token_mask=...)`): after the gate, an
image row of the residual stream keeps the value it entered the Engram with."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class EngramImageRows(OpWrapper):
    """`(gated [n, hc, d], residual [n, hc, d], image bool [n])` -> `residual` on image rows, `gated` elsewhere, as
    stock's `torch.where(token_mask[:, None, None], output, hidden_states)`. Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="an image prompt's chunk at the model's width (hc 4, d 5120)", spec=dict(n=300, hc=4, d=5120)),
        Cell(why="no image row", spec=dict(n=16, hc=4, d=5120, no_image=True)),
    )

    def __call__(self, gated: torch.Tensor, residual: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        return torch.where(image[:, None, None], residual, gated)

    def reference(self, gated: torch.Tensor, residual: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
        out = gated.clone()
        out[image] = residual[image]
        return out

    def is_valid(self, gated: torch.Tensor, residual: torch.Tensor, image: torch.Tensor) -> None:
        assert gated.shape == residual.shape and gated.dtype == residual.dtype
        assert image.dtype == torch.bool and image.shape == (gated.shape[0],)


engram_image_rows = EngramImageRows()

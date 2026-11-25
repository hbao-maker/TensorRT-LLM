# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The rows one image occupies in the token stream (stock's `DeepseekV41ForConditionalGeneration
.encode_multimodal_inputs`): the image-start embedding, each projected row of the image followed by the newline
embedding, then the image-end embedding. The processor reserves exactly `rows * (columns + 1) + 2` image tokens."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class ImageSpan(OpWrapper):
    """`features [rows * columns, H]` (row-major) and the `[H]` start / newline / end embeddings ->
    `[rows * (columns + 1) + 2, H]` in `features`' dtype. Moves bits: exact."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="a 12 x 12 image at the text width", spec=dict(rows=12, columns=12, hidden=5120)),
        Cell(why="a 1 x 1 image: the shortest span (4 tokens)", spec=dict(rows=1, columns=1, hidden=5120)),
        Cell(why="a wide 8 x 30 image", spec=dict(rows=8, columns=30, hidden=5120)),
    )

    def __call__(self, features: torch.Tensor, rows: int, columns: int, start: torch.Tensor, newline: torch.Tensor,
                 end: torch.Tensor) -> torch.Tensor:
        grid = features.reshape(rows, columns, -1)
        nl = newline.to(grid.dtype).expand(rows, 1, -1)
        return torch.cat(
            (start.to(grid.dtype).unsqueeze(0), torch.cat((grid, nl), dim=1).flatten(0, 1), end.to(grid.dtype).unsqueeze(0))
        )

    def reference(self, features: torch.Tensor, rows: int, columns: int, start: torch.Tensor, newline: torch.Tensor,
                  end: torch.Tensor) -> torch.Tensor:
        out = [start.to(features.dtype)]
        for r in range(rows):
            out.extend(features[r * columns + c] for c in range(columns))
            out.append(newline.to(features.dtype))
        out.append(end.to(features.dtype))
        return torch.stack(out)

    def is_valid(self, features: torch.Tensor, rows: int, columns: int, start: torch.Tensor, newline: torch.Tensor,
                 end: torch.Tensor) -> None:
        assert features.dim() == 2 and features.shape[0] == rows * columns and rows > 0 and columns > 0
        assert start.shape == newline.shape == end.shape == (features.shape[1],)


image_span = ImageSpan()

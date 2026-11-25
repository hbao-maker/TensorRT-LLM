# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`engram_image_rows`: image rows keep the residual they entered the Engram with, text rows take the gated value
(stock's `torch.where(token_mask, output, hidden_states)`). Exact."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_image_rows import (
    engram_image_rows,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA")


@pytest.mark.parametrize("cell", engram_image_rows.CELLS, ids=[c.why[:44] for c in engram_image_rows.CELLS])
def test_cells(cell, certifying):
    certifying(engram_image_rows)
    s, gen = cell.spec, gen_for(cell)
    gated, residual = (torch.randn(s["n"], s["hc"], s["d"], generator=gen, device=DEV).bfloat16() for _ in range(2))
    image = torch.zeros(s["n"], dtype=torch.bool, device=DEV)
    if not s.get("no_image"):
        image[5 : s["n"] - 9] = True
    out = engram_image_rows(gated, residual, image)
    assert torch.equal(out, engram_image_rows.reference(gated, residual, image))
    assert torch.equal(out[~image], gated[~image]) and torch.equal(out[image], residual[image])


def test_rejections():
    x = torch.zeros(3, 4, 8, device=DEV)
    with pytest.raises(AssertionError):
        engram_image_rows.is_valid(x, x, torch.ones(3, device=DEV))

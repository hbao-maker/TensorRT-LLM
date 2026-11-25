# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The step's token positions of the target's core: a generation step reads the CSA2 metadata's live query positions
(`csa2_positions`, as stock's CSA2 layers), not the engine's `position_ids`, which differ for the warm-up's max-length
dummy request (position 0 at a cache length of max_seq_len - 1); steps with context rows keep `position_ids`."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="imports the target (CUDA build)")
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__default.attention_tp.core")


@pytest.mark.parametrize("rows", [1, 2, 3])
def test_generation_step_reads_the_live_positions(rows):
    live = torch.tensor([996593] + [5 + r for r in range(1, rows)] + [0, 0], dtype=torch.int32)  # padded buffer
    ids = torch.zeros(1, rows, dtype=torch.long)  # the engine's position_ids of the dummy: 0
    got = mod._step_positions(NS(csa2_positions=live, num_contexts=0), ids, rows)
    assert got.dtype == torch.int32 and torch.equal(got, live[:rows])


def test_context_step_reads_position_ids():
    ids = torch.arange(7).view(1, -1)
    live = torch.full((7,), 9, dtype=torch.int32)  # possibly not prepared yet (stock defers it to its first layer)
    got = mod._step_positions(NS(csa2_positions=live, num_contexts=1), ids, 7)
    assert torch.equal(got, ids.view(-1).int())
    assert torch.equal(mod._step_positions(NS(num_contexts=0), ids, 7), ids.view(-1).int())

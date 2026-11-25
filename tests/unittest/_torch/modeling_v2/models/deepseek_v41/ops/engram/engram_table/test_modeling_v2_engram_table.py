# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Test for the host-resident engram table: a lookup through `ShardedFp8MultiHeadEmbedding` against a dense dequantization.

The table lives in pinned host memory and is read by the gather kernel through UVA; its bucket offsets are a buffer on
the device. Row shards (heads not divisible by the table's ranks) return the rows the rank owns and zero elsewhere; head
shards return only the rank's own heads.
"""

import pytest
import torch
from _v41_util import DEV

__extra_import_path__ = ["../../../../.."]

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
)

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] == 10),
    reason="the engram lookup is a Blackwell (sm_10x) kernel",
)

DIM, BLOCK = 128, 32
BUCKETS = [37, 41, 43, 47]  # one bucket per head


def _checkpoint_rows():
    gen = torch.Generator(device="cpu").manual_seed(7)
    rows = sum(BUCKETS)
    weight = (torch.randn(rows, DIM, generator=gen) * 4).to(torch.float8_e4m3fn)
    scale = torch.randint(118, 134, (rows, DIM // BLOCK), generator=gen, dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    return weight, scale


def _dense(weight, scale):
    values = weight.float().unflatten(-1, (-1, BLOCK))
    exponent = (scale.view(torch.uint8).float() - 127).exp2().unsqueeze(-1)
    return (values * exponent).flatten(-2).to(torch.bfloat16)


@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1), (3, 0), (3, 2)], ids=str)
def test_lookup_matches_dense_dequantization(tp_size, tp_rank):
    weight, scale = _checkpoint_rows()
    table = ShardedFp8MultiHeadEmbedding(
        list_of_N=BUCKETS, D=DIM, block_size=BLOCK, dtype=torch.bfloat16, tp_size=tp_size, tp_rank=tp_rank
    )
    table.to(DEV)  # what a target does once the module exists: the offsets buffer goes to the GPU
    table.load_weights({"weight": weight, "scale": scale})
    gen = torch.Generator(device="cpu").manual_seed(11)
    ids = torch.stack([torch.randint(0, n, (64,), generator=gen) for n in BUCKETS], dim=1).to(DEV)
    got = table(ids)

    dense = _dense(weight, scale).to(DEV)
    offsets = torch.tensor([0, *torch.tensor(BUCKETS).cumsum(0)[:-1].tolist()], device=DEV)
    rows = ids + offsets  # global rows, [64, heads]
    expect = dense[rows]  # [64, heads, DIM]
    if table.shard_heads:
        expect = expect[:, table.head_start : table.head_start + table.num_local_heads]
    else:
        owned = (rows >= table.vocab_start_idx) & (rows < table.vocab_end_idx)
        expect = expect.masked_fill(~owned.unsqueeze(-1), 0)
    assert got.shape == expect.shape and got.dtype == torch.bfloat16
    torch.testing.assert_close(got, expect, rtol=0, atol=0)

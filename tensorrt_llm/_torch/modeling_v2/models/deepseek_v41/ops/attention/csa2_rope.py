# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Rotary embedding on the trailing `rd` dims of a latent head (plain torch)."""

from __future__ import annotations

import math

import numpy as np
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


def csa2_rope_table(
    rd: int,
    seqlen: int,
    original_seq_len: int,
    base: float,
    factor: float,
    beta_fast: float,
    beta_slow: float,
    device,
) -> torch.Tensor:
    """`[seqlen, rd/2, 2]` float32 (cos, sin); YaRN when `original_seq_len > 0` (the reference's rule).

    A setup function, not an entry: it builds a constant table once when the target loads its
    weights, and no step calls it. The table is an argument of `csa2_rope_tail`. Built bit for bit as the
    engine builds its CSA2 layers' `RotaryEmbedding.rotary_cos_sin` (`RopeEmbeddingUtils`): a compressed layer's
    YaRN table (`create_sinusoidal_positions_yarn`: torch fp32, mscale 1) and a window-only layer's plain table
    (`create_sinusoidal_positions_for_attention_plugin`: numpy fp32, numpy's cos / sin); the fp32 rounding of the
    angle grows with the position, so a table built another way differs in the last bit at most positions. The
    test checks it against the reference's complex-exponential form.
    """
    if original_seq_len > 0:

        def corrected_dim(rot):
            return rd * math.log(original_seq_len / (rot * 2 * math.pi)) / (2 * math.log(base))

        low = max(math.floor(corrected_dim(beta_fast)), 0)
        high = min(math.ceil(corrected_dim(beta_slow)), rd - 1)
        if low == high:
            high += 0.001
        pos_freqs = base ** (torch.arange(0, rd, 2, dtype=torch.float32) / rd)
        freq_extra = 1.0 / pos_freqs
        freq_inter = 1.0 / (factor * pos_freqs)
        mask = 1 - torch.clamp((torch.arange(rd // 2, dtype=torch.float32) - low) / (high - low), 0, 1)
        inv_freq = freq_inter * (1 - mask) + freq_extra * mask
        # the engine's shapes too ([seqlen, rd, 1], the angles twice): CPU cos / sin split their work by the tensor's
        # size, and an element's vector or scalar path decides its last bit
        ang = torch.einsum("i,j -> ij", torch.arange(seqlen, dtype=torch.float32), inv_freq).unsqueeze(-1)
        ang = torch.cat((ang, ang), dim=-2)
        table = torch.cat((torch.cos(ang) * 1.0, torch.sin(ang) * 1.0), dim=-1)  # (the engine's mscale ratio, 1)
        return table[:, : rd // 2].contiguous().to(device)
    inv_freq = 1.0 / (base ** (np.arange(0, rd, 2) / rd)).astype(np.float32)
    ang = np.expand_dims(np.einsum("i , j -> i j", np.arange(seqlen, dtype=np.float32), inv_freq, dtype=np.float32), axis=-1)
    ang = np.concatenate((ang, ang), axis=-2)
    table = np.concatenate((np.cos(ang), np.sin(ang)), axis=-1).astype(np.float32)
    return torch.from_numpy(np.ascontiguousarray(table[:, : rd // 2])).to(device)


class Csa2RopeTail(OpWrapper):
    """Rotate `x[..., -rd:]` by the angle of `pos` (adjacent element pairs); returns a new tensor.

    `x` is `[n, D]` or `[n, h, D]`; `pos` is `[n]` (int32/int64) indexing `table`
    (`[S, rd/2, 2]` fp32, from `csa2_rope_table`). `inverse` rotates the other way.
    The part of `x` before the tail is returned untouched.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="query heads of a tp8 rank at decode: 8 heads, head dim 512, rd 64, base-theta table",
            spec=dict(n=4, heads=8, dim=512, dtype=torch.bfloat16, original=0, inverse=False),
        ),
        Cell(
            why="all 64 heads (dep2 attention is replicated) at a prefill chunk, YaRN table",
            spec=dict(n=300, heads=64, dim=512, dtype=torch.bfloat16, original=65536, inverse=False),
        ),
        Cell(
            why="the single kv head row [n, D] (no head axis), base table",
            spec=dict(n=16, heads=None, dim=512, dtype=torch.bfloat16, original=0, inverse=False),
        ),
        Cell(
            why="index keys: 128-wide rows, YaRN table (the compressed-KV rope)",
            spec=dict(n=64, heads=None, dim=128, dtype=torch.bfloat16, original=65536, inverse=False),
        ),
        Cell(
            why="attention output de-rotated: inverse=True over 64 heads",
            spec=dict(n=8, heads=64, dim=512, dtype=torch.bfloat16, original=0, inverse=True),
        ),
        Cell(
            why="fp32 input: output stays fp32",
            spec=dict(n=8, heads=8, dim=512, dtype=torch.float32, original=65536, inverse=False),
        ),
    )
    note = """
    `rd` is 64 at every cell (the checkpoint's `qk_rope_head_dim`); other even values are
    not driven. Positions at the table's last row are covered only through the cell with
    n=300 (positions are 0..299 of a 4096-row table).
    """

    def __call__(
        self,
        x: torch.Tensor,
        table: torch.Tensor,
        pos: torch.Tensor,
        rd: int,
        inverse: bool = False,
    ) -> torch.Tensor:
        cs = table[pos.long()]  # [n, rd/2, 2]
        c, s = cs[..., 0], cs[..., 1]
        if inverse:
            s = -s
        if x.dim() == 3:
            c, s = c.unsqueeze(1), s.unsqueeze(1)
        tail = x[..., -rd:].float().unflatten(-1, (rd // 2, 2))
        a, b = tail[..., 0], tail[..., 1]
        rot = torch.stack([a * c - b * s, a * s + b * c], dim=-1).flatten(-2)
        return torch.cat([x[..., :-rd], rot.to(x.dtype)], dim=-1)

    def reference(
        self,
        x: torch.Tensor,
        table: torch.Tensor,
        pos: torch.Tensor,
        rd: int,
        inverse: bool = False,
    ) -> torch.Tensor:
        """The checkpoint reference's form: multiply the tail, viewed as complex numbers, by
        `exp(i * angle)` (conjugated for `inverse`)."""
        fc = torch.view_as_complex(table[pos.long()].contiguous())  # [n, rd/2]
        if inverse:
            fc = fc.conj()
        if x.dim() == 3:
            fc = fc.unsqueeze(1)
        xc = torch.view_as_complex(x[..., -rd:].float().contiguous().unflatten(-1, (-1, 2)))
        rot = torch.view_as_real(xc * fc).flatten(-2).to(x.dtype)
        return torch.cat([x[..., :-rd], rot], dim=-1)

    def is_valid(
        self,
        x: torch.Tensor,
        table: torch.Tensor,
        pos: torch.Tensor,
        rd: int,
        inverse: bool = False,
    ) -> None:
        assert x.dim() in (2, 3), "x must be [n, D] or [n, h, D]; other ranks broadcast wrongly"
        assert pos.dim() == 1 and pos.shape[0] == x.shape[0], (
            "pos must be [n]; a single position would be broadcast over every row"
        )
        assert pos.dtype in (torch.int32, torch.int64), (
            "pos must be integer: a float position is truncated when it indexes the table"
        )
        assert table.dim() == 3 and table.shape[1:] == (rd // 2, 2) and table.dtype == torch.float32
        assert rd % 2 == 0 and x.shape[-1] >= rd


csa2_rope_tail = Csa2RopeTail()

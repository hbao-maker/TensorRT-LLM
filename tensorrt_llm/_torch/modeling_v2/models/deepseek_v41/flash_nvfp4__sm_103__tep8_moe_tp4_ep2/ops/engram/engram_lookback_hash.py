# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hash ids of a step from the target's per-request look-back ring, in two launches (decode sizes).

The plain-torch form (`modeling._EngramHash.lookback`) is ~30 launches; the engine's own decode hash
(`kernels/blackwell/engram/decode_hash.py`) reads a full-length per-request history, which this target does not keep."""

from __future__ import annotations

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

MAX_ROWS = 1024  # each program holds the whole step's rows


class EngramLookbackHash(OpWrapper):
    """`-> [n, L, (max_n - 1) * H]` int64 bucket-local hash ids; also writes the step's compressed ids into `hist`.
    The ids are stored layer-major (a `[L, n, cols]` buffer, returned as its `[n, L, cols]` transpose): each Engram
    layer's `ids[:, l]` is contiguous, as its table lookup reads it (no per-layer copy).

    Token `r` (request row `tok_req[r]`, ring slot `s = req_slot[tok_req[r]]`, position `p`): `c0 = token_map[ids[r]]`;
    look-back `c_b` (b = 1..max_n-1) is the compressed id at `p - b`: the step's own row `r - b` when that row is the same
    slot at `p - b`, else `hist[s, (p - b) % ring]`, and the pad id when `p - b < 0`. Hash of layer `l`, order `i`, head
    `h`: `(XOR_{b <= i} c_b * mults[l, b]) mod primes[l, i - 1, h]` (int64 wrap-around products, non-negative remainder).
    After every read, `hist[s, p % ring] = c0` for the last `ring` tokens of each request in the step; the others write
    the dummy row `hist.shape[0] - 1`.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: one request, 1 + 5 draft tokens, deep in its history", spec=dict(reqs=1, per=6, start=900)),
        Cell(why="request start: look-back positions before 0 read the pad id", spec=dict(reqs=1, per=6, start=1)),
        Cell(why="four requests of 6 rows, two of them in the dummy slot", spec=dict(reqs=4, per=6, start=200, dummy=2)),
        Cell(why="a 40-row chunk: longer than the ring, only its last 16 rows are written", spec=dict(reqs=1, per=40, start=3)),
    )
    note = "Exact (integer arithmetic): the test gates hash ids and the ring bit for bit against `reference`."

    def __call__(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        import triton

        from ...kernels.engram_hash import engram_hash_kernel, engram_ring_write_kernel

        n = input_ids.shape[0]
        layers, max_n = mults.shape
        heads = primes.shape[-1]
        out = torch.empty(layers, n, (max_n - 1) * heads, dtype=torch.int64, device=input_ids.device)
        if n:
            block = max(16, triton.next_power_of_2(n))
            engram_hash_kernel[(layers * (max_n - 1) * heads,)](
                input_ids, positions, tok_req, req_slot, token_map, hist, mults, primes, out,
                n, token_map.shape[0], pad_id,
                RING=hist.shape[1], MAX_N=max_n, L=layers, H=heads, BLOCK=block, num_warps=4,
            )  # fmt: skip
            engram_ring_write_kernel[(1,)](
                input_ids, positions, tok_req, req_slot, token_map, hist, n, token_map.shape[0], hist.shape[0] - 1,
                RING=hist.shape[1], BLOCK=block, num_warps=4,
            )  # fmt: skip
        return out.transpose(0, 1)

    def reference(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
    ) -> torch.Tensor:
        """The target's plain-torch look-back (`_EngramHash.lookback`), statement for statement."""
        ring = hist.shape[1]
        max_n = mults.shape[1]
        slot = req_slot[tok_req].long()
        comp = token_map[input_ids.long()]
        pos = positions.long()
        n = comp.shape[0]
        rows = torch.arange(n, device=comp.device)
        cols = [comp]
        for back in range(1, max_n):
            pb = pos - back
            j = (rows - back).clamp_min(0)
            inside = (rows >= back) & (slot[j] == slot) & (pos[j] == pb)
            v = torch.where(inside, comp[j], hist[slot, pb % ring].long())
            cols.append(torch.where(pb < 0, torch.full_like(v, pad_id), v))
        later = (rows + ring).clamp_max(max(n - 1, 0))
        last = (rows + ring >= n) | (slot[later] != slot) | (pos[later] != pos + ring)
        hist[torch.where(last, slot, hist.shape[0] - 1), pos % ring] = comp.to(hist.dtype)
        toks = torch.stack(cols, dim=-1)
        prods = toks.unsqueeze(1) * mults
        rolling, hashes = prods[..., 0], []
        for i in range(1, max_n):
            rolling = torch.bitwise_xor(rolling, prods[..., i])
            hashes.append(rolling.unsqueeze(-1) % primes[:, i - 1])
        return torch.cat(hashes, dim=-1)

    def is_valid(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        tok_req: torch.Tensor,
        req_slot: torch.Tensor,
        token_map: torch.Tensor,
        hist: torch.Tensor,
        mults: torch.Tensor,
        primes: torch.Tensor,
        pad_id: int,
    ) -> None:
        n = input_ids.shape[0]
        assert n <= MAX_ROWS, "a program holds the step's rows: wider steps take the plain-torch look-back"
        assert positions.dtype == torch.int32 and positions.shape == (n,) and tok_req.dtype == torch.int64
        assert req_slot.dtype == torch.int32 and token_map.dtype == torch.int64
        assert hist.dtype == torch.int32 and hist.is_contiguous()
        assert mults.dtype == primes.dtype == torch.int64 and mults.is_contiguous() and primes.is_contiguous()
        assert primes.shape[:2] == (mults.shape[0], mults.shape[1] - 1)
        assert all(t.is_contiguous() for t in (input_ids, positions, tok_req, req_slot, token_map))


engram_lookback_hash = EngramLookbackHash()

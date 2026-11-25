# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Engram n-gram hash of a decode-sized step over the target's look-back ring (see `ops/engram/engram_lookback_hash.py`
for the contract), in two launches: `engram_hash_kernel` (one program per (layer, order, head) over all tokens of the
step: each program does one int64 modulus per token instead of one thread doing all of them) reads the ring, then
`engram_ring_write_kernel` (one program) writes the step's ids into it. Stream order puts every read before the writes."""

import triton
import triton.language as tl


@triton.jit
def _floor_mod(x, p):
    r = x % p
    return tl.where(r < 0, r + p, r)


@triton.jit
def _compressed(ids_ptr, token_map_ptr, j, ok, vocab):
    t = tl.load(ids_ptr + j, mask=ok, other=0).to(tl.int64)
    t = tl.minimum(tl.maximum(t, 0), vocab - 1)
    return tl.load(token_map_ptr + t, mask=ok, other=0)


@triton.jit
def _slot(tok_req_ptr, req_slot_ptr, j, ok):
    return tl.load(req_slot_ptr + tl.load(tok_req_ptr + j, mask=ok, other=0), mask=ok, other=-1).to(tl.int64)


@triton.jit
def engram_hash_kernel(
    ids_ptr,  # [n] token ids (int32 / int64)
    pos_ptr,  # [n] int32 absolute positions
    tok_req_ptr,  # [n] int64 request row of each token
    req_slot_ptr,  # [R] int32 ring slot of each request row
    token_map_ptr,  # [V] int64 compressed vocabulary
    hist_ptr,  # [S + 1, RING] int32 look-back ring (read only here)
    mults_ptr,  # [L, MAX_N] int64
    primes_ptr,  # [L, MAX_N - 1, H] int64
    out_ptr,  # [L, n, (MAX_N - 1) * H] int64 (layer-major: each layer's ids are one contiguous [n, cols] block)
    n,
    vocab,
    pad_id,
    RING: tl.constexpr,
    MAX_N: tl.constexpr,
    L: tl.constexpr,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)  # (layer, order - 1, head)
    head = pid % H
    order = (pid // H) % (MAX_N - 1) + 1
    layer = pid // (H * (MAX_N - 1))
    r = tl.arange(0, BLOCK)
    ok = r < n
    comp = _compressed(ids_ptr, token_map_ptr, r, ok, vocab)
    pos = tl.load(pos_ptr + r, mask=ok, other=0).to(tl.int64)
    slot = _slot(tok_req_ptr, req_slot_ptr, r, ok)
    rolling = comp * tl.load(mults_ptr + layer * MAX_N)
    for back in tl.static_range(1, MAX_N):
        if back <= order:
            # the step's own row when it is the same request's previous position, else the ring; pad before 0
            pb = pos - back
            j = tl.maximum(r - back, 0)
            inside = (r >= back) & (_slot(tok_req_ptr, req_slot_ptr, j, ok) == slot) & (
                tl.load(pos_ptr + j, mask=ok, other=-1).to(tl.int64) == pb)
            cj = _compressed(ids_ptr, token_map_ptr, j, ok & inside, vocab)
            ring = tl.load(hist_ptr + slot * RING + _floor_mod(pb, RING), mask=ok & (pb >= 0) & ~inside, other=0).to(tl.int64)
            v = tl.where(pb < 0, pad_id, tl.where(inside, cj, ring))
            rolling = rolling ^ (v * tl.load(mults_ptr + layer * MAX_N + back))
    p = tl.load(primes_ptr + (layer * (MAX_N - 1) + (order - 1)) * H + head)
    COLS: tl.constexpr = (MAX_N - 1) * H
    tl.store(out_ptr + (layer * n + r) * COLS + (order - 1) * H + head, _floor_mod(rolling, p), mask=ok)


@triton.jit
def engram_ring_write_kernel(
    ids_ptr, pos_ptr, tok_req_ptr, req_slot_ptr, token_map_ptr, hist_ptr, n, vocab, dummy,
    RING: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Only each request's last RING tokens of the step are written (the others go to the dummy slot)."""
    r = tl.arange(0, BLOCK)
    ok = r < n
    comp = _compressed(ids_ptr, token_map_ptr, r, ok, vocab)
    pos = tl.load(pos_ptr + r, mask=ok, other=0).to(tl.int64)
    slot = _slot(tok_req_ptr, req_slot_ptr, r, ok)
    later = tl.minimum(r + RING, tl.maximum(n - 1, 0))
    s_l = _slot(tok_req_ptr, req_slot_ptr, later, ok)
    p_l = tl.load(pos_ptr + later, mask=ok, other=-1).to(tl.int64)
    last = (r + RING >= n) | (s_l != slot) | (p_l != pos + RING)
    dst = tl.where(last, slot, dummy)
    tl.store(hist_ptr + dst * RING + _floor_mod(pos, RING), comp.to(tl.int32), mask=ok)

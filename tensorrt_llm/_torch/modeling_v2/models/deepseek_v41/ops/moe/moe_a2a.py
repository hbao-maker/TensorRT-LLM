# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The expert all-to-all over NVLink symmetric memory: the engine's one-sided `moe_comm` kernels, as stock's context
role runs them under attention DP (`NVLinkOneSided` with its "auto" sync protocol, no EPLB, no rank mask): the sentinel
transport (`moeA2ASentinelDispatchKernel`, `moeA2ASentinelCombineKernel`) for steps whose busiest rank has at most the
workspace's `sentinel_max_rows` rows, the fence transport (`moeA2ADispatchKernel`, `moeA2ASanitizeExpertIdsKernel`,
`moeA2ACombineKernel`) for larger ones.

Under attention DP every rank routes and quantizes its own rows. `moe_a2a_dispatch` puts each row into the
symmetric receive buffer of every rank that holds one of its experts (once per rank, not per expert), so a rank
receives `[ep_size, max_rows, ...]` rows of which the first `count[src]` per source are live; the rest have their
expert ids set to -1 (no expert); a sentinel step does not compact (row `src * max_rows + t` is token `t` of source
`src`, -1 ids where that token is not routed here). The experts run over the received rows (each rank its own window
of experts, non-local ids contribute nothing) and `moe_a2a_combine` sums every row's partial results back on the
rank that owns the row. With two ranks and the experts' output written straight into the workspace's combine region
(`moe_a2a_combine_payload`), `moe_a2a_pull_combine` replaces the native combine as stock does for large steps
(`pull_combine.py`): a handshake, then one kernel per row reads its two partials (the peer's over NVLink) and adds
them, optionally with an addend (the shared expert's output).

`moe_a2a_workspace` is a build-time helper of the owning target, not an entry: the **collective** allocation of
the symmetric workspace (every EP rank calls it, at build time, in the same order) and the layout every call
addresses. The target keeps the returned `MoeA2AWorkspace` and passes its fields to every call (layers run one
after another, so they share it). It also holds the outcome of stock's sentinel vote (`NVLinkOneSided._vote_sentinel`
under the default "auto" protocol): every rank votes whether its GPU runs the sentinel kernels for these rows and its SM
count, and where all can, the step limit is stock's `default_sentinel_max_tokens` and the sentinel regions are appended
to the layout. Dispatch and combine take that limit and pick the transport from the step's `max_rows`, which every rank
passes alike, so every rank takes the same one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import triton
import triton.language as tl

import tensorrt_llm._torch.custom_ops  # noqa: F401 -- registers torch.ops.trtllm.*
from tensorrt_llm.bindings import internal as _tllm_internal
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

INVALID_EXPERT = -1  # the id a dead receive row carries: `moe_sort` routes it nowhere
_ALIGN = 1024  # per-payload slack of the receive region (the op aligns each payload's start to a cache line)
_FLAG_BYTES = 256  # the pull combine's handshake flag, past the native layout in the same allocation
# stock's floor (PULL_COMBINE_MIN_TOKENS): steps whose busiest rank has fewer rows combine natively
PULL_MIN_ROWS = 4096
_SENTINEL_FORMAT = "NVFP4"  # the sentinel payload format of the rows this family dispatches (stock's for NVFP4 layers)
_SF_VEC = 16  # NVFP4 elements per scale byte


def _pad(nbytes: int) -> int:
    return -(-nbytes // _ALIGN) * _ALIGN


@dataclass
class MoeA2AWorkspace:
    """The symmetric workspace `[ep_size, bytes_per_rank]` uint8, its layout (`metainfo`), the MNNVL allocation
    that backs it (kept alive with it), the per-rank row capacity it was sized for, and the largest step the sentinel
    transport carries (`sentinel_max_rows`, 0 where the vote turned it off)."""

    workspace: torch.Tensor
    metainfo: torch.Tensor
    memory: object
    ep_rank: int
    ep_size: int
    top_k: int
    max_rows: int
    sentinel_max_rows: int
    # the pull combine's operands: (flag-value offset, target-ranks offset, send-indices offset, handshake flag offset)
    # into a rank's slice, and this rank's handshake epoch (int32 [1] on the device, so a graph can replay it)
    offsets: Tuple[int, int, int, int] = (0, 0, 0, 0)
    epoch: Optional[torch.Tensor] = None


def _sentinel_vote(mapping, metainfo: torch.Tensor, max_rows: int, top_k: int, hidden: int, num_experts: int) -> int:
    """Stock's sentinel vote (`NVLinkOneSided._vote_sentinel`, "auto" protocol) for NVFP4 rows of `hidden`: the step
    limit, with the sentinel regions appended to `metainfo`, or 0 (fence only) when a rank cannot run the sentinel
    kernels. The reasons are stock's `_sentinel_unsupported_reason` for this family's fixed settings (BF16 expert
    outputs, no FP8 combine, no watchdog or rank mask). Collective: every rank votes."""
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
        default_sentinel_max_tokens,
        sentinel_row,
    )

    thop = _tllm_internal.thop
    ep = mapping.moe_ep_size
    ok = bool(torch.ops.trtllm.moe_a2a_sentinel_available()) and ep <= int(thop.MOE_A2A_SENTINEL_MAX_RANKS)
    ok = ok and num_experts % ep == 0
    row = None
    if ok:
        try:
            row = sentinel_row(_SENTINEL_FORMAT, hidden, top_k)
        except RuntimeError:  # the native row check: the kernels do not take this hidden size
            ok = False
    sms = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count if ok else None
    votes = MnnvlMemory.get_comm(mapping).allgather((ok, sms))
    if not all(v[0] for v in votes):
        return 0
    limit = min(default_sentinel_max_tokens(ep, min(v[1] for v in votes), row), max_rows)
    torch.ops.trtllm.moe_a2a_sentinel_extend_workspace_layout(
        metainfo, limit, int(getattr(thop, f"MOE_A2A_SENTINEL_FORMAT_{_SENTINEL_FORMAT}")), hidden
    )
    return limit


def moe_a2a_workspace(
    mapping, max_rows: int, top_k: int, row_bytes: Sequence[int], combine_row_bytes: int, hidden: int,
    num_experts: int,
) -> MoeA2AWorkspace:
    """Allocate and initialize the workspace for dispatches of payloads of `row_bytes` bytes per row each (NVFP4 rows
    of `hidden` and their scales first, then ids and weights) and combines of `combine_row_bytes` per row, at most
    `max_rows` rows per rank, with the sentinel regions where every rank votes for them (`_sentinel_vote`).

    **Collective** over the expert-parallel group: the vote gathers every rank's, the MNNVL allocation exchanges
    handles and the initialization ends in a barrier, so every rank calls it once, at build time, never inside a
    forward or a graph capture.
    """
    from tensorrt_llm._torch.distributed.mnnvl_memory import MnnvlMemory

    assert list(row_bytes[:2]) == [hidden // 2, hidden // _SF_VEC], "NVFP4 rows of `hidden` and their scales first"
    ep, rank = mapping.moe_ep_size, mapping.moe_ep_rank
    rows = ep * max_rows
    dispatch = sum(_pad(rows * b) for b in row_bytes)
    combine = _pad(rows * combine_row_bytes)
    metainfo = torch.ops.trtllm.moe_a2a_get_workspace_layout(ep, max_rows, top_k, dispatch, combine, None)
    MnnvlMemory.initialize()
    sentinel_max_rows = _sentinel_vote(mapping, metainfo, max_rows, top_k, hidden, num_experts)
    thop = _tllm_internal.thop
    size = int(metainfo[int(thop.MOE_A2A_WORKSPACE_SIZE_INDEX)])
    flag = _pad(size)
    memory = MnnvlMemory(mapping, flag + _FLAG_BYTES)
    workspace = memory.as_torch_strided_tensor(torch.uint8)
    workspace[rank, flag : flag + _FLAG_BYTES].zero_()
    torch.ops.trtllm.moe_a2a_initialize(workspace, metainfo, rank, ep)  # ends in a barrier: both flags are zero now
    offsets = tuple(int(metainfo[int(getattr(thop, f"MOE_A2A_{k}_INDEX"))]) for k in (
        "FLAG_VAL_OFFSET", "TOPK_TARGET_RANKS_OFFSET", "TOPK_TARGET_INDICES_OFFSET"))  # fmt: skip
    epoch = torch.zeros(1, dtype=torch.int32, device=workspace.device)
    return MoeA2AWorkspace(workspace, metainfo, memory, rank, ep, top_k, max_rows, sentinel_max_rows, (*offsets, flag),
                           epoch)


def moe_a2a_combine_payload(workspace: torch.Tensor, metainfo: torch.Tensor, ep_rank: int, ep_size: int, max_rows: int,
                            hidden: int) -> torch.Tensor:
    """The combine region of this rank's slice as `[ep_size * max_rows, hidden]` bf16: the experts write their output
    here (`cute_dsl_moe_nvfp4(..., out=)`), so neither combine stages it. Valid after this step's dispatch."""
    return torch.ops.trtllm.moe_a2a_get_combine_payload_tensor(
        workspace, metainfo, ep_rank, ep_size, max_rows, torch.bfloat16, hidden
    ).view(ep_size * max_rows, hidden)


class MoeA2ADispatch(OpWrapper):
    """`(ids [n, top_k] int32, payloads [n, *] each, ...)` -> `(received payloads [ep_size * max_rows, *] each, combine
    offset)`.

    Row `t` of this rank goes to every rank whose expert window `[r * E / ep, (r + 1) * E / ep)` holds one of
    `ids[t]` (an id of -1 goes nowhere); every payload of a row travels with it. `max_rows` is the largest row count of
    any rank this step (the same on every rank), and it picks the transport as stock's `use_sentinel_for_dispatch`
    does: the sentinel one up to `sentinel_max_rows` (the workspace's), else the fence one.

    - Fence: the rows from source `s` are packed into `[s * max_rows, s * max_rows + count)` in an unspecified order,
      and the expert ids of the dead rows past `count` are set to -1 (`moe_a2a_sanitize_expert_ids`).
    - Sentinel: row `s * max_rows + t` holds token `t` of source `s`; a row whose token is not routed here (or that
      `s` does not have) has -1 in every id slot. NVFP4 codes arrive canonicalized (a -0 nibble as +0, the same value).

    Either way the experts skip the rows with -1 ids. The returned tensors are views of the workspace, valid until the
    next dispatch; the offset goes to `moe_a2a_combine`, which must get the same `max_rows` and `sentinel_max_rows`.

    `payloads` must include `ids` itself (at `ids_index`): it is the id payload that is received and sanitized. A
    sentinel step needs the payloads in stock's order and format: NVFP4 codes `[n, H / 2]` and scales `[n, H / 16]`
    (bytes), `ids` at index 2, then 4-byte or bf16 weights `[n, top_k]` (the workspace's sentinel layout).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="dep2 context step: EP2, 2048 rows per rank, top-6 of 384, NVFP4 rows (H 5120), scales, ids, weights",
             spec=dict(ep=2, rows=(2048, 2048), top_k=6, experts=384, hidden=5120)),
        Cell(why="dep2 uneven ranks (the decoder half of a bounded replay): 300 and 1 rows",
             spec=dict(ep=2, rows=(300, 1), top_k=6, experts=384, hidden=5120)),
        Cell(why="dep2 rows whose experts all sit on one rank, and rows routed nowhere (-1)",
             spec=dict(ep=2, rows=(64, 64), top_k=6, experts=384, hidden=5120, skew=True)),
        Cell(why="dep2 decoder pass graph, one prompt's 140-row replay window per rank (the bucket both ranks replay)",
             spec=dict(ep=2, rows=(140, 140), top_k=6, experts=384, hidden=5120)),
        Cell(why="dep2 decoder pass graph, three windows per rank (420-row bucket)",
             spec=dict(ep=2, rows=(420, 420), top_k=6, experts=384, hidden=5120, skew=True)),
    )
    note = """
    Two transports, as stock's "auto" protocol. Fence (steps past `sentinel_max_rows`): the dispatch kernel writes
    into the peers' receive buffers and publishes a completion flag behind a system fence; the sanitize kernel waits on
    the received counts. Sentinel (smaller steps): one dispatch and one combine kernel; every 16-byte vector carries its
    own arrival (a receiver polls the data for the armed sentinel word), so neither waits for a flag or a fence, and the
    dispatch fills the ids of rows without a token itself. A sentinel dispatch must be followed by its combine (the
    native op checks it). Under a CUDA graph the host arguments (row counts) are baked in, as for any collective. The op
    allocates nothing and syncs nothing on the host.
    """

    def __call__(self, ids: torch.Tensor, payloads: List[torch.Tensor], ids_index: int, workspace: torch.Tensor,
                 metainfo: torch.Tensor, max_rows: int, ep_rank: int, ep_size: int, num_experts: int,
                 sentinel_max_rows: int) -> Tuple[List[torch.Tensor], int]:
        top_k = ids.shape[1]
        sentinel = max_rows <= sentinel_max_rows
        recv, combine_offset, _ = torch.ops.trtllm.moe_a2a_dispatch(
            ids, payloads, workspace, metainfo, max_rows, ep_rank, ep_size, top_k, num_experts,
            eplb_local_stats=None, expert_id_payload_index=ids_index if sentinel else None,
            invalid_token_expert_id=INVALID_EXPERT if sentinel else None,
            enable_rank_mask=False, active_rank_mask=None, use_sentinel=sentinel,
        )  # fmt: skip
        if not sentinel:
            torch.ops.trtllm.moe_a2a_sanitize_expert_ids(recv[ids_index], workspace, metainfo, ep_rank, INVALID_EXPERT)
        return [t.view(ep_size * max_rows, t.shape[-1]) for t in recv], int(combine_offset)

    def reference(self, all_ids: Sequence[torch.Tensor], all_payloads: Sequence[Sequence[torch.Tensor]], ids_index: int,
                  ep_rank: int, ep_size: int, num_experts: int) -> List[List[torch.Tensor]]:
        """What this rank receives, per source rank: the payload rows (in source order) of the rows that hold an
        expert of this rank's window. The collective exception (see `OpWrapper.reference`): it takes every rank's
        `ids` and `payloads` in rank order. The op's order within a source is unspecified, so a test compares the
        received live rows to these as sets (on a sentinel step a live row also sits at its token's index)."""
        lo = (num_experts * ep_rank) // ep_size
        hi = (num_experts * (ep_rank + 1)) // ep_size
        out = []
        for ids, payloads in zip(all_ids, all_payloads):
            keep = ((ids >= lo) & (ids < hi)).any(dim=1)
            out.append([p[keep] for p in payloads])
        return out

    def is_valid(self, ids: torch.Tensor, payloads: List[torch.Tensor], ids_index: int, workspace: torch.Tensor,
                 metainfo: torch.Tensor, max_rows: int, ep_rank: int, ep_size: int, num_experts: int,
                 sentinel_max_rows: int) -> None:
        n, top_k = ids.shape
        assert ids.dtype == torch.int32 and ids.is_contiguous()
        assert payloads[ids_index] is ids, "the id payload is the one the sanitize pass rewrites"
        assert 0 < n <= max_rows, "every rank dispatches at least one row (a dummy row routed nowhere if it has none)"
        for p in payloads:
            assert p.dim() == 2 and p.shape[0] == n and p.is_contiguous() and p.data_ptr() % 16 == 0
        assert num_experts % ep_size == 0, "this target's windows are equal (the kernel itself takes uneven ones)"
        regions = int(metainfo[int(_tllm_internal.thop.MOE_A2A_SENTINEL_MAX_NUM_TOKENS_INDEX)])
        assert 0 <= sentinel_max_rows <= regions, (
            "the workspace's sentinel limit (`MoeA2AWorkspace.sentinel_max_rows`), or 0 for the fence transport only"
        )
        if max_rows <= sentinel_max_rows:  # stock's `use_sentinel_for_dispatch` row checks
            xq, xsf, w = payloads[0], payloads[1], payloads[3] if len(payloads) > 3 else None
            hidden = 2 * xq.shape[1]
            assert ids_index == 2 and len(payloads) in (3, 4), "codes, scales, ids, weights (stock's order)"
            assert xq.element_size() == 1 and xsf.element_size() == 1 and xsf.shape[1] == hidden // _SF_VEC
            assert w is None or (w.shape[1] == top_k and (w.element_size() == 4 or w.dtype == torch.bfloat16))


moe_a2a_dispatch = MoeA2ADispatch()


class MoeA2ACombine(OpWrapper):
    """`partial [ep_size * max_rows, H]` bf16 (the experts' output over the received rows) -> `[n, H]` bf16: row `t`
    of this rank is the sum of the partial results its row got on every rank it was dispatched to (zero for a row
    routed nowhere). `n`, `max_rows`, `combine_offset` and `sentinel_max_rows` are the matching dispatch's: they pick
    the same transport. The partial rows of rows the experts skipped (-1 ids) are not read."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = MoeA2ADispatch.CELLS
    note = """
    Fence: the partials are first staged into the combine region of the workspace unless they already are that region
    (`in_workspace`, `moe_a2a_combine_payload`), then each rank reads its rows' partials from every peer and sums
    them in fp32, rounding once to bf16. Sentinel (after a sentinel dispatch): each rank pushes the partials of the rows
    it received into their owners' landing rows, and each owner polls its rows' partials and sums them in the fence
    combine's order (fp32, rounded once), so both transports give the same bits. Allocates its output only and syncs
    nothing on the host.
    """

    def __call__(self, partial: torch.Tensor, n: int, workspace: torch.Tensor, metainfo: torch.Tensor, max_rows: int,
                 ep_rank: int, ep_size: int, top_k: int, combine_offset: int, sentinel_max_rows: int,
                 in_workspace: bool = False) -> torch.Tensor:
        return torch.ops.trtllm.moe_a2a_combine(
            partial.view(ep_size, max_rows, partial.shape[-1]), n, workspace, metainfo, max_rows, ep_rank, ep_size,
            top_k, combine_offset, in_workspace, use_low_precision=False, enable_rank_mask=False,
            active_rank_mask=None, use_sentinel=max_rows <= sentinel_max_rows,
        )  # fmt: skip

    def reference(self, all_contributions: Sequence[torch.Tensor]) -> torch.Tensor:
        """The fp32 sum of the partial rows this rank's rows got on each rank (zeros where a row was not sent), in
        rank order, rounded once. The collective exception: a test regenerates every rank's contribution."""
        acc = torch.zeros(all_contributions[0].shape, dtype=torch.float32, device=all_contributions[0].device)
        for c in all_contributions:
            acc += c.float()
        return acc.to(torch.bfloat16)

    def is_valid(self, partial: torch.Tensor, n: int, workspace: torch.Tensor, metainfo: torch.Tensor, max_rows: int,
                 ep_rank: int, ep_size: int, top_k: int, combine_offset: int, sentinel_max_rows: int,
                 in_workspace: bool = False) -> None:
        assert partial.dtype == torch.bfloat16 and partial.is_contiguous()
        assert partial.shape[0] == ep_size * max_rows, "the dispatch's receive layout"
        assert 0 < n <= max_rows
        own = workspace[ep_rank, combine_offset:].data_ptr()
        assert (partial.data_ptr() == own) == in_workspace, "in_workspace says whether partial is the combine region"


moe_a2a_combine = MoeA2ACombine()


@triton.jit
def _handshake_kernel(epoch_ptr, peer_flag_ptr, own_flag_ptr):
    # Both ranks run the same sequence of handshakes; this rank's count is the epoch. The system release publishes
    # this rank's earlier work (its experts' rows) before the flag; the acquire after the wait orders the following
    # reads of the peer's rows after the peer's experts.
    epoch = tl.load(epoch_ptr) + 1
    tl.atomic_xchg(peer_flag_ptr, epoch, sem="release", scope="sys")
    seen = tl.load(own_flag_ptr, volatile=True)
    while seen < epoch:
        seen = tl.load(own_flag_ptr, volatile=True)
    tl.atomic_add(own_flag_ptr, 0, sem="acquire", scope="sys")
    tl.store(epoch_ptr, epoch)


@triton.jit
def _pull_reduce_kernel(own_ptr, peer_ptr, ranks_ptr, indices_ptr, addend_ptr, out_ptr, hidden, ep_rank,
                        TOP_K: tl.constexpr, BLOCK: tl.constexpr, HAS_ADDEND: tl.constexpr):
    # A duplicate target rank carries send index -1, so each rank holds at most one valid slot of a row.
    token = tl.program_id(0).to(tl.int64)
    own_row = -1
    peer_row = -1
    for k in tl.static_range(TOP_K):
        rank = tl.load(ranks_ptr + token * TOP_K + k)
        index = tl.load(indices_ptr + token * TOP_K + k)
        own_row = tl.where((index >= 0) & (rank == ep_rank), index, own_row)
        peer_row = tl.where((index >= 0) & (rank != ep_rank), index, peer_row)
    own_base = own_ptr + tl.maximum(own_row, 0).to(tl.int64) * hidden
    peer_base = peer_ptr + tl.maximum(peer_row, 0).to(tl.int64) * hidden
    for start in range(0, hidden, BLOCK):
        offsets = start + tl.arange(0, BLOCK)
        mask = offsets < hidden
        own = tl.load(own_base + offsets, mask=mask & (own_row >= 0), other=0.0).to(tl.float32)
        peer = tl.load(peer_base + offsets, mask=mask & (peer_row >= 0), other=0.0).to(tl.float32)
        out = (own + peer).to(tl.bfloat16)
        if HAS_ADDEND:
            shared = tl.load(addend_ptr + token * hidden + offsets, mask=mask, other=0.0)
            out = (shared.to(tl.float32) + out.to(tl.float32)).to(tl.bfloat16)
        tl.store(out_ptr + token * hidden + offsets, out, mask=mask)


class MoeA2APullCombine(OpWrapper):
    """`partial [2 * max_rows, H]` bf16, the experts' output written in this rank's combine region
    (`moe_a2a_combine_payload`) -> `[n, H]` bf16: row `t` of this rank is `bf16(own + peer)` of its two partial
    results (the one it got here and the one it got on the peer, read over NVLink; zero where it was not sent),
    plus `addend[t]` with the rounding of a separate bf16 add when an addend is given. Two ranks only. Stock's
    pull combine (`pull_combine.py`: `_handshake_kernel` + `_pull_reduce_kernel`).

    `offsets` and `epoch` are the workspace's (`MoeA2AWorkspace`); `n`, `max_rows`, `combine_offset` the matching
    dispatch's, which took the fence transport (its floor is past any sentinel step, as stock's
    `pull_combine_applies`). Both ranks must call it in the same order (the handshake counts calls).
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="dep2 context step at stock's floor: EP2, 4096 rows per rank, top-6 of 384, H 5120, shared addend",
             spec=dict(ep=2, rows=(4096, 4096), top_k=6, experts=384, hidden=5120, addend=True)),
        Cell(why="dep2 uneven ranks past the floor: 6000 and 4100 rows, no addend",
             spec=dict(ep=2, rows=(6000, 4100), top_k=6, experts=384, hidden=5120, addend=False)),
        Cell(why="dep2 rows whose experts all sit on one rank, and rows routed nowhere (-1)",
             spec=dict(ep=2, rows=(4096, 4096), top_k=6, experts=384, hidden=5120, skew=True, addend=True)),
    )
    note = """
    The handshake (one single-warp kernel) raises the peer's flag with a system-scope release after this rank's
    experts and waits for its own; the epoch lives on the device, so the pair is graph capturable. The flag-value
    word of the native layout is advanced as the native prepare-combine would, so native dispatch / combine rounds
    that follow keep their parity. The sum is fp32, rounded once to bf16, the bits of the native combine (which also
    sums a row's partials in fp32 with zeros for absent ones); with an addend the result is the bits of the native
    combine followed by a bf16 add. Allocates its output only and syncs nothing on the host.
    """

    def __call__(self, partial: torch.Tensor, n: int, workspace: torch.Tensor, max_rows: int, ep_rank: int,
                 top_k: int, combine_offset: int, offsets: Tuple[int, int, int, int], epoch: torch.Tensor,
                 addend: Optional[torch.Tensor] = None) -> torch.Tensor:
        flag_val, ranks_at, indices_at, flag_at = offsets
        hidden = partial.shape[-1]
        workspace[ep_rank, flag_val : flag_val + 4].view(torch.int32).add_(1)  # the native prepare-combine's round step
        flag = lambda r: workspace[r, flag_at : flag_at + 4].view(torch.int32)  # noqa: E731
        _handshake_kernel[(1,)](epoch, flag(1 - ep_rank), flag(ep_rank), num_warps=1)
        nbytes = 2 * max_rows * hidden * 2
        peer = workspace[1 - ep_rank, combine_offset : combine_offset + nbytes].view(torch.bfloat16)
        peer = peer.view(2, max_rows, hidden)[ep_rank]
        own = partial.view(2, max_rows, hidden)[ep_rank]
        nmeta = n * top_k * 4
        meta = [workspace[ep_rank, at : at + nmeta].view(torch.int32).view(n, top_k) for at in (ranks_at, indices_at)]
        out = torch.empty(n, hidden, dtype=torch.bfloat16, device=partial.device)
        _pull_reduce_kernel[(n,)](
            own, peer, *meta, addend if addend is not None else out, out, hidden, ep_rank,
            TOP_K=top_k, BLOCK=1024, HAS_ADDEND=addend is not None, num_warps=4,
        )  # fmt: skip
        return out

    def reference(self, all_contributions: Sequence[torch.Tensor],
                  addend: Optional[torch.Tensor] = None) -> torch.Tensor:
        """`moe_a2a_combine`'s reference (the fp32 sum of the rows' partials, rounded once), then the addend in a
        separate bf16 add. The collective exception, as `moe_a2a_combine`."""
        routed = moe_a2a_combine.reference(all_contributions)
        return routed if addend is None else (routed.float() + addend.float()).to(torch.bfloat16)

    def is_valid(self, partial: torch.Tensor, n: int, workspace: torch.Tensor, max_rows: int, ep_rank: int,
                 top_k: int, combine_offset: int, offsets: Tuple[int, int, int, int], epoch: torch.Tensor,
                 addend: Optional[torch.Tensor] = None) -> None:
        assert workspace.shape[0] == 2, "two ranks: a row has at most its own and the peer's partial"
        assert partial.dtype == torch.bfloat16 and partial.shape[0] == 2 * max_rows and partial.is_contiguous()
        assert partial.data_ptr() == workspace[ep_rank, combine_offset:].data_ptr(), (
            "the peer reads this rank's partials at the same offset: they must be its combine region"
        )
        assert 0 < n <= max_rows and epoch.dtype == torch.int32 and epoch.is_cuda
        assert max_rows > int(_tllm_internal.thop.MOE_A2A_SENTINEL_MAX_TOKENS_PER_RANK), (
            "a step the sentinel transport could carry ends in its own combine (stock's `pull_combine_applies`)"
        )
        if addend is not None:
            assert addend.shape == (n, partial.shape[-1]) and addend.dtype == torch.bfloat16 and addend.is_contiguous()


moe_a2a_pull_combine = MoeA2APullCombine()

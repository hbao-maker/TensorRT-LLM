# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU certification matrix for the `moe_a2a` entries on two GPUs (EP2, attention-DP rows).

The kernels write into the peer's symmetric memory over NVLink, so they cannot run in one process: this script is
its own launcher. Run it plainly and it re-executes itself under `mpirun`, one rank per device named in
CUDA_VISIBLE_DEVICES, under a deadline the parent enforces by killing the process group (a wedged completion flag
hangs rather than raising).

    CUDA_VISIBLE_DEVICES=0,1 python _moe_a2a_op_matrix.py

Covered: `moe_a2a_workspace` (the collective allocation), `moe_a2a_dispatch` and `moe_a2a_combine` over every cell of
`MoeA2ADispatch.CELLS` / `MoeA2ACombine.CELLS`, `moe_a2a_pull_combine` over `MoeA2APullCombine.CELLS` (and the native
combine of the same partials written in the workspace, interleaved with it), and the dep2 target's composition (route-quantized rows -> dispatch ->
`cute_dsl_moe_nvfp4` -> combine) against the all-gather / reduce-scatter path it replaces. The collected entry point
is `test_modeling_v2_moe_a2a.py`.

Every rank regenerates every rank's inputs from the cell's seed, so the references need no exchange. Each cell runs on
the transport its rows pick (the sentinel one up to the workspace's `sentinel_max_rows`, the fence one past it), and
two more cells sit at the limit and one past it. Dispatch: the live rows a rank receives from each source are the
source's rows holding one of its experts, every payload travelling with its row (as a set on a fence step, at the
token's index on a sentinel step, whose NVFP4 codes arrive with -0 nibbles as +0), and the other rows carry ids of -1.
Combine: the partial of a received row is an exact function of the row's e2m1 values and the rank, so the sum is
checked bitwise.
"""

import os
import signal
import subprocess
import sys
import zlib
from pathlib import Path

import torch

_WORKER_FLAG = "--rank-worker"
DEADLINE_S = 1800
WORLD = 2
MAX_ROWS = 8192
HIDDEN, TOP_K, EXPERTS = 5120, 6, 384
RANK = 0


def _say(msg: str) -> None:
    if RANK == 0:
        print(f"[a2a] {msg}", flush=True)


def _gen(cell_why: str, rank: int) -> torch.Generator:
    return torch.Generator(device="cuda").manual_seed((zlib.crc32(cell_why.encode()) + 7919 * rank) % 2**31)


def _inputs(spec: dict, why: str, src: int):
    """Rank `src`'s rows of a cell: NVFP4-shaped bytes, scale bytes, ids, bf16 weights (a pure function of the cell)."""
    gen = _gen(why, src)
    n, e = spec["rows"][src], spec["experts"]
    ids = torch.argsort(torch.rand(n, e, generator=gen, device="cuda"), dim=1)[:, : spec["top_k"]].to(torch.int32)
    if spec.get("skew"):
        half = e // 2
        ids[0::4] = ids[0::4] % half  # all on rank 0's window
        ids[1::4] = ids[1::4] % half + half  # all on rank 1's window
        ids[2::8] = -1  # routed nowhere
    xq = torch.randint(0, 256, (n, spec["hidden"] // 2), generator=gen, device="cuda", dtype=torch.uint8)
    xq[:, :8] = _key_bytes(n, src)  # row id
    xsf = torch.randint(0, 256, (n, spec["hidden"] // 16), generator=gen, device="cuda", dtype=torch.uint8)
    w = torch.rand(n, spec["top_k"], generator=gen, device="cuda").to(torch.bfloat16)
    return [xq, xsf, ids, w]


def _key_bytes(n: int, src: int) -> torch.Tensor:
    """8 bytes naming row `t` of rank `src`, three bits per nibble: no -0 nibble (1000b), which the sentinel transport
    would rewrite, so the key arrives as sent on both transports."""
    v = torch.arange(n, device="cuda", dtype=torch.int64) + (src << 40)
    nib = torch.stack([(v >> (3 * k)) & 7 for k in range(16)], dim=1)
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).to(torch.uint8)


def _row_key(xq: torch.Tensor) -> torch.Tensor:
    return xq[:, :8].contiguous().view(torch.int64).view(-1)


def _canonical(xq: torch.Tensor) -> torch.Tensor:
    """NVFP4 codes as the sentinel transport delivers them: every -0 nibble (1000b) as +0, the same e2m1 value."""
    lo, hi = xq & 0x0F, xq >> 4
    return torch.where(lo == 8, 0, lo) | (torch.where(hi == 8, 0, hi) << 4)


_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _partial(rows: torch.Tensor, rank: int) -> torch.Tensor:
    """An exact bf16 function of a received row's e2m1 values and the rank: `(rank + 1) * e2m1(low nibble)` of each
    byte, over both halves. A function of the values, not the bits: -0 and +0 give the same (+0)."""
    code = rows.to(torch.int64) & 0x0F
    mag = torch.tensor(_E2M1, device=rows.device)[code & 7]
    v = torch.where(code >= 8, -mag, mag) * (rank + 1) + 0.0  # + 0.0: -0 -> +0
    return torch.cat([v, v], dim=1).to(torch.bfloat16)


def _live(recv_ids: torch.Tensor, src: int, rows: int, src_ids: torch.Tensor, sentinel: bool):
    """Where source `src`'s live rows sit in the receive buffer (indices into it), in the order of the reference's rows
    (sentinel), or the first `count` (fence, packed in an unspecified order)."""
    lo, hi = EXPERTS * RANK // WORLD, EXPERTS * (RANK + 1) // WORLD
    keep = ((src_ids >= lo) & (src_ids < hi)).any(dim=1)
    if sentinel:
        return src * rows + keep.nonzero().squeeze(1)
    return src * rows + torch.arange(int(keep.sum()), device="cuda")


def check_cells(M) -> None:
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import INVALID_EXPERT

    disp, comb, ws = M["dispatch"], M["combine"], M["ws"]
    assert comb.CELLS == disp.CELLS
    limit = ws.sentinel_max_rows
    assert limit > 0, "the sentinel vote turned the transport off on these GPUs: the matrix covers both transports"
    spec = dict(ep=2, top_k=TOP_K, experts=EXPERTS, hidden=HIDDEN, skew=True)
    cells = [(c.why, c.spec) for c in disp.CELLS] + [
        ("at the sentinel limit", dict(spec, rows=(limit, limit - 9))),
        ("one past the sentinel limit (fence)", dict(spec, rows=(limit + 1, 3))),
    ]
    seen = set()
    for why, s in cells:
        allin = [_inputs(s, why, r) for r in range(WORLD)]
        mine = allin[RANK]
        rows = max(max(1, c) for c in s["rows"])
        sentinel = rows <= limit
        seen.add(sentinel)
        _say(f"{why} {s['rows']}: {'sentinel' if sentinel else 'fence'}")
        disp.is_valid(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"], limit)
        recv, offset = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"], limit)
        expected = disp.reference([a[2] for a in allin], allin, 2, RANK, WORLD, s["experts"])
        live_rows = []
        for src in range(WORLD):
            want = list(expected[src])
            if sentinel:
                want[0] = _canonical(want[0])
            at = _live(recv[2], src, rows, allin[src][2], sentinel)
            live_rows.append(at)
            got = [t[at] for t in recv]
            order = slice(None) if sentinel else torch.argsort(_row_key(got[0]))
            worder = slice(None) if sentinel else torch.argsort(_row_key(want[0]))
            for g, w_ in zip(got, want):
                assert torch.equal(g[order], w_[worder]), f"dispatch: payload rows from rank {src} differ"
            dead = torch.ones(rows, dtype=torch.bool, device="cuda")
            dead[at - src * rows] = False
            assert bool((recv[2][src * rows : (src + 1) * rows][dead] == INVALID_EXPERT).all()), (
                "dispatch: a dead row kept expert ids"
            )
        # combine: every received live row carries an exact partial; dead rows carry garbage the combine must skip
        partial = torch.full((WORLD * rows, HIDDEN), 1.0e4, dtype=torch.bfloat16, device="cuda")
        for at in live_rows:
            partial[at] = _partial(recv[0][at], RANK)
        n = mine[0].shape[0]
        comb.is_valid(partial, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, offset, limit)
        out = comb(partial, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, offset, limit)
        contributions = []
        for r in range(WORLD):
            lo, hi = EXPERTS * r // WORLD, EXPERTS * (r + 1) // WORLD
            sent = ((mine[2] >= lo) & (mine[2] < hi)).any(dim=1, keepdim=True)
            contributions.append(torch.where(sent, _partial(mine[0], r).float(), 0.0).to(torch.bfloat16))
        ref = comb.reference(contributions)
        assert out.shape == ref.shape and torch.equal(out, ref), "combine: sums differ"
        torch.cuda.synchronize()
    assert seen == {True, False}, "the cells cover both transports"


def _fill_region(M, s, why, rows, mine, recv, addend_on):
    """Dispatch done: the partials of the live received rows written into this rank's combine region (dead rows carry
    garbage), and the reference of the rows' combined sums (+ an exact bf16 addend)."""
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import moe_a2a_combine_payload

    ws, disp, comb = M["ws"], M["dispatch"], M["combine"]
    allin = [_inputs(s, why, r) for r in range(WORLD)]
    expected = disp.reference([a[2] for a in allin], allin, 2, RANK, WORLD, s["experts"])
    region = moe_a2a_combine_payload(ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN)
    region.fill_(1.0e4)
    for src in range(WORLD):
        c = expected[src][0].shape[0]
        region[src * rows : src * rows + c] = _partial(recv[0][src * rows : src * rows + c], RANK)
    contributions = []
    for r in range(WORLD):
        lo, hi = EXPERTS * r // WORLD, EXPERTS * (r + 1) // WORLD
        sent = ((mine[2] >= lo) & (mine[2] < hi)).any(dim=1, keepdim=True)
        contributions.append(torch.where(sent, _partial(mine[0], r).float(), 0.0).to(torch.bfloat16))
    n = mine[0].shape[0]
    addend = None
    if addend_on:
        addend = (torch.arange(n * HIDDEN, device="cuda").view(n, HIDDEN) % 13 - 6).to(torch.bfloat16) / 8
    return region, contributions, addend


def check_pull_combine(M) -> None:
    """Every pull cell: the pull combine and, on a second round, the native combine of partials already in the
    workspace, bitwise against the reference (the two kinds of round interleave, as in the target)."""
    ws, disp, comb, pull = M["ws"], M["dispatch"], M["combine"], M["pull"]
    for cell in pull.CELLS:
        s = cell.spec
        _say(cell.why)
        mine = _inputs(s, cell.why, RANK)
        rows, n = max(s["rows"]), mine[0].shape[0]
        recv, off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"],
                         ws.sentinel_max_rows)
        region, contributions, addend = _fill_region(M, s, cell.why, rows, mine, recv, s["addend"])
        args = (region, n, ws.workspace, rows, RANK, TOP_K, off, ws.offsets, ws.epoch, addend)
        pull.is_valid(*args)
        out = pull(*args)
        ref = pull.reference(contributions, addend)
        assert torch.equal(out, ref), f"pull combine differs: {cell.why}"
        recv, off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"],
                         ws.sentinel_max_rows)
        region, contributions, _ = _fill_region(M, s, cell.why, rows, mine, recv, False)
        comb.is_valid(region, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, off, ws.sentinel_max_rows, True)
        native = comb(region, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, off, ws.sentinel_max_rows, True)
        assert torch.equal(native, comb.reference(contributions)), f"in-workspace native combine differs: {cell.why}"
        torch.cuda.synchronize()
    # a captured dispatch + experts-in-place + pull replays the eager result
    cell = pull.CELLS[1]
    s = cell.spec
    mine = _inputs(s, cell.why, RANK)
    rows, n = max(s["rows"]), mine[0].shape[0]

    def body():
        from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import moe_a2a_combine_payload

        recv, off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"],
                         ws.sentinel_max_rows)
        region = moe_a2a_combine_payload(ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN)
        region[:, : HIDDEN // 2] = (recv[0].float() % 16).to(torch.bfloat16)
        region[:, HIDDEN // 2 :] = (recv[0].float() % 16).to(torch.bfloat16)
        return pull(region, n, ws.workspace, rows, RANK, TOP_K, off, ws.offsets, ws.epoch, None)

    eager = body().clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = body()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager), "pull combine: graph replay differs from the eager result"


def check_target_composition(M) -> None:
    """The dep2 target's experts over the all-to-all equal the all-gather / reduce-scatter path, row for row: a large
    step (fence transport, pull combine with the shared expert) and a decoder-pass bucket (sentinel transport, whose
    rows are not compacted, native combine)."""
    from mpi4py import MPI

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.cute_dsl_moe_nvfp4 import (
        cute_dsl_moe_layout,
        cute_dsl_moe_nvfp4,
    )

    ws, disp, comb = M["ws"], M["dispatch"], M["combine"]
    e_local, inter = EXPERTS // WORLD, 2304
    gen = torch.Generator(device="cuda").manual_seed(1234 + RANK)

    def nvfp4(rows_, cols):
        codes = torch.randint(0, 256, (e_local, rows_, cols // 2), generator=gen, device="cuda", dtype=torch.uint8)
        exps = torch.randint(-3, 1, (e_local, rows_, cols // 16), generator=gen, device="cuda").float()
        return codes, torch.exp2(exps).to(torch.float8_e4m3fn).view(torch.uint8)

    up, gate, down = nvfp4(inter, HIDDEN), nvfp4(inter, HIDDEN), nvfp4(HIDDEN, inter)
    lay = [cute_dsl_moe_layout(up[0][i], up[1][i], gate[0][i], gate[1][i], down[0][i], down[1][i])
           for i in range(e_local)]  # fmt: skip
    fc1_w, fc1_s, fc2_w, fc2_s = (torch.stack([l[j] for l in lay]) for j in range(4))
    del lay
    a1 = torch.full((e_local,), 0.02, device="cuda")
    a2 = torch.full((e_local,), 0.01, device="cuda")
    g2 = torch.tensor([50.0], device="cuda")
    offset_e = RANK * e_local
    def step_inputs(spec, why):
        allin = [_inputs(spec, why, r) for r in range(WORLD)]
        for src, a in enumerate(allin):  # plausible NVFP4 scales (e4m3 powers of two), the same on every rank
            g = _gen(why + " scales", src)
            e = torch.randint(-2, 1, a[1].shape, generator=g, device="cuda").float()
            a[1].copy_(torch.exp2(e).to(torch.float8_e4m3fn).view(torch.uint8))
        return allin

    spec = dict(rows=(4500, 2000), top_k=TOP_K, experts=EXPERTS, hidden=HIDDEN)
    allin = step_inputs(spec, "target composition")
    mine = allin[RANK]
    n = mine[0].shape[0]
    rows = max(spec["rows"])
    assert rows > ws.sentinel_max_rows

    def experts(xq, xsf, ids, w):
        return cute_dsl_moe_nvfp4(xq, xsf, ids, w, fc1_w, fc1_s, fc2_w, fc2_s, a1, a2, g2, 10.0, EXPERTS, offset_e)

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.moe_a2a import moe_a2a_combine_payload

    def experts_into(xq, xsf, ids, w, out):
        return cute_dsl_moe_nvfp4(xq, xsf, ids, w, fc1_w, fc1_s, fc2_w, fc2_s, a1, a2, g2, 10.0, EXPERTS, offset_e, out=out)

    shared = torch.randn(n, HIDDEN, generator=gen, device="cuda").bfloat16()
    # the target's path: experts written into the combine region, pulled with the shared expert added
    (rq, rsf, rids, rw), off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, EXPERTS,
                                    ws.sentinel_max_rows)
    region = moe_a2a_combine_payload(ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN)
    partial = experts_into(rq, rsf, rids, rw, region)
    out = M["pull"](partial, n, ws.workspace, rows, RANK, TOP_K, off, ws.offsets, ws.epoch, shared)
    # the path it replaces: this rank's experts over every rank's rows, partial sums exchanged and added
    full = experts(*(torch.cat([a[i] for a in allin]) for i in range(4)))
    parts = MPI.COMM_WORLD.allgather(full.float().cpu())
    start = sum(spec["rows"][:RANK])
    ref = sum(p[start : start + n] for p in parts).to(torch.bfloat16).cuda()
    ref = (ref.float() + shared.float()).to(torch.bfloat16)
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
    _say(f"composition: relative RMS vs all-gather path {float(err):.2e}")
    assert float(err) < 4e-3, f"all-to-all composition differs from the all-gather path: {float(err):.3e}"

    # a decoder-pass bucket on the sentinel transport: every rank replays 140 rows (131 live on rank 1)
    spec = dict(rows=(140, 131), top_k=TOP_K, experts=EXPERTS, hidden=HIDDEN)
    allin = step_inputs(spec, "target composition, sentinel")
    mine = allin[RANK]
    n, rows = mine[0].shape[0], max(spec["rows"])
    assert rows <= ws.sentinel_max_rows
    (rq, rsf, rids, rw), off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, EXPERTS,
                                    ws.sentinel_max_rows)
    region = moe_a2a_combine_payload(ws.workspace, ws.metainfo, RANK, WORLD, rows, HIDDEN)
    partial = experts_into(rq, rsf, rids, rw, region)
    out = comb(partial, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, off, ws.sentinel_max_rows, True)
    # the reference sees the codes as the sentinel transport delivers them (-0 nibbles as +0, the same values)
    full = experts(_canonical(torch.cat([a[0] for a in allin])), *(torch.cat([a[i] for a in allin]) for i in (1, 2, 3)))
    parts = MPI.COMM_WORLD.allgather(full.float().cpu())
    start = sum(spec["rows"][:RANK])
    ref = sum(p[start : start + n] for p in parts).to(torch.bfloat16).cuda()
    assert torch.isfinite(out.float()).all()
    err = (out.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)
    _say(f"composition (sentinel, 140 rows): relative RMS vs all-gather path {float(err):.2e}")
    assert float(err) < 4e-3, f"sentinel all-to-all composition differs from the all-gather path: {float(err):.3e}"


def check_graph_replay(M) -> None:
    """Captured dispatch + combine rounds replay the eager results (a collective under a graph: same counts): a
    sentinel graph (300 / 1 rows) and a fence graph (2048 rows) on the same workspace, replayed in turn, as the dep2
    decoder- and encoder-pass graphs are."""
    ws, disp, comb = M["ws"], M["dispatch"], M["combine"]
    limit = ws.sentinel_max_rows

    def body_of(cell):
        s = cell.spec
        mine = _inputs(s, cell.why, RANK)
        rows = max(max(1, c) for c in s["rows"])
        n = mine[0].shape[0]

        def body():
            recv, off = disp(mine[2], mine, 2, ws.workspace, ws.metainfo, rows, RANK, WORLD, s["experts"], limit)
            partial = torch.zeros(WORLD * rows, HIDDEN, dtype=torch.bfloat16, device="cuda")
            partial[:, : HIDDEN // 2] = (recv[0].float() % 16).to(torch.bfloat16)
            partial[:, HIDDEN // 2 :] = (recv[0].float() % 16).to(torch.bfloat16)
            return comb(partial, n, ws.workspace, ws.metainfo, rows, RANK, WORLD, TOP_K, off, limit)

        return body, rows <= limit

    graphs = []
    for cell in (disp.CELLS[1], disp.CELLS[0]):
        body, sentinel = body_of(cell)
        eager = body().clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = body()
        graphs.append((graph, captured, eager, sentinel, body))  # `body` keeps the graph's inputs alive
    assert [g[3] for g in graphs] == [True, False], "one sentinel and one fence graph"
    for _ in range(3):
        for graph, captured, eager, sentinel, _ in graphs:
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(captured, eager), (
                f"{'sentinel' if sentinel else 'fence'} graph replay differs from the eager result"
            )


def _run_one_rank() -> int:
    global RANK
    from mpi4py import MPI

    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe import moe_a2a
    from tensorrt_llm.mapping import Mapping

    comm = MPI.COMM_WORLD
    RANK = comm.Get_rank()
    assert comm.Get_size() == WORLD, f"this matrix is written for {WORLD} ranks"
    torch.cuda.set_device(RANK)
    mapping = Mapping(world_size=WORLD, rank=RANK, gpus_per_node=WORLD, tp_size=WORLD, moe_ep_size=WORLD,
                      enable_attention_dp=True)  # fmt: skip
    h, k = HIDDEN, TOP_K
    ws = moe_a2a.moe_a2a_workspace(mapping, MAX_ROWS, k, [h // 2, h // 16, 4 * k, 2 * k], 2 * h, h, EXPERTS)
    _say(f"sentinel transport up to {ws.sentinel_max_rows} rows per rank")
    M = dict(ws=ws, dispatch=moe_a2a.moe_a2a_dispatch, combine=moe_a2a.moe_a2a_combine, pull=moe_a2a.moe_a2a_pull_combine)
    for check in (check_cells, check_pull_combine, check_target_composition, check_graph_replay):
        try:
            _say(check.__name__)
            check(M)
            comm.Barrier()
        except BaseException:
            import traceback

            print(f"[rank {RANK}] FAILED {check.__name__}", flush=True)
            traceback.print_exc()
            sys.stdout.flush()
            sys.stderr.flush()
            comm.Abort(1)
    print(f"[rank {RANK}] all checks passed", flush=True)
    return 0


def _spawn_ranks() -> None:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    assert visible, "set CUDA_VISIBLE_DEVICES to the two devices this run owns, e.g. 0,1"
    n = len([d for d in visible.split(",") if d.strip()])
    assert n == WORLD, f"this matrix runs on {WORLD} devices; CUDA_VISIBLE_DEVICES names {n}"
    command = ["mpirun", "--oversubscribe", "--allow-run-as-root", "-n", str(WORLD), sys.executable,
               str(Path(__file__).resolve()), _WORKER_FLAG]  # fmt: skip
    print(f"[launcher] {' '.join(command)}", flush=True)
    process = subprocess.Popen(command, start_new_session=True)
    try:
        code = process.wait(timeout=DEADLINE_S)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()
        raise AssertionError(f"the {WORLD}-rank run did not finish in {DEADLINE_S}s (wedged)") from None
    assert code == 0, f"the {WORLD}-rank run exited {code}"


if __name__ == "__main__":
    if _WORKER_FLAG in sys.argv:
        sys.exit(_run_one_rank())
    _spawn_ranks()
    print("OK")

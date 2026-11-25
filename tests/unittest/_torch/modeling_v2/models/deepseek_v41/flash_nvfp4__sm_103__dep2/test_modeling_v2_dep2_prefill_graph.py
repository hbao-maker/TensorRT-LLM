# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2 prefill CUDA graph's host side: which steps run as the padded static body, and its per-token inputs."""

import importlib
from types import SimpleNamespace as NS

import numpy as np
import torch
import pytest

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling")


def _core(enabled=True):
    return NS(_prefill_graph=enabled)


def _rows(counts, enabled=True):
    return modeling.ModelingV2Core._graph_rows(_core(enabled), NS(all_rank_num_tokens=counts))


@pytest.mark.parametrize(
    "counts,want",
    [([1, 0], 128), ([128, 5], 128), ([129, 0], 256), ([2000, 2048], 2048), ([4096, 1], 4096), ([4097, 0], None),
     ([8192, 8192], None), ([0, 0], None), (None, None)],
)
def test_bucket_follows_the_largest_rank(counts, want):
    assert _rows(counts) == want


@pytest.mark.parametrize("counts", [[16, 16], [1, 1], [8, 0], [128, 5]])
def test_cuda_graph_steps_never_take_the_static_body(counts):
    """A step on the engine's CUDA-graph metadata (a generation graph's warm-up runs and its capture) runs the forward
    the graph records. Before, the warm-up runs of a decode graph (rows within a prefill bucket) took the static body
    while the capture did not, so the decode indexer's self-sampling top-k first ran under capture, where it refuses
    to build its launcher ("varlen launcher not compiled for this shape"): engine start failed with CUDA graphs on."""
    assert _rows(counts) is not None  # an eager step of these counts still takes the static body
    graph_step = NS(all_rank_num_tokens=counts, is_cuda_graph=True)
    assert modeling.ModelingV2Core._graph_rows(_core(), graph_step) is None


def test_switch_off():
    assert _rows([100, 100], enabled=False) is None


def test_static_inputs_of_two_requests_and_padding():
    out = modeling._static_host_rows([3, 2], [10, 0], [8, 0], 8)
    assert out.dtype == np.int32 and out.shape == (7, 8)
    assert out[0].tolist() == [10, 11, 12, 0, 1, 1, 1, 1]  # positions; padding repeats the last row
    assert out[1].tolist() == [0, 0, 0, 1, 1, 1, 1, 1]  # request rows
    assert out[2].tolist() == [10, 10, 10, 0, 0, 0, 0, 0]  # first position in the step
    assert out[3, :5].tolist() == [13, 13, 13, 2, 2] and (out[3, 5:] == 1 << 30).all()  # one past the last; padding keeps no window row
    assert out[4].tolist() == [8, 8, 8, 0, 0, 0, 0, 0]  # SWA floors
    assert out[5].tolist() == [1, 1, 1, 1, 1, 0, 0, 0]  # live rows
    assert (out[6] == 0).all()  # no Encoder recovery: no Encoder SWA floor


def test_static_inputs_of_an_encoder_replay():
    """Request 0 replays its Encoder from position 6 (Global end 10, then a 3-row chunk): its rows below 10 publish no
    compressed rows and its Encoder SWA reads start at 6; request 1 has no recovery."""
    out = modeling._static_host_rows([7, 2], [6, 0], [6, 0], 12, recovery=([10, 0], [6, 0]))
    assert out.shape == (7, 12)
    assert out[0, :9].tolist() == [6, 7, 8, 9, 10, 11, 12, 0, 1]
    assert out[5].tolist() == [0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0]  # historical rows and padding publish nothing
    assert out[6, :9].tolist() == [6] * 7 + [0, 0]  # Encoder SWA floor per row

def test_static_inputs_of_an_idle_rank():
    out = modeling._static_host_rows([], [], None, 4)
    assert (out[[0, 1, 2, 4, 5, 6]] == 0).all() and (out[3] == 1 << 30).all()


def test_too_many_rows_rejected():
    with pytest.raises(AssertionError):
        modeling._static_host_rows([5], [0], None, 4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
@pytest.mark.parametrize("bucket", [8192, 32768, None])
def test_precapture_walks_every_bucket_alike(bucket):
    """`_precapture_static`: every rank runs every (rows, span) bucket once eagerly in the same order (whatever its own
    span, so the experts' collectives line up), and captures the span buckets that cover its own step; the captures
    replay against the persistent inputs."""
    src = torch.arange(8, dtype=torch.float32, device="cuda")
    runs = []

    def body(kvm, r, span, ncap, world):
        if not torch.cuda.is_current_stream_capturing():
            runs.append((r, span))
        return src * r + span

    core = NS(_graphs={}, _graph_pool=None, _static_body=body)
    modeling.ModelingV2Core._precapture_static(core, None, bucket, 64, 2, True)
    walk = [(r, b) for r in reversed(modeling._GRAPH_ROWS) for b in reversed(modeling._GRAPH_SPANS)]
    assert [r for r, _ in runs] == [r for r, _ in walk]  # the same eager sequence on every rank
    covered = [(r, b) for r, b in walk if bucket is not None and b >= bucket]
    assert sorted(core._graphs) == sorted((r, b, 64) for r, b in covered)
    assert all(span == (b if (r, b) in covered else (bucket or b)) for (r, b), (_, span) in zip(walk, runs))
    if covered:
        r, b = covered[0]
        src.copy_(torch.arange(8, 16, dtype=torch.float32, device="cuda"))
        graph, out = core._graphs[(r, b, 64)]
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, src * r + b)


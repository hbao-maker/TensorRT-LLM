# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the `moe_a2a` entries' 2-GPU certification matrix.

The matrix itself is `_moe_a2a_op_matrix.py` beside this file, which is its own 2-rank launcher (the dispatch and
combine kernels write into the peer's memory over NVLink, so they cannot run in one process). It covers
`moe_a2a_workspace`, `moe_a2a_dispatch`, `moe_a2a_combine` and `moe_a2a_pull_combine` (every cell of their `CELLS`),
the dep2 target's composition (experts written into the combine region by `cute_dsl_moe_nvfp4`, pulled with the shared
expert added) against the all-gather path it replaces, and CUDA-graph replays.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.device_count() < 2
    or torch.cuda.get_device_capability() != (10, 3),
    reason="the moe_a2a matrix needs 2 SM103 GPUs (EP2 over NVLink)",
)

_LAUNCHER = Path(__file__).resolve().with_name("_moe_a2a_op_matrix.py")
_WORLD = 2


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= _WORLD, f"the matrix needs {_WORLD} devices; {len(devices)} visible"
    return ",".join(devices[:_WORLD])


# Starts its own 2-rank mpirun: it must run alone, not under xdist.
@pytest.mark.no_xdist
def test_moe_a2a_op_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ under {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    # The launcher must not import tensorrt_llm (MPI_Init); it is started in a fresh interpreter.
    done = subprocess.run([sys.executable, str(_LAUNCHER)], env=env, capture_output=True, text=True, timeout=2400)
    assert done.returncode == 0, (
        f"{_LAUNCHER.name} exited {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    )

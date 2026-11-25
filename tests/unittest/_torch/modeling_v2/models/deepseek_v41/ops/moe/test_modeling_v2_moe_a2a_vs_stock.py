# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for dep2's expert all-to-all against the engine's `NVLinkOneSided` on 2 GPUs.

The check itself is `_moe_a2a_vs_stock.py` beside this file, its own 2-rank launcher: the same dispatch input through
`moe_a2a_dispatch` / `moe_a2a_combine` and through `NVLinkOneSided`, both taking the sentinel transport for steps up to
the voted limit (`sentinel_max_rows` / `sentinel_max_tokens`, which must agree) and the fence one past it, and through
ours on the fence transport at every step; the combined rows must be bitwise equal. It also prints the kernel time per
round of ours on each transport (CUDA graph).
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
    reason="the all-to-all comparison needs 2 SM103 GPUs (EP2 over NVLink)",
)

_LAUNCHER = Path(__file__).resolve().with_name("_moe_a2a_vs_stock.py")
_WORLD = 2


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= _WORLD, f"the check needs {_WORLD} devices; {len(devices)} visible"
    return ",".join(devices[:_WORLD])


# Starts its own 2-rank mpirun: it must run alone, not under xdist.
@pytest.mark.no_xdist
def test_moe_a2a_vs_stock() -> None:
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ under {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    # The launcher must not import tensorrt_llm (MPI_Init); it is started in a fresh interpreter.
    done = subprocess.run([sys.executable, str(_LAUNCHER)], env=env, capture_output=True, text=True, timeout=1200)
    assert done.returncode == 0, (
        f"{_LAUNCHER.name} exited {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    )

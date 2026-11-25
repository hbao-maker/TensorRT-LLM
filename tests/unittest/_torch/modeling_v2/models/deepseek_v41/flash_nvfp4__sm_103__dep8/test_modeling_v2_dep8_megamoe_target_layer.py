# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the generation targets' MegaMoE layer matrix (4 GPUs, EP4).

`_megamoe_target_layer_matrix.py` beside this file is its own 4-rank launcher; it drives the dep8 and the dep32 target's
`megamoe.py` and `weights.py` end to end for one layer against a plain-torch reference (see its docstring).
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="the MegaMoE layer matrix needs 4 GPUs (EP4 over NVLink)"
)

_LAUNCHER = Path(__file__).resolve().parents[1] / "_megamoe_target_layer_matrix.py"


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= 4, f"the matrix needs 4 devices; {len(devices)} visible"
    return ",".join(devices[:4])


# Starts its own 4-rank mpirun over four devices: it must run alone, not under xdist.
@pytest.mark.no_xdist
@pytest.mark.parametrize("target", ["dep8"])
def test_megamoe_target_layer_matrix(target) -> None:
    repo_root = Path(__file__).resolve().parents[7]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ under {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices(), TARGET=target)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    # The launcher must not import tensorrt_llm (MPI_Init); it is started in a fresh interpreter.
    done = subprocess.run([sys.executable, str(_LAUNCHER)], env=env, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, (
        f"{_LAUNCHER.name} exited {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    )

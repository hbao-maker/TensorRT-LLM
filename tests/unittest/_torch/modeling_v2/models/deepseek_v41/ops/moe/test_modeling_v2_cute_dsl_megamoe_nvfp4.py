# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collected entry point for the MegaMoE entries' 4-GPU certification matrix.

The matrix itself is `_cute_dsl_megamoe_nvfp4_op_matrix.py` beside this file, which is its own
4-rank launcher (the kernel dispatches tokens over NVLink, so it cannot run in one process). It
covers `cute_dsl_megamoe_nvfp4` (every cell of its `CELLS`) and `megamoe_workspace`, and runs the other `megamoe_*` entries as
the composition a target uses.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 4, reason="the MegaMoE matrix needs 4 GPUs (EP4 over NVLink)"
)

_LAUNCHER = Path(__file__).resolve().with_name("_cute_dsl_megamoe_nvfp4_op_matrix.py")
_WORLD = 4


def _devices() -> str:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    devices = (
        [d for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(torch.cuda.device_count())]
    )
    assert len(devices) >= _WORLD, f"the matrix needs {_WORLD} devices; {len(devices)} visible"
    return ",".join(devices[:_WORLD])


# Starts its own 4-rank mpirun over every visible device: it must run alone, not under xdist.
@pytest.mark.no_xdist
def test_cute_dsl_megamoe_nvfp4_op_matrix() -> None:
    repo_root = Path(__file__).resolve().parents[8]
    assert (repo_root / "tensorrt_llm").is_dir(), f"no tensorrt_llm/ under {repo_root}"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=_devices())
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo_root), env.get("PYTHONPATH", "")) if p)
    # The launcher must not import tensorrt_llm (MPI_Init); it is started in a fresh interpreter.
    done = subprocess.run([sys.executable, str(_LAUNCHER)], env=env, capture_output=True, text=True, timeout=3600)
    assert done.returncode == 0, (
        f"{_LAUNCHER.name} exited {done.returncode}\n--- stdout ---\n{done.stdout}\n--- stderr ---\n{done.stderr}"
    )

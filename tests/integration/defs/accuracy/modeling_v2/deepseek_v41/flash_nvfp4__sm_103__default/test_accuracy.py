# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Whole-model gate for the deepseek_v41 modeling_v2 target flash_nvfp4__sm_103__default; the harness is in ``../_common.py``."""

import pytest

from tensorrt_llm.llmapi import CudaGraphConfig, DSparkDecodingConfig, KvCacheConfig, MoeConfig

from .._common import _DeepseekV41Gate, skip_not_sm103

# id -> (world, attention DP, moe tp, moe ep): routing's `_DEFAULT_LAYOUTS`. Four ranks fit one GB300 tray, eight span
# two; dep6 takes six ranks of two trays and dep16 four trays, so those two are run by name (no CI stage has them).
_LAYOUTS = {
    "dep4": (4, True, 1, 4),
    "tp4_ep4": (4, False, 1, 4),
    "tp4_moetp4": (4, False, 4, 1),
    "tp4_moetp2_ep2": (4, False, 2, 2),
    "dep6": (6, True, 1, 6),
    "tp8_ep8": (8, False, 1, 8),
    "tp8_moetp8": (8, False, 8, 1),
    "tp8_moetp2_ep4": (8, False, 2, 4),
    "dep16": (16, True, 1, 16),
}


def _ranks(world: int):
    # one tray: its GPUs; more: the MPI world the multi-node launcher gives
    return pytest.mark.skip_less_device(world) if world <= 4 else pytest.mark.skip_less_mpi_world_size(world)


class TestModelingV2DeepseekV41FlashNvfp4Sm103Default(_DeepseekV41Gate):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / default: the layouts no layout-specific target takes, sharded from the mapping.

    One aggregated engine per gated layout over the full GSM8K split (chunked prefill and CUDA-graph decode), with DSpark
    (lossless, so the floor is the same row as without it; the speculative run proves the draft reads the right hidden
    states). The attention-DP layouts split the LM head over the ranks as the deployed generation roles do; the
    attention-TP layouts run it column-parallel.
    """

    @skip_not_sm103
    @pytest.mark.timeout(7200)
    @pytest.mark.parametrize("layout", [pytest.param(k, marks=_ranks(v[0])) for k, v in _LAYOUTS.items()])
    def test_gsm8k(self, mocker, layout):
        world, adp, moe_tp, moe_ep = _LAYOUTS[layout]
        self._gsm8k(
            mocker,
            tensor_parallel_size=world,
            moe_expert_parallel_size=moe_ep,
            moe_tensor_parallel_size=moe_tp,
            enable_attention_dp=adp,
            enable_lm_head_tp_in_adp=adp,
            moe_config=MoeConfig(backend="TRTLLM", use_low_precision_moe_combine=True),
            cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16], enable_padding=True),
            max_batch_size=16,
            max_num_tokens=8192,
            max_seq_len=8192,
            kv_cache_config=KvCacheConfig(
                dtype="fp8",
                tokens_per_block=128,
                enable_block_reuse=False,
                free_gpu_memory_fraction=0.6,
            ),
            speculative_config=DSparkDecodingConfig(max_draft_len=3, block_size=3, speculative_model=self.MODEL_PATH),
        )

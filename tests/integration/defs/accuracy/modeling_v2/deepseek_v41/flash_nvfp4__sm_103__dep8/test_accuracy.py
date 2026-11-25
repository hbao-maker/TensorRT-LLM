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
"""Whole-model gate for the deepseek_v41 modeling_v2 target flash_nvfp4__sm_103__dep8; the harness is in ``../_common.py``."""

import pytest

from .._common import _AttentionDpGenerationGate, skip_not_sm103


class TestModelingV2DeepseekV41FlashNvfp4Sm103Dep8(_AttentionDpGenerationGate):
    """deepseek-v4.1-flash-nvfp4 / sm_103 / dep8: TP8 + attention DP + MoE EP8, DSpark, lm_head TP in ADP (gen role).

    Two GB300 trays (8 ranks). With and without speculation, like tep8: one floor, and the speculative run
    proves the draft reads the right hidden states. The deployed batch is 320 with DSpark 3/3; the gate runs
    the small batch the other gates use.
    """

    WORLD = 8
    DRAFT_LEN = 3

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.timeout(7200)
    @pytest.mark.parametrize("dspark", [False, True], ids=["no_spec", "dspark"])
    def test_gsm8k(self, mocker, dspark):
        self._run(mocker, dspark)

    @skip_not_sm103
    @pytest.mark.skip_less_mpi_world_size(8)
    @pytest.mark.timeout(7200)
    def test_mmmu(self):
        """Image + text requests through the image class (see ``../_common.py``)."""
        self._run_mmmu()

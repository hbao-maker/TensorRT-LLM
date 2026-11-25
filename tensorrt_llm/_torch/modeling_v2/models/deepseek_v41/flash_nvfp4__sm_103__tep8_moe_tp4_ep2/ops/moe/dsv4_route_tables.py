# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Decode-sized routed experts on the TRTLLM-Gen runner without its routing kernel, as stock's TRTLLM-Gen NVFP4 MoE
does (`moe/fused_moe/trtllm_gen/trtllm_nvfp4.py::try_fused_route_quant`): the routing + NVFP4 quantization launch also
writes the runner's routing tables (`with_routing_tables`, this rank's expert window), and the runner takes them
instead of routing (`routing_tables` / `routing_tiles` of `torch.ops.trtllm.fp4_block_scale_moe_runner`).

Three entries: `dsv4_route_nvfp4_quant_tables` (the launch), `dsv4_route_tables` (the routing and tables alone, for rows
the FFN boundary already quantized: stock's `try_fused_route`) and `fp4_moe_runner_tables` (the runner on the tables)."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modeling_v2.catalog.moe.fp4_block_scale_moe_runner import fp4_block_scale_moe_runner
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route_nvfp4_quant,
)

ROUTING_TABLES_MAX_TOKENS = 8  # the engine's `dsv4_route_quant.ROUTING_TABLES_MAX_TOKENS`
ROUTING_TABLE_TILES = (8, 16, 32)  # the engine's `dsv4_route_quant.ROUTING_TABLE_TILES`


_TABLE_ROW_ALIGN = 4  # int32 entries: every table row starts 16 bytes aligned


def _tables(num_tokens: int, top_k: int, local_count: int, total_experts: int, tiles: Sequence[int], device) -> torch.Tensor:
    """The launch's `[len(tiles), 5, width]` tables buffer, `width` the engine's (`_routing_tables`: the widest route map,
    read one past its padded rows) rounded up to 4 entries: an unfinalized run's permutation is row 0 of its tile's table
    (`table_permutation`), and the boundary kernel that finalizes it reads it as a 16-byte aligned tensor. The engine's
    width alone puts tile i's rows at `i * 5 * width * 4` bytes, which a width not a multiple of 4 leaves unaligned for
    every tile but the first (the launch and the runner take any width: compact `[tiles, 5, width]`)."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import _max_ctas

    width = max([num_tokens * top_k] + [_max_ctas(num_tokens, top_k, local_count, t) * t + 1 for t in tiles])
    width = -(-width // _TABLE_ROW_ALIGN) * _TABLE_ROW_ALIGN
    return torch.empty((len(tiles), 5, width), dtype=torch.int32, device=device)


def _route_launch(logits, selection_bias, x, global_scale, top_k, routed_scaling_factor, local_first, local_count, tiles,
                  hidden=None):
    """The engine's routing launch with tables (`dsv4_route_nvfp4_quant` with `x`, else `dsv4_route`), bit for bit the
    same arguments to the same compiled kernel, on the 16-byte aligned tables buffer of `_tables`."""
    from tensorrt_llm._torch.moe.kernels.blackwell import dsv4_route_quant as rq

    if not routing_tables_fit(logits.shape[0], local_count, logits.shape[1]) or not set(tiles) <= set(ROUTING_TABLE_TILES):
        raise ValueError(f"no routing tables for {logits.shape[0]} tokens over {local_count} experts, tiles {tiles}")
    n, experts = logits.shape
    dev = logits.device
    tiles = tuple(tiles)
    ids = torch.empty((n, top_k), dtype=torch.int32, device=dev)
    weights = torch.empty((n, top_k), dtype=torch.bfloat16, device=dev)
    local_start, num_local = rq._local_range((local_first, local_count), experts)
    tables = _tables(n, top_k, num_local, experts, tiles, dev)
    if x is None:
        rq._compile(
            experts, top_k, routed_scaling_factor, hidden, torch.bfloat16, tiles, quant=False, num_local=num_local,
            pdl=rq.TRTLLM_ENABLE_PDL, cta_scoring=rq._cta_scoring(n, experts),
        )(logits, selection_bias, None, None, ids, weights, None, None, tables, local_start,
          None, None, None, None, None, None, 0, None, None)  # fmt: skip
        return ids, weights, tables
    hidden = x.shape[1]
    x_fp4 = torch.empty((n, hidden // 8), dtype=torch.int32, device=dev)
    x_sf = torch.empty((n, hidden // 16), dtype=torch.uint8, device=dev)
    rq._compile(
        experts, top_k, routed_scaling_factor, hidden, torch.bfloat16, tiles, None, False, False, False, False, False,
        num_local=num_local, cta_scoring=rq._cta_scoring(n, experts), splits=1,
    )(logits, selection_bias, x, global_scale.view(1), ids, weights, x_fp4, x_sf, tables, local_start,
      None, None, None, None, None, None, 0, None, None)  # fmt: skip
    return ids, weights, x_fp4.view(torch.uint8), x_sf, tables


def routing_tables_fit(num_tokens: int, local_experts: int, total_experts: int) -> bool:
    """Whether the launch can write tables for this shape (a setup-free predicate; shapes only)."""
    from tensorrt_llm._torch.moe.kernels.blackwell.dsv4_route_quant import can_build_routing_tables

    return can_build_routing_tables(num_tokens, local_experts, total_experts)


class Dsv4RouteNvfp4QuantTables(OpWrapper):
    """`dsv4_route_nvfp4_quant` (bit for bit: ids `[n, k]` int32 global expert ids, bf16 weights, NVFP4 rows and
    row-major scales) plus `tables [len(tiles), 5, width]` int32: the runner's routing tables for tiles `tiles` over
    the experts `[local_first, local_first + local_count)`, numbered from 0 (-1: another rank's expert)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: six rows, this EP rank's 192 of 384 experts, top-6", spec=dict(n=6, experts=384, first=192, count=192)),
        Cell(why="one row, the first EP window", spec=dict(n=1, experts=384, first=0, count=192)),
        Cell(why="eight rows: the launch's table limit", spec=dict(n=8, experts=384, first=192, count=192)),
    )
    note = "The tables' format is the runner's; the runner entry's test proves them by running the experts on them."

    def __call__(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _route_launch(logits, selection_bias, x, global_scale, top_k, routed_scaling_factor, local_first,
                             local_count, tiles)

    def reference(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ):
        """The family entry's routing and quantization; the tables have no plain-torch form (None)."""
        return (*dsv4_route_nvfp4_quant.reference(logits, selection_bias, x, global_scale, top_k, routed_scaling_factor), None)

    def is_valid(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        x: torch.Tensor,
        global_scale: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ) -> None:
        dsv4_route_nvfp4_quant.is_valid(logits, selection_bias, x, global_scale, top_k, routed_scaling_factor)
        n, experts = logits.shape
        assert routing_tables_fit(n, local_count, experts), "the launch writes tables for at most 8 rows, 256..1024 experts"
        assert 0 <= local_first and local_first + local_count <= experts
        assert tiles and set(tiles) <= set(ROUTING_TABLE_TILES)


class Dsv4RouteTables(OpWrapper):
    """`dsv4_route_nvfp4_quant_tables` without the quantization, `(ids [n, k] int32, bf16 weights, tables)`, for rows
    their producer already quantized (stock's `try_fused_route` after the FFN boundary wrote the NVFP4 copy). One CTA
    per launch, issued as a programmatic dependent of the router GEMM (stock's `dsv4_route`)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = Dsv4RouteNvfp4QuantTables.CELLS
    note = "Bit for bit the routing and tables of `dsv4_route_nvfp4_quant_tables` (the test compares them)."

    def __call__(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _route_launch(logits, selection_bias, None, None, top_k, routed_scaling_factor, local_first, local_count,
                             tiles, hidden)

    def reference(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ):
        """The family routing entry; the tables have no plain-torch form (None)."""
        from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import dsv4_route

        return (*dsv4_route.reference(logits, selection_bias, top_k, routed_scaling_factor, hidden), None)

    def is_valid(
        self,
        logits: torch.Tensor,
        selection_bias: torch.Tensor,
        top_k: int,
        routed_scaling_factor: float,
        hidden: int,
        local_first: int,
        local_count: int,
        tiles: Sequence[int],
    ) -> None:
        assert logits.dtype == torch.float32 and logits.dim() == 2 and logits.is_contiguous()
        assert selection_bias.dtype == torch.float32 and selection_bias.shape == (logits.shape[1],)
        n, experts = logits.shape
        assert 0 < top_k <= experts and hidden > 0
        assert routing_tables_fit(n, local_count, experts), "the launch writes tables for at most 8 rows, 256..1024 experts"
        assert 0 <= local_first and local_first + local_count <= experts
        assert tiles and set(tiles) <= set(ROUTING_TABLE_TILES)


class Fp4MoeRunnerTables(OpWrapper):
    """The TRTLLM-Gen NVFP4 MoE runner (the catalog's `fp4_block_scale_moe_runner`, same arguments) on routing tables:
    a tactic whose tile is in `routing_tiles` skips the runner's routing kernel and reads `routing_tables`; any other
    tactic routes itself from `topk_ids` (global ids). Unfinalized, a run on the tables returns an empty `[0, tile]`
    permutation: the permutation is row 0 of that tile's table (`table_permutation`).

    `routing_method_type` / `n_group` / `topk_group` / `routed_scaling_factor`: what the runner is told about the routing
    (the ids and weights are the caller's). The runner's own routing kernel (tactics off the tables) and the autotuner's
    dummy routing follow them, so a target passes what the engine's MoE passes for its model."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="tep8 decode: six rows over this rank's 128-expert window of 256, unfinalized", spec=dict(n=6, finalize=False)),
        Cell(why="six rows, finalized in the runner", spec=dict(n=6, finalize=True)),
        Cell(why="one row", spec=dict(n=1, finalize=False)),
    )
    note = "Same kernels and tactic as the runner's own routing: the test gates the outputs bit for bit against it."

    def __call__(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
    ) -> List[torch.Tensor]:
        return torch.ops.trtllm.fp4_block_scale_moe_runner(
            None, None, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, None, None, None,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, None, output1_scale_scalar, output1_scale_gate_scalar,
            output2_scale_scalar, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
            local_num_experts, routed_scaling_factor, routing_method_type, do_finalize, act_type, topk_weights=topk_weights,
            topk_ids=topk_ids, routing_tables=routing_tables, routing_tiles=list(routing_tiles),
        )  # fmt: skip

    def reference(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
    ) -> List[torch.Tensor]:
        """The catalog runner on the same operands, routing itself (the computation the tables must reproduce)."""
        return fp4_block_scale_moe_runner(
            None, None, hidden_states, hidden_states_scale, gemm1_weights, gemm1_weights_scale, None, None, None,
            gemm1_clamp_limit, gemm2_weights, gemm2_weights_scale, None, output1_scale_scalar, output1_scale_gate_scalar,
            output2_scale_scalar, num_experts, top_k, n_group, topk_group, intermediate_size, local_expert_offset,
            local_num_experts, routed_scaling_factor, routing_method_type, do_finalize, act_type, topk_weights=topk_weights,
            topk_ids=topk_ids,
        )  # fmt: skip

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm1_weights_scale: torch.Tensor,
        gemm1_clamp_limit: Optional[torch.Tensor],
        gemm2_weights: torch.Tensor,
        gemm2_weights_scale: torch.Tensor,
        output1_scale_scalar: torch.Tensor,
        output1_scale_gate_scalar: torch.Tensor,
        output2_scale_scalar: torch.Tensor,
        num_experts: int,
        top_k: int,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        do_finalize: bool,
        act_type: int,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        routing_tables: torch.Tensor,
        routing_tiles: List[int],
        n_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        routed_scaling_factor: Optional[float] = None,
    ) -> None:
        n = hidden_states.shape[0]
        assert topk_ids.shape == topk_weights.shape == (n, top_k) and topk_ids.dtype == torch.int32
        assert routing_tables.dtype == torch.int32 and routing_tables.dim() == 3 and routing_tables.shape[:2] == (
            len(routing_tiles), 5), "one [5, width] table per tile, from `dsv4_route_nvfp4_quant_tables` of this window"
        assert routing_tiles and set(routing_tiles) <= set(ROUTING_TABLE_TILES)


def table_permutation(outputs: List[torch.Tensor], tables: torch.Tensor, tiles: Sequence[int], n: int, top_k: int):
    """-> (the unfinalized run's `[n, top_k]` expanded -> permuted row index, the tile it ran on or None when the runner
    routed itself). A setup-free helper over the runner's outputs (no launch: a view of the table's head)."""
    perm = outputs[2]
    if perm.numel() == 0:
        tile = perm.shape[1]
        return tables[list(tiles).index(tile), 0][: n * top_k].view(n, top_k), tile
    return perm, None


dsv4_route_nvfp4_quant_tables = Dsv4RouteNvfp4QuantTables()
dsv4_route_tables = Dsv4RouteTables()
fp4_moe_runner_tables = Fp4MoeRunnerTables()

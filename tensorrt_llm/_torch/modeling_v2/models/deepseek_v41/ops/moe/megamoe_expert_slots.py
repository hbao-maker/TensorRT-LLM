# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expert placement for the MegaMoE kernel: which expert each slot holds, and the router's
expert ids mapped onto slots (plain torch).

`megamoe_experts_to_slots` is the entry (a device op of every forward, graph-safe). The placement
helpers run on the host when the target is built: `megamoe_default_placement` (the engine's placement when
nothing is configured: rank `r` holds `num_slots // ep_size` consecutive experts from `r * E // ep_size`,
wrapping modulo `E`, so with extra slots each rank also holds the first experts of the next rank's window),
`megamoe_check_placement` (every expert held, never twice on one rank) and `megamoe_slot_table` (the int32
device table the entry reads; rewrite it in place with `table.copy_` after moving experts so CUDA graphs that
captured it route to the new slots).
"""

from __future__ import annotations

from typing import List

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

_COPIES_SHIFT = 16
_HASH = 0x9E3779B1
_MAX_SLOTS = 1 << 16


def megamoe_default_placement(num_experts: int, num_slots: int, ep_size: int) -> List[int]:
    """Slot -> expert list of the engine's default (no learned statistics) placement.

    Rank `r` holds `num_slots // ep_size` consecutive experts starting at expert
    `r * num_experts // ep_size`, wrapping modulo `num_experts`. With `num_slots == num_experts` that
    is the contiguous one-copy split; with extra slots (`num_slots = num_experts + ep_size * x`) each
    rank also holds the first experts of the next rank's window, so every expert is held and no rank
    holds an expert twice.
    """
    if num_slots % ep_size or num_slots < num_experts:
        raise ValueError(
            f"num_slots {num_slots} must be a multiple of ep_size {ep_size} and >= {num_experts}"
        )
    local = num_slots // ep_size
    return [(r * num_experts // ep_size + j) % num_experts for r in range(ep_size) for j in range(local)]


def megamoe_check_placement(
    assignments: List[int], num_experts: int, ep_size: int
) -> None:
    """Raise unless `assignments` (slot -> expert) holds every expert, never twice on one rank."""
    if len(assignments) % ep_size:
        raise ValueError("number of slots must be a multiple of ep_size")
    if len(assignments) >= _MAX_SLOTS:
        raise ValueError("MegaMoE replica placement must hold every expert in < 65536 slots")
    if set(assignments) != set(range(num_experts)):
        raise ValueError("placement must hold every expert at least once")
    local = len(assignments) // ep_size
    for r in range(ep_size):
        mine = assignments[r * local : (r + 1) * local]
        if len(set(mine)) != len(mine):
            raise ValueError(f"rank {r} holds an expert twice")


def megamoe_slot_table(
    assignments: List[int], num_experts: int, ep_rank: int, device="cuda"
) -> torch.Tensor:
    """The int32 device table `megamoe_experts_to_slots` reads, for the placement `assignments`.

    One copy per expert (`len(assignments) == num_experts`): `table[e]` is `e`'s slot, `[num_experts]`.
    Copies: `num_experts` words `(copies << 16) | start` followed by every expert's slots, rotated by
    `ep_rank` so ranks start on different copies, `[num_experts + len(assignments)]`. Built on the host
    once; rewrite it in place (`table.copy_(megamoe_slot_table(...))`) when experts move, so CUDA
    graphs that captured it route to the new slots.
    """
    if len(assignments) == num_experts:
        table = [0] * num_experts
        for slot, expert in enumerate(assignments):
            table[expert] = slot
        return torch.tensor(table, dtype=torch.int32, device=device)
    slots_of: List[List[int]] = [[] for _ in range(num_experts)]
    for slot, expert in enumerate(assignments):
        slots_of[int(expert)].append(slot)
    meta, flat = [], []
    for slots in slots_of:
        if not slots or len(assignments) >= _MAX_SLOTS:
            raise ValueError("MegaMoE replica placement must hold every expert in < 65536 slots")
        k = ep_rank % len(slots)
        meta.append((len(slots) << _COPIES_SHIFT) | len(flat))
        flat.extend(slots[k:] + slots[:k])
    return torch.tensor(meta + flat, dtype=torch.int32, device=device)


class MegaMoeExpertsToSlots(OpWrapper):
    """Router expert ids `[rows, top_k]` -> slot ids `[rows, top_k]` int32, on the device.

    `table` is `megamoe_slot_table`'s. With copies, row `r`'s pick of an expert takes copy `h(r) % copies` with
    `h(r) = ((r * 0x9E3779B1) mod 2**32) >> 16`, so a hot expert's rows spread over its copies (all copies hold
    the same bytes). Pure device torch, static shapes: graph-safe.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(
            why="dep8: 384 experts on 384 slots (one copy), a 1280-row step",
            spec=dict(num_experts=384, num_slots=384, ep_size=8, rows=1280, top_k=6),
        ),
        Cell(
            why="dep32: 384 experts on 416 slots (32 experts with a second copy), a 150-row step",
            spec=dict(num_experts=384, num_slots=416, ep_size=32, rows=150, top_k=6),
        ),
        Cell(
            why="EP4 test geometry: 128 experts on 136 slots",
            spec=dict(num_experts=128, num_slots=136, ep_size=4, rows=64, top_k=6),
        ),
    )
    note = "Expert ids outside `[0, num_experts)` are not certified (`is_valid` rejects them)."

    def __call__(self, table: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
        if table.numel() == num_experts:
            return table[expert_ids.long()]
        rows = torch.arange(expert_ids.shape[0], device=expert_ids.device, dtype=torch.int64)[:, None]
        meta = table[expert_ids.long()].long()
        copy = (((rows * _HASH) & 0xFFFFFFFF) >> _COPIES_SHIFT) % (meta >> _COPIES_SHIFT)
        return table[num_experts + (meta & 0xFFFF) + copy].to(torch.int32)

    def reference(self, assignments: List[int], expert_ids: torch.Tensor, ep_rank: int) -> torch.Tensor:
        """Slot ids as a host loop over rows: copy `h(r) % copies` of the expert, copies rotated by `ep_rank`.

        Takes the placement itself (`assignments`, slot -> expert) where `__call__` takes the table built from it,
        and the rank the table was built for.
        """
        slots_of = {}
        for slot, e in enumerate(assignments):
            slots_of.setdefault(int(e), []).append(slot)
        ids = expert_ids.cpu().tolist()
        out = []
        for r, row in enumerate(ids):
            picks = []
            for e in row:
                s = slots_of[e]
                s = s[ep_rank % len(s) :] + s[: ep_rank % len(s)]
                picks.append(s[(((r * _HASH) & 0xFFFFFFFF) >> _COPIES_SHIFT) % len(s)])
            out.append(picks)
        return torch.tensor(out, dtype=torch.int32, device=expert_ids.device)

    def is_valid(self, table: torch.Tensor, expert_ids: torch.Tensor, num_experts: int) -> None:
        assert table.dtype == torch.int32 and table.numel() >= num_experts, (
            "table is megamoe_slot_table's int32 [E] (one copy) or [E + slots] (copies)"
        )
        assert expert_ids.dim() == 2
        assert bool(((expert_ids >= 0) & (expert_ids < num_experts)).all()), (
            "an expert id outside [0, num_experts) reads the table out of range"
        )


megamoe_experts_to_slots = MegaMoeExpertsToSlots()

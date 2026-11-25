# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Disaggregated remote-tail replay (`bounded_replay_on_generation`, docs/source/features/disagg-serving.md, "DeepSeek-V4.1
context decoder weights"): stock's switches and contract as setup helpers, and the destination's decoder SWA floors as an
entry. Each helper calls or restates the engine's code (`configs/deepseek_v41.py`, `DeepseekV41Model.__init__` /
`_remote_tail_boundary` / `_enter_remote_tail_boundary`, `attention/backends/sparse/csa2/decoder_replay.py::
enter_remote_tail_decoder`); the targets read no switch themselves.

The flow (the engine's executor, cache manager and transceiver drive it; the model only follows the step's mode):

* source (the context worker, `LLMREQUEST_TYPE_CONTEXT_ONLY` requests): the request's context ends at
  `remote_tail_start = max(0, prompt_len - window)` (the whole prompt when that is 0). The forward runs the encoder
  layers, publishes the boundary layer's (the last unpooled KV owner's, layer 20's) GLOBAL KV and index keys from the
  boundary state collapsed with its pre-mix and normalized by the layer's input norm, and returns that collapse (no
  decoder layer, no final norm; the token sampled from it is discarded by the handoff). A context-only worker
  (`TRTLLM_DISAGG_ROLE=context`) also omits the decoder weights and the head, and its cache omits the decoder SWA.
* destination (the generation worker): the transfer brings the prefix's encoder SWA and every GLOBAL / index /
  compressor row, not the decoder SWA. Once received the request is scheduled again as a context chunk from
  `remote_tail_start` to the prompt end; the forward runs every layer over these rows, publishes the boundary owner's
  rows at the boundary as a bounded replay does, and from there hides each context's decoder SWA below its
  `remote_tail_start` (the floors below). Generation rows keep their visibility. The first token is sampled here."""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

SOURCE, DESTINATION = "source", "destination"


def csa2_disagg_role() -> Optional[str]:
    """The serving role of this worker (`TRTLLM_DISAGG_ROLE`: `trtllm-serve --server_role` sets it before the model is
    built, the Python API's caller sets it by hand); a setup helper. Stock's policy reads the same variable."""
    return os.environ.get("TRTLLM_DISAGG_ROLE")


def csa2_context_decoder_skipping(bounded_replay_on_generation: bool) -> bool:
    """Whether this worker omits its decoder weights (stock's `disagg_context_decoder_skipping_enabled`): the switch,
    decoder bounded replay on, and the context role; a setup helper."""
    from tensorrt_llm._torch.configs.deepseek_v41 import disagg_context_decoder_skipping_enabled

    return disagg_context_decoder_skipping_enabled(bool(bounded_replay_on_generation))


def csa2_remote_tail_switch(model_config) -> bool:
    """`bounded_replay_on_generation` as the engine's model loader hands it to the model (`extra_attrs`)."""
    return bool((getattr(model_config, "extra_attrs", None) or {}).get("bounded_replay_on_generation", False))


def csa2_remote_tail_refusal(model_config) -> Optional[str]:
    """Why stock's replay policy (`_resolve_replay_policy`) declines the replay under attention DP with the switch on,
    or None: the speculative worker must be embedded DSpark, and the role explicit (generation, or a context worker that
    skips its decoder weights) so that every attention-DP rank stops at the same layer. A setup helper, called by
    `csa2_replay_policy` before its other checks."""
    mapping = model_config.mapping
    if not (mapping.enable_attention_dp and csa2_remote_tail_switch(model_config)):
        return None
    spec = getattr(model_config, "spec_config", None)
    if spec is not None and not (spec.spec_dec_mode.is_dspark() and spec.draft_is_embedded_in_target):
        return "attention-DP remote-tail replay requires embedded DSpark speculation"
    if csa2_disagg_role() != "generation" and not csa2_context_decoder_skipping(True):
        return ("attention-DP remote-tail replay requires an explicit generation role or a context role with decoder "
                "weight skipping enabled")
    return None


def csa2_remote_tail_policy(model_config, decoder_replay_split: Optional[int]) -> Tuple[bool, bool]:
    """`(disagg_remote_tail_replay, disagg_context_only)` of a model whose bounded replay starts at
    `decoder_replay_split` (None: no replay), with stock's errors (`DeepseekV41Model.__init__`) where the switch cannot
    be honoured; a setup helper. The executor reads both names off the model: the first makes it attach the remote-tail
    contract to disaggregated requests, the second marks a worker whose decoder weights and decoder SWA are omitted."""
    switch = csa2_remote_tail_switch(model_config)
    if switch and decoder_replay_split is None:
        raise ValueError("bounded_replay_on_generation requires supported decoder bounded replay")
    context_only = csa2_context_decoder_skipping(switch)
    if context_only and not switch:
        raise ValueError("Context decoder skipping requires a supported remote-tail replay boundary")
    if context_only and getattr(model_config, "spec_config", None) is not None:
        raise ValueError("Context decoder skipping requires speculative decoding to be disabled")
    if switch:
        logger.info(
            "DeepSeek-V4.1 disaggregated remote-tail replay enabled: eligible context workers stop before layer "
            f"{decoder_replay_split} after the prompt prefix, and generation workers replay the final prompt tokens "
            "through the full model. Set bounded_replay_on_generation=false to disable."
        )
    return switch, context_only


def csa2_remote_tail_mode(attn_metadata) -> Optional[str]:
    """This step's remote-tail mode (`"source"`, `"destination"` or None), as the engine's input preparation sets it from
    the step's context requests (a step is homogeneous); a setup helper."""
    mode = getattr(attn_metadata, "csa2_remote_tail_mode", None)
    if mode not in (None, SOURCE, DESTINATION):
        raise ValueError(f"unknown CSA2 remote-tail mode {mode!r}")
    return mode


def csa2_enter_context_only(attn_metadata) -> None:
    """A context-only worker's forward (stock's `DeepseekV41Model.forward`): every batch is a source, profiling and
    warm-up batches included (they carry no handoff metadata, and every attention-DP rank must stop at the same layer);
    a destination is refused."""
    if csa2_remote_tail_mode(attn_metadata) not in (None, SOURCE):
        raise ValueError("A context-only model cannot execute a remote-tail destination")
    attn_metadata.csa2_remote_tail_mode = SOURCE


class Csa2RemoteTailFloors(OpWrapper):
    """`(starts, num_contexts, nreq, prior=None) -> int64 [nreq]` (host): the first position each request row's decoder
    SWA reads may see from the boundary on, at a destination (stock's `enter_remote_tail_decoder`): a context row's
    `remote_tail_start` (the decoder SWA below it was never transferred), raised to its `prior` floor (an earlier
    floor of the step, e.g. a replay start); generation rows and padding rows keep their prior floor (0 without one).
    `starts` lists one start per context request, in the step's request order (contexts first)."""

    ARCHS = frozenset({Arch.SM_103})
    CELLS = (
        Cell(why="one handed-off tail beside a generation row", spec=dict(starts=(1024,), nctx=1, nreq=2, prior=None)),
        Cell(why="a short prompt replayed whole (start 0) and a long one", spec=dict(starts=(0, 8052), nctx=2, nreq=2,
                                                                                    prior=None)),
        Cell(why="padding request rows past the step's requests, and a prior floor above a start",
             spec=dict(starts=(140, 300), nctx=2, nreq=5, prior=(500, 0, 7, 0, 0))),
    )
    note = "Host integers; exact."

    def __call__(self, starts: Sequence[int], num_contexts: int, nreq: int,
                 prior: Optional[Sequence[int]] = None) -> torch.Tensor:
        if len(starts) != num_contexts:
            raise ValueError("remote-tail starts must match the context rows")
        floors = torch.zeros(nreq, dtype=torch.int64)
        floors[:num_contexts] = torch.tensor(list(starts), dtype=torch.int64)
        if prior is not None:
            torch.maximum(floors, torch.tensor(list(prior)[:nreq], dtype=torch.int64), out=floors)
        return floors

    def reference(self, starts: Sequence[int], num_contexts: int, nreq: int,
                  prior: Optional[Sequence[int]] = None) -> List[int]:
        own = list(starts) + [0] * (nreq - num_contexts)
        base = [0] * nreq if prior is None else list(prior)[:nreq]
        return [max(a, b) for a, b in zip(own, base)]

    def is_valid(self, starts: Sequence[int], num_contexts: int, nreq: int,
                 prior: Optional[Sequence[int]] = None) -> None:
        assert num_contexts <= nreq and all(int(s) >= 0 for s in starts)
        assert prior is None or len(prior) >= nreq


csa2_remote_tail_floors = Csa2RemoteTailFloors()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""ModelingV2 target: deepseek-v4.1-flash-nvfp4 / sm_103 / default.

The fallback target of the family: it derives every share from the engine's `Mapping` and runs the layouts the
family's `route` admits for it when no layout-specific target (dep2, dep8, dep32, tep8_moe_tp4_ep2) takes the
deployment. DeepSeek-V4.1-Flash, modelopt MIXED_PRECISION export: 40 layers, hidden 5120, 64 query heads of dim 512
(one shared KV head) in 8 output groups, q-LoRA 1280, 384 routed NVFP4 experts at top-6 plus one FP8 shared expert,
Engram n-gram tables at layers 1 and 14, a 4-copy Hyper-Connection residual stream.

Two forward paths, one per attention layout, chosen at construction from `mapping.enable_attention_dp`:

* `attention_dp/` -- attention data parallel (`enable_attention_dp: true`, TP group = world). Every rank owns its
  requests and holds the whole attention block and the shared expert; the routed experts are split
  `moe_tp_size x moe_ep_size` ways and the ranks' rows meet them through an all-gather and a reduce-scatter (any
  split), or, with `moe_config.backend: MEGAMOE_CUTEDSL` and the experts split only across ranks, through one fused
  MegaMoE launch per layer. The Engram tables are head- or row-sharded over the group and exchanged in peer memory.
  The LM head stays whole or is split over the ranks (`enable_lm_head_tp_in_adp`), as the engine runs it.
* `attention_tp/` -- tensor-parallel attention (no attention DP, TP group = world, at most the 8 output groups):
  every rank sees the same tokens, attention heads and output groups are split `tp_size` ways, the shared expert's
  intermediate width likewise, the routed experts `moe_tp_size x moe_ep_size` ways; attention and FFN hand back
  partial sums the next Hyper-Connection boundary all-reduces (in the same launch where the one-shot workspace is
  there). The LM head is column-parallel.

Both paths run the TRTLLM-Gen NVFP4 MoE runner (stock's `AUTO` / `TRTLLM` backend on SM10x), DSpark speculative
decoding through the engine's one-model draft worker (`SpecDecOneEngineForCausalLM`), chunked prefill, the bounded
decoder replay of context chunks (stock's policy), and CUDA graphs of generation steps. Each path's code is the
measured code of the family's layout-specific targets of the same attention layout (dep8 / dep32 for attention DP,
tep8_moe_tp4_ep2 for attention TP), whose sharding already follows the mapping; this target owns its copy (targets do
not import each other).

What is not here, and is refused by `route` rather than run: pipeline / context parallelism, attention TP wider than
8 or not dividing the heads, a moe cluster, MoE backends other than TRTLLM (and MegaMoE outside pure expert
parallelism), either side of the disaggregated remote-tail replay (`bounded_replay_on_generation`: the dep2 target is
its context-only worker, the dep8 / dep32 / tep8 targets its generation side), and a vision checkpoint served with its
image encoder (the layout-specific targets have image classes; this one is text only). Served with `disable_mm_encoder`
it refuses a context request carrying image input (`validate_request`) and hashes the image tokens of a generation-only
request's prompt as stock does (the Engram n-gram break).
"""

import contextlib
from types import ModuleType

import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.models.modeling_utils import DecoderModel
from tensorrt_llm._torch.speculative.dspark import DSparkSpecMetadata

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_decoder_replay import (
    csa2_replay_plan,
)

from ..image_input import refuse_image_request, refuse_image_step


def _path(mapping) -> ModuleType:
    """The forward path of this layout: the attention-DP path or the attention-TP path (each imported only when
    built)."""
    if mapping.enable_attention_dp:
        from .attention_dp import core
    else:
        from .attention_tp import core
    return core


class ModelingV2DeepseekV41FlashNvfp4Sm103Default(SpecDecOneEngineForCausalLM[DecoderModel, PretrainedConfig]):
    def __init__(self, model_config: ModelConfig):
        core = _path(model_config.mapping)
        model_config = core._text_model_config(model_config)
        cfg = model_config.pretrained_config
        assert cfg is not None
        # The inherited shell builds the DSpark draft model and its worker from `speculative_config`; the draft stages
        # live in the same checkpoint (`mtp.*`).
        super().__init__(core.ModelingV2Core(model_config), model_config, hidden_size=cfg.hidden_size,
                         vocab_size=cfg.vocab_size)
        self._weights = core._weights
        core._opt_into_small_m_gemm(self)

    def forward(self, *args, return_context_logits: bool = False, **kwargs) -> torch.Tensor:
        """Tell the core whether anything downstream reads non-final rows (stock's `DeepseekV41ForCausalLM.forward`):
        context logits, or the explicit full-context demand of a DSpark gather. The engine's forward consumes
        `return_context_logits` itself, so the core gets it as `all_token_states_required`; a bounded replay is refused
        for such a step."""
        required = bool(return_context_logits)
        spec_metadata = kwargs.get("spec_metadata")
        if (isinstance(spec_metadata, DSparkSpecMetadata) and spec_metadata.gather_ids is not None
                and spec_metadata._requires_all_context_logits is not None):
            required = spec_metadata._requires_all_context_logits
        kwargs["all_token_states_required"] = required
        refuse_image_step(kwargs.get("multimodal_params"))
        return super().forward(*args, return_context_logits=return_context_logits, **kwargs)

    def validate_request(self, request) -> None:
        """Engine hook at admission: this target serves text only (no image class; `route` declines a vision checkpoint
        served with its image encoder), so a context request with image input is refused
        (`image_input.refuse_image_request`); a generation-only request of an image prompt is admitted."""
        refuse_image_request(request)

    def prepare_adp_inputs(self, attn_metadata, *, all_token_states_required: bool, requests=None) -> None:
        """Engine hook, before the attention-DP token-count exchange (or a breakable prefill graph's padding choice):
        fix which rows of each context chunk run the decoder half (stock's `prepare_adp_inputs`), so the exchange
        carries every rank's decoder row counts."""
        attn_metadata.decoder_replay_plan = csa2_replay_plan(self.model, attn_metadata, all_token_states_required, requests)

    def release_request_state(self, request_id: int) -> None:
        """Engine hook (the executor's request release: completion, cancellation, a warm-up dummy), as stock's: the
        request's Engram look-back row is freed for the next request."""
        self.model.release_request_state(request_id)

    def prepare_request_inputs(self, scheduled_requests, attn_metadata, promoted_context_request_ids):
        """Engine hook, called outside graph capture before the step's inputs are built."""
        # The engine may call this hook on another stream than the forward runs on. The page tables, slot map and
        # history ring are read by the forward (or a graph replay) of the previous step, so every device write
        # here goes on the forward's own stream: stream order keeps it after that step and before this one.
        stream = self.model._fwd_stream
        with torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext():
            self.model.on_step(scheduled_requests, attn_metadata.kv_cache_manager)

    def load_weights(self, weights, *args, **kwargs):
        self.model.load(weights)
        self._weights.load_head(self, weights)

    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load()

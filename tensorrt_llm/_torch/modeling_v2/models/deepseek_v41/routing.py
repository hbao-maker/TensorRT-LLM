# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where a DeepseekV41ForCausalLM config lands. Read this file and you know.

``route`` is one forward-reading decision: the facts that fix which code runs,
compared against the deployments the targets below were written for. When none
fits it raises ``NoTarget`` stating the criterion that failed. In ``auto`` the
engine then uses the built-in DeepseekV41 implementation; in ``require`` it
raises that error.

The same checkpoint is published with a vision tower, whose ``architectures[0]``
is ``DeepseekV41ForConditionalGeneration`` and whose text model is nested under
``text_config``; both names land here and the text model is what is routed on.
Served with the image encoder on (``disable_mm_encoder`` off), such a checkpoint
gets the target's image class (``<target class>Images``: the family's image
encoder in front of the text target, ``vision/image_model.py``); with it off, the
text class, which refuses image input.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from tensorrt_llm.quantization.mode import QuantAlgo

from ..._router_index import NoTarget, engine_kv_cache_quant_algo, routes

# sm is part of a target's identity, not a knob: the targets read and write
# the CSA2 pools through plain-torch ops and run the MoE through trtllm-gen
# kernels, both measured on GB300 only.
_SM = (10, 3)

# (num_hidden_layers, hidden_size, n_routed_experts, q_lora_rank) of
# DeepSeek-V4.1-Flash. q_lora_rank is in the shape because it changes the
# attention weight layout.
_SHAPE = (40, 5120, 384, 1280)

if TYPE_CHECKING:
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs


def _serves_images(model_config: ModelConfig, llm_args: TorchLlmArgs) -> bool:
    """Whether this deployment takes image input: a checkpoint with a vision tower, served with the image encoder."""
    vision = getattr(model_config.pretrained_config, "vision_config", None)
    if vision is None or getattr(vision, "num_hidden_layers", 0) <= 0:
        return False
    return not (getattr(model_config, "disable_mm_encoder", False) or getattr(llm_args, "disable_mm_encoder", False))


@routes("DeepseekV41ForCausalLM", "DeepseekV41ForConditionalGeneration")
def route(
    model_config: ModelConfig, llm_args: TorchLlmArgs, compute_capability: Tuple[int, int]
) -> type:
    """Return the target class for this deployment, or raise ``NoTarget`` saying why none fits.

    Each target is imported where it is returned, so declining costs no import
    of modeling code.
    """
    pretrained_config = model_config.pretrained_config
    # A vision-bearing checkpoint arrives as a composite config: the text model is nested.
    text = getattr(pretrained_config, "text_config", None) or pretrained_config
    mapping = model_config.mapping
    spec_config = model_config.spec_config
    kv_cache_quant_algo = engine_kv_cache_quant_algo(model_config, llm_args)
    images = _serves_images(model_config, llm_args)

    if compute_capability != _SM:
        raise NoTarget(
            f"compute capability {compute_capability}; the deepseek_v41 targets are written "
            f"for sm{_SM[0]}{_SM[1]} (GB300)"
        )

    shape = (
        getattr(text, "num_hidden_layers", None),
        getattr(text, "hidden_size", None),
        getattr(text, "n_routed_experts", None),
        getattr(text, "q_lora_rank", None),
    )
    if shape != _SHAPE:
        raise NoTarget(
            f"config shape {shape} (num_hidden_layers, hidden_size, n_routed_experts, "
            f"q_lora_rank); the deepseek_v41 targets are DeepSeek-V4.1-Flash-NVFP4, {_SHAPE}"
        )

    # The pools are the engine's CSA2CacheManager's, in the byte formats the
    # target's ops write (an fp8 sliding window, fp4 compressed rows).
    if kv_cache_quant_algo != QuantAlgo.FP8:
        raise NoTarget(
            f"KV cache {kv_cache_quant_algo or 'bf16'}; the deepseek_v41 targets run an fp8 "
            "KV cache (kv_cache_config.dtype='fp8')"
        )
    sparse = getattr(model_config, "sparse_attention_config", None) or getattr(
        llm_args, "sparse_attention_config", None
    )
    algorithm = getattr(sparse, "algorithm", None)
    if algorithm != "csa2":
        raise NoTarget(
            f"sparse attention {algorithm}; the deepseek_v41 targets allocate their cache "
            "pools from CSA2CacheManager (sparse_attention_config.algorithm='csa2')"
        )

    # SWA scratch reuse (`kv_cache_config.enable_swa_scratch_reuse`, the built-in model's default): the manager maps a
    # prefill chunk's blocks that leave the window within the step to per-layer scratch slots instead of pages. The
    # targets take it as stock does: they read the step's own rows from a step-local copy and write only the rows that
    # stay in the window (real pages) through the manager's converted tables, which map the scratch range.

    split = (
        f"world_size={mapping.world_size} tp={mapping.tp_size} pp={mapping.pp_size} "
        f"cp={mapping.cp_size} attention_dp={mapping.enable_attention_dp} "
        f"moe_tp={mapping.moe_tp_size} moe_ep={mapping.moe_ep_size}"
    )
    if mapping.pp_size != 1 or mapping.cp_size != 1:
        raise NoTarget(
            f"parallel layout {split}; no deepseek_v41 target implements pipeline or "
            "context parallelism"
        )
    # A moe cluster is a different expert split than any target implements; refused
    # here so that ``auto`` falls back instead of crashing in the model.
    lm_head_tp_in_adp = getattr(mapping, "enable_lm_head_tp_in_adp", False)
    if mapping.moe_cluster_size != 1:
        raise NoTarget(f"parallel layout {split} with moe_cluster_size={mapping.moe_cluster_size}")

    # ``moe_config.backend: MEGAMOE_CUTEDSL`` is a different MoE implementation, not a tuning knob: the dep8 and
    # dep32 targets implement it (static expert placement), the others run the TRTLLM fp4 runner and would silently
    # ignore it, so they decline it here and ``auto`` falls back to the built-in model.
    moe_config = getattr(llm_args, "moe_config", None)
    megamoe = getattr(moe_config, "backend", None) == "MEGAMOE_CUTEDSL"

    # The attention-DP topologies: every rank is a TP rank and holds the whole
    # attention block, the experts are split EP-wide (moe_tp 1).
    adp = (
        mapping.enable_attention_dp
        and mapping.tp_size == mapping.world_size
        and mapping.moe_ep_size == mapping.world_size
        and mapping.moe_tp_size == 1
    )

    # `bounded_replay_on_generation` is read from the engine arguments, and from `extra_attrs`, where stock's loader
    # hands it to the model.
    bounded_replay = bool(getattr(llm_args, "bounded_replay_on_generation", False)) or bool(
        (getattr(model_config, "extra_attrs", None) or {}).get("bounded_replay_on_generation", False)
    )
    remote_tail = _remote_tail_refusal(bounded_replay, mapping.world_size == 2 and adp, split)
    if remote_tail is not None:
        raise NoTarget(remote_tail)

    # dep2: the context role. Attention replicated on each of two ranks, experts
    # split two ways. No speculative decoding on this role, and the lm_head stays
    # whole (the engine's lm_head split across attention-DP ranks is a generation-role
    # switch: the dep8 / dep32 targets run it, this one does not).
    if mapping.world_size == 2 and adp:
        if megamoe:
            raise NoTarget(
                f"moe_config.backend MEGAMOE_CUTEDSL at parallel layout {split}; the dep2 target runs the "
                "TRTLLM fp4 MoE runner (MegaMoE is implemented by the dep8 and dep32 targets)"
            )
        if lm_head_tp_in_adp:
            raise NoTarget(
                f"parallel layout {split} with enable_lm_head_tp_in_adp; the dep2 target "
                "keeps the lm_head whole under attention DP"
            )
        if spec_config is not None:
            raise NoTarget(
                f"speculative decoding {spec_config.spec_dec_mode.name} at parallel layout "
                f"{split}; the dep2 target is the context role and has no speculative path"
            )
        # fmt: off
        # isort: off
        from .flash_nvfp4__sm_103__dep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep2
        from .flash_nvfp4__sm_103__dep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep2Images
        # isort: on
        # fmt: on
        if images:
            return ModelingV2DeepseekV41FlashNvfp4Sm103Dep2Images
        return ModelingV2DeepseekV41FlashNvfp4Sm103Dep2

    # dep8 and dep32: the generation roles (batch 320 and batch 25). The same
    # attention-DP layout at 8 and 32 ranks, each its own target. They drive DSpark
    # through the engine's one-model draft worker, or nothing, and run the lm_head
    # split across the attention-DP ranks when the engine is asked to
    # (``enable_lm_head_tp_in_adp``), which is a runtime switch of the role rather
    # than part of its identity.
    if adp and mapping.world_size in (8, 32):
        balancer_mode = getattr(getattr(moe_config, "load_balancer", None), "mode", "standard")
        if megamoe and balancer_mode != "standard":
            raise NoTarget(
                f"moe_config.load_balancer.mode {balancer_mode!r} with MEGAMOE_CUTEDSL; the dep8 and dep32 "
                "targets implement the static expert placement only"
            )
        dspark_or_none = spec_config is None or spec_config.spec_dec_mode.is_dspark()
        if not dspark_or_none:
            raise NoTarget(
                f"speculative decoding {spec_config.spec_dec_mode.name} at parallel layout "
                f"{split}; the dep8 and dep32 targets implement DSpark or no speculative decoding"
            )
        if mapping.world_size == 8:
            # fmt: off
            # isort: off
            from .flash_nvfp4__sm_103__dep8.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep8
            from .flash_nvfp4__sm_103__dep8.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep8Images
            # isort: on
            # fmt: on
            if images:
                return ModelingV2DeepseekV41FlashNvfp4Sm103Dep8Images
            return ModelingV2DeepseekV41FlashNvfp4Sm103Dep8
        # fmt: off
        # isort: off
        from .flash_nvfp4__sm_103__dep32.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep32
        from .flash_nvfp4__sm_103__dep32.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Dep32Images
        # isort: on
        # fmt: on
        if images:
            return ModelingV2DeepseekV41FlashNvfp4Sm103Dep32Images
        return ModelingV2DeepseekV41FlashNvfp4Sm103Dep32

    # tep8 with the experts split 4 (tensor) x 2 (expert): the generation role.
    # Attention heads are split eight ways, so attention DP must be off. It
    # drives DSpark through the engine's one-model draft worker, or nothing.
    if (
        mapping.world_size == 8
        and mapping.tp_size == 8
        and not mapping.enable_attention_dp
        and mapping.moe_ep_size == 2
        and mapping.moe_tp_size == 4
    ):
        if megamoe:
            raise NoTarget(
                f"moe_config.backend MEGAMOE_CUTEDSL at parallel layout {split}; the tep8 target runs the "
                "TRTLLM fp4 MoE runner (MegaMoE needs pure DEP: the dep8 and dep32 targets)"
            )
        dspark_or_none = spec_config is None or spec_config.spec_dec_mode.is_dspark()
        if not dspark_or_none:
            raise NoTarget(
                f"speculative decoding {spec_config.spec_dec_mode.name} at parallel layout "
                f"{split}; the tep8 target implements DSpark or no speculative decoding"
            )
        # fmt: off
        # isort: off
        from .flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2
        from .flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2Images
        # isort: on
        # fmt: on
        if images:
            return ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2Images
        return ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2

    # No layout-specific target: the default target, which derives its shares from the mapping, at the layouts it has
    # been measured at and without the features it does not run.
    why = _default_refusals(model_config, llm_args, mapping, spec_config, megamoe, bounded_replay)
    if why:
        raise NoTarget(
            f"parallel layout {split}; the deepseek_v41 targets are dep2 (2 ranks, attention DP, moe ep2: the context "
            "role), dep8 and dep32 (8 / 32 ranks, attention DP, moe ep = world size: generation roles), tep8 (8 ranks, "
            "tensor-parallel attention, moe tp4 x ep2: the generation role) and the default target, which declines: "
            + "; ".join(why)
        )
    # fmt: off
    # isort: off
    from .flash_nvfp4__sm_103__default.modeling import ModelingV2DeepseekV41FlashNvfp4Sm103Default
    # isort: on
    # fmt: on
    return ModelingV2DeepseekV41FlashNvfp4Sm103Default


# The layouts the default target has passed its gates on, as (world_size, attention_dp, moe_tp, moe_ep): per-layer
# outputs against stock's, the needle and many-request checks, GSM8K paired with stock (real DSpark acceptance) and a
# serving sweep against stock. Its forward computes others too (any attention-DP world with any expert split, attention
# TP 2 / 4 / 8 with any expert split), but a layout is admitted only once it has been measured, so ``require`` cannot
# quietly run one nobody checked. A GB300 tray holds four GPUs: the 8-rank layouts span two trays and dep16 four, in one
# NVLink domain. dep2, dep8, dep32 and tep8 with moe tp4 x ep2 have their own targets above.
_DEFAULT_LAYOUTS = frozenset(
    {
        (4, True, 1, 4),  # dep4
        (6, True, 1, 6),  # dep6
        (16, True, 1, 16),  # dep16
        (4, False, 1, 4),  # tp4 + ep4
        (4, False, 4, 1),  # tp4, experts split by intermediate
        (4, False, 2, 2),  # tp4, experts tp2 x ep2
        (8, False, 1, 8),  # tp8 + ep8
        (8, False, 8, 1),  # tp8, experts split by intermediate
        (8, False, 2, 4),  # tp8, experts tp2 x ep4
    }
)
# Attention TP splits the 64 heads and the 8 output groups of the attention block.
_ATTENTION_TP = (2, 4, 8)


def _default_refusals(model_config, llm_args, mapping, spec_config, megamoe, bounded_replay: bool) -> list:
    """Why the default target does not run this deployment (empty: it does)."""
    why = []
    layout = (mapping.world_size, bool(mapping.enable_attention_dp), mapping.moe_tp_size, mapping.moe_ep_size)
    if getattr(mapping, "enable_lm_head_tp_in_adp", False) and not mapping.enable_attention_dp:
        why.append("enable_lm_head_tp_in_adp without attention DP")
    if mapping.tp_size != mapping.world_size:
        why.append(f"tp_size {mapping.tp_size} != world_size {mapping.world_size}")
    if not mapping.enable_attention_dp and mapping.tp_size not in _ATTENTION_TP:
        why.append(f"attention TP over {mapping.tp_size} ranks (the default target splits the 64 heads and 8 output "
                   f"groups over {', '.join(map(str, _ATTENTION_TP))} ranks, or runs attention DP)")
    if layout not in _DEFAULT_LAYOUTS:
        why.append("a layout the default target has not been measured at (it is gated on dep4, dep6, dep16, tp4 with "
                   "moe ep4, tp4 or tp2 x ep2, and tp8 with moe ep8, tp8 or tp2 x ep4)")
    # The MoE implementation: stock's AUTO resolves to TRTLLM on SM10x, which this target runs; MegaMoE is the
    # attention-DP path's with the experts split only across ranks. Other backends would be silently ignored.
    moe_config = getattr(llm_args, "moe_config", None)
    backend = str(getattr(moe_config, "backend", None) or "AUTO").upper()
    pure_ep = mapping.enable_attention_dp and mapping.moe_tp_size == 1 and mapping.moe_ep_size == mapping.world_size
    if megamoe:
        balancer_mode = getattr(getattr(moe_config, "load_balancer", None), "mode", "standard")
        if not pure_ep:
            why.append("moe_config.backend MEGAMOE_CUTEDSL outside attention DP with the experts split only across ranks")
        elif balancer_mode != "standard":
            why.append(f"moe_config.load_balancer.mode {balancer_mode!r} with MEGAMOE_CUTEDSL (static placement only)")
    elif backend not in ("AUTO", "TRTLLM"):
        why.append(f"moe_config.backend {backend} (the default target runs the TRTLLM-Gen NVFP4 MoE, or MegaMoE)")
    elif getattr(moe_config, "load_balancer", None) is not None:
        why.append("moe_config.load_balancer without MEGAMOE_CUTEDSL (the TRTLLM runner path places experts statically)")
    if spec_config is not None and not spec_config.spec_dec_mode.is_dspark():
        why.append(f"speculative decoding {spec_config.spec_dec_mode.name} (the default target runs DSpark or none)")
    # Disaggregated remote-tail replay: the dep2 target is its context-only worker and dep8 / dep32 / tep8 its
    # generation side; the default target implements neither side, so it declines the switch rather than run without
    # it (the engine would omit the decoder SWA it reads).
    if bounded_replay:
        why.append("bounded_replay_on_generation (the default target has no generation-side remote-tail replay)")
    # A checkpoint with a vision tower served with the image encoder gets a layout-specific target's image class
    # (`vision/image_model.py`); the default target has none (no vision-bias routing nor image-row Engram in its cores),
    # so it declines it, and an encoder-only worker. Served with disable_mm_encoder it runs as the text class, which
    # refuses image requests itself (`image_input.py`).
    if getattr(llm_args, "mm_encoder_only", False) or _serves_images(model_config, llm_args):
        why.append("a checkpoint with a vision tower served with its image encoder (the default target has no image "
                   "class; set disable_mm_encoder=True to run it on text requests)")
    return why


def _remote_tail_refusal(bounded_replay: bool, dep2: bool, split: str):
    """Why no target runs this side of `bounded_replay_on_generation` (disaggregated remote-tail replay), or None.
    The dep2 target is its context-only worker (the context role: decoder weights omitted, every step a source), the
    generation targets its generation side (the tail replay). Deployments stock supports beyond that fall back."""
    if not bounded_replay:
        return None
    # fmt: off
    # isort: off
    from .ops.attention.csa2_remote_tail import csa2_context_decoder_skipping, csa2_disagg_role
    # isort: on
    # fmt: on
    context_only = csa2_context_decoder_skipping(True)
    if dep2 and not context_only:
        return (
            f"bounded_replay_on_generation at parallel layout {split} with TRTLLM_DISAGG_ROLE="
            f"{csa2_disagg_role()!r}; the dep2 target runs it as the context-only worker (the context role, "
            "trtllm-serve --server_role CONTEXT), the generation-side tail replay is the dep8 / dep32 / tep8 targets'"
        )
    if not dep2 and context_only:
        return (
            f"bounded_replay_on_generation on the context role at parallel layout {split}; the context-only worker "
            "(decoder weights omitted) is the dep2 target's"
        )
    return None

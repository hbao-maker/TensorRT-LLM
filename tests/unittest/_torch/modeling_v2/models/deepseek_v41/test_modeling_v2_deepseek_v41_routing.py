# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1-Flash target routing, driven by synthetic configs on CPU.

Proved here: the four shipped targets (dep2 context role; tep8 moe tp4 x ep2, dep8 and dep32 generation
roles) are claimed by their layouts under ``auto`` (also when ``modeling_v2`` is left unset: they are on by
default) and ``require`` and declined under ``off``, the vision-bearing composite config routes like the
text-only one, ``enable_lm_head_tp_in_adp`` is accepted by the generation roles that run it
and declined by the others, ``bounded_replay_on_generation`` is claimed by role (dep2 the context-only worker, the
generation targets the tail replay), a vision checkpoint served with its image encoder gets the target's image class,
and every near miss is declined with a message that names the criterion that failed. The default target takes exactly
the measured layouts no layout-specific target claims (with or without DSpark), never one a specific target claims, and
declines what it does not run (other MoE backends, a balancer without MegaMoE, other speculative modes,
bounded_replay_on_generation, a vision checkpoint served with its image encoder).
"""

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.modeling_v2._router_index import modeling_v2_resolve
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.llmapi.llm_args import (
    CSA2SparseAttentionConfig,
    DSparkDecodingConfig,
    KvCacheConfig,
    MTPDecodingConfig,
    NGramDecodingConfig,
    TorchLlmArgs,
)
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantConfig
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

_DUMMY_MODEL = "/tmp/dummy_model"
_DEP2 = dict(world_size=2, tp_size=2, enable_attention_dp=True, moe_ep_size=2, moe_tp_size=1)
_TEP8 = dict(world_size=8, tp_size=8, moe_ep_size=2, moe_tp_size=4)
_DEP8 = dict(world_size=8, tp_size=8, enable_attention_dp=True, moe_ep_size=8, moe_tp_size=1)
_DEP32 = dict(world_size=32, tp_size=32, enable_attention_dp=True, moe_ep_size=32, moe_tp_size=1)
_DEP2_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2"
)
_TEP8_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2"
)
_DEP8_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__dep8.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Dep8"
)
_DEP32_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__dep32.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Dep32"
)
_DEFAULT_TARGET = (
    "models.deepseek_v41.flash_nvfp4__sm_103__default.modeling."
    "ModelingV2DeepseekV41FlashNvfp4Sm103Default"
)
_DSPARK = DSparkDecodingConfig(max_draft_len=5)
_NGRAM = NGramDecodingConfig(max_draft_len=3, max_matching_ngram_size=2)


def _qualified(target) -> str:
    """``models.<family>.<dir>.modeling.<Class>`` for a resolved target class."""
    return f"{target.__module__.split('.modeling_v2.')[1]}.{target.__name__}"


@pytest.fixture(autouse=True)
def _on_sm103(monkeypatch):
    """Route as if this were a GB300, wherever the test actually runs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 3))


def _llm_args(mode="auto", **kwargs) -> TorchLlmArgs:
    """``mode`` is the ``modeling_v2`` option; ``"unset"`` leaves it at its default, as a deployment that never
    names it."""
    kwargs.setdefault("kv_cache_config", KvCacheConfig(dtype="fp8"))
    if mode != "unset":
        kwargs["modeling_v2"] = mode
    return TorchLlmArgs(model=_DUMMY_MODEL, **kwargs)


def _config(overrides=None, composite=False):
    """The shape DeepSeek-V4.1-Flash-NVFP4's own config.json declares."""
    fields = dict(
        architectures=["DeepseekV41ForCausalLM"],
        num_hidden_layers=40,
        hidden_size=5120,
        n_routed_experts=384,
        q_lora_rank=1280,
    )
    fields.update(overrides or {})
    text = PretrainedConfig(**fields)
    if not composite:
        return text
    # The released checkpoint carries a vision tower: the text model is nested.
    return PretrainedConfig(
        architectures=["DeepseekV41ForConditionalGeneration"], text_config=text
    )


def _model_config(
    pretrained_config, spec_config=None, sparse="csa2", kv=QuantAlgo.FP8, **mapping_kwargs
):
    mapping = Mapping(**mapping_kwargs) if mapping_kwargs else Mapping()
    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        spec_config=spec_config,
        sparse_attention_config=CSA2SparseAttentionConfig() if sparse == "csa2" else None,
        quant_config=QuantConfig(kv_cache_quant_algo=kv),
    )


@pytest.mark.parametrize("mode", ["unset", "auto", "require"])
@pytest.mark.parametrize("composite", [False, True], ids=["text", "vision-composite"])
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [
        (_DEP2, None, _DEP2_TARGET),
        (_TEP8, None, _TEP8_TARGET),
        (_TEP8, _DSPARK, _TEP8_TARGET),
        (_DEP8, None, _DEP8_TARGET),
        (_DEP8, _DSPARK, _DEP8_TARGET),
        (dict(_DEP8, enable_lm_head_tp_in_adp=True), _DSPARK, _DEP8_TARGET),
        (_DEP32, None, _DEP32_TARGET),
        (_DEP32, _DSPARK, _DEP32_TARGET),
        (dict(_DEP32, enable_lm_head_tp_in_adp=True), _DSPARK, _DEP32_TARGET),
    ],
    ids=[
        "dep2", "tep8", "tep8-dspark", "dep8", "dep8-dspark", "dep8-lm-head-tp", "dep32",
        "dep32-dspark", "dep32-lm-head-tp",
    ],
)  # fmt: skip
def test_targets_match(mode, composite, mapping_kwargs, spec_config, target):
    config = _model_config(_config(composite=composite), spec_config, **mapping_kwargs)
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == target


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config",
    [(_DEP2, None), (_TEP8, _DSPARK), (_DEP8, _DSPARK), (_DEP32, _DSPARK)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)
def test_off_keeps_the_built_in_model(mapping_kwargs, spec_config):
    """The targets are on by default (no ``@disable_v2_by_default``): ``modeling_v2="off"`` is how a matching
    deployment keeps the built-in model."""
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("off")) is None


@pytest.mark.parametrize(
    "config_kwargs, mapping_kwargs, spec_config, missed",
    [
        (dict(num_hidden_layers=61), _DEP2, None, r"shape \(61, 5120, 384, 1280\)"),
        (dict(hidden_size=7168), _TEP8, None, r"shape \(40, 7168, 384, 1280\)"),
        (dict(), dict(_DEP2, enable_attention_dp=False), None, "attention_dp=False"),
        (dict(), dict(_TEP8, enable_attention_dp=True), None, "attention_dp=True"),
        (dict(), dict(world_size=4, tp_size=4, enable_attention_dp=True, moe_ep_size=2, moe_tp_size=2), _DSPARK,
         "moe_tp=2.*not been measured"),
        (dict(), dict(world_size=2, tp_size=2, moe_ep_size=2), None, "world_size=2.*not been measured"),
        (dict(), _DEP2, _DSPARK, "DSPARK.*context role"),
        (dict(), _TEP8, MTPDecodingConfig(max_draft_len=3), "MTP.*DSpark"),
        (dict(), _TEP8, _NGRAM, "NGRAM.*DSpark"),
        (dict(), dict(_DEP8, moe_ep_size=4, moe_tp_size=2), None, "moe_ep=4"),
        (dict(), dict(world_size=16, tp_size=16, moe_ep_size=16), None, "attention TP over 16 ranks"),
        (dict(), _DEP8, MTPDecodingConfig(max_draft_len=3), "MTP.*DSpark"),
        (dict(), _DEP32, _NGRAM, "NGRAM.*DSpark"),
        (dict(hidden_size=7168), _DEP32, None, r"shape \(40, 7168, 384, 1280\)"),
    ],
    ids=[
        "other-depth", "other-width", "dep2-without-adp", "tep8-with-adp", "dep4-moe-tp2-ep2-unmeasured", "tp2",
        "ctx-with-dspark", "mtp", "ngram", "dep8-moe-tp2-ep4", "tp16",
        "dep8-mtp", "dep32-ngram", "dep32-other-width",
    ],
)  # fmt: skip
def test_near_misses_do_not_match(config_kwargs, mapping_kwargs, spec_config, missed):
    config = _model_config(_config(config_kwargs), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


def test_other_sm_does_not_match(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (10, 0))
    config = _model_config(_config(), **_DEP2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match=r"compute capability \(10, 0\)"):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize(
    "kv, dtype", [(None, "auto"), (QuantAlgo.NVFP4, "auto"), (QuantAlgo.FP8, "nvfp4")]
)
def test_only_an_fp8_kv_cache_matches(kv, dtype):
    config = _model_config(_config(), kv=kv, **_DEP2)
    args = dict(kv_cache_config=KvCacheConfig(dtype=dtype))
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="KV cache"):
        modeling_v2_resolve(config, _llm_args("require", **args))


def test_the_kv_cache_dtype_argument_overrides_the_checkpoint():
    config = _model_config(_config(), kv=None, **_DEP2)
    args = _llm_args("require", kv_cache_config=KvCacheConfig(dtype="fp8"))
    assert _qualified(modeling_v2_resolve(config, args)) == _DEP2_TARGET


def test_swa_scratch_reuse_is_taken():
    # The engine's DeepSeek-V4.1 default (the official deployments leave it unset): the targets write only window rows.
    config = _model_config(_config(), **_DEP2)
    for value in (True, False):
        args = _llm_args("require", kv_cache_config=KvCacheConfig(dtype="fp8", enable_swa_scratch_reuse=value))
        assert _qualified(modeling_v2_resolve(config, args)) == _DEP2_TARGET


def test_csa2_sparse_attention_is_required():
    config = _model_config(_config(), sparse=None, **_DEP2)
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="CSA2CacheManager"):
        modeling_v2_resolve(config, _llm_args("require"))


def test_lm_head_tp_in_adp_is_declined():
    config = _model_config(_config(), **dict(_DEP2, enable_lm_head_tp_in_adp=True))
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="enable_lm_head_tp_in_adp"):
        modeling_v2_resolve(config, _llm_args("require"))


_MEGAMOE = dict(moe_config={"backend": "MEGAMOE_CUTEDSL"})


@pytest.mark.parametrize("mode", ["auto", "require"])
@pytest.mark.parametrize(
    "mapping_kwargs, target",
    [(_DEP8, _DEP8_TARGET), (_DEP32, _DEP32_TARGET)],
    ids=["dep8", "dep32"],
)
def test_megamoe_backend_is_claimed_by_the_dep_generation_targets(mode, mapping_kwargs, target):
    config = _model_config(_config(), _DSPARK, **dict(mapping_kwargs, enable_lm_head_tp_in_adp=True))
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode, **_MEGAMOE))) == target


@pytest.mark.parametrize(
    "mapping_kwargs, spec_config",
    [(_DEP2, None), (_TEP8, None), (_TEP8, _DSPARK)],
    ids=["dep2", "tep8", "tep8-dspark"],
)
def test_megamoe_backend_is_declined_where_it_is_not_implemented(mapping_kwargs, spec_config):
    # These targets run the TRTLLM fp4 runner: honouring the config silently would be a different MoE.
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    assert modeling_v2_resolve(config, _llm_args("auto", **_MEGAMOE)) is None
    with pytest.raises(ValueError, match="MEGAMOE_CUTEDSL"):
        modeling_v2_resolve(config, _llm_args("require", **_MEGAMOE))
    # and the same layout without the backend still routes
    assert modeling_v2_resolve(config, _llm_args("auto")) is not None


def test_per_layer_dynamic_placement_is_declined_with_megamoe():
    balancer = {"mode": "per_layer_dynamic", "num_slots": 416}
    config = _model_config(_config(), _DSPARK, **_DEP32)
    args = dict(moe_config={"backend": "MEGAMOE_CUTEDSL", "load_balancer": balancer})
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match="per_layer_dynamic"):
        modeling_v2_resolve(config, _llm_args("require", **args))


def test_the_static_balancer_config_of_the_gen_deployments_is_accepted():
    warmup = dict(
        count_every_steps=4, decay=0.5, interval_steps=12000, max_migrations=100000, max_moved_fraction=0.1,
        min_gain=0.01, min_pos=2, min_round_gain=0.5, min_rows=1048576, planner="cost", start_step=0,
    )  # fmt: skip  (the gen-B deployment's block)
    balancer = {"num_slots": 416, "layer_updates_per_iter": 0, "warmup_placement": warmup}
    config = _model_config(_config(), _DSPARK, **_DEP32)
    args = dict(moe_config={"backend": "MEGAMOE_CUTEDSL", "load_balancer": balancer})
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", **args))) == _DEP32_TARGET


@pytest.mark.parametrize("role", ["context", "generation", None])
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [(_DEP2, None, _DEP2_TARGET), (_TEP8, _DSPARK, _TEP8_TARGET), (_DEP8, _DSPARK, _DEP8_TARGET),
     (_DEP32, None, _DEP32_TARGET)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)
def test_bounded_replay_on_generation_routes_by_role(monkeypatch, role, mapping_kwargs, spec_config, target):
    """`bounded_replay_on_generation` (disaggregated remote-tail replay): the dep2 target is its context-only worker
    (the context role), the generation targets its generation side; each declines the other role."""
    monkeypatch.delenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", raising=False)
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    config = _model_config(_config(), spec_config, **mapping_kwargs)
    dep2 = mapping_kwargs is _DEP2
    claimed = (role == "context") == dep2
    for mode in ("auto", "require"):
        args = _llm_args(mode, bounded_replay_on_generation=True)
        if claimed:
            assert _qualified(modeling_v2_resolve(config, args)) == target
        elif mode == "auto":
            assert modeling_v2_resolve(config, args) is None
        else:
            with pytest.raises(ValueError, match="context-only worker"):
                modeling_v2_resolve(config, args)
    # the switch off: every target as before, whatever the role
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == target


def _vision_config(composite: bool, layers: int = 32):
    """The checkpoint as the loader builds it with its vision tower: the composite (text nested) or the flat text
    config, either carrying the vision tower's config and the image token."""
    vision = PretrainedConfig(num_hidden_layers=layers, hidden_size=1024)
    if composite:
        return PretrainedConfig(architectures=["DeepseekV41ForConditionalGeneration"], text_config=_config(),
                                vision_config=vision, image_token_id=129264)
    return _config(dict(architectures=["DeepseekV41ForConditionalGeneration"], vision_config=vision,
                        image_token_id=129264))


@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize(
    "mapping_kwargs, spec_config, target",
    [(_DEP2, None, _DEP2_TARGET), (_TEP8, _DSPARK, _TEP8_TARGET), (_DEP8, _DSPARK, _DEP8_TARGET),
     (_DEP32, _DSPARK, _DEP32_TARGET)],
    ids=["dep2", "tep8", "dep8", "dep32"],
)  # fmt: skip
def test_vision_checkpoint_with_the_image_encoder_gets_the_image_class(composite, mapping_kwargs, spec_config, target):
    """A checkpoint with a vision tower, served with the image encoder on, routes to the target's image class (the
    family's image encoder in front of the text target); with ``disable_mm_encoder`` it routes to the text class."""
    config = _model_config(_vision_config(composite), spec_config, **mapping_kwargs)
    images = modeling_v2_resolve(config, _llm_args("require"))
    assert _qualified(images) == target + "Images"
    assert images.TEXT_TARGET.__name__ == target.rsplit(".", 1)[1]
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", disable_mm_encoder=True))) == target


def test_vision_tower_of_zero_layers_is_text_only():
    """The loader reads a checkpoint whose vision config has no layer as text only; so does routing."""
    config = _model_config(_vision_config(composite=False, layers=0), **_DEP2)
    assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) == _DEP2_TARGET


# The layouts the default target is gated on (routing's ``_DEFAULT_LAYOUTS``), as mapping kwargs.
_DEFAULT_LAYOUTS = {
    "dep4": dict(world_size=4, tp_size=4, enable_attention_dp=True, moe_ep_size=4, moe_tp_size=1),
    "dep6": dict(world_size=6, tp_size=6, enable_attention_dp=True, moe_ep_size=6, moe_tp_size=1),
    "dep16": dict(world_size=16, tp_size=16, enable_attention_dp=True, moe_ep_size=16, moe_tp_size=1),
    "tp4-ep4": dict(world_size=4, tp_size=4, moe_ep_size=4, moe_tp_size=1),
    "tp4-moe-tp4": dict(world_size=4, tp_size=4, moe_ep_size=1, moe_tp_size=4),
    "tp4-moe-tp2-ep2": dict(world_size=4, tp_size=4, moe_ep_size=2, moe_tp_size=2),
    "tp8-ep8": dict(world_size=8, tp_size=8, moe_ep_size=8, moe_tp_size=1),
    "tp8-moe-tp8": dict(world_size=8, tp_size=8, moe_ep_size=1, moe_tp_size=8),
    "tp8-moe-tp2-ep4": dict(world_size=8, tp_size=8, moe_ep_size=4, moe_tp_size=2),
}


@pytest.mark.parametrize("mode", ["unset", "auto", "require"])
@pytest.mark.parametrize("spec_config", [None, _DSPARK], ids=["no-spec", "dspark"])
@pytest.mark.parametrize("layout", sorted(_DEFAULT_LAYOUTS))
def test_the_default_target_takes_the_measured_layouts_no_specific_target_claims(mode, spec_config, layout):
    config = _model_config(_config(), spec_config, **_DEFAULT_LAYOUTS[layout])
    assert _qualified(modeling_v2_resolve(config, _llm_args(mode))) == _DEFAULT_TARGET


@pytest.mark.parametrize("layout", ["dep4", "dep16"])
def test_the_default_target_runs_megamoe_and_the_split_head_under_pure_expert_parallelism(layout):
    config = _model_config(_config(), _DSPARK, **dict(_DEFAULT_LAYOUTS[layout], enable_lm_head_tp_in_adp=True))
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", **_MEGAMOE))) == _DEFAULT_TARGET


def test_specific_targets_come_first():
    """A layout with its own target never lands on the default target, which would also compute it."""
    for kwargs in (_DEP2, _TEP8, _DEP8, _DEP32):
        config = _model_config(_config(), None, **kwargs)
        assert _qualified(modeling_v2_resolve(config, _llm_args("require"))) != _DEFAULT_TARGET


@pytest.mark.parametrize(
    "layout, args, missed",
    [
        ("tp4-ep4", _MEGAMOE, "MEGAMOE_CUTEDSL outside attention DP"),
        ("tp4-moe-tp2-ep2", dict(moe_config={"backend": "CUTLASS"}), "moe_config.backend CUTLASS"),
        ("dep4", dict(moe_config={"backend": "CUTEDSL"}), "moe_config.backend CUTEDSL"),
        ("dep4", dict(moe_config={"load_balancer": {"num_slots": 384}}), "load_balancer without MEGAMOE"),
        ("dep6", dict(bounded_replay_on_generation=True), "bounded_replay_on_generation"),
        ("tp8-ep8", dict(bounded_replay_on_generation=True), "bounded_replay_on_generation"),
    ],
    ids=["megamoe-under-tp", "cutlass", "cutedsl", "balancer-without-megamoe", "brg-dep6", "brg-tp8"],
)
def test_the_default_target_refuses_what_it_does_not_run(layout, args, missed):
    config = _model_config(_config(), None, **_DEFAULT_LAYOUTS[layout])
    assert modeling_v2_resolve(config, _llm_args("auto", **args)) is None
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require", **args))


@pytest.mark.parametrize("spec_config, missed", [(MTPDecodingConfig(max_draft_len=3), "MTP"), (_NGRAM, "NGRAM")])
def test_the_default_target_runs_dspark_only(spec_config, missed):
    config = _model_config(_config(), spec_config, **_DEFAULT_LAYOUTS["dep4"])
    with pytest.raises(ValueError, match=missed):
        modeling_v2_resolve(config, _llm_args("require"))


@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize("layout", ["tp4-ep4", "dep4"])
def test_the_default_target_is_text_only(composite, layout):
    """A checkpoint with a vision tower routes to it only with the encoder disabled (it has no image class; its text
    class refuses image requests, test_modeling_v2_deepseek_v41_image_input.py); an encoder-only worker is declined; a
    vision tower of no layer is text only."""
    config = _model_config(_vision_config(composite), None, **_DEFAULT_LAYOUTS[layout])
    assert modeling_v2_resolve(config, _llm_args("auto")) is None
    with pytest.raises(ValueError, match="vision tower served with its image encoder"):
        modeling_v2_resolve(config, _llm_args("require"))
    with pytest.raises(ValueError, match="vision tower served with its image encoder"):
        modeling_v2_resolve(config, _llm_args("require", mm_encoder_only=True))
    assert _qualified(modeling_v2_resolve(config, _llm_args("require", disable_mm_encoder=True))) == _DEFAULT_TARGET
    text_only = _model_config(_vision_config(composite, layers=0), None, **_DEFAULT_LAYOUTS[layout])
    assert _qualified(modeling_v2_resolve(text_only, _llm_args("require"))) == _DEFAULT_TARGET


@pytest.mark.parametrize("role", ["context", "generation", None])
@pytest.mark.parametrize("layout", ["dep6", "tp8-ep8"])
def test_the_default_target_declines_both_sides_of_the_remote_tail_replay(monkeypatch, role, layout):
    """`bounded_replay_on_generation`: neither the context-only worker (dep2's) nor the generation-side tail replay
    (dep8 / dep32 / tep8) runs on the default target, whatever the role."""
    monkeypatch.delenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", raising=False)
    if role is None:
        monkeypatch.delenv("TRTLLM_DISAGG_ROLE", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_DISAGG_ROLE", role)
    config = _model_config(_config(), None, **_DEFAULT_LAYOUTS[layout])
    assert modeling_v2_resolve(config, _llm_args("auto", bounded_replay_on_generation=True)) is None
    with pytest.raises(ValueError, match="bounded_replay_on_generation"):
        modeling_v2_resolve(config, _llm_args("require", bounded_replay_on_generation=True))
    extra = _model_config(_config(), None, **_DEFAULT_LAYOUTS[layout])
    extra.extra_attrs["bounded_replay_on_generation"] = True  # as stock's loader hands it to the model
    assert modeling_v2_resolve(extra, _llm_args("auto")) is None

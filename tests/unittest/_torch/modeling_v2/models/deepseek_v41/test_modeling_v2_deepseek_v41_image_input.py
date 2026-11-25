# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image input on the DeepSeek-V4.1 targets, on CPU: a text class (no image encoder in front) refuses a context request
that carries image input with a per-request error, admits a generation-only request of an image prompt and every text
request; behind the image model (`serves_images`) image requests are admitted; each target reads the image token of a
vision checkpoint for its Engram hash."""

import importlib
from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.image_input import (
    refuse_image_request,
    refuse_image_step,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.inputs.multimodal import MultimodalParams

pytestmark = pytest.mark.cpu_only

_IMG = 129264
_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
_TARGETS = {
    "dep2": ("flash_nvfp4__sm_103__dep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2"),
    "dep8": ("flash_nvfp4__sm_103__dep8", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep8"),
    "dep32": ("flash_nvfp4__sm_103__dep32", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep32"),
    "tep8": ("flash_nvfp4__sm_103__tep8_moe_tp4_ep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2"),
}


def _module(name):
    return importlib.import_module(f"{_PKG}.{_TARGETS[name][0]}.modeling")


def _request(data, generation_only=False):
    return SimpleNamespace(py_multimodal_data=data, is_generation_only_request=generation_only)


_IMAGE = {"image": {"pixel_values": [torch.zeros(1)]}}


def test_refuse_image_input():
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request(_IMAGE))
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request({"multimodal_embedding": torch.zeros(4, 8)}))
    refuse_image_request(_request(_IMAGE, generation_only=True))  # its prompt ran on the context server
    refuse_image_request(_request(None))
    refuse_image_request(_request({}))
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_step([MultimodalParams(multimodal_data=_IMAGE)])
    refuse_image_step(None)
    refuse_image_step([])
    refuse_image_step([MultimodalParams()])
    # behind the image model the encoder is in front: nothing is refused
    refuse_image_request(_request(_IMAGE), serves_images=True)
    refuse_image_step([MultimodalParams(multimodal_data=_IMAGE)], serves_images=True)


@pytest.mark.parametrize("name", list(_TARGETS))
def test_each_target_refuses_image_context_requests_at_admission(name):
    """The engine calls `validate_request` on every new request (a ValueError fails that request only)."""
    target = getattr(_module(name), _TARGETS[name][1])
    text = SimpleNamespace(model=SimpleNamespace(serves_images=False))  # the text class
    with pytest.raises(ValueError, match="serves text only"):
        target.validate_request(text, _request(_IMAGE))
    target.validate_request(text, _request(_IMAGE, generation_only=True))
    target.validate_request(text, _request(None))
    # the image class's text target (`serve_images`): the image model encodes the request
    target.validate_request(SimpleNamespace(model=SimpleNamespace(serves_images=True)), _request(_IMAGE))


@pytest.mark.parametrize("name", list(_TARGETS))
@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize("layers, image", [(32, _IMG), (0, None)], ids=["vision", "vision-0-layers"])
def test_engram_reads_the_image_token_of_a_vision_checkpoint(name, composite, layers, image):
    """`_text_model_config` hands the Engram hash the image token (hashed as the pad id: the n-gram break stock
    applies) only for a checkpoint with a vision tower, the composite and the flat form alike."""
    vision = PretrainedConfig(num_hidden_layers=layers)
    text = PretrainedConfig(architectures=["DeepseekV41ForCausalLM"], hidden_size=8)
    if composite:
        cfg = PretrainedConfig(text_config=text, vision_config=vision, image_token_id=_IMG)
    else:
        cfg = text
        cfg.vision_config, cfg.image_token_id = vision, _IMG
    out = _module(name)._text_model_config(ModelConfig(pretrained_config=cfg))
    assert out.pretrained_config is text
    assert text._v41_image_token_id == image


@pytest.mark.parametrize("name", list(_TARGETS))
def test_text_checkpoint_has_no_image_token(name):
    text = PretrainedConfig(architectures=["DeepseekV41ForCausalLM"], hidden_size=8)
    _module(name)._text_model_config(ModelConfig(pretrained_config=text))
    assert text._v41_image_token_id is None


_DEFAULT = "flash_nvfp4__sm_103__default"


def test_the_default_target_refuses_image_context_requests_at_admission():
    """The default target is text only (no image class): its `validate_request` refuses a context request with image
    input and admits a generation-only request of an image prompt and every text request."""
    target = importlib.import_module(f"{_PKG}.{_DEFAULT}.modeling").ModelingV2DeepseekV41FlashNvfp4Sm103Default
    with pytest.raises(ValueError, match="serves text only"):
        target.validate_request(None, _request(_IMAGE))
    target.validate_request(None, _request(_IMAGE, generation_only=True))
    target.validate_request(None, _request(None))


@pytest.mark.parametrize("path", ["attention_dp", "attention_tp"])
@pytest.mark.parametrize("composite", [False, True], ids=["flat", "composite"])
@pytest.mark.parametrize("layers, image", [(32, _IMG), (0, None)], ids=["vision", "vision-0-layers"])
def test_the_default_target_engram_reads_the_image_token(path, composite, layers, image):
    """Both forward paths of the default target hand their Engram hash the image token of a vision checkpoint (the
    n-gram break of a generation-only request's image prompt), the composite and the flat form alike."""
    core = importlib.import_module(f"{_PKG}.{_DEFAULT}.{path}.core")
    vision = PretrainedConfig(num_hidden_layers=layers)
    text = PretrainedConfig(architectures=["DeepseekV41ForCausalLM"], hidden_size=8)
    if composite:
        cfg = PretrainedConfig(text_config=text, vision_config=vision, image_token_id=_IMG)
    else:
        cfg = text
        cfg.vision_config, cfg.image_token_id = vision, _IMG
    out = core._text_model_config(ModelConfig(pretrained_config=cfg))
    assert out.pretrained_config is text
    assert text._v41_image_token_id == image

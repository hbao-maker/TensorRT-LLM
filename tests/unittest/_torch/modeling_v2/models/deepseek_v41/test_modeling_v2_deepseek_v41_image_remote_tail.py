# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An image prompt with `bounded_replay_on_generation` on the DeepSeek-V4.1 targets, as stock declines it.

Stock does not run disaggregated remote-tail replay for a request with image input: the executor's admission
(`PyExecutor._configure_csa2_remote_tail`) fails such a request on either side, the context-only source and the
generation-side destination whose tail replay could contain image tokens ("CSA2 disaggregated remote-tail replay
currently supports text prompts only"). The check reads the model's remote-tail switches through
`model_engine.model.model`; the targets' text classes and their image classes (which hand the engine the text target's
attributes) expose the same switches, so the same per-request error reaches every target. Text requests are attached
their remote-tail contract as before. Served with `disable_mm_encoder`, a request carries no image input at all (the
text classes refuse an image context request themselves: test_modeling_v2_deepseek_v41_image_input.py)."""

import importlib
from types import SimpleNamespace as NS

import pytest
from torch import nn

from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestType
from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor

_PKG = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
_IMAGE_CLASSES = {
    "dep2": ("flash_nvfp4__sm_103__dep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep2Images"),
    "dep8": ("flash_nvfp4__sm_103__dep8", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep8Images"),
    "dep32": ("flash_nvfp4__sm_103__dep32", "ModelingV2DeepseekV41FlashNvfp4Sm103Dep32Images"),
    "tep8": ("flash_nvfp4__sm_103__tep8_moe_tp4_ep2", "ModelingV2DeepseekV41FlashNvfp4Sm103Tep8MoeTp4Ep2Images"),
}
_WINDOW, _PROMPT = 140, 1000


def _core(context_only: bool) -> nn.Module:
    """The text target's core as the executor reads it (the switches `csa2_remote_tail_policy` sets)."""
    core = nn.Module()
    core.disagg_remote_tail_replay, core.disagg_context_only = True, context_only
    core.decoder_replay_split, core.decoder_replay_window = 20, _WINDOW
    return core


def _image_model(name: str, core: nn.Module) -> nn.Module:
    """The target's image class around a text target whose core is `core` (no weights: only attribute delegation)."""
    mod, cls = _IMAGE_CLASSES[name]
    images = object.__new__(getattr(importlib.import_module(f"{_PKG}.{mod}.modeling"), cls))
    nn.Module.__init__(images)
    llm = nn.Module()
    llm.model = core
    images.llm = llm
    return images


def _executor(model) -> NS:
    return NS(model_engine=NS(model=model), kv_cache_transceiver=NS(pipeline_transfer_enabled=False),
              scheduler=NS(max_context_length=None), max_num_tokens=8192, is_warmup=False)


def _request(source: bool, image: bool) -> NS:
    params = NS(remote_tail_start=None if source else _PROMPT - _WINDOW, schedule_style=None)
    kind = LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY if source else LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY
    return NS(py_disaggregated_params=params, py_llm_request_type=kind, prompt_len=_PROMPT, py_beam_width=1,
              py_return_context_logits=False, py_multimodal_data={"image": {"pixel_values": [1]}} if image else None)


@pytest.mark.parametrize("name", list(_IMAGE_CLASSES))
@pytest.mark.parametrize("source", [True, False], ids=["source", "destination"])
def test_an_image_request_is_declined_per_request(name, source):
    """An image request is failed at admission on the context-only source (dep2's role) and on the generation side
    (the tail replay); a text request gets its remote-tail contract, through the image class as through the text one."""
    images = _image_model(name, _core(context_only=source and name == "dep2"))
    assert images.model is images.llm.model  # the engine reads the text target's switches through the image class
    ex = _executor(images)
    with pytest.raises(ValueError, match="supports text prompts only"):
        PyExecutor._configure_csa2_remote_tail(ex, _request(source, image=True))
    text = _request(source, image=False)
    PyExecutor._configure_csa2_remote_tail(ex, text)
    assert text.py_csa2_remote_tail_mode == ("source" if source else "destination")
    assert text.py_csa2_remote_tail_start == _PROMPT - _WINDOW

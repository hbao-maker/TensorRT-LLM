# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The image model's step plumbing, on CPU: the image-row mask, what the text target is handed, and the text targets'
refusal of image input without the image encoder."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.vision.image_model import (
    DeepseekV41ImageModel,
    image_rows_mask,
    refuse_image_request,
    refuse_image_step,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams

pytestmark = pytest.mark.cpu_only

_IMG = 129264


def test_mask_from_the_executor_indices_and_from_the_ids():
    ids = torch.tensor([5, _IMG, _IMG, _IMG, _IMG, 9, 7])
    by_index = image_rows_mask(ids, torch.tensor([1, 2, 3, 4]), _IMG, "cpu")
    by_id = image_rows_mask(ids, None, _IMG, "cpu")
    assert by_index.tolist() == by_id.tolist() == [False, True, True, True, True, False, False]
    assert not image_rows_mask(ids, torch.tensor([], dtype=torch.long), _IMG, "cpu").any()


def _bare_model():
    model = DeepseekV41ImageModel.__new__(DeepseekV41ImageModel)
    torch.nn.Module.__init__(model)
    model.image_token_id = _IMG
    return model


def test_text_target_gets_raw_ids_embeddings_and_mask():
    ids = torch.tensor([5, _IMG, _IMG, _IMG, _IMG, 9])
    embeds = torch.randn(6, 8)
    kwargs = _bare_model().get_language_model_forward_kwargs(
        attn_metadata="md", input_ids=None, raw_input_ids=ids, position_ids="pos", inputs_embeds=embeds,
        mm_inputs=None, return_context_logits=False, spec_metadata="spec", resource_manager="rm",
        context_requests=["r"], mm_token_indices=torch.tensor([1, 2, 3, 4]), text_token_indices=torch.tensor([0, 5]),
        num_generation_requests=0, multimodal_params=[],
    )
    assert kwargs["input_ids"] is ids and kwargs["inputs_embeds"] is embeds
    assert kwargs["image_mask"].tolist() == [False, True, True, True, True, False]
    assert kwargs["spec_metadata"] == "spec" and kwargs["resource_manager"] == "rm" and kwargs["context_requests"] == ["r"]
    assert not {"mm_token_indices", "text_token_indices", "num_generation_requests", "multimodal_params"} & set(kwargs)


def test_text_step_has_no_mask():
    ids = torch.tensor([5, 6])
    kwargs = _bare_model().get_language_model_forward_kwargs(
        attn_metadata="md", input_ids=ids, raw_input_ids=ids, position_ids="pos", inputs_embeds=None,
        mm_inputs=None, return_context_logits=False,
    )
    assert kwargs["input_ids"] is ids and kwargs["inputs_embeds"] is None and "image_mask" not in kwargs


def _request(data, generation_only=False):
    return SimpleNamespace(py_multimodal_data=data, is_generation_only_request=generation_only)


def test_text_targets_refuse_image_input_without_the_encoder():
    image = {"image": {"pixel_values": [torch.zeros(1)]}}
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request(image), serves_images=False)
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_request(_request({"multimodal_embedding": torch.zeros(4, 8)}), serves_images=False)
    refuse_image_request(_request(image), serves_images=True)
    refuse_image_request(_request(image, generation_only=True), serves_images=False)  # its prompt ran elsewhere
    refuse_image_request(_request(None), serves_images=False)
    refuse_image_request(_request({}), serves_images=False)
    step = [MultimodalParams(multimodal_data=image)]
    with pytest.raises(ValueError, match="disable_mm_encoder"):
        refuse_image_step(step, serves_images=False)
    refuse_image_step(step, serves_images=True)
    refuse_image_step([], serves_images=False)
    refuse_image_step([MultimodalParams()], serves_images=False)

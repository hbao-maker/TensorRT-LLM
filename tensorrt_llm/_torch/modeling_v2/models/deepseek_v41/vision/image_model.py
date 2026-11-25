# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image + text requests on a DeepSeek-V4.1 target: the family's image encoder in front of one of its text targets.

Each target declares its image class as a subclass of `DeepseekV41ImageModel` naming its text class
(`TEXT_TARGET`), and the family's `route` returns that class for a vision checkpoint served with the image encoder
on (`disable_mm_encoder` off). The flow is stock's (`DeepseekV41ForConditionalGeneration`):

* The input processor is the engine's for the checkpoint (stock's `DeepseekV41InputProcessor`): each image becomes
  `rows * (columns + 1) + 2` copies of `image_token_id` in the prompt, plus its patches.
* `MultimodalModelMixin` (the engine's contract for a multimodal model, as `DecoderModelForCausalLM` is for a text
  model) gathers each context request's image rows -- from the encoder cache, or by calling
  `encode_multimodal_inputs` (`DeepseekV41ImageEncoder`, the family's) -- keeps the rows of the current chunk, and
  writes them over the text embeddings of the image tokens.
* The text target runs on those embeddings with the raw token ids and the step's image-row mask, which select stock's
  three image behaviours: image rows route their experts with the vision bias (`ffn.gate.bias_vl`), take no Engram
  delta, and break the Engram n-grams of the text that follows them.

Text-only steps run the text target unchanged (no mask, no extra launch); generation steps are text. With
`disable_mm_encoder` the family routes to the text class itself, which refuses a step that carries image input.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin, PreparedLlmInputs
from tensorrt_llm._torch.models.modeling_utils import DecoderModelForCausalLM
from tensorrt_llm.inputs.multimodal import MultimodalParams

from ..image_input import refuse_image_request, refuse_image_step  # noqa: F401 (re-exported to the targets)
from .encoder import DeepseekV41ImageEncoder

# Forward keywords the mixin consumes or adds for itself, never handed to the text target.
_MM_ONLY = frozenset({"mm_token_indices", "text_token_indices", "num_generation_requests", "multimodal_params"})


def image_rows_mask(input_ids: torch.Tensor, mm_token_indices: Optional[torch.Tensor], image_token_id: int,
                    device) -> torch.Tensor:
    """bool `[len(input_ids)]`, True at the step's image-token rows: the executor's precomputed `mm_token_indices` (as
    stock builds its mask), else the rows whose raw id is the image token (the same rows, without the index list)."""
    if mm_token_indices is None:
        return input_ids.to(device) == image_token_id
    mask = torch.zeros(input_ids.shape[0], dtype=torch.bool, device=device)
    if mm_token_indices.numel():
        mask[mm_token_indices.to(device=device, non_blocking=True).long()] = True
    return mask


class DeepseekV41ImageModel(MultimodalModelMixin, nn.Module):
    """A text target (`TEXT_TARGET`, the subclass sets it) behind the family's image encoder."""

    TEXT_TARGET: type = None

    def __init__(self, model_config):
        nn.Module.__init__(self)
        assert self.TEXT_TARGET is not None, "an image class names its text target"
        config = model_config.pretrained_config
        self.model_config = model_config
        self.config = config
        self.image_token_id = int(config.image_token_id)
        if issubclass(self.TEXT_TARGET, DecoderModelForCausalLM) and not model_config.skip_create_weights_in_init:
            # What `AutoModelForCausalLM.from_config` sets for a text target built directly: its modules create their
            # weights after its constructor (`__post_init__`), once it has configured them.
            model_config._frozen = False
            model_config.skip_create_weights_in_init = True
            model_config._frozen = True
        self.llm = self.TEXT_TARGET(model_config)
        # Before the weights load: the text target then loads the vision routing bias and takes image steps eagerly.
        self.llm.model.serve_images()
        self.encoder = DeepseekV41ImageEncoder(config.vision_config, self.embedding_dim, self.embedding_dtype)

    # ---- the text target, as the engine reads it -------------------------------------------------------------
    def __getattr__(self, name: str):
        """What the image model does not define is the text target's (its engine hooks, draft model, lm_head, ...),
        so the engine drives the text target exactly as when it is the model itself."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name in ("llm", "encoder") or name.startswith("__"):
                raise
            return getattr(super().__getattr__("llm"), name)

    @property
    def language_model(self):
        return self.llm

    @property
    def samples_from_vocab_shards(self) -> bool:
        return self.llm.samples_from_vocab_shards

    @property
    def text_embedding_layer(self):
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.llm.model.cfg.hidden_size

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    @property
    def dtype(self) -> torch.dtype:
        """The model's dtype as the engine reads it off a multimodal model (stock's wrapper is a `PreTrainedModel`),
        e.g. for the encoder's dummy inputs of the memory estimate: the text embedding's."""
        return self.embedding_dtype

    @property
    def device(self) -> torch.device:
        return self.text_embedding_layer.weight.device

    @property
    def multimodal_token_ids(self):
        return torch.tensor([self.image_token_id], dtype=torch.int32, device="cpu")

    @property
    def mm_token_ids(self):
        return self.multimodal_token_ids

    @property
    def multimodal_data_device_paths(self) -> list[str]:
        return ["image.pixel_values", "multimodal_embedding"]

    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return cls.TEXT_TARGET.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None):
        return cls.TEXT_TARGET.get_preferred_kv_cache_manager_version(pretrained_config)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None):
        return cls.TEXT_TARGET.get_preferred_transceiver_runtime(pretrained_config)

    # ---- weights --------------------------------------------------------------------------------------------
    def load_weights(self, weights, weight_mapper=None) -> None:
        device = torch.device("cuda", torch.cuda.current_device())
        self.encoder.load(weights, device)
        self.llm.load_weights(weights)

    def post_load_weights(self) -> None:
        self.llm.post_load_weights()

    # ---- images ---------------------------------------------------------------------------------------------
    @torch.inference_mode()
    def encode_multimodal_inputs(self, multimodal_params: Sequence[MultimodalParams]) -> torch.Tensor:
        """The image rows of `multimodal_params`, request after request, image after image (stock's)."""
        spans = []
        device = self.text_embedding_layer.weight.device
        for param in multimodal_params:
            images = param.multimodal_data["image"]
            patches, grids = images["pixel_values"], images["image_grid_hw"]
            llm_grids, lengths = images["llm_grid_hw"], images["num_tokens"]
            if not (len(patches) == len(grids) == len(llm_grids) == len(lengths)):
                raise ValueError("DeepSeek-V4.1 image metadata lengths disagree")
            for pixels, (height, width), (rows, columns), length in zip(patches, grids, llm_grids, lengths):
                pixels = pixels.to(device=device, dtype=self.embedding_dtype)
                spans.append(self.encoder(pixels, int(height), int(width), int(rows), int(columns), int(length)))
        if not spans:
            raise ValueError("No images supplied to DeepSeek-V4.1 encoder")
        return torch.cat(spans)

    def get_language_model_forward_kwargs(
        self,
        *,
        attn_metadata: Any,
        input_ids: Optional[torch.Tensor],
        raw_input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        inputs_embeds: Optional[torch.Tensor],
        mm_inputs: PreparedLlmInputs,
        return_context_logits: bool,
        **forward_kwargs: Any,
    ) -> dict[str, Any]:
        """The text target's forward arguments: the engine's own, with the raw token ids kept (the Engram hashes them
        and the draft reads them, as stock's `orig_input_ids`), plus, on a step with image rows, their embeddings
        and the image-row mask."""
        kwargs = {k: v for k, v in forward_kwargs.items() if k not in _MM_ONLY}
        kwargs.update(
            attn_metadata=attn_metadata,
            input_ids=raw_input_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            return_context_logits=return_context_logits,
        )
        if inputs_embeds is not None:
            if raw_input_ids is None or raw_input_ids.shape[0] != inputs_embeds.shape[0]:
                raise ValueError("DeepSeek-V4.1 image steps need one raw token id per input embedding")
            kwargs["image_mask"] = image_rows_mask(
                raw_input_ids, forward_kwargs.get("mm_token_indices"), self.image_token_id, inputs_embeds.device
            )
        return kwargs

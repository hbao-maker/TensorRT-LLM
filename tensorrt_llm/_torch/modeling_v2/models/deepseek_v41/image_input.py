# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Image input on the DeepSeek-V4.1 targets: who encodes it, and who refuses it.

A checkpoint with a vision tower served with the image encoder on is routed to the target's image class
(`vision/image_model.py`: the family's image encoder in front of the text target), whose text target runs with
`serves_images` set. Served with `disable_mm_encoder`, it reaches the text class, which has no encoder in front and so
refuses a context request carrying image input instead of answering it from placeholder embeddings of the image tokens.
A generation-only request of an image prompt is admitted either way: its prompt ran on the context server, and the image
tokens in its look-back break the Engram n-grams as they do there (each target's `_EngramHash` hashes the image token as
the pad id).

With `bounded_replay_on_generation` an image request runs on neither side: stock's executor fails it at admission
(`PyExecutor._configure_csa2_remote_tail`: "currently supports text prompts only"), on the context-only source and on
the generation side whose tail replay could hold image tokens, and it reads the switches the text and image classes
expose alike (test_modeling_v2_deepseek_v41_image_remote_tail.py).
"""

from __future__ import annotations

_NO_ENCODER = (
    "this DeepSeek-V4.1 modeling_v2 target serves text only: the engine runs with disable_mm_encoder, so no image "
    "encoder is built and the image input would be embedded as placeholder tokens. Serve images with "
    "disable_mm_encoder off (routing then builds the target's image class, the family's image encoder in front)."
)


def refuse_image_request(request, serves_images: bool = False) -> None:
    """A target's admission check (`validate_request`): without the image encoder in front (`serves_images` off), a
    context request whose prompt still has to be encoded (image pixels or precomputed image embeddings) is refused, as a
    per-request error. A generation-only request is admitted."""
    if serves_images or request.is_generation_only_request:
        return
    data = getattr(request, "py_multimodal_data", None) or {}
    if data.get("image") is not None or data.get("multimodal_embedding") is not None:
        raise ValueError(_NO_ENCODER)


def refuse_image_step(multimodal_params, serves_images: bool = False) -> None:
    """The forward-time form of `refuse_image_request`, for a step that carries image input anyway."""
    if not serves_images and multimodal_params and any(p is not None and p.has_content() for p in multimodal_params):
        raise ValueError(_NO_ENCODER)

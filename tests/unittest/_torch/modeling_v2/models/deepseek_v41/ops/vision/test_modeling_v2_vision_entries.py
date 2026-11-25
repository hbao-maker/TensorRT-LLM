# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The image encoder's entries (`ops/vision`): every certified cell against its reference.

Gates: the bit-moving entries (`vision_patch_merge`, `image_span`) are exact; `vision_rms_norm` and `vision_rope_2d`
compute in fp32 and round once, so they sit within one bf16 step of their float64 references; `vision_linear`
(cuBLAS, fp32 accumulation) and `vision_attention` (SDPA) are gated by relative RMS against fp32 references.
"""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.image_span import image_span
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_attention import (
    vision_attention,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_linear import vision_linear
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_patch_merge import (
    vision_patch_merge,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_rms_norm import (
    vision_rms_norm,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.vision.vision_rope_2d import (
    vision_rope_2d,
    vision_rope_tables,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3), reason="sm_103 (GB300)"
)

_BF16_STEP = 2.0**-7  # one bf16 ulp at 1.0, relative


def _rel_rms(out, ref):
    out, ref = out.double(), ref.double()
    return ((out - ref).pow(2).mean().sqrt() / ref.pow(2).mean().sqrt().clamp_min(1e-30)).item()


def _ids(entry):
    return [c.why[:44] for c in entry.CELLS]


@pytest.mark.parametrize("cell", vision_linear.CELLS, ids=_ids(vision_linear))
def test_vision_linear(cell, certifying):
    certifying(vision_linear)
    s, gen = cell.spec, gen_for(cell)
    x = torch.randn(s["m"], s["k"], generator=gen, device=DEV).bfloat16()
    w = (torch.randn(s["n"], s["k"], generator=gen, device=DEV) * s["k"] ** -0.5).bfloat16()
    b = torch.randn(s["n"], generator=gen, device=DEV).bfloat16() if s["bias"] else None
    out = vision_linear(x, w, b)
    assert out.dtype == torch.bfloat16 and out.shape == (s["m"], s["n"])
    assert torch.equal(out, torch.nn.functional.linear(x, w, b)), "the GEMM stock's nn.Linear runs"
    assert _rel_rms(out, vision_linear.reference(x, w, b)) < 4e-3


@pytest.mark.parametrize("cell", vision_rms_norm.CELLS, ids=_ids(vision_rms_norm))
def test_vision_rms_norm(cell, certifying):
    certifying(vision_rms_norm)
    s, gen = cell.spec, gen_for(cell)
    x = (torch.randn(s["m"], s["d"], generator=gen, device=DEV) * 3).bfloat16()
    w = torch.rand(s["d"], generator=gen, device=DEV) + 0.5
    out, ref = vision_rms_norm(x, w, 1e-6), vision_rms_norm.reference(x, w, 1e-6)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_BF16_STEP, atol=0)


@pytest.mark.parametrize("cell", vision_rope_2d.CELLS, ids=_ids(vision_rope_2d))
def test_vision_rope_2d(cell, certifying):
    certifying(vision_rope_2d)
    s, gen = cell.spec, gen_for(cell)
    n = s["n_h"] * s["n_w"]
    x = torch.randn(n, s["heads"], s["head_dim"], generator=gen, device=DEV).bfloat16()
    cos, sin = vision_rope_tables(s["n_h"], s["n_w"], s["head_dim"] // 2, 10000.0, DEV)
    assert cos.shape == (n, 1, s["head_dim"] // 2)
    out, ref = vision_rope_2d(x, cos, sin), vision_rope_2d.reference(x, cos, sin)
    torch.testing.assert_close(out.float(), ref.float(), rtol=_BF16_STEP, atol=_BF16_STEP * 2**-6)
    # the first half of the frequencies follow the row, the second the column: a patch's angles in closed form
    r, c, half = s["n_h"] - 1, s["n_w"] - 1, s["head_dim"] // 4
    inv = 1.0 / (10000.0 ** (torch.arange(0, s["head_dim"] // 2, 2, device=DEV, dtype=torch.float32) / (s["head_dim"] // 2)))
    assert torch.allclose(cos[-1, 0, :half], torch.cos(r * inv)) and torch.allclose(cos[-1, 0, half:], torch.cos(c * inv))


@pytest.mark.parametrize("cell", vision_attention.CELLS, ids=_ids(vision_attention))
def test_vision_attention(cell, certifying):
    certifying(vision_attention)
    s, gen = cell.spec, gen_for(cell)
    q, k, v = (torch.randn(s["n"], s["heads"], s["head_dim"], generator=gen, device=DEV).bfloat16() for _ in range(3))
    out = vision_attention(q, k, v)
    assert out.shape == (s["n"], s["heads"] * s["head_dim"]) and out.dtype == torch.bfloat16
    assert _rel_rms(out, vision_attention.reference(q, k, v)) < 1e-2


@pytest.mark.parametrize("cell", vision_patch_merge.CELLS, ids=_ids(vision_patch_merge))
def test_vision_patch_merge(cell, certifying):
    certifying(vision_patch_merge)
    s, gen = cell.spec, gen_for(cell)
    x = torch.randn(s["n_h"] * s["n_w"], s["c"], generator=gen, device=DEV).bfloat16()
    out = vision_patch_merge(x, s["n_h"], s["n_w"], s["r"])
    rows, cols = -(-s["n_h"] // s["r"]), -(-s["n_w"] // s["r"])
    assert out.shape == (rows * cols, s["c"] * s["r"] ** 2)
    assert torch.equal(out, vision_patch_merge.reference(x, s["n_h"], s["n_w"], s["r"]))


@pytest.mark.parametrize("cell", image_span.CELLS, ids=_ids(image_span))
def test_image_span(cell, certifying):
    certifying(image_span)
    s, gen = cell.spec, gen_for(cell)
    f = torch.randn(s["rows"] * s["columns"], s["hidden"], generator=gen, device=DEV).bfloat16()
    start, newline, end = (torch.randn(s["hidden"], generator=gen, device=DEV).bfloat16() for _ in range(3))
    out = image_span(f, s["rows"], s["columns"], start, newline, end)
    assert out.shape[0] == s["rows"] * (s["columns"] + 1) + 2
    assert torch.equal(out, image_span.reference(f, s["rows"], s["columns"], start, newline, end))


def test_rejections():
    x = torch.zeros(4, 8, dtype=torch.bfloat16, device=DEV)
    with pytest.raises(AssertionError):
        vision_rms_norm.is_valid(x, torch.ones(8, dtype=torch.bfloat16, device=DEV), 1e-6)  # stock's weight is fp32
    with pytest.raises(AssertionError):
        vision_linear.is_valid(x.float(), torch.zeros(3, 8, device=DEV))
    with pytest.raises(AssertionError):
        vision_patch_merge.is_valid(x, 3, 3, 2)  # 4 rows are not a 3 x 3 grid
    with pytest.raises(AssertionError):
        image_span.is_valid(x, 2, 3, x[0], x[0], x[0])  # 4 rows are not a 2 x 3 image

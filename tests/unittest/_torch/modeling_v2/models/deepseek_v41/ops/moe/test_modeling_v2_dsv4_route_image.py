# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The image-row routing entries: each row picks its experts with its own selection bias (the vision bias on image
rows), against the plain-torch per-row-bias routing, and the in-place row select.

Gate (as `dsv4_route_nvfp4_quant`'s): each row picks the same expert set, weights within one bf16 step; the quantized
rows are the text launch's, bit for bit; a step without image rows is exactly the text launch."""

import pytest
import torch
from _v41_util import DEV, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_image import (
    dsv4_route_image,
    dsv4_route_nvfp4_quant_image,
    image_rows_select,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.moe.dsv4_route_nvfp4_quant import (
    dsv4_route,
    dsv4_route_nvfp4_quant,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10, reason="CuTe-DSL kernel, SM100 family"
)

_E, _H, _K, _SCALE = 384, 5120, 6, 1.5


def _inputs(cell):
    s, gen = cell.spec, gen_for(cell)
    n = s["n"]
    logits = torch.randn(n, _E, generator=gen, device=DEV) * 3
    bias = torch.randn(_E, generator=gen, device=DEV) * 0.5
    bias_vl = torch.randn(_E, generator=gen, device=DEV) * 0.5  # a different bias: image rows pick other experts
    image = torch.zeros(n, dtype=torch.bool, device=DEV)
    image[s["image"][0] : s["image"][1]] = True
    x = (torch.randn(n, _H, generator=gen, device=DEV) * 2).bfloat16()
    g = (448.0 * 6.0 / x.float().abs().max()).reshape(1).to(DEV)
    return logits, bias, bias_vl, image, x, g


def _same_routing(ids, w, rids, rw):
    order, rorder = ids.long().argsort(-1), rids.long().argsort(-1)
    assert torch.equal(ids.long().gather(1, order), rids.long().gather(1, rorder)), "different experts picked"
    assert (w.float().gather(1, order) - rw.float().gather(1, rorder)).abs().max() <= 0.008 * rw.float().max()


@pytest.mark.parametrize("cell", dsv4_route_image.CELLS, ids=[c.why[:44] for c in dsv4_route_image.CELLS])
def test_route_image(cell, certifying):
    certifying(dsv4_route_image)
    logits, bias, bias_vl, image, x, _ = _inputs(cell)
    ids, w = dsv4_route_image(logits, bias, bias_vl, image, _K, _SCALE, _H, torch.float32)
    rids, rw = dsv4_route_image.reference(logits, bias, bias_vl, image, _K, _SCALE, _H, torch.float32)
    _same_routing(ids, w, rids, rw)
    text_ids, text_w = dsv4_route(logits, bias, _K, _SCALE, _H, torch.float32)
    assert torch.equal(ids[~image], text_ids[~image]) and torch.equal(w[~image], text_w[~image])


@pytest.mark.parametrize("cell", dsv4_route_nvfp4_quant_image.CELLS,
                         ids=[c.why[:44] for c in dsv4_route_nvfp4_quant_image.CELLS])  # fmt: skip
def test_route_nvfp4_quant_image(cell, certifying):
    certifying(dsv4_route_nvfp4_quant_image)
    logits, bias, bias_vl, image, x, g = _inputs(cell)
    ids, w, xq, sf = dsv4_route_nvfp4_quant_image(logits, bias, bias_vl, image, x, g, _K, _SCALE)
    rids, rw, _, _ = dsv4_route_nvfp4_quant_image.reference(logits, bias, bias_vl, image, x, g, _K, _SCALE)
    _same_routing(ids, w, rids, rw)
    tids, tw, txq, tsf = dsv4_route_nvfp4_quant(logits, bias, x, g, _K, _SCALE)
    assert torch.equal(xq, txq) and torch.equal(sf, tsf), "the input is quantized by the text launch"
    assert torch.equal(ids[~image], tids[~image]) and torch.equal(w[~image], tw[~image])


@pytest.mark.parametrize("cell", image_rows_select.CELLS, ids=[c.why[:44] for c in image_rows_select.CELLS])
def test_image_rows_select(cell, certifying):
    certifying(image_rows_select)
    s, gen = cell.spec, gen_for(cell)
    n, k = s["n"], s["top_k"]
    ids = torch.randint(0, 48, (n + 7, k), generator=gen, device=DEV)  # a bucket: live rows first
    w = torch.rand(n + 7, k, generator=gen, device=DEV)
    iid, iw = torch.randint(0, 48, (n, k), generator=gen, device=DEV), torch.rand(n, k, generator=gen, device=DEV)
    image = torch.rand(n, generator=gen, device=DEV) < 0.5
    ref = image_rows_select.reference(ids[:n], w[:n], iid, iw, image)
    tail = (ids[n:].clone(), w[n:].clone())
    image_rows_select(ids[:n], w[:n], iid, iw, image)
    assert torch.equal(ids[:n], ref[0]) and torch.equal(w[:n], ref[1])
    assert torch.equal(ids[n:], tail[0]) and torch.equal(w[n:], tail[1]), "rows past the live ones are untouched"


def test_rejections():
    cell = dsv4_route_image.CELLS[0]
    logits, bias, bias_vl, image, x, g = _inputs(cell)
    dsv4_route_image.is_valid(logits, bias, bias_vl, image, _K, _SCALE, _H)
    with pytest.raises(AssertionError):
        dsv4_route_image.is_valid(logits, bias, bias_vl, image.int(), _K, _SCALE, _H)
    with pytest.raises(AssertionError):
        dsv4_route_nvfp4_quant_image.is_valid(logits, bias, bias_vl.bfloat16(), image, x, g, _K, _SCALE)

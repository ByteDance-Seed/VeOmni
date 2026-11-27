# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Qwen4-Exp ``qsa_indexer`` eager row."""

import pytest
import torch

from veomni.ops import resolve_op
from veomni.utils.device import IS_NPU_AVAILABLE


def _select():
    return resolve_op("qsa_indexer", "standard", "eager").wrapper


@pytest.mark.parametrize(
    "device_type",
    ["cpu", pytest.param("npu", marks=pytest.mark.skipif(not IS_NPU_AVAILABLE, reason="requires Ascend"))],
)
@pytest.mark.parametrize("key_length", [511, 513, 29166])
@pytest.mark.parametrize("quantized", [False, True])
def test_eager_matches_inf_reference(device_type: str, key_length: int, quantized: bool):
    """Finite -1.0 masking matches an -inf reference, including cutoff ties."""
    rows, k = 143, min(512, key_length)
    scores = torch.rand((1, rows, key_length), generator=torch.Generator().manual_seed(38))
    if quantized:
        scores = (scores * 4).floor()
    visible = torch.zeros_like(scores, dtype=torch.bool)
    for row in range(rows - 6):
        count = min((1951 + row) // 4, key_length)
        visible[0, row, key_length - count :] = True
    visible[0, -3:, -1] = True
    reference = scores.masked_fill(~visible, float("-inf")).sort(dim=-1, descending=True).values[..., :k]

    picked, valid = _select()(scores.to(device_type), visible.to(device_type), k)
    picked, valid = picked.cpu(), valid.cpu()
    values = scores.gather(-1, picked).masked_fill(~valid, float("-inf"))

    torch.testing.assert_close(values.sort(dim=-1, descending=True).values, reference, rtol=0, atol=0)
    assert torch.equal(valid.sum(dim=-1), visible.sum(dim=-1).clamp(max=k))
    assert not valid[0, -6:-3].any()


def test_cpu_validity_gather_stays_boolean(monkeypatch):
    """The int8 gather exists for NPU bool gather, so CPU must not take it."""
    scores = torch.tensor([[[0.2, 0.0, 1.0]]])
    visible = torch.tensor([[[True, False, True]]])
    original = torch.Tensor.to
    casts: list[object] = []

    def recording_to(self, *args, **kwargs):
        if args and args[0] is torch.int8:
            casts.append(self.dtype)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", recording_to)
    blocks, valid = _select()(scores, visible, 2)

    assert casts == []
    assert valid.dtype is torch.bool
    assert blocks.shape == valid.shape == (1, 1, 2)
    assert valid.all()

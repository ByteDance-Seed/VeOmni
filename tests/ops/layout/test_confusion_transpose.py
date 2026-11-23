# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Contract for the Wan confusion-transpose helper."""

import pytest
import torch

from veomni.ops.layout.confusion_transpose import confusion_transpose
from veomni.utils.device import IS_NPU_AVAILABLE


def _npu_ready() -> bool:
    return bool(IS_NPU_AVAILABLE and torch.npu.is_available())


def test_confusion_transpose_matches_flatten_on_cpu(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("CPU input reached npu_confusion_transpose")

    monkeypatch.setattr(
        "veomni.ops.layout.confusion_transpose._npu_confusion_transpose",
        reject,
    )
    hidden = torch.randn(2, 3, 2, 4, 5)
    actual = confusion_transpose(hidden)
    expected = hidden.flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 40, 3)


@pytest.mark.skipif(not _npu_ready(), reason="NPU confusion transpose needs an available NPU")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_confusion_transpose_matches_flatten_on_npu(dtype: torch.dtype):
    hidden = torch.randn(2, 3, 2, 4, 5, device="npu", dtype=dtype, requires_grad=True)
    actual = confusion_transpose(hidden)
    reference_input = hidden.detach().to("cpu").requires_grad_(True)
    reference = reference_input.flatten(2).transpose(1, 2)
    tolerance = {"atol": 1e-5, "rtol": 1e-5} if dtype == torch.float32 else {"atol": 1e-2, "rtol": 1e-2}
    torch.testing.assert_close(actual.detach().cpu(), reference.detach(), **tolerance)

    grad_out = torch.randn_like(reference)
    actual.backward(grad_out.to(device="npu", dtype=dtype))
    reference.backward(grad_out)
    assert hidden.grad is not None
    torch.testing.assert_close(hidden.grad.cpu(), reference_input.grad, **tolerance)

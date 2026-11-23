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

"""CPU contract for the Wan confusion-transpose helper."""

import pytest
import torch

from veomni.ops.layout.confusion_transpose import confusion_transpose


def test_confusion_transpose_matches_flatten_on_cpu(monkeypatch):
    def reject(*args, **kwargs):
        pytest.fail("CPU input reached npu_confusion_transpose")

    monkeypatch.setattr(
        "veomni.ops.layout.confusion_transpose._npu_confusion_transpose",
        reject,
    )
    hidden = torch.randn(2, 3, 2, 2, 2)
    actual = confusion_transpose(hidden)
    expected = hidden.flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 8, 3)

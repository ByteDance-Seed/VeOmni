# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""rms_norm torch_npu adapter: same-dtype contract and empty fallback.

``aclnnRmsNorm`` has no mixed-dtype overload, so the NPU adapter rejects
misaligned ``x``/``weight`` dtypes loudly instead of silently casting
(callers align on their side); empty inputs fall back to the eager pair.
Both branches are host-side decisions and must hold on any device.
"""

import pytest
import torch

from veomni.ops.kernels.rms_norm.standard import torch_npu as adapter


def test_mixed_dtypes_are_rejected():
    x = torch.randn(2, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.float32)
    with pytest.raises(TypeError, match="share one dtype"):
        adapter.forward(x, weight, eps=1e-6)

    x = torch.randn(2, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.bfloat16)
    with pytest.raises(TypeError, match="share one dtype"):
        adapter.forward(x, weight, eps=1e-6)


def test_empty_input_falls_back_to_eager_pair():
    """Empty ``x`` takes the eager route: no dtype constraint, no NPU import."""
    x = torch.empty(0, 8, dtype=torch.bfloat16)
    weight = torch.randn(8, dtype=torch.float32)
    out, saved = adapter.forward(x, weight, eps=1e-6)
    assert out.shape == (0, 8)
    assert saved.metadata.empty

    grad_out = torch.empty(0, 8, dtype=torch.bfloat16)
    grad_x, grad_weight = adapter.backward(grad_out, saved)
    assert grad_x.shape == (0, 8)
    assert grad_weight.shape == (8,)

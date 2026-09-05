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

"""Numerical alignment tests for KERNEL_REGISTRY-registered kernels.

For each (op_name, variant, impl_name) tuple that ships a non-eager kernel,
bind an OpSlot to that kernel and compare its output against the canonical
eager implementation on random inputs. This is what guards against the
KernelRegistry silently routing the wrong implementation into an OpSlot
(e.g. a standard-variant kernel into a qwen3_5 slot).
"""

import pytest
import torch

import veomni.ops  # noqa: F401 - trigger kernel registrations
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


pytestmark = pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="kernels require CUDA")

DEVICE = get_device_type()


def _fresh_slot(op_name, variant, impl_name):
    slot = OpSlot(op_name, variant)
    slot.bind(impl_name)
    return slot


def test_load_balancing_loss_triton_matches_eager():
    from veomni.ops.kernels.load_balancing_loss.eager import load_balancing_loss_pytorch

    slot = _fresh_slot("load_balancing_loss", "standard", "triton")
    num_experts, top_k, num_layers, N = 8, 2, 3, 256
    gate_logits = tuple(torch.randn(N, num_experts, device=DEVICE, dtype=torch.float32) for _ in range(num_layers))
    out_kernel = slot(gate_logits, num_experts, top_k, None)
    out_eager = load_balancing_loss_pytorch(gate_logits, num_experts, top_k, None)
    assert torch.allclose(out_kernel, out_eager, atol=1e-4, rtol=1e-4)

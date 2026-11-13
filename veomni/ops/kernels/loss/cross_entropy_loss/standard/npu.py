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

"""Optional Ascend CE for the standard chunk implementation."""

import torch
import torch.nn.functional as F
from torch import Tensor

from veomni.utils.device import IS_NPU_AVAILABLE


if IS_NPU_AVAILABLE:
    import torch_npu


def can_use_fused_ce(hidden: Tensor, weight: Tensor, labels: Tensor) -> bool:
    """Conservatively stay within the native kernel's documented contract."""
    return (
        IS_NPU_AVAILABLE
        and not torch.compiler.is_compiling()
        and hidden.device.type == "npu"
        and weight.device == hidden.device == labels.device
        and hidden.dtype in (torch.float16, torch.bfloat16, torch.float32)
        and weight.dtype == hidden.dtype
        and labels.dtype == torch.int64
        and hidden.ndim == weight.ndim == 2
        and hidden.shape[1] == weight.shape[1]
        and weight.shape[0] > 0
        and labels.ndim == 1
        and labels.numel() == hidden.shape[0]
        and hidden.shape[0] <= 200000
    )


class _NPUCrossEntropy(torch.autograd.Function):
    """First-order native CE with the setup_context API required by torch.func."""

    @staticmethod
    def forward(logits: Tensor, labels: Tensor):
        """Return summed native CE and the log probabilities needed for backward."""
        loss, log_prob, _, _ = torch_npu.npu_cross_entropy_loss(logits, labels, None, "sum", -100, 0.0, 0.0, False)
        return loss.squeeze(), log_prob

    @staticmethod
    def setup_context(ctx, inputs, output):
        """Save labels and native log probabilities without differentiating the latter."""
        _, labels = inputs
        _, log_prob = output
        ctx.save_for_backward(log_prob, labels)
        ctx.mark_non_differentiable(log_prob)

    @staticmethod
    def backward(ctx, grad_loss, _grad_log_prob):
        """Return logits gradients scaled by the incoming scalar loss gradient."""
        log_prob, labels = ctx.saved_tensors
        grad_logits = torch_npu.npu_cross_entropy_loss_backward(
            grad_loss.reshape(1), log_prob, labels, None, None, None, "sum", -100, 0.0, 0.0
        )
        return grad_logits, None


def fused_linear_ce(
    hidden: Tensor, weight: Tensor, labels: Tensor, denominator: int | Tensor, ignore_index: int
) -> Tensor:
    """Compact ignored rows before projection, keeping an all-ignore zero graph."""
    valid = labels != ignore_index
    labels = labels[valid]
    if ignore_index != -100 and not bool(((labels >= 0) & (labels < weight.shape[0])).all()):
        # Native's fixed sentinel must not silently ignore an invalid -100
        # target when the caller selected a different ignore index.
        raise IndexError("Target out of bounds for the vocabulary")
    logits = F.linear(hidden[valid], weight).float()
    if labels.numel() == 0:
        loss = logits.sum()
    else:
        # Every ignored label has been removed. A fixed negative sentinel avoids
        # native-kernel constraints on caller-supplied positive ignore indices.
        loss, _ = _NPUCrossEntropy.apply(logits, labels)
    if isinstance(denominator, Tensor):
        denominator = denominator.to(device=loss.device, dtype=loss.dtype)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
    else:
        denominator = 1 if denominator == 0 else denominator
    return loss / denominator

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

"""standard token-level log-probability ``torch_npu`` cross-entropy adapter."""

from __future__ import annotations

import torch
from torch import Tensor

from veomni.utils.device import IS_NPU_AVAILABLE


def wrapper(
    logits: Tensor,
    labels: Tensor,
    *,
    ignore_index: int = -100,
) -> Tensor:
    """Per-token ``log p(y_t)`` via ``torch_npu.npu_cross_entropy_loss``.

    Fuses ``log_softmax``, target gather, and ``ignore_index`` zero-masking into
    a single Ascend NPU kernel call instead of decomposing into separate
    ``clamp + log_softmax + gather + where`` ops per chunk.
    Lazy-imports ``torch_npu`` when ``IS_NPU_AVAILABLE`` holds.
    """
    if not IS_NPU_AVAILABLE:
        raise RuntimeError("token_logprobs torch_npu implementation requires Ascend NPU runtime (IS_NPU_AVAILABLE).")

    import torch_npu

    batch_shape = logits.shape[:-1]
    logits_2d = logits.reshape(-1, logits.shape[-1])
    labels_1d = labels.reshape(-1).to(device=logits.device, dtype=torch.int64)
    outputs = torch_npu.npu_cross_entropy_loss(
        logits_2d,
        labels_1d,
        reduction="none",
        ignore_index=ignore_index,
    )
    loss = outputs[0] if isinstance(outputs, tuple) else outputs
    return -loss.view(batch_shape)

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

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.loss.loss_utils import LOSS_MAPPING, fixed_cross_entropy

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import reduce_sequence_parallel_loss
from ..utils import logging
from ..utils.import_utils import is_liger_kernel_available, is_torch_npu_available


if is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss

    liger_kernel_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")

    def fused_liger_kernel_cross_entropy(
        logits: torch.Tensor = None,
        labels: torch.Tensor = None,
        vocab_size: int = None,
        num_items_in_batch: Optional[int] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        weights = kwargs.pop("weights")
        hidden_states = kwargs.pop("hidden_states")
        return liger_kernel_cross_entropy(weights, hidden_states, labels), logits


logger = logging.get_logger(__name__)


def eager_cross_entropy(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    if logits is None:
        hidden_states = kwargs.pop("hidden_states")
        weights = kwargs.pop("weights")
        logits = F.linear(hidden_states, weights).float()
        logits = logits.view(-1, vocab_size)
    return fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs), logits


_cross_entropy = eager_cross_entropy


def ForCausalLMLoss(
    logits: torch.Tensor = None,
    labels: torch.Tensor = None,
    vocab_size: int = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # pop fused loss kwargs
    hidden_states = kwargs.pop("hidden_states", None)
    weights = kwargs.pop("weights", None)

    assert hidden_states is not None or logits is not None, "hidden_states or logits must be provided."

    device = logits.device if logits is not None else hidden_states.device
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    if logits is not None:
        logits = logits.float()

    sp_enabled = get_parallel_state().sp_enabled

    # veomni sp patch
    if not sp_enabled:
        # Shift so that tokens < n predict n
        if shift_labels is None:
            labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()
    else:
        if shift_labels is not None:
            logger.warning_once("labels have been shifted in dataloader when `sp_enabeld=True`, ignore shift_labels.")
        shift_labels = labels

    # Flatten the tokens
    shift_labels = shift_labels.view(-1)
    if hidden_states is not None:
        hidden_states = hidden_states.view(-1, hidden_states.size(-1))
    if logits is not None:
        logits = logits.view(-1, vocab_size)
    # Enable model parallelism
    shift_labels = shift_labels.to(device)

    if hidden_states is None or weights is None:
        logger.warning_once(
            "hidden_states or weights is None, use eager loss implementation."
            "To enable fused linear cross entropy loss, please patch modeling.py `forward` function "
            "to pass `hidden_states` and `weights` to `loss_function`."
        )
        loss_func = eager_cross_entropy
    else:
        loss_func = _cross_entropy
    loss, logits = loss_func(
        logits,
        shift_labels,
        vocab_size,
        num_items_in_batch,
        ignore_index,
        shift_labels,
        hidden_states=hidden_states,
        weights=weights,
        **kwargs,
    )

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (labels != IGNORE_INDEX).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)
    return loss, logits


def apply_veomni_loss_patch():
    LOSS_MAPPING["ForCausalLM"] = ForCausalLMLoss
    global _cross_entropy
    if is_torch_npu_available():
        _cross_entropy = eager_cross_entropy
    elif is_liger_kernel_available() and os.environ.get("USE_LIGER_KERNEL", "1") == "1":
        _cross_entropy = fused_liger_kernel_cross_entropy
    else:
        _cross_entropy = eager_cross_entropy

    logger.info_rank0(
        f"✅ Transformers LOSS_MAPPING['ForCausalLM'] patched with new ForCausalLM in VeOmni, using {_cross_entropy.__name__} for celoss kernel"
    )

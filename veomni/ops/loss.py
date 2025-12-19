from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from ..distributed.sequence_parallel import reduce_sequence_parallel_loss
from ..utils import logging
from ..utils.import_utils import is_liger_kernel_available, is_seed_kernels_available


logger = logging.get_logger(__name__)


def fixed_cross_entropy(
    source: torch.Tensor,
    target: torch.Tensor,
    num_items_in_batch: Optional[torch.Tensor] = None,
    ignore_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    reduction = "sum" if num_items_in_batch is not None else "mean"
    loss = nn.functional.cross_entropy(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
        if torch.is_tensor(num_items_in_batch):
            num_items_in_batch = num_items_in_batch.to(loss.device)
        loss = loss / num_items_in_batch
    return loss


fused_linear_cross_entropy = None

if is_seed_kernels_available():
    from seed_kernels.transformers.functional import seed_fused_linear_cross_entropy

    fused_linear_cross_entropy = seed_fused_linear_cross_entropy
elif is_liger_kernel_available():
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss  # type: ignore

    fused_linear_cross_entropy = LigerFusedLinearCrossEntropyLoss(reduction="mean")


def causallm_loss_function(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: Optional[int] = None,
    num_items_in_batch: Optional[int] = None,
    ignore_index: int = -100,
    shift_labels: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    # We don't use shift_labels in causallm
    assert shift_labels is None

    loss = None
    logits = None

    if labels is None:
        logits = F.linear(hidden_states, weight)
        return loss, logits

    sp_enabled = get_parallel_state().sp_enabled

    # Shift the labels and hidden_states so that tokens < n predict n
    if not sp_enabled:
        labels = labels[..., 1:].contiguous()
        hidden_states = hidden_states[..., :-1, :].contiguous()

    # Flatten the labels and hidden_states
    labels = labels.view(-1)
    hidden_states = hidden_states.view(-1, hidden_states.size(-1))

    # Calculate loss
    if fused_linear_cross_entropy is not None:  # use kernels
        if is_seed_kernels_available():
            loss = fused_linear_cross_entropy(hidden_states, weight, labels, ignore_index=ignore_index)
        elif is_liger_kernel_available():
            loss = fused_linear_cross_entropy(weight, hidden_states, labels)
    else:
        logits = F.linear(hidden_states, weight).float()
        loss = fixed_cross_entropy(logits, labels, num_items_in_batch, ignore_index, **kwargs)

    # Reduce loss when using sp
    if sp_enabled:
        num_valid_tokens = (labels != IGNORE_INDEX).sum()
        loss = reduce_sequence_parallel_loss(loss, num_valid_tokens)

    return loss, logits


def seqcls_token_loss_sp_aware(
    logits: torch.Tensor,  # [N, C]
    labels: torch.Tensor,  # [N]
    loss_fct: nn.Module,
    sp_group,
) -> torch.Tensor:
    # todo move to input peremeters
    IGNORE_INDEX = -100
    # local sum loss
    # CrossEntropyLoss(reduction="none") + mask + sum
    per = loss_fct(logits, labels)  # [N] if reduction="none"
    valid = labels != IGNORE_INDEX
    loss_sum = (per * valid).sum()
    cnt = valid.sum().to(dtype=loss_sum.dtype)

    if sp_group is not None:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM, group=sp_group)
        dist.all_reduce(cnt, op=dist.ReduceOp.SUM, group=sp_group)
    return loss_sum / cnt.clamp_min(1.0)

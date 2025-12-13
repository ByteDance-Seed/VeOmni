from typing import Optional, Tuple

import torch
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


def _last_token_index_varlen(
    cu_seqlens: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    In the Varlen scenario, the index of the last token of each sample (the index after flattening) is calculated based on cu_seqlens.
    cu_seqlens: [B+1], where cu_seqlens[i] is the starting offset of the i-th sample in the flattened sequence.
    """
    return (cu_seqlens[1:].to(device) - 1).long()  # [B]


def _last_token_index_padded(
    input_ids: Optional[torch.LongTensor],
    seq_len: int,
    pad_token_id: Optional[int],
    device: torch.device,
) -> torch.Tensor:
    """
    In a padding scenario, the position (index on the seq_len dimension) of the last valid token for each sample is calculated as follows:
    - If pad_token_id exists: take the rightmost non-pad position for each sample;
    - If pad_token_id does not exist: the last position can only be used when batch_size == 1; otherwise, an error is thrown.
    """
    if input_ids is None:
        batch_size = 1
        return torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)

    batch_size = input_ids.shape[0]

    if pad_token_id is None and batch_size != 1:
        raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")

    if pad_token_id is None:
        return torch.full((batch_size,), seq_len - 1, dtype=torch.long, device=device)

    non_pad_mask = (input_ids != pad_token_id).to(device=device, dtype=torch.int32)  # [B, L]
    token_indices = torch.arange(seq_len, device=device, dtype=torch.int32)  # [L]
    last_non_pad_token = (token_indices * non_pad_mask).argmax(dim=-1)  # [B]

    return last_non_pad_token.long()


def seqcls_last_token_loss_and_logits(
    hidden_states: torch.Tensor,
    classifier: nn.Module,
    labels: Optional[torch.Tensor] = None,
    *,
    cu_seqlens: Optional[torch.Tensor] = None,
    input_ids: Optional[torch.LongTensor] = None,
    pad_token_id: Optional[int] = None,
    loss_fct: Optional[nn.Module] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    The general category head includes:
    - Supports two input formats: varlen (cu_seqlens) and padding;
    - Returns (loss, pooled_logits)

    Parameters:
        hidden_states: In a varlen scenario, it's typically [T_total, H], while in a padding scenario it's [B, L, H]
        classifier: A linear layer
        labels: Single-label integers of shape [B], [B, 1], or any shape
        cu_seqlens: [B+1] prefix sum in varlen mode; if None, padding logic is used
        input_ids: IDs used to find the last non-pad token in padding mode
        pad_token_id: Token ID used to identify padding
        loss_fct: Such as nn.CrossEntropyLoss; if None, a default value is created

    Returns:
        loss: None if labels is None; otherwise, a scalar loss
        pooled_logits: [B, C], the classification logits after pooling the last token
    """
    device = hidden_states.device
    hidden_size = hidden_states.size(-1)

    # calculate pooled_logits
    if cu_seqlens is not None:
        # ---- varlen ----
        flat_hidden = hidden_states.view(-1, hidden_size)  # [T_total, H]
        flat_logits = classifier(flat_hidden)  # [T_total, C]

        last_idx = _last_token_index_varlen(
            cu_seqlens=cu_seqlens,
            device=device,
        )  # [B]

        pooled_logits = flat_logits[last_idx]  # [B, C]
    else:
        # ---- padding ----
        if hidden_states.dim() != 3:
            raise ValueError(
                f"Expected hidden_states with shape [batch, seq_len, hidden_size] for padded input, "
                f"but got shape {hidden_states.shape}."
            )

        batch_size, seq_len, _ = hidden_states.shape
        logits = classifier(hidden_states)  # [B, L, C]

        last_pos = _last_token_index_padded(
            input_ids=input_ids,
            seq_len=seq_len,
            pad_token_id=pad_token_id,
            device=device,
        )  # [B]

        batch_idx = torch.arange(batch_size, device=device)
        pooled_logits = logits[batch_idx, last_pos]  # [B, C]

    # if there are no labels, return logits directly
    if labels is None:
        return None, pooled_logits

    # Calculate loss
    if loss_fct is None:
        loss_fct = nn.CrossEntropyLoss()

    if torch.is_floating_point(labels):
        labels = labels.to(torch.long)

    if labels.ndim > 1:
        labels = labels.view(-1)

    num_labels = pooled_logits.size(-1)
    loss = loss_fct(
        pooled_logits.view(-1, num_labels),
        labels.view(-1),
    )

    return loss, pooled_logits

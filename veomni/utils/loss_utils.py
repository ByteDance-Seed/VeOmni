from collections import defaultdict
from typing import Union

import torch

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from .device import get_device_type
from .dist_utils import all_reduce


def count_loss_token(batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]):
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len = defaultdict(int)
    for batch in batches:
        token_len["foundation_tokens"] += torch.sum(batch["labels"] != IGNORE_INDEX)
        if "image_output_mask" in batch:
            token_len["image_decoder_tokens"] += torch.sum(batch["image_output_mask"])
    return token_len


def mean_global_loss(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    micro_batch_token_len: dict[str, torch.Tensor],
    micro_batches_token_len: dict[str, torch.Tensor],
):
    """Calcuate the global mean loss. Avg on all_reduced_token_num instead of on dp_size.
    - cur_losses[key] = cur_loss * cur_token_num / global_batches_token_num * get_parallel_state().fsdp_size
    - loss_bwd = sum(dict(cur_losses))
    """
    loss_bwd = torch.tensor(0.0, device=get_device_type())
    loss_dict = {}

    if isinstance(losses, torch.Tensor):  # text loss only
        losses = {"foundation_loss": losses}

    for key, cur_loss in losses.items():
        loss_name = key.split("_loss")[0]  # foundation/image_decoder/**

        cur_token_len = micro_batch_token_len[f"{loss_name}_tokens"]
        if get_parallel_state().sp_enabled:
            cur_token_len = all_reduce(cur_token_len.item(), op="sum", group=get_parallel_state().sp_group)

        all_reduced_len = all_reduce((micro_batches_token_len[f"{loss_name}_tokens"].item()), op="sum")

        if all_reduced_len != 0:
            cur_loss = cur_loss * cur_token_len / all_reduced_len * get_parallel_state().fsdp_size
        else:  # no loss
            assert torch.allclose(cur_loss, torch.zeros_like(cur_loss)), f"cur_loss: {cur_loss}"

        if get_parallel_state().sp_enabled:
            cur_loss = cur_loss / get_parallel_state().sp_size

        loss_bwd += cur_loss

        loss_dict[key] = cur_loss.item()

    return loss_bwd, loss_dict

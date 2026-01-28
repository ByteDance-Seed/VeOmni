from collections import defaultdict
from typing import Any, Union

import torch

from ..data.constants import IGNORE_INDEX
from ..distributed.parallel_state import get_parallel_state
from . import helper
from .device import get_device_type
from .dist_utils import all_reduce

logger = helper.create_logger(__name__)


def count_loss_token(batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]):
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len = defaultdict(int)
    for batch in batches:
        token_len["foundation_tokens"] += torch.sum(batch["labels"] != IGNORE_INDEX)  # text tokens
        if "image_output_mask" in batch:
            token_len["image_decoder_tokens"] += torch.sum(batch["image_output_mask"])  # image generation tokens
    return token_len


def mean_global_loss(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    micro_batch_token_len: dict[str, torch.Tensor],
    micro_batches_token_len: dict[str, torch.Tensor],
):
    """Calcuate the global mean loss. Avg on all_reduced_token_num instead of on dp_size.
    - cur_losses[key] = cur_loss * cur_token_num / global_batches_token_num * get_parallel_state().fsdp_size
    # fsdp by default divides gradients by its size, so we need to multiply by fsdp_size
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
        else:
            if not torch.allclose(cur_loss, torch.zeros_like(cur_loss)):
                raise ValueError(
                    f"The all_reduced_len for {loss_name}_tokens is 0, but the cur_loss is not 0: {cur_loss}"
                )

        if get_parallel_state().sp_enabled:
            cur_loss = cur_loss / get_parallel_state().sp_size

        loss_bwd += cur_loss

        loss_dict[key] = cur_loss.item()

    return loss_bwd, loss_dict


def calc_validation_metrics(
    model: Any,
    val_dataloader: Any,
    val_steps: int,
):
    if get_parallel_state().sp_enabled:
        raise ValueError("Validation currently is not supported for SP.")
    logger.info_rank0("running validation step")

    model.eval()
    val_metrics = {}

    # total_label_token_num is the total num of tokens across all validation steps and all micro batches
    # total_loss is the total sum of loss across all validation steps and micro batches
    total_accumulated_loss, total_label_token_num = 0.0, 0.0

    val_iter = iter(val_dataloader)
    for _ in range(val_steps):
        micro_batches: list[dict[str, Any]] = next(val_iter)
        micro_batches_token_num = count_loss_token(micro_batches)

        loss_name = "foundation"
        all_reduced_len = all_reduce(
            (micro_batches_token_num[f"{loss_name}_tokens"].item()), op="sum", group=get_parallel_state().fsdp_group
        )
        total_label_token_num += all_reduced_len

        for micro_batch in micro_batches:
            micro_batch_token_num = count_loss_token(micro_batch)
            micro_batch = {
                k: v.to(get_device_type(), non_blocking=True) if isinstance(v, torch.Tensor) else v
                for k, v in micro_batch.items()
            }
            with torch.no_grad():
                # set output_router_logits=False because moe lb loss is not needed
                outputs = model(**micro_batch, use_cache=False, output_router_logits=False)
                total_accumulated_loss += outputs.loss.item() * micro_batch_token_num[f"{loss_name}_tokens"].item()
            del micro_batch

    # synchronize after calculating all micro batches for all steps
    torch.cuda.synchronize()

    # calculate the validation loss across all ranks
    total_accumulated_loss = all_reduce(total_accumulated_loss, op="sum", group=get_parallel_state().fsdp_group)
    loss = total_accumulated_loss / total_label_token_num

    val_metrics.update({"validation/loss": loss})
    logger.info_rank0(f"Validation finished. {loss=}")

    model.train()
    return val_metrics

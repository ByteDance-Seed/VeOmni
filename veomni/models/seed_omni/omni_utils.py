from collections import defaultdict
from typing import List, Union

import torch
import torch.distributed as dist

from veomni.utils.dist_utils import all_reduce

from ...data.constants import IGNORE_INDEX
from ...utils.device import get_device_type
from .modeling_seed_omni import SeedOmniModel


def omni_token_meter(batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]):
    """Calculate the total number of text_tokens/image_tokens in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_meter = defaultdict(int)
    for batch in batches:
        token_meter["text_tokens"] += torch.sum(batch["labels"] != IGNORE_INDEX)
        if "image_output_mask" in batch:
            token_meter["image_tokens"] += torch.sum(batch["image_output_mask"])
    return token_meter


def mean_global_batch_loss(
    losses: dict[str, int], micro_batch_meter: dict[str, torch.Tensor], micro_batches_meter: dict[str, int]
):
    """Calculate the mean loss in a global batch. Avg on token_num instead of on len(global_batch).
    For text_loss:
    - √ text_loss = text_loss * micro_batch_text_token_num / global_batch_text_token_num
    - x text_loss = text_loss / (len(global_batch))
    For image loss:
    - √ image_loss = image_loss * micro_batch_image_token_num / global_batch_image_token_num
    - x image_loss = image_loss / (len(global_batch))
    """
    loss = torch.tensor(0.0, device=get_device_type())
    for key in losses.keys():
        if "foundation" in key:
            losses[key] *= micro_batch_meter["text_tokens"]
            if micro_batches_meter["text_tokens"] != 0:
                losses[key] /= micro_batches_meter["text_tokens"]

        elif "image_decoder" in key:
            losses[key] *= micro_batch_meter["image_tokens"]
            if micro_batches_meter["image_tokens"] != 0:
                losses[key] /= micro_batches_meter["image_tokens"]
        else:
            raise ValueError(f"Unrecognized loss key: {key}")
        loss += losses[key]
    return loss


def mean_dp_loss(
    losses: dict[str, int], micro_batches_meter: dict[str, torch.Tensor], group: dist.ProcessGroup = None
):
    """Calcuate the mean loss in a dp group. Avg on all_reduced_token_num instead of on dp_size.
    For text_loss:
    - √ text_loss = all_reduce(text_loss * micro_batch_text_token_num, op="sum") / all_reduce(global_batch_text_token_num, op="sum")
    - x text_loss = all_reduce(text_loss, op="sum") / dp_size
    For image loss:
    - √ image_loss = all_reduce(image_loss * micro_batch_image_token_num, op="sum") / all_reduce(global_batch_image_token_num, op="sum")
    - x image_loss = all_reduce(image_loss, op="sum") / dp_size
    """
    total_loss = 0
    total_losses = {}
    for key, item in losses.items():
        if "foundation" in key:
            item *= micro_batches_meter["text_tokens"]
            all_reduced_loss = all_reduce((item.item()), op="sum", group=group)
            all_reduced_length = all_reduce((micro_batches_meter["text_tokens"].item()), op="sum", group=group)
            if all_reduced_length != 0:
                all_reduced_loss /= all_reduced_length
            total_losses[key] = all_reduced_loss
            total_loss += all_reduced_loss
        elif "image_decoder" in key:
            item *= micro_batches_meter["image_tokens"]
            all_reduced_loss = all_reduce((item.item()), op="sum", group=group)
            all_reduced_length = all_reduced_length = all_reduce(
                (micro_batches_meter["image_tokens"].item()), op="sum", group=group
            )
            if all_reduced_length != 0:
                all_reduced_loss /= all_reduced_length
            total_losses[key] = all_reduced_loss
            total_loss += all_reduced_loss
        else:
            raise ValueError(f"Unrecognized loss key: {key}")
    return total_loss, total_losses


def get_embed_token_grad_mask_func(
    model: SeedOmniModel, trained_embedding_indices: List[int], global_rank: int, world_size: int
):
    embed_dim = model.encoder.text_encoder.weight.shape[-1]
    train_indices = []
    for indice in trained_embedding_indices:
        train_indices.append((indice * embed_dim, indice * embed_dim + embed_dim))

    def mask_embedding_grad(embedding: torch.nn.Embedding):
        grad = embedding.weight.grad
        device = embedding.weight.device
        if grad is None:
            local_numel = torch.tensor([0], device=device)
        else:
            local_numel = torch.tensor([grad.numel()], device=device)

        all_numel = [torch.zeros_like(local_numel) for _ in range(world_size)]
        dist.all_gather(all_numel, local_numel)

        if grad is None:
            return
        shard_sizes = [x.item() for x in all_numel]
        offsets = [0]
        for sz in shard_sizes:
            offsets.append(offsets[-1] + sz)

        start_offset = offsets[global_rank]
        end_offset = offsets[global_rank + 1]

        local_mask = torch.zeros_like(grad, dtype=torch.bool, device=device)
        for start, end in train_indices:
            if end >= start_offset and start < end_offset:
                local_mask[max(start, start_offset) - start_offset : min(end, end_offset) - start_offset] = True
        grad.mul_(local_mask)

    return mask_embedding_grad

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

from typing import Union

import torch

from ..distributed.parallel_state import get_parallel_state
from .constants import IGNORE_INDEX
from .device import get_device_type
from .dist_utils import all_reduce


def is_padding_microbatch(batch) -> bool:
    return isinstance(batch, dict) and bool(batch.get("padding_flag", False))


def contains_padding(batch) -> bool:
    if isinstance(batch, (list, tuple)):
        return any(contains_padding(item) for item in batch)
    return is_padding_microbatch(batch)


def _distributed_batch_status(batch, invalid: bool, has_error: bool = False, has_auxiliary_targets: bool = False):
    count = len(batch) if isinstance(batch, (list, tuple)) else 1
    device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
    flags = torch.tensor(
        [int(invalid), int(has_error), int(has_auxiliary_targets), count, -count], dtype=torch.int32, device=device
    )
    torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
    values = flags.tolist()
    return bool(values[0]) or values[3] != -values[4], bool(values[1]), bool(values[2])


def reject_objective_padding(batch, objective: str) -> None:
    error = None
    try:
        found = contains_padding(batch) or (isinstance(batch, (list, tuple)) and not batch)
    except Exception as exc:  # noqa: BLE001 - peers must reach the validation collective
        error = exc
        found = True
    distributed = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
    if distributed:
        found, has_error, _ = _distributed_batch_status(batch, found, error is not None)
    else:
        has_error = error is not None
    if found:
        if has_error:
            if not distributed:
                raise error
            raise RuntimeError(f"Invalid {objective} padding metadata on a rank") from error
        raise ValueError(
            f"{objective} does not support padded micro-batches, an empty step, or unequal microbatch counts across ranks"
        )


def synchronize_step_begin_error(error: Exception | None, objective: str) -> None:
    """Propagate real callback errors before distributed model collectives."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
        failed = torch.tensor([int(error is not None)], dtype=torch.int32, device=device)
        torch.distributed.all_reduce(failed, op=torch.distributed.ReduceOp.MAX)
        if failed.item():
            raise RuntimeError(f"{objective} step-begin callback failed on a rank") from error
    elif error is not None:
        if isinstance(error, StopIteration):
            raise RuntimeError(f"{objective} step-begin callback raised StopIteration") from error
        raise error


def _has_auxiliary_labels(batch) -> bool:
    if isinstance(batch, dict):
        return any(key.endswith("_labels") for key in batch)
    if isinstance(batch, (list, tuple)):
        return any(_has_auxiliary_labels(item) for item in batch)
    return False


def reject_distributed_padding(batch) -> bool:
    """Reject unequal/padded steps and return whether any rank has auxiliary labels."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        error = None
        try:
            invalid = contains_padding(batch) or (isinstance(batch, (list, tuple)) and not batch)
            has_auxiliary_targets = _has_auxiliary_labels(batch)
        except Exception as exc:  # noqa: BLE001 - peers must reach the validation collective
            error = exc
            invalid = True
            has_auxiliary_targets = False
        found, has_error, has_auxiliary_targets = _distributed_batch_status(
            batch, invalid, error is not None, has_auxiliary_targets
        )
        if found:
            if has_error:
                raise RuntimeError("Invalid padding metadata on a rank") from error
            raise RuntimeError(
                "Distributed training does not support padded micro-batches, an empty step, "
                "or unequal microbatch counts across ranks"
            )
        return has_auxiliary_targets
    return False


def count_loss_token(batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]):
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len = {
        "foundation_tokens": torch.tensor(0),
    }

    def _count(obj):
        if isinstance(obj, dict):
            if is_padding_microbatch(obj):
                for key in obj:
                    if key.endswith("_labels"):
                        token_len.setdefault(f"{key.split('_labels')[0]}_tokens", torch.tensor(0))
                return
            token_len["foundation_tokens"] += torch.sum(obj["labels"] != IGNORE_INDEX)  # text tokens

            for key in obj.keys():
                if key.endswith("_labels"):
                    token_name = key.split("_labels")[0]
                    token_len[f"{token_name}_tokens"] = token_len.get(
                        f"{token_name}_tokens", torch.tensor(0)
                    ) + torch.sum(obj[key] != IGNORE_INDEX)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                _count(item)
        else:
            raise TypeError(f"Unsupported batch type: {type(obj)}")

    _count(batches)
    return token_len


def synchronize_auxiliary_loss_tokens(token_len):
    """Reduce the same ordered token keys when different ranks see different modalities."""
    key_lists = [None] * torch.distributed.get_world_size()
    torch.distributed.all_gather_object(key_lists, sorted(token_len))
    for key in sorted({key for keys in key_lists for key in keys}):
        token_len.setdefault(key, torch.tensor(0))
    return token_len


def mean_global_loss(
    losses: Union[dict[str, torch.Tensor], torch.Tensor],
    micro_batch_token_len: dict[str, torch.Tensor],
    micro_batches_token_len: dict[str, torch.Tensor],
    global_micro_batches_token_len: dict[str, float] | None = None,
):
    """Calcuate the global mean loss. Avg on all_reduced_token_num instead of on dp_size.
    - cur_losses[key] = cur_loss * cur_token_num / global_batches_token_num * get_parallel_state().fsdp_size
    # fsdp by default divides gradients by its size, so we need to multiply by fsdp_size
    """
    loss_dict = {}

    if isinstance(losses, torch.Tensor):  # text loss only
        losses = {"foundation_loss": losses}

    for key, cur_loss in losses.items():
        loss_name = key.split("_loss")[0]  # foundation/image_decoder/**

        cur_token_len = micro_batch_token_len[f"{loss_name}_tokens"]
        if get_parallel_state().sp_enabled:
            cur_token_len = all_reduce(cur_token_len.item(), op="sum", group=get_parallel_state().sp_group)

        if global_micro_batches_token_len is None:
            all_reduced_len = all_reduce((micro_batches_token_len[f"{loss_name}_tokens"].item()), op="sum")
        else:
            all_reduced_len = global_micro_batches_token_len[f"{loss_name}_tokens"]

        if all_reduced_len != 0:
            cur_loss = cur_loss * cur_token_len / all_reduced_len * get_parallel_state().fsdp_size
        else:
            if not torch.allclose(cur_loss, torch.zeros_like(cur_loss)):
                raise ValueError(
                    f"The all_reduced_len for {loss_name}_tokens is 0, but the cur_loss is not 0: {cur_loss}"
                )

        if get_parallel_state().sp_enabled:
            cur_loss = cur_loss / get_parallel_state().sp_size

        loss_dict[key] = cur_loss

    return loss_dict


def reduce_global_loss_token(
    micro_batches_token_len: dict[str, torch.Tensor], *, error: Exception | None = None, sync_auxiliary: bool = False
) -> dict[str, float]:
    """All-reduce per-step loss-token denominators once for reuse across micro-batches."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        # Reuse the existing foundation-token reduction to propagate a
        # rank-local callback/counting error without another collective.
        device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
        local = torch.tensor(
            [micro_batches_token_len["foundation_tokens"].item(), int(error is not None)],
            dtype=torch.float,
            device=device,
        )
        foundation, failures = all_reduce(local, op="sum")
        if failures:
            raise RuntimeError("Step-begin callback or loss-token counting failed on a rank") from error
        if sync_auxiliary:
            synchronize_auxiliary_loss_tokens(micro_batches_token_len)
        reduced = {"foundation_tokens": foundation}
        reduced.update(
            {
                key: all_reduce(value.item(), op="sum")
                for key, value in sorted(micro_batches_token_len.items())
                if key != "foundation_tokens"
            }
        )
        return reduced
    if error is not None:
        if isinstance(error, StopIteration):
            raise RuntimeError("Step-begin callback raised StopIteration") from error
        raise error
    return {key: all_reduce(value.item(), op="sum") for key, value in sorted(micro_batches_token_len.items())}

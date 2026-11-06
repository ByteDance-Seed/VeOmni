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


def contains_loss_weights(value):
    if isinstance(value, (list, tuple)):
        return any(contains_loss_weights(item) for item in value)
    return isinstance(value, dict) and (value.get("loss_weight") is not None or value.get("loss_weights") is not None)


def fetch_training_micro_batches(data_iterator, *, objective=None, on_fetched=None):
    """Coordinate input errors and uneven exhaustion before any model collective."""
    error = None
    try:
        batches = next(data_iterator)
        if on_fetched is not None:
            on_fetched()
        if not isinstance(batches, (list, tuple)) or not batches:
            raise ValueError("Training loader must return a nonempty list of micro-batches")
        if contains_padding(batches):
            raise ValueError(
                f"{objective} does not support padded micro-batches"
                if objective
                else "Synthetic padding requires the padding-safe trainer"
            )
    except Exception as exc:  # noqa: BLE001 - all ranks must agree before forward
        batches, error = None, exc
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
        count = len(batches) if batches is not None else 0
        flags = torch.tensor(
            [
                int(error is None),
                int(isinstance(error, StopIteration)),
                int(error is not None and not isinstance(error, StopIteration)),
                count,
                -count,
            ],
            dtype=torch.int32,
            device=device,
        )
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        success, eof, failed, max_count, neg_min_count = flags.tolist()
        if failed:
            raise RuntimeError("Training data loading failed on a rank") from error
        if eof:
            if not success:
                raise StopIteration
            raise RuntimeError("Training ranks reached the end of data at different times") from error
        if max_count != -neg_min_count:
            raise RuntimeError("Training ranks must execute the same number of micro-batches")
    elif error is not None:
        raise error
    return batches


def synchronize_training_error(error, objective, *, microbatch_count=None):
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
        values = [int(error is not None)]
        if microbatch_count is not None:
            values.extend([microbatch_count, -microbatch_count])
        flags = torch.tensor(values, dtype=torch.int32, device=device)
        torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
        values = flags.tolist()
        if values[0]:
            raise RuntimeError(f"{objective} preforward validation failed on a rank") from error
        if microbatch_count is not None and values[1] != -values[2]:
            raise RuntimeError(f"{objective} ranks must execute the same number of micro-batches")
    elif error is not None:
        if isinstance(error, StopIteration):
            raise RuntimeError(f"{objective} step-begin callback raised StopIteration") from error
        raise error


def _distributed_batch_status(
    batch, invalid: bool, has_auxiliary_targets: bool = False, *, check_fetch=False, fetch_error=None
):
    count = len(batch) if isinstance(batch, (list, tuple)) else 1
    device = "cpu" if torch.distributed.get_backend() == "gloo" else get_device_type()
    values = [int(invalid), int(has_auxiliary_targets), count, -count]
    if check_fetch:
        values.extend(
            [
                int(fetch_error is None),
                int(isinstance(fetch_error, StopIteration)),
                int(fetch_error is not None and not isinstance(fetch_error, StopIteration)),
            ]
        )
    flags = torch.tensor(values, dtype=torch.int32, device=device)
    torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.MAX)
    values = flags.tolist()
    if check_fetch:
        any_loaded, any_eof, any_error = values[4:]
        if any_error:
            raise RuntimeError("SFT data loading failed on a rank") from fetch_error
        if any_eof:
            if not any_loaded:
                raise StopIteration
            raise RuntimeError("SFT ranks reached the end of the data at different times") from fetch_error
    return bool(values[0]) or values[2] != -values[3], bool(values[1])


def reject_objective_padding(batch, objective: str) -> None:
    validation_error = None
    try:
        found = contains_padding(batch) or (isinstance(batch, (list, tuple)) and not batch)
    except Exception as exc:  # noqa: BLE001 - coordinate invalid metadata before objective collectives
        found, validation_error = True, exc
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        found, _ = _distributed_batch_status(batch, found)
    if found:
        raise ValueError(
            f"{objective} does not support padded micro-batches, an empty step, or unequal microbatch counts across ranks"
        ) from validation_error


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


def reject_distributed_padding(batch, *, check_fetch=False, fetch_error=None) -> bool:
    """Reject unequal/padded steps and return whether any rank has auxiliary labels."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        validation_error = None
        try:
            invalid = (
                contains_padding(batch)
                or (isinstance(batch, (list, tuple)) and not batch)
                or (check_fetch and fetch_error is None and not isinstance(batch, (list, tuple)))
            )
            has_auxiliary_targets = _has_auxiliary_labels(batch)
        except Exception as exc:  # noqa: BLE001 - peers must reach the same status collective
            invalid, has_auxiliary_targets, validation_error = True, False, exc
        found, has_auxiliary_targets = _distributed_batch_status(
            batch,
            invalid,
            has_auxiliary_targets=has_auxiliary_targets,
            check_fetch=check_fetch,
            fetch_error=fetch_error,
        )
        if found:
            raise RuntimeError(
                "Distributed training does not support padded micro-batches, an empty step, "
                "unequal microbatch counts across ranks, or malformed batch metadata"
            ) from validation_error
        return has_auxiliary_targets
    if check_fetch and fetch_error is not None:
        raise fetch_error
    if check_fetch and not isinstance(batch, (list, tuple)):
        raise ValueError("SFT data loader must return a list of micro-batches")
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
    return {key: all_reduce(value.item(), op="sum") for key, value in micro_batches_token_len.items()}

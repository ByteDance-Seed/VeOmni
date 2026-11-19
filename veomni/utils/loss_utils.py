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
from .env import get_env


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


def fetch_training_micro_batches(data_iterator, *, objective=None, on_fetched=None, reject_loss_weights=False):
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
        if reject_loss_weights and contains_loss_weights(batches):
            raise ValueError(f"{objective} does not support token loss_weights")
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
    found = contains_padding(batch) or (isinstance(batch, (list, tuple)) and not batch)
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        found, _ = _distributed_batch_status(batch, found)
    if found:
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


def reject_distributed_padding(batch, *, check_fetch=False, fetch_error=None) -> bool:
    """Reject unequal/padded steps and return whether any rank has auxiliary labels."""
    if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
        invalid = (
            contains_padding(batch)
            or (isinstance(batch, (list, tuple)) and not batch)
            or (check_fetch and fetch_error is None and not isinstance(batch, (list, tuple)))
        )
        has_auxiliary_targets = _has_auxiliary_labels(batch)
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
                "or unequal microbatch counts across ranks"
            )
        return has_auxiliary_targets
    if check_fetch and fetch_error is not None:
        raise fetch_error
    if check_fetch and not isinstance(batch, (list, tuple)):
        raise ValueError("SFT data loader must return a list of micro-batches")
    return False


def count_loss_token(
    batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]], *, weighted_normalization: bool = False
):
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
            valid = obj["labels"] != IGNORE_INDEX
            weights = obj.get("loss_weights")
            if weights is None:
                if weighted_normalization and not get_parallel_state().sp_enabled and valid.ndim:
                    valid = valid[..., 1:]
                token_len["foundation_tokens"] = token_len["foundation_tokens"] + valid.sum()
            else:
                if weights.shape != valid.shape:
                    raise ValueError("loss_weights must match labels")
                if not valid.ndim:
                    raise ValueError("loss_weights require sequence labels")
                # SP inputs are already shifted; otherwise causal CE discards the first target.
                if not get_parallel_state().sp_enabled:
                    weights, valid = weights[..., 1:], valid[..., 1:]
                token_len["foundation_tokens"] = token_len["foundation_tokens"] + (weights.float() * valid).sum()

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


def _iter_feature_dicts(value):
    """Walk collated batches and uncollated RL micro-batches alike."""
    if isinstance(value, dict):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_feature_dicts(item)
    else:
        raise TypeError(f"Unsupported batch type: {type(value)}")


def has_weighted_loss_in_step(batches: list[dict[str, torch.Tensor]]) -> bool:
    """Synchronize opt-in across DP ranks before counting non-SP causal targets."""
    local_has_weights = any(batch.get("loss_weights") is not None for batch in _iter_feature_dicts(batches))
    if get_parallel_state().sp_enabled or not torch.distributed.is_initialized():
        return local_has_weights
    return bool(all_reduce(int(local_has_weights), op="max"))


def prepare_loss_token_counts(
    batches: list[dict[str, torch.Tensor]], model_runtime=None, *, error: Exception | None = None, sync_auxiliary=False
):
    """Reuse the denominator reduction to agree on weighted causal normalization.

    For non-SP training a weighted rank can coexist with an unweighted rank.
    Reduce both denominator variants and the opt-in flag together so that all
    ranks choose the same normalization without an additional collective.
    """
    legacy = shifted = {"foundation_tokens": torch.tensor(0)}
    local_weighted = local_mtp = local_uncollated = local_router = invalid_router = False
    unsupported_model = explicit_shift = explicit_count = scalar_labels = local_auxiliary = False
    if error is None:
        try:
            features = [batch for batch in _iter_feature_dicts(batches) if not is_padding_microbatch(batch)]
            local_weighted = any(batch.get("loss_weights") is not None for batch in features)
            for batch in features:
                if batch.get("loss_weights") is None:
                    continue
                weights = batch["loss_weights"]
                if not isinstance(weights, torch.Tensor) or weights.is_complex():
                    raise TypeError("loss_weights must be a real tensor")
                valid = (
                    torch.isfinite(weights)
                    & (weights >= 0)
                    & (weights <= 1)
                    & ((weights == 0) | (weights >= torch.finfo(torch.float32).tiny))
                )
                if not bool(valid.all()):
                    raise ValueError("loss_weights must be in [0, 1] and positive weights must be normal float32")
            local_mtp = any("mtp_labels" in batch for batch in features)
            local_auxiliary = any(any(key.endswith("_labels") for key in batch) for batch in features)
            scalar_labels = any(
                isinstance(batch.get("labels"), torch.Tensor) and batch["labels"].ndim == 0 for batch in features
            )
            explicit_shift = any(batch.get("shift_labels") is not None for batch in features)
            explicit_count = any(batch.get("num_items_in_batch") is not None for batch in features)
            # RL packs list-valued micro-batches inside preforward and computes
            # its own unweighted loss; weighted SFT cannot use that path.
            local_uncollated = isinstance(batches, (list, tuple)) and any(
                isinstance(batch, (list, tuple)) for batch in batches
            )
            router_flags = [batch.get("output_router_logits", False) for batch in features]
            local_router = any(value is True for value in router_flags)
            invalid_router = any(type(value) is not bool for value in router_flags)
            legacy = count_loss_token(batches)
            shifted = (
                legacy if get_parallel_state().sp_enabled else count_loss_token(batches, weighted_normalization=True)
            )
            if model_runtime is not None:
                try:
                    unsupported_model = not (
                        model_runtime.model_config.model_type == "qwen4_exp"
                        and get_env("MODELING_BACKEND") != "hf"
                        and model_runtime.unwrapped_module.veomni_ce.impl == "chunk_loss"
                    )
                except (AttributeError, TypeError):
                    unsupported_model = True
        except Exception as exc:  # noqa: BLE001 - all ranks must leave before forward
            error = exc
    if error is not None and not torch.distributed.is_initialized():
        if isinstance(error, StopIteration):
            raise RuntimeError("Step-begin callback raised StopIteration") from error
        raise error
    reduced = all_reduce(
        [
            legacy["foundation_tokens"].item(),
            shifted["foundation_tokens"].item(),
            int(local_weighted),
            int(local_router),
            int(invalid_router),
            int(local_mtp),
            int(local_uncollated),
            int(unsupported_model),
            int(explicit_shift),
            int(explicit_count),
            int(scalar_labels),
            int(local_auxiliary),
            int(error is not None),
        ],
        op="sum",
    )
    if reduced[12]:
        raise RuntimeError("Step-begin callback or loss-token counting failed on a rank") from error
    weighted = bool(reduced[2])
    if weighted and reduced[4]:
        raise ValueError("output_router_logits must be boolean when using loss_weights")
    if weighted and reduced[3]:
        raise ValueError("loss_weights cannot be combined with router auxiliary loss")
    if weighted and reduced[5]:
        raise ValueError("loss_weights cannot be combined with MTP labels in the same step")
    if weighted and reduced[6]:
        raise ValueError("loss_weights are not supported for uncollated list-valued micro-batches")
    if weighted and reduced[7]:
        raise ValueError("loss_weights require the VeOmni qwen4_exp backend with chunk_loss")
    if weighted and reduced[8]:
        raise ValueError("loss_weights cannot be combined with explicit shift_labels")
    if weighted and reduced[9]:
        raise ValueError("loss_weights cannot be combined with num_items_in_batch")
    if weighted and reduced[10]:
        raise ValueError("loss_weights require sequence labels on every rank")
    counts = shifted if weighted else legacy
    if (sync_auxiliary or reduced[11]) and torch.distributed.is_initialized():
        synchronize_auxiliary_loss_tokens(counts)
    global_counts = {"foundation_tokens": reduced[1] if weighted else reduced[0]}
    global_counts.update(
        {
            key: all_reduce(value.item(), op="sum")
            for key, value in sorted(counts.items())
            if key != "foundation_tokens"
        }
    )
    if not any(value > 0 for value in global_counts.values()):
        raise ValueError("Training step has no supervised target mass")
    return weighted, counts, global_counts


def validate_weighted_loss_model(weighted: bool, model_type: str):
    """Do not silently ignore supervision weights on models without explicit support."""
    if weighted and (model_type != "qwen4_exp" or get_env("MODELING_BACKEND") == "hf"):
        raise ValueError("loss_weights require the VeOmni qwen4_exp model backend")

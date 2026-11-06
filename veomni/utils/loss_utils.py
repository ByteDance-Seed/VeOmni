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
from .dist_utils import all_reduce
from .env import get_env


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
        """Accumulate valid labels or weighted target mass in nested batches."""
        if isinstance(obj, dict) and not obj.get("padding_flag", False):
            valid = obj["labels"] != IGNORE_INDEX
            weights = obj.get("loss_weights")
            if weights is None:
                if weighted_normalization and not get_parallel_state().sp_enabled:
                    valid = valid[..., 1:]
                token_len["foundation_tokens"] = token_len["foundation_tokens"] + valid.sum()
            else:
                if weights.shape != valid.shape:
                    raise ValueError("loss_weights must match labels")
                # SP collators already shift labels and weights; without SP the
                # causal loss shifts in the model, so the first target is unused.
                if not get_parallel_state().sp_enabled:
                    weights, valid = weights[..., 1:], valid[..., 1:]
                token_len["foundation_tokens"] = token_len["foundation_tokens"] + (weights * valid).sum()

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


def reduce_global_loss_token(micro_batches_token_len: dict[str, torch.Tensor]) -> dict[str, float]:
    """All-reduce per-step loss-token denominators once for reuse across micro-batches."""
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
    local_has_weights = any("loss_weights" in batch for batch in _iter_feature_dicts(batches))
    if get_parallel_state().sp_enabled or not torch.distributed.is_initialized():
        return local_has_weights
    return bool(all_reduce(int(local_has_weights), op="max"))


def prepare_loss_token_counts(batches: list[dict[str, torch.Tensor]], model_runtime=None):
    """Reuse the denominator reduction to agree on weighted causal normalization.

    For non-SP training a weighted rank can coexist with an unweighted rank.
    Reduce both denominator variants and the opt-in flag together so that all
    ranks choose the same normalization without an additional collective.
    """
    features = list(_iter_feature_dicts(batches))
    local_weighted = any("loss_weights" in batch for batch in features)
    local_mtp = any("mtp_labels" in batch for batch in features)
    # BaseRLTrainer packs list-valued micro-batches inside preforward and
    # computes its own unweighted loss; weighted SFT cannot use that path.
    local_uncollated = isinstance(batches, (list, tuple)) and any(
        isinstance(batch, (list, tuple)) for batch in batches
    )
    router_flags = [batch.get("output_router_logits", False) for batch in features]
    local_router = any(value is True for value in router_flags)
    invalid_router = any(type(value) is not bool for value in router_flags)
    legacy = count_loss_token(batches)
    shifted = legacy if get_parallel_state().sp_enabled else count_loss_token(batches, weighted_normalization=True)
    reduced = all_reduce(
        [
            legacy["foundation_tokens"].item(),
            shifted["foundation_tokens"].item(),
            int(local_weighted),
            int(local_router),
            int(invalid_router),
            int(local_mtp),
            int(local_uncollated),
        ],
        op="sum",
    )
    weighted = bool(reduced[2])
    if weighted and reduced[4]:
        raise ValueError("output_router_logits must be boolean when using loss_weights")
    if weighted and reduced[3]:
        raise ValueError("loss_weights cannot be combined with router auxiliary loss")
    if weighted and reduced[5]:
        raise ValueError("loss_weights cannot be combined with MTP labels in the same step")
    if weighted and reduced[6]:
        raise ValueError("loss_weights are not supported for uncollated list-valued micro-batches")
    if weighted and model_runtime is not None:
        try:
            supported = (
                model_runtime.model_config.model_type == "qwen4_exp"
                and get_env("MODELING_BACKEND") != "hf"
                and model_runtime.unwrapped_module.veomni_ce.impl == "chunk_loss"
            )
        except (AttributeError, TypeError):
            supported = False
        if not supported:
            raise ValueError("loss_weights require the VeOmni qwen4_exp backend with chunk_loss")
    counts = shifted if weighted else legacy
    global_counts = {"foundation_tokens": reduced[1] if weighted else reduced[0]}
    global_counts.update(
        {key: all_reduce(value.item(), op="sum") for key, value in counts.items() if key != "foundation_tokens"}
    )
    return weighted, counts, global_counts


def validate_weighted_loss_model(weighted: bool, model_type: str):
    """Do not silently ignore supervision weights on models without explicit support."""
    if weighted and (model_type != "qwen4_exp" or get_env("MODELING_BACKEND") == "hf"):
        raise ValueError("loss_weights require the VeOmni qwen4_exp model backend")

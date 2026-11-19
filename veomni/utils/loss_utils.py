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


def count_loss_token(batches: Union[list[dict[str, torch.Tensor]], dict[str, torch.Tensor]]):
    """Calculate the total number of text_tokens/image_tokens/** for loss in a global batch, or one micro batch."""
    if isinstance(batches, dict):
        batches = [batches]
    token_len = {
        "foundation_tokens": torch.tensor(0),
    }

    def _count(obj):
        if isinstance(obj, dict) and not obj.get("padding_flag", False):
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


def dllm_mean_global_loss(
    local_weighted_sum: torch.Tensor,
    micro_batches_num_samples: int,
    sum_spans_sp_group: bool = False,
):
    """Global normalization for the dLLM masked-CE loss.

    The dLLM objective normalizes by the **number of samples** (``B``), not by
    token count (cf. ``mean_global_loss``). ``dllm_masked_cross_entropy`` with
    ``reduction="sum"`` returns the *local* weighted sum ``Σ_i w_i · CE_i`` for one
    micro-batch (``w_i = 1/(t_s·N_s)`` already baked in by the collator). This
    function turns that into the contribution to the global per-sample mean::

        loss = local_weighted_sum / global_num_samples * fsdp_size

    where ``global_num_samples`` is the all-reduced (over DP) total sample count
    across the whole optimizer step, and the ``* fsdp_size`` undoes FSDP's
    gradient averaging (same rationale as ``mean_global_loss``).

    Args:
        local_weighted_sum: this micro-batch's ``Σ w_i·CE_i`` (scalar tensor).
        micro_batches_num_samples: total sample count of all micro-batches in
            this step on the local DP rank (summed before the call).
        sum_spans_sp_group: the caller asserts that ``local_weighted_sum`` is the
            *whole* sequence-parallel group's sum rather than this rank's shard of
            it. Only then does the formula above hold unchanged under SP; see
            below. Left ``False`` by default because it is a property of the
            model, not of the topology, and the wrong answer here is a silently
            mis-scaled loss.

    Returns:
        The normalized scalar loss contribution for this micro-batch.

    Under sequence parallelism the formula needs no ``sp_size`` term, and it is
    worth showing why rather than trusting it. With ``dp_shard = D`` and
    ``ulysses = S``: ``dp_rank`` excludes the Ulysses dimension, so the ``S`` peers
    of a group are handed the *same* micro-batch and report the same
    ``micro_batches_num_samples``; the all-reduce below is over the world group,
    so it counts every sample ``S`` times and ``global_num_samples`` is ``S`` times
    the step's true count. ``fsdp_size`` is ``world_size // (pp * tp)``, which is
    ``D * S``, so it carries the same factor. The two cancel, leaving exactly what
    ``sp_size = 1`` over ``D`` ranks produces.

    That cancellation rests entirely on ``local_weighted_sum`` being the group's
    whole sum. A model that computes its loss on the sharded sequence hands in a
    *shard* of the sum, and then the ``S`` in ``fsdp_size`` has nothing to cancel
    against and the loss comes out ``S`` times small. DiffusionGemma qualifies for
    the opposite reason on each of its two terms: the diffusion term runs on
    canvases that SP replicates, and the AR term is passed through
    ``sequence_parallel.sum_over_ulysses_shards`` before it arrives. Hence the flag.
    """
    if get_parallel_state().sp_enabled and not sum_spans_sp_group:
        raise NotImplementedError(
            "dLLM loss reduction under sequence parallelism needs the loss to be whole across the SP "
            "group, and this caller has not said that it is. A model that computes its loss on the "
            "sharded sequence must reduce it over the group first (see "
            "`sequence_parallel.sum_over_ulysses_shards`) and pass `sum_spans_sp_group=True`; otherwise run "
            "with `ulysses_size: 1`."
        )

    global_num_samples = all_reduce(int(micro_batches_num_samples), op="sum")
    if global_num_samples == 0:
        # Keep the autograd graph intact while contributing zero.
        return local_weighted_sum * 0.0
    return local_weighted_sum / global_num_samples * get_parallel_state().fsdp_size

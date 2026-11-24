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

"""Training hooks that turn :class:`LladaOmni`'s local loss sum into the global mean.

The dLLM objective divides by the **step's** sample count rather than by a token
count: ``dllm_masked_cross_entropy(reduction="sum")`` already returns
``sum_i w_i * CE_i`` with ``w_i = 1/(t_s * N_s)``, and the trainer supplies the
``B`` in ``Eq. 13``'s ``1/B``. SeedOmni's own decode losses take the other
convention (``reduce_sequence_parallel_loss`` divides by valid tokens), so the
division has to happen here rather than in the executor.

Two pieces of state make that possible:

* ``on_micro_batches_ready`` receives the whole accumulation window and records
  how many samples the step holds. Per micro-batch alone this number is wrong by
  the accumulation factor, which is why the hook exists at all.
* ``_normalize_loss`` divides by it, once per micro-batch. Because that divisor
  spans the whole window, the loss leaving this module is already the step's
  mean, so ``loss_is_step_mean`` tells the trainer to skip its own
  ``1 / num_micro_steps`` on the backward. Without that claim the gradient would
  come out ``num_micro_steps`` times too small -- the executor's scaling applies
  to modules whose loss is a per-micro-batch mean.
"""

from __future__ import annotations

from typing import Any

import torch

from ......data.data_collator import add_flash_attention_kwargs_from_position_ids
from ......distributed.parallel_state import get_parallel_state
from ......utils.loss_utils import count_supervised_samples, dllm_mean_global_loss
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ..modeling import LladaOmni


class TrainingMixin(TrainingModuleMixin):
    """Per-step sample normalization for a single-node LLaDA graph."""

    #: Samples in this optimizer step, summed over every micro-batch. Set by
    #: :meth:`on_micro_batches_ready` before the loop starts; a forward outside
    #: ``train_step`` (a dummy pass, an eval) sees the initial ``1`` and only ever
    #: produces a loss nobody divides by.
    micro_batches_num_samples: int = 1

    #: Micro-batches in this step, so an aux term joins once per step rather than
    #: once per micro-batch. Default 1 covers a forward outside `train_step`.
    _num_micro_batches: int = 1

    #: ``dllm_mean_global_loss`` divides by the accumulation window's global sample
    #: count, so the loss leaving this module is already the step's mean and the
    #: trainer must not apply its ``1 / num_micro_steps`` on top.
    loss_is_step_mean: bool = True

    @pre_forward("forward")
    def batch_pre(self, conversation_list: list[Any] = None, **kwargs: Any) -> dict[str, Any]:
        """Pack the CPU-preprocessed items into the single row a mask predictor reads.

        ``SeedOmniCollator`` deliberately does not stack or pad -- each module
        owns its own batching -- so this is where the per-sample ``input_ids`` /
        ``labels`` / ``loss_weight`` tensors become one batch.

        The samples are **packed**, not padded into a ``[B, L]`` block, because
        the configured backend is ``flash_attention_2``: for a varlen backend
        ``bidirectional_attention_mask`` returns ``None`` and the sample
        boundaries come from ``cu_seq_lens_*``, which this module derives from
        ``position_ids``. A ``[B, L]`` block without those boundaries would let
        every row attend across the whole batch, and no mask would say otherwise.
        Packing is also what ``MainCollator`` does (it packs
        several samples into one row and marks each start with ``position_ids ==
        0``), so the two paths describe the same batch.

        The tensors are built here, inside the graph, after
        ``OmniTrainer.forward_backward_step`` has already moved the batch: the
        carrier's items stay where the CPU preprocessor built them, so this
        method places the batch itself instead of relying on that earlier move.

        ``answer_mask`` is deliberately not returned. It exists only so
        ``on_micro_batches_ready`` can count masked-nothing draws, and the forward
        would otherwise receive it as an unexpected keyword and pass it down to
        the decoder layers.

        Sequence parallelism is refused rather than silently mis-computed. Ulysses
        and context parallelism require this module to pad the packed row and hand
        each rank its own ``1 / sp_size`` chunk of ``input_ids`` / ``labels`` /
        ``loss_weight`` / ``position_ids``, exactly as the Qwen3 LLM module does;
        every tensor below is handed over whole, so a rank would supervise the
        entire row and its gradient would then be scaled by the group size. That
        path is unimplemented, and failing here is better than training on a
        wrong loss.
        """
        del kwargs
        state = get_parallel_state()
        if state.sp_enabled:
            raise NotImplementedError(
                "LladaOmni does not implement sequence parallelism yet: `batch_pre` hands the whole packed "
                f"row to every rank, so the loss would be wrong with sp_size={state.sp_size}. Run with "
                "`ulysses_size: 1` and `cp_size: 1`, or slice the batch in `batch_pre` first."
            )
        if conversation_list is None:
            return {}

        rows = []
        for sample in conversation_list:
            processed = None
            for item in sample:
                if isinstance(item.value, dict) and "input_ids" in item.value:
                    processed = item.value
                    break
            if processed is None:
                raise ValueError(
                    "LladaOmni: no processed item in this sample. The module's preprocessor "
                    "must run before the forward; check that `data_type: seedomni` and the "
                    "module's `processing.py` are both wired."
                )
            rows.append(processed)

        # The carrier's items were built by the CPU preprocessor and were never
        # moved, so the batch is placed on the module's device here.
        device = self.get_input_embeddings().weight.device

        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        loss_weight: list[torch.Tensor] = []
        position_ids: list[torch.Tensor] = []
        for r in rows:
            ids = r["input_ids"].to(device=device, dtype=torch.long).reshape(-1)
            n = int(ids.numel())
            input_ids.append(ids)
            attention_mask.append(r["attention_mask"].to(device=device, dtype=torch.long).reshape(-1))
            labels.append(r["labels"].to(device=device, dtype=torch.long).reshape(-1))
            loss_weight.append(r["loss_weight"].to(device=device, dtype=torch.float32).reshape(-1))
            # A packed row restarts at 0 for each sample; the varlen backend reads
            # the starts and builds `cu_seq_lens_*` from them.
            position_ids.append(torch.arange(n, device=device, dtype=torch.long))

        batch: dict[str, Any] = {
            "input_ids": torch.cat(input_ids).unsqueeze(0),
            "attention_mask": torch.cat(attention_mask).unsqueeze(0),
            "labels": torch.cat(labels).unsqueeze(0),
            "loss_weight": torch.cat(loss_weight).unsqueeze(0),
            "position_ids": torch.cat(position_ids).unsqueeze(0),
        }
        add_flash_attention_kwargs_from_position_ids(batch)
        return batch

    def on_micro_batches_ready(self, micro_batches: list[dict[str, Any]]) -> None:
        """Record this step's supervised sample count.

        The count is what ``dllm_mean_global_loss`` divides by, so it has to match
        the single-model training step exactly:
        :func:`~veomni.utils.loss_utils.count_supervised_samples`
        marks a sample when it has any *answer* token, not when its weights are
        non-zero. A LLaDA draw masks nothing in about ``1/(N+1)`` of samples with
        an ``N``-token answer; those contribute zero to the sum but are still one
        sample of the mean, and dropping them would bias the loss up by that
        fraction.

        Not floored at 1: ``dllm_mean_global_loss`` all-reduces this across data
        parallel ranks and handles a global zero, so a rank whose samples all
        degraded reports 0 and lets the other ranks carry the step.
        """
        self.micro_batches_num_samples = sum(self._micro_batch_sample_count(mb) for mb in micro_batches)
        self._num_micro_batches = len(micro_batches)

    @staticmethod
    def _micro_batch_sample_count(micro_batch: dict[str, Any]) -> int:
        """This micro-batch's supervised sample count.

        Runs on the *collated* batch, before ``batch_pre`` has stacked anything:
        ``SeedOmniCollator`` hands over ``conversation_list`` with each sample's
        tensors still inside its item, so the count comes from the items rather
        than from a ``loss_weight`` key that does not exist yet at this point.

        The per-item dicts carry the same ``answer_mask`` /
        ``loss_weight`` / ``position_ids`` keys the shared collator produces, so
        the shared predicate decides the count unchanged. LLaDA is a single
        stream with no canvas expansion, so no ``canvas_length`` reshaping
        applies.
        """
        conversation_list = micro_batch.get("conversation_list")
        if conversation_list is None:
            return 0
        count = 0
        for sample in conversation_list:
            for item in sample:
                value = getattr(item, "value", None)
                if not isinstance(value, dict) or "loss_weight" not in value:
                    continue
                count += count_supervised_samples(value)
                break
        return count

    @post_forward("forward")
    def normalize_loss(self, **outputs: Any) -> dict[str, Any]:
        """Divide the local sum by the step's global sample count.

        ``dllm_mean_global_loss`` all-reduces that count over data parallel ranks
        and multiplies by ``fsdp_size`` to undo FSDP's gradient averaging -- the
        same reduction the single-model training step performs for the LLaDA family.

        A MoE router's load-balancing term is already a per-token mean, so it is
        added after this normalization rather than divided by the sample count,
        which is what that step does too.

        The divisor covers the whole accumulation window, so the result is the
        step's mean rather than one micro-batch's contribution. That is what
        ``loss_is_step_mean = True`` declares to the trainer.

        No sequence-parallel branch: ``batch_pre`` refuses SP, so the local sum is
        already this rank's whole contribution.
        """
        loss = outputs.get("_loss")
        if not isinstance(loss, torch.Tensor):
            return outputs

        loss = dllm_mean_global_loss(loss, self.micro_batches_num_samples)

        aux_loss = outputs.get("aux_loss")
        if isinstance(aux_loss, torch.Tensor):
            aux_coef = float(getattr(self.config, "router_aux_loss_coef", 0.0) or 0.0)
            loss = loss + aux_coef * aux_loss.to(loss.device) / max(self._num_micro_batches, 1)
            del outputs["aux_loss"]
        outputs["_loss"] = loss
        return outputs


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceModuleMixin):
    pass


class LladaOmniAccelerated(VeOmniMixin, LladaOmni):
    pass


__all__ = ["LladaOmniAccelerated"]

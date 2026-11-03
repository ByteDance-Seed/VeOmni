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

"""Trainer-layer callback that schedules model checkpoint I/O.

This owns the every-N-steps / epochs cadence and the one-shot sidecar export.
*What* is written is :meth:`BaseTrainer.save_dcp` /
:meth:`~BaseTrainer.save_hf_or_lora` / :meth:`~BaseTrainer.load` /
:meth:`~BaseTrainer.save_model_assets`, which fan out to the model handles.
*How* belongs to each model's
:class:`~veomni.models.checkpoint.ModelCheckpointManager`.

DCP and HF/LoRA share this callback because they share a manager; each format
still has its own cadence knobs and its own last-saved step so a DCP write
does not suppress an HF export or the reverse.

Job-level global_state — where the dataloader is, the rng, the meters — is not written
here. It has its own schedule and its own files, in
:mod:`~veomni.trainer.callbacks.global_state_callback`.
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch.distributed as dist

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


def _single_multi_checkpoint(trainer):
    """Limit the same-step receipt to the synchronous single-model route."""
    runtime = getattr(trainer, "model", None)
    if isinstance(getattr(runtime, "module_runtimes", None), Mapping):
        return None
    optimizer = getattr(runtime, "optimizer", None)
    if getattr(optimizer, "_is_multi_optimizer", False) is not True:
        return None
    return runtime, optimizer, getattr(runtime, "checkpoint", None)


def _agree_multi_value(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def _current_multi_receipt(runtime, optimizer, checkpoint, step):
    from ...models.checkpoint.manager import ModelCheckpointManager

    if (
        not isinstance(checkpoint, ModelCheckpointManager)
        or getattr(checkpoint, "runtime", None) is not runtime
        or getattr(runtime, "optimizer", None) is not optimizer
    ):
        return None
    ref = getattr(checkpoint, "_last_saved_multi_optimizer_ref", None)
    generation = getattr(checkpoint, "_last_saved_optimizer_generation", None)
    if (
        getattr(checkpoint, "_last_saved_step", None) != step
        or not callable(ref)
        or ref() is not optimizer
        or not isinstance(generation, str)
        or not generation
    ):
        return None
    return generation


class CheckpointCallback(Callback):
    """Schedule DCP / HF / LoRA I/O and the one-shot tokenizer/config export."""

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        ckpt = args.train.checkpoint
        self.dcp_every_n_steps = ckpt.save_steps
        self.dcp_every_n_epochs = ckpt.save_epochs
        self.save_hf_weights = ckpt.save_hf_weights
        self.hf_every_n_steps = ckpt.hf_save_steps
        self.hf_every_n_epochs = ckpt.hf_save_epochs
        self._last_dcp_step: int = -1
        self._last_hf_step: int = -1
        self._last_multi_receipt = None
        self._uses_single_multi_checkpoint = _single_multi_checkpoint(trainer) is not None

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.trainer.save_model_assets()
        self.trainer.load()
        helper.empty_cache()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if self.save_hf_weights:
            if state.global_step != self._last_hf_step:
                self._save_hf(state, stage="train_end")
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )
                self.trainer.wait_for_pending_save()
        else:
            self.trainer.wait_for_pending_save()

    def on_step_end(self, state: TrainerState, **kwargs):
        if (
            self.dcp_every_n_steps
            and state.global_step % self.dcp_every_n_steps == 0
            and state.global_step != self._last_dcp_step
        ):
            self._save_dcp(state)
        if self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0:
            # HF/LoRA export requires a same-step DCP. When epoch DCP saves
            # are enabled, schedule that DCP through this callback so the
            # epoch-end cadence reuses it instead of overwriting its manifest.
            if self.dcp_every_n_epochs and state.global_step != self._last_dcp_step:
                if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                    self.trainer.wait_for_pending_save()
                self._save_dcp(state)
            try:
                self._save_hf(state)
            except BaseException:
                # An asynchronous DCP may have failed while export waited for
                # it; do not treat that submission as a completed save.
                if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                    self._last_dcp_step = -1
                raise

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0:
            if self._uses_single_multi_checkpoint:
                multi = _single_multi_checkpoint(self.trainer)
                recorded = self._last_multi_receipt if self._last_dcp_step == state.global_step else None
                current = _current_multi_receipt(*multi, state.global_step) if multi is not None else None
                values = _agree_multi_value((state.global_step, recorded, current))
                if any(value[1] is not None for value in values) and (
                    any(value != values[0] for value in values) or recorded is None or recorded != current
                ):
                    raise RuntimeError("Same-step MultiOptimizer checkpoint receipt changed across ranks")
            if state.global_step != self._last_dcp_step:
                self._save_dcp(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )
        if self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0:
            if state.global_step != self._last_hf_step:
                self._save_hf(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def _save_dcp(self, state: TrainerState):
        self.trainer.save_dcp(state)
        self._last_dcp_step = state.global_step
        self._last_multi_receipt = None

    def _save_hf(self, state: TrainerState, stage: str = "step_end"):
        if self._uses_single_multi_checkpoint:
            from ...models.checkpoint.manager import ModelCheckpointManager

            multi = _single_multi_checkpoint(self.trainer)
            runtime, optimizer, checkpoint = multi if multi is not None else (None, None, None)
            known_owner = (
                isinstance(checkpoint, ModelCheckpointManager)
                and checkpoint.runtime is runtime
                and not getattr(checkpoint.config, "save_async", False)
            )
            needs_save = known_owner and checkpoint._last_saved_step != state.global_step
            decisions = _agree_multi_value((state.global_step, known_owner, needs_save))
            if not known_owner or any(decision != decisions[0] for decision in decisions):
                raise RuntimeError("MultiOptimizer export save decision differs across ranks")
            if needs_save:
                self.trainer.save_dcp(state)
                checkpoint = getattr(runtime, "checkpoint", None)
            generation = _current_multi_receipt(runtime, optimizer, checkpoint, state.global_step)
            receipts = _agree_multi_value((state.global_step, generation))
            if generation is None or any(receipt != receipts[0] for receipt in receipts):
                raise RuntimeError("MultiOptimizer export has no matching same-step checkpoint receipt")
            self._last_dcp_step = state.global_step
            self._last_multi_receipt = generation
            # At train end the exporter may release the optimizer before the HF
            # gather. Do not keep it alive through local strong references.
            del optimizer, multi
        self.trainer.save_hf_or_lora(state, stage=stage)
        self._last_hf_step = state.global_step


__all__ = ["CheckpointCallback"]

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

import os
from collections.abc import Mapping
from typing import TYPE_CHECKING

from ...checkpoint import layout
from ...utils import helper
from ...utils.dist_utils import raise_if_any_rank_failed
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


def _export_wrote_dcp(trainer, step):
    """Reuse a completed export only when each trainable owner saved this step."""
    root = getattr(trainer, "model", None)
    modules = getattr(root, "module_runtimes", None)
    runtimes = list(modules.values()) if isinstance(modules, Mapping) and modules else [root]
    owners = []
    for runtime in runtimes:
        checkpoint = getattr(runtime, "checkpoint", None)
        if checkpoint is None and getattr(runtime, "has_trainable_parameters", None) is False:
            continue
        if checkpoint is None or getattr(checkpoint, "_last_saved_step", None) != step:
            return False
        owners.append(checkpoint)
    return bool(owners)


def _has_multi_optimizer(trainer):
    root = getattr(trainer, "model", None)
    modules = getattr(root, "module_runtimes", None)
    runtimes = list(modules.values()) if isinstance(modules, Mapping) and modules else [root]
    return any(
        getattr(getattr(runtime, "optimizer", None), "_is_multi_optimizer", False) is True for runtime in runtimes
    )


def _check_multi_destination(trainer, step):
    """Reject an existing transaction before any mixed-owner write begins."""
    if not _has_multi_optimizer(trainer):
        return
    error = None
    try:
        step_root = layout.step_dir(trainer.args.train.checkpoint.save_path, step)
        try:
            os.stat(layout.manifest_path(step_root))
        except FileNotFoundError:
            pass
        else:
            raise RuntimeError("Cannot overwrite a completed checkpoint transaction with MultiOptimizer state")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "checking MultiOptimizer checkpoint destination")


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
        self._async_save_failed = False
        self._external_pending_dcp = False
        self._external_pending_hf = False
        self._external_checkpoint_event = None

    def _check_async_save(self):
        if self._async_save_failed:
            raise RuntimeError("A previous asynchronous checkpoint save failed; restart from a complete checkpoint")

    def _wait_for_pending_save(self):
        try:
            self.trainer.wait_for_pending_save()
        except BaseException:
            if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                self._async_save_failed = True
                self._last_dcp_step = -1
            raise

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        from .global_state_callback import prepare_external_data_contract

        prepare_external_data_contract(self.trainer, install=True)
        self.trainer.save_model_assets()
        self.trainer.load()
        helper.empty_cache()

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        self._check_async_save()
        if getattr(self.trainer, "_veomni_require_exact_data_resume", False) is True:
            self._external_checkpoint_event = None
            hf_due = self.save_hf_weights and state.global_step != self._last_hf_step
            if self._external_pending_dcp or self._external_pending_hf or hf_due:
                self._save_exact_external_event(state, "train_end", False, bool(hf_due))
                if self._external_pending_dcp or self._external_pending_hf:
                    raise RuntimeError("Training ended before the exact data checkpoint became ready")
            else:
                self._wait_for_pending_save()
                self._external_checkpoint_event = ("train_end", state.global_step, "none", False)
            return
        if self.save_hf_weights:
            if state.global_step != self._last_hf_step:
                self._save_hf(state, stage="train_end")
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )
                self._wait_for_pending_save()
        else:
            self._wait_for_pending_save()

    def on_step_end(self, state: TrainerState, **kwargs):
        self._check_async_save()
        dcp_due = self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0
        hf_due = self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0
        if getattr(self.trainer, "_veomni_require_exact_data_resume", False) is True:
            self._external_checkpoint_event = None
            if dcp_due or hf_due or self._external_pending_dcp or self._external_pending_hf:
                self._save_exact_external_event(state, "step_end", bool(dcp_due), bool(hf_due))
            else:
                self._external_checkpoint_event = ("step_end", state.global_step, "none", False)
            return
        if dcp_due and state.global_step != self._last_dcp_step:
            self._save_dcp(state)
        if hf_due:
            # Export needs a same-step DCP. Schedule it through the trainer so
            # all model owners share the receipt used by the epoch cadence.
            if self.dcp_every_n_epochs and state.global_step != self._last_dcp_step:
                if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                    self._wait_for_pending_save()
                self._save_dcp(state)
            self._save_hf(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        self._check_async_save()
        if getattr(self.trainer, "_veomni_require_exact_data_resume", False) is True:
            self._external_checkpoint_event = None
            dcp_due = self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0
            hf_due = (
                self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0
            )
            if dcp_due or hf_due:
                self._save_exact_external_event(state, "epoch_end", bool(dcp_due), bool(hf_due))
            else:
                self._external_checkpoint_event = ("epoch_end", state.global_step, "none", False)
            return
        if self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0:
            if state.global_step != self._last_dcp_step:
                self._save_dcp(state)
            else:
                if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                    self._wait_for_pending_save()
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

    def _save_exact_external_event(self, state, event, dcp_due, hf_due):
        """Defer all pieces together until every data provider is ready."""
        from .global_state_callback import external_data_checkpoint_ready

        dcp_due = (dcp_due or self._external_pending_dcp) and state.global_step != self._last_dcp_step
        hf_due = (hf_due or self._external_pending_hf) and state.global_step != self._last_hf_step
        if not dcp_due and not hf_due:
            self._external_pending_dcp = False
            self._external_pending_hf = False
            self._external_checkpoint_event = (event, state.global_step, "none", False)
            return
        if not external_data_checkpoint_ready(self.trainer, state):
            self._external_pending_dcp = dcp_due
            self._external_pending_hf = hf_due
            self._external_checkpoint_event = (event, state.global_step, "deferred", False)
            logger.warning_rank0(f"Deferring exact-data checkpoint at step {state.global_step} until replay drains.")
            return
        if dcp_due and state.global_step != self._last_dcp_step:
            self._save_dcp(state)
        if hf_due and state.global_step != self._last_hf_step:
            self._save_hf(state, stage=event)
        self._external_pending_dcp = False
        self._external_pending_hf = False
        self._external_checkpoint_event = (event, state.global_step, "saved", self._last_dcp_step == state.global_step)

    def _save_dcp(self, state: TrainerState):
        try:
            from .global_state_callback import prepare_external_data_contract

            prepare_external_data_contract(self.trainer, state)
            _check_multi_destination(self.trainer, state.global_step)
            self.trainer.save_dcp(state)
        except BaseException:
            if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                self._async_save_failed = True
                self._last_dcp_step = -1
            raise
        self._last_dcp_step = state.global_step

    def _save_hf(self, state: TrainerState, stage: str = "step_end"):
        try:
            from .global_state_callback import prepare_external_data_contract

            prepare_external_data_contract(self.trainer, state)
            if _has_multi_optimizer(self.trainer) and self._last_dcp_step != state.global_step:
                self._save_dcp(state)
            self.trainer.save_hf_or_lora(state, stage=stage)
        except BaseException:
            if getattr(self.trainer.args.train.checkpoint, "save_async", False):
                self._async_save_failed = True
                self._last_dcp_step = -1
            raise
        self._last_hf_step = state.global_step
        if _export_wrote_dcp(self.trainer, state.global_step):
            self._last_dcp_step = state.global_step


__all__ = ["CheckpointCallback"]

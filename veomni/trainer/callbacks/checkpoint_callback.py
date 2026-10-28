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

import math
from typing import TYPE_CHECKING

from torch.distributed.checkpoint.api import CheckpointException

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


def _validate_file_profile_checkpoint_schedule(trainer):
    """Reject schedules that would require an unvalidated epoch-boundary resume."""
    args = trainer.args
    checkpoint = args.train.checkpoint
    if checkpoint.save_epochs or checkpoint.save_hf_weights:
        raise ValueError(
            "File-backed exact resume does not support epoch/final checkpoint saves; "
            "set save_epochs=0 and save_hf_weights=false and use a safe step cadence"
        )
    steps, interval, epochs = args.train_steps, checkpoint.save_steps, args.train.num_train_epochs
    verified_stop = getattr(trainer, "_veomni_exact_verified_stop_after_steps", None)
    if verified_stop is not None and (
        type(verified_stop) is not int
        or verified_stop < 0
        or verified_stop != getattr(args.train, "stop_after_steps", None)
    ):
        raise ValueError("File-backed checkpoint stop bound is not attested by the active trainer")
    if interval and all(type(value) is int and value > 0 for value in (steps, interval, epochs)):
        final_step = steps * epochs
        if verified_stop:
            final_step = min(final_step, verified_stop)
        if math.lcm(steps, interval) <= final_step:
            raise ValueError(
                "File-backed checkpoint step cadence reaches an unvalidated epoch boundary; "
                "choose save_steps that avoids every planned epoch boundary"
            )


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

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        from .global_state_callback import (
            _exact_checkpoint_contract,
            _has_file_profile,
            _resume_identity,
            _validate_exact_environ_meter,
            validate_resumed_prefetch_is_drained,
        )

        if _has_file_profile(self.trainer):
            from ...utils.dist_utils import raise_if_any_rank_failed, require_same_rank_value

            error, identity, runtime_identity = None, None, None
            try:
                _validate_file_profile_checkpoint_schedule(self.trainer)
                contract = _exact_checkpoint_contract(self.trainer)
                contract.install()
                identity = _exact_checkpoint_contract(self.trainer, require_identity=True).checkpoint_identity()
                runtime_identity = _resume_identity(self.trainer)
                _validate_exact_environ_meter(self.trainer, self.trainer.environ_meter.state_dict(), 0)
            except (Exception, CheckpointException) as exc:
                error = exc
            raise_if_any_rank_failed(error, "installing the exact data checkpoint adapter")
            require_same_rank_value(
                (
                    self.trainer.args.train_steps,
                    self.trainer.args.train.num_train_epochs,
                    self.trainer.args.train.checkpoint.save_steps,
                    getattr(self.trainer, "_veomni_exact_verified_stop_after_steps", None),
                    getattr(self.trainer.args.train, "stop_after_steps", None),
                ),
                "exact checkpoint preflight cadence",
            )
            require_same_rank_value(identity, "data checkpoint contract identity")
            require_same_rank_value(runtime_identity, "exact resume runtime identity")

        # Validate an opt-in exact-resume loader before a full model checkpoint
        # is loaded or a fresh run spends time training without a safe save.
        validate_resumed_prefetch_is_drained(self.trainer)
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
        if self.dcp_every_n_steps and state.global_step % self.dcp_every_n_steps == 0:
            self._save_dcp(state)
        if self.save_hf_weights and self.hf_every_n_steps and state.global_step % self.hf_every_n_steps == 0:
            self._save_hf(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0:
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
        from .global_state_callback import (
            _has_file_profile,
            _validate_exact_epoch,
            validate_resumed_prefetch_is_drained,
        )

        validate_resumed_prefetch_is_drained(self.trainer, state)
        if _has_file_profile(self.trainer):
            from ...utils.dist_utils import raise_if_any_rank_failed

            error = None
            try:
                _validate_exact_epoch(self.trainer, state.global_step, state.epoch)
            except Exception as exc:
                error = exc
            raise_if_any_rank_failed(error, "validating file-backed checkpoint epoch")
        self.trainer.save_dcp(state)
        self._last_dcp_step = state.global_step

    def _save_hf(self, state: TrainerState, stage: str = "step_end"):
        from .global_state_callback import validate_resumed_prefetch_is_drained

        validate_resumed_prefetch_is_drained(self.trainer, state)
        self.trainer.save_hf_or_lora(state, stage=stage)
        self._last_hf_step = state.global_step
        # A model checkpoint manager may have written DCP as part of HF/LoRA
        # export. Record it only when the manager confirms completion; other
        # trainers may export directly without a DCP.
        manager = getattr(getattr(self.trainer, "model", None), "checkpoint", None)
        if getattr(manager, "_last_saved_step", None) == state.global_step:
            self._last_dcp_step = state.global_step


__all__ = ["CheckpointCallback"]

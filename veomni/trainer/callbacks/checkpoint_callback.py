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

"""Trainer-layer callbacks that schedule model checkpoint I/O.

These own the every-N-steps / epochs cadence and nothing else. *What* is written
is :meth:`BaseTrainer.save_dcp` / :meth:`~BaseTrainer.save_hf_or_lora` /
:meth:`~BaseTrainer.load`, which fan out to the model handles; *how* belongs to
each model's :class:`~veomni.models.checkpoint_manager.ModelCheckpointManager`.

Job-level state — where the dataloader is, the rng, the meters — is not written
here. It has its own schedule and its own files, in
:mod:`~veomni.trainer.callbacks.global_state_callback`.
"""

from typing import TYPE_CHECKING

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


class ModelDcpCallback(Callback):
    """Schedule the resumable checkpoint via :meth:`BaseTrainer.load` / ``save_dcp``."""

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        # Tracked per callback rather than read off the model: the DCP and HF
        # callbacks save on independent cadences, and each one's "did I already
        # save this step?" answer has to be about its own writes.
        self._last_saved_step: int = -1

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.trainer.load()
        # Free transient buffers from DCP materialization before the first train
        # step. Large MoE resumes are often near GPU capacity; leftover allocator
        # fragments after load can OOM the first NCCL collective (e.g. the
        # grad-norm all-reduce).
        helper.empty_cache()

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        self.trainer.save_dcp(state)
        self._last_saved_step = state.global_step


class ModelHfCallback(Callback):
    """Schedule the HF / LoRA export; the model picks which format it writes."""

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.save_hf_weights = args.train.checkpoint.save_hf_weights
        self.every_n_steps = args.train.checkpoint.hf_save_steps
        self.every_n_epochs = args.train.checkpoint.hf_save_epochs
        self._last_saved_step: int = -1

    def on_train_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state, stage="train_end")
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at train_end (global_step {state.global_step} "
                    f"already saved)."
                )

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self._save_checkpoint(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate HF checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        """Export the weights, in HF safetensors or PEFT layout as the model requires."""
        self.trainer.save_hf_or_lora(state, stage=stage)
        self._last_saved_step = state.global_step


__all__ = ["ModelDcpCallback", "ModelHfCallback"]

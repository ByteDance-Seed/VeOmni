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

"""Trainer-layer callbacks that schedule composed-model checkpoint I/O.

Per-module looping lives on :class:`~OmniModelRuntime`; these callbacks only own
the every-N-steps / epochs cadence and call :meth:`OmniTrainer.save_dcp` /
:meth:`OmniTrainer.save_hf_or_lora` / :meth:`OmniTrainer.load`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


class OmniModuleDcpCallback(Callback):
    """Schedule DCP load/save via :meth:`OmniTrainer.load` / :meth:`OmniTrainer.save_dcp`."""

    trainer: OmniTrainer

    def __init__(self, trainer: OmniTrainer) -> None:
        super().__init__(trainer)
        ckpt = trainer.args.train.checkpoint
        self.every_n_steps = ckpt.save_steps
        self.every_n_epochs = ckpt.save_epochs

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.trainer.load()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if not self.every_n_steps or state.global_step % self.every_n_steps != 0:
            return
        self.trainer.save_dcp(state)

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        if not self.every_n_epochs or (state.epoch + 1) % self.every_n_epochs != 0:
            return
        self.trainer.save_dcp(state)


class OmniModuleHfCallback(Callback):
    """Schedule HF / LoRA export via :meth:`OmniTrainer.save_hf_or_lora`."""

    trainer: OmniTrainer

    def __init__(self, trainer: OmniTrainer) -> None:
        super().__init__(trainer)
        ckpt = trainer.args.train.checkpoint
        self.save_hf_weights = ckpt.save_hf_weights
        self.every_n_steps = ckpt.hf_save_steps
        self.every_n_epochs = ckpt.hf_save_epochs

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        if not self.save_hf_weights:
            return
        self.trainer.save_hf_or_lora(state)

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if not self.save_hf_weights or not self.every_n_steps:
            return
        if state.global_step % self.every_n_steps != 0:
            return
        self.trainer.save_hf_or_lora(state)

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        if not self.save_hf_weights or not self.every_n_epochs:
            return
        if (state.epoch + 1) % self.every_n_epochs != 0:
            return
        self.trainer.save_hf_or_lora(state)


__all__ = ["OmniModuleDcpCallback", "OmniModuleHfCallback"]

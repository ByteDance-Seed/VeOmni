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

"""Orchestrator-level checkpoint callbacks (distinct from per-module runtime I/O)."""

from __future__ import annotations

import os
import random
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.distributed as dist

from ....utils import logging
from ..base import Callback, TrainerState


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


logger = logging.get_logger(__name__)


def global_state_path(root: str, global_step: int) -> str:
    return os.path.join(root, f"global_step_{global_step}", "trainer_state.pt")


class OmniRootAssetsCallback(Callback):
    """Scheduler: export omni-root HF layout once at train begin."""

    trainer: OmniTrainer

    def on_train_begin(self, state, **kwargs) -> None:
        self.trainer.save_model_assets()


class OmniGlobalStateCallback(Callback):
    """Save / resume module-agnostic orchestrator state (step, dataloader, meter, RNG).

    Per-module weights and optimizers are scheduled by :class:`OmniModuleDcpCallback`
    / :class:`OmniModuleHfCallback`, which call :meth:`OmniTrainer.save_dcp` /
    :meth:`OmniTrainer.save_hf_or_lora` / :meth:`OmniTrainer.load` (fan-out lives
    on :class:`~OmniModelRuntime`).
    Omni-root HF layout export is :class:`OmniRootAssetsCallback` +
    :meth:`OmniTrainer.save_model_assets`.
    """

    trainer: OmniTrainer

    def __init__(self, trainer: OmniTrainer) -> None:
        super().__init__(trainer)
        args = trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1

    def _train_dataloader_state(self) -> dict[str, Any]:
        if self.trainer.data_iterator is not None and hasattr(self.trainer.data_iterator, "state_dict"):
            return self.trainer.data_iterator.state_dict()
        if self.trainer.train_dataloader is not None:
            return self.trainer.train_dataloader.state_dict()
        return {}

    def save_global_state(self, global_step: int) -> None:
        """Persist orchestrator state to ``trainer_state.pt`` under the step folder."""
        args = self.trainer.args
        if args.train.global_rank == 0:
            state_path = global_state_path(args.train.checkpoint.save_path, global_step)
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            torch.save(
                {
                    "global_step": global_step,
                    "train_dataloader": self._train_dataloader_state(),
                    "environ_meter": self.trainer.environ_meter.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "numpy_rng_state": np.random.get_state(),
                    "python_rng_state": random.getstate(),
                },
                state_path,
            )
            logger.info_rank0(f"OmniGlobalStateCallback: saved orchestrator state → {state_path}")
        if dist.is_initialized():
            dist.barrier()

    def load_global_state(self) -> None:
        """Restore orchestrator state from ``load_path/trainer_state.pt``."""
        trainer = self.trainer
        args = trainer.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return
        trainer_state_path = os.path.join(load_path, "trainer_state.pt")
        if not os.path.exists(trainer_state_path):
            return
        ts = torch.load(trainer_state_path, map_location="cpu", weights_only=False)
        trainer.state.global_step = ts["global_step"]
        trainer.start_epoch = trainer.state.global_step // args.train_steps
        trainer.start_step = trainer.state.global_step % args.train_steps
        if trainer.train_dataloader is not None and ts.get("train_dataloader"):
            trainer.train_dataloader.load_state_dict(ts["train_dataloader"])
        if ts.get("environ_meter") is not None:
            trainer.environ_meter.load_state_dict(ts["environ_meter"])
        torch.set_rng_state(ts["torch_rng_state"])
        if ts.get("numpy_rng_state") is not None:
            np.random.set_state(ts["numpy_rng_state"])
        if ts.get("python_rng_state") is not None:
            random.setstate(ts["python_rng_state"])
        if trainer.start_step == 0 and trainer.train_dataloader is not None:
            iter(trainer.train_dataloader)
        logger.info_rank0(
            f"OmniGlobalStateCallback: restored orchestrator state from {trainer_state_path} "
            f"(global_step={trainer.state.global_step}, start_epoch={trainer.start_epoch}, "
            f"start_step={trainer.start_step})."
        )

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self.save_global_state(state.global_step)
            self._last_saved_step = state.global_step

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self.save_global_state(state.global_step)
                self._last_saved_step = state.global_step

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.load_global_state()


__all__ = ["OmniGlobalStateCallback", "OmniRootAssetsCallback", "global_state_path"]

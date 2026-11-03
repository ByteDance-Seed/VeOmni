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

import torch
import torch.distributed as dist

from ...utils import helper
from ...utils.device import get_device_type
from ...utils.dist_utils import raise_if_any_rank_failed
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


def _confirmed_export_optimizer(checkpoint, step):
    if not hasattr(checkpoint, "_last_export_validated_step"):
        return True  # Compatibility with checkpoint providers that only expose a receipt.
    if checkpoint._last_export_validated_step != step:
        return False
    optimizer = getattr(getattr(checkpoint, "runtime", None), "optimizer", None)
    if optimizer is None:
        return getattr(checkpoint, "_last_export_released_optimizer", False) is True
    saved_ref = getattr(checkpoint, "_last_saved_optimizer_ref", None)
    return callable(saved_ref) and saved_ref() is optimizer


def _check_same_step_receipt(trainer, step, same_step):
    """Never skip a same-step save after optimizer identity has changed."""
    # This is an actual save receipt, which may differ after a rank-local I/O
    # failure; it is not a comparison of the shared checkpoint schedule.
    if dist.is_initialized():
        decisions = [None] * dist.get_world_size()
        dist.all_gather_object(decisions, (step, same_step))
        if any(decision != (step, same_step) for decision in decisions):
            raise RuntimeError("Same-step checkpoint save decision differs across ranks")
    if not same_step:
        return
    root = getattr(trainer, "model", None)
    modules = getattr(root, "module_runtimes", None)
    runtimes = list(modules.values()) if isinstance(modules, Mapping) and modules else [root]
    error = None
    try:
        for runtime in runtimes:
            checkpoint = getattr(runtime, "checkpoint", None)
            check = getattr(type(checkpoint), "checkpoint_receipt_is_current", None)
            if callable(check) and not check(checkpoint, step):
                raise RuntimeError("Optimizer or checkpoint changed after the same-step DCP save")
    except BaseException as exc:
        error = exc
    raise_if_any_rank_failed(error, "validating the same-step checkpoint")


def _export_wrote_dcp(trainer, step):
    """Only reuse an export's DCP when every model manager confirms its step."""
    root = getattr(trainer, "model", None)
    modules = getattr(root, "module_runtimes", None)
    runtimes = list(modules.values()) if isinstance(modules, Mapping) and modules else [root]
    # Omni intentionally has no manager for fully frozen modules; unknown or
    # trainable modules with no manager must still prevent deduplication.
    checkpoints = [
        getattr(runtime, "checkpoint", None)
        for runtime in runtimes
        if getattr(runtime, "checkpoint", None) is not None
        or getattr(runtime, "has_trainable_parameters", None) is not False
    ]
    saved = bool(checkpoints) and all(
        checkpoint is not None
        and getattr(getattr(checkpoint, "checkpointer", None), "_async_save_failed", False) is not True
        and _confirmed_export_optimizer(checkpoint, step)
        and type(getattr(checkpoint, "_last_saved_step", None)) is int
        and checkpoint._last_saved_step == step
        and (
            (
                isinstance(getattr(checkpoint, "_last_saved_optimizer_generation", None), str)
                and bool(checkpoint._last_saved_optimizer_generation)
            )
            or (
                type(getattr(checkpoint, "_last_legacy_export_dcp_step", None)) is int
                and checkpoint._last_legacy_export_dcp_step == step
            )
        )
        for checkpoint in checkpoints
    )
    if dist.is_initialized():
        group = dist.group.WORLD
        device = "cpu" if dist.get_backend(group) == "gloo" else get_device_type()
        flag = torch.tensor([int(saved)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
        saved = bool(flag.item())
    return saved


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
        dcp_due = bool(self.dcp_every_n_epochs and (state.epoch + 1) % self.dcp_every_n_epochs == 0)
        hf_due = bool(
            self.save_hf_weights and self.hf_every_n_epochs and (state.epoch + 1) % self.hf_every_n_epochs == 0
        )
        if dcp_due:
            _check_same_step_receipt(self.trainer, state.global_step, state.global_step == self._last_dcp_step)
            if state.global_step != self._last_dcp_step:
                self._save_dcp(state)
            else:
                logger.info_rank0(
                    f"Skipping duplicate checkpoint save at epoch_end (global_step {state.global_step} "
                    f"already saved at step_end)."
                )
        if hf_due:
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

    def _save_hf(self, state: TrainerState, stage: str = "step_end"):
        self.trainer.save_hf_or_lora(state, stage=stage)
        self._last_hf_step = state.global_step
        if _export_wrote_dcp(self.trainer, state.global_step):
            self._last_dcp_step = state.global_step


__all__ = ["CheckpointCallback"]

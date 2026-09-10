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

"""Job-level checkpoint callback, as distinct from per-model checkpoint I/O.

Nothing here belongs to a model: where the dataloader is, the rng, the metric
meters. Model weights, optimizer, HF/LoRA export, and the tokenizer/config
sidecars are scheduled by :mod:`~veomni.trainer.callbacks.checkpoint_callback`.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

from ...utils import helper
from ...utils.device import get_device_type
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)

_GLOBAL_STATE_FORMAT = "trainer_state_rank_{}.pt"


def global_state_path(root: str, rank: int) -> str:
    return os.path.join(root, _GLOBAL_STATE_FORMAT.format(rank))


class GlobalStateCallback(Callback):
    """Save and resume the state that belongs to the job rather than to a model.

    Written per rank, not once on rank 0. The cursor in here is rank-local by
    construction: iterable datasets are ``split_dataset_by_node``-sharded on
    ``dp_rank``, the multisource sampler filters on ``_global_sample_idx %
    dp_size == dp_rank``, and Energon takes ``dp_rank`` in its ``WorkerConfig``.
    Restoring one rank's cursor everywhere would make every rank resume on rank
    0's shard — replaying that slice and skipping the rest.
    """

    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1

    @property
    def rank(self) -> int:
        return self.trainer.args.train.global_rank

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.load_global_state()

    def on_step_end(self, state: TrainerState, **kwargs) -> None:
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self.save_global_state(state)

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            if state.global_step != self._last_saved_step:
                self.save_global_state(state)

    def state_dict(self, state: TrainerState) -> Dict[str, Any]:
        if hasattr(self.trainer, "data_iterator") and hasattr(self.trainer.data_iterator, "state_dict"):
            train_dataloader_state = self.trainer.data_iterator.state_dict()
        elif self.trainer.train_dataloader is not None:
            train_dataloader_state = self.trainer.train_dataloader.state_dict()
        else:
            train_dataloader_state = {}

        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        channel_loss_state = channel_loss_callback.state_dict() if channel_loss_callback is not None else {}

        return {
            "global_step": state.global_step,
            "train_dataloader": train_dataloader_state,
            "environ_meter": self.trainer.environ_meter.state_dict(),
            "channel_loss_callback": channel_loss_state,
            "torch_rng_state": torch.get_rng_state(),
        }

    def save_global_state(self, state: TrainerState) -> None:
        # Drain a pending async DCP save first. CheckpointCallback returns while
        # that write is still in flight; a cursor file that lands before the
        # shards would resume a step whose weights never made it to disk.
        checkpoint = getattr(self.trainer, "checkpoint", None)
        if checkpoint is not None:
            checkpoint.wait_for_pending_save()

        args: "VeOmniArguments" = self.trainer.args
        step_dir = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        torch.save(self.state_dict(state), global_state_path(step_dir, self.rank))
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = state.global_step

    def load_global_state(self) -> Optional[Dict[str, Any]]:
        args: "VeOmniArguments" = self.trainer.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return None

        state_path = global_state_path(load_path, self.rank)
        found = os.path.exists(state_path)
        if dist.is_initialized():
            flag = torch.tensor([int(found)], dtype=torch.int32, device=get_device_type())
            dist.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            found = bool(flag.item())
            if not found:
                logger.warning_rank0("Trainer state missing on at least one rank; resuming weights only.")
                return None
        elif not found:
            logger.warning(f"No trainer state at {state_path}; resuming weights only.")
            return None

        global_state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.trainer.state.global_step = global_state["global_step"]
        self._restore_position(global_state)

        channel_loss_state = global_state.get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        if self.trainer.train_dataloader is not None and global_state.get("train_dataloader") is not None:
            self.trainer.train_dataloader.load_state_dict(global_state["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(global_state["environ_meter"])
        torch.set_rng_state(global_state["torch_rng_state"])
        if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
            iter(self.trainer.train_dataloader)

        logger.info_rank0(
            f"Restored trainer state from {state_path} (global_step={self.trainer.state.global_step}, "
            f"start_epoch={self.trainer.start_epoch}, start_step={self.trainer.start_step})."
        )
        return global_state

    def _restore_position(self, global_state: Dict[str, Any]) -> None:
        args: "VeOmniArguments" = self.trainer.args
        global_step = global_state["global_step"]
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps


__all__ = ["GlobalStateCallback", "global_state_path"]

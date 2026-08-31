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

"""Job-level checkpoint callbacks, as distinct from per-model checkpoint I/O.

Nothing here belongs to a model: where the dataloader is, the rng, the metric
meters. The sidecars an export needs are the model's; :class:`RootAssetsCallback`
only decides *when*. Model weights and optimizers are scheduled by
:mod:`~veomni.trainer.callbacks.checkpoint_callback`.
"""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)

_GLOBAL_STATE_FORMAT = "trainer_state_rank_{}.pt"


def global_state_path(root: str, rank: int) -> str:
    return os.path.join(root, _GLOBAL_STATE_FORMAT.format(rank))


class RootAssetsCallback(Callback):
    """Export the config / tokenizer / processor sidecars once, at train begin."""

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self.trainer.model.save_model_assets()


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
        """The job-level state to persist for this step."""
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
        """Write this rank's job state beside the step's model checkpoint."""
        args: "VeOmniArguments" = self.trainer.args
        step_dir = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        os.makedirs(step_dir, exist_ok=True)
        torch.save(self.state_dict(state), global_state_path(step_dir, self.rank))
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = state.global_step

    def load_global_state(self) -> Optional[Dict[str, Any]]:
        """Restore this rank's job state from ``load_path``, if there is one."""
        args: "VeOmniArguments" = self.trainer.args
        load_path = args.train.checkpoint.load_path
        if load_path is None:
            return None

        state_path = global_state_path(load_path, self.rank)
        if not os.path.exists(state_path):
            logger.warning_rank0(f"No trainer state at {state_path}; resuming weights only.")
            return None

        global_state = torch.load(state_path, map_location="cpu", weights_only=False)
        self.trainer.state.global_step = global_state["global_step"]
        self._restore_position(global_state)

        channel_loss_state = global_state.get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        # dataloader may only init on sp_rank_0 to save memory
        if self.trainer.train_dataloader is not None and global_state.get("train_dataloader") is not None:
            self.trainer.train_dataloader.load_state_dict(global_state["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(global_state["environ_meter"])
        torch.set_rng_state(global_state["torch_rng_state"])
        if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        logger.info_rank0(
            f"Restored trainer state from {state_path} (global_step={self.trainer.state.global_step}, "
            f"start_epoch={self.trainer.start_epoch}, start_step={self.trainer.start_step})."
        )
        return global_state

    def _restore_position(self, global_state: Dict[str, Any]) -> None:
        """Place the resumed run back in the epoch/step grid.

        Takes the whole blob rather than the step it needs, because this is the
        seam for a subclass that stored a finer-grained cursor (see
        ``StepAwareTestGlobalStateCallback``): only the subclass knows which key
        it wrote, and ``start_step`` has to settle here, before
        :meth:`load_global_state` decides whether to prefetch.
        """
        args: "VeOmniArguments" = self.trainer.args
        global_step = global_state["global_step"]
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps


__all__ = ["GlobalStateCallback", "RootAssetsCallback", "global_state_path"]

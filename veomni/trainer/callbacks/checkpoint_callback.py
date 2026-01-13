import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ...checkpoint import CheckpointerBase, build_checkpointer, ckpt_to_state_dict
from ...models import save_model_assets, save_model_weights
from ...utils import helper
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import Arguments, BaseTrainer


logger = helper.create_logger(__name__)


class CheckpointerCallback(Callback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "Arguments" = self.trainer.args
        self.every_n_steps = args.train.save_steps
        self.every_n_epochs = args.train.save_epochs
        self.trainer.checkpointer: CheckpointerBase = build_checkpointer(
            dist_backend=args.train.data_parallel_mode, ckpt_manager=args.train.ckpt_manager
        )

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            self._save_checkpoint(state)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint from path."""
        args: "Arguments" = self.trainer.args
        if args.train.load_checkpoint_path is None:
            return

        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }
        self.trainer.checkpointer.load(args.train.load_checkpoint_path, state)

        self.trainer.state.global_step = state["extra_state"]["global_step"]
        self.trainer.start_epoch = self.trainer.state.global_step // args.train.train_steps
        self.trainer.start_step = self.trainer.state.global_step % args.train.train_steps

        self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
        self.trainer.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])
        self.trainer.environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])

        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.load_checkpoint_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "Arguments" = self.trainer.args

        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{state.global_step}")

        ckpt_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {
                "global_step": state.global_step,
                "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
                "train_dataloader": self.trainer.train_dataloader.state_dict(),
                "environ_meter": self.trainer.environ_meter.state_dict(),
                "torch_rng_state": torch.get_rng_state(),
            },
        }
        self.trainer.checkpointer.save(save_checkpoint_path, ckpt_state, save_async=args.train.save_async)

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


class HuggingfaceCkptCallback(CheckpointerCallback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "Arguments" = self.trainer.args
        self.save_hf_weights = args.train.save_hf_weights
        self.every_n_steps = args.train.hf_save_steps
        self.every_n_epochs = args.train.hf_save_epochs

    def on_train_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights:
            self._save_checkpoint(state)

    def on_step_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_steps and state.global_step % self.every_n_steps == 0:
            self._save_checkpoint(state)

    def on_epoch_end(self, state: TrainerState, **kwargs):
        if self.save_hf_weights and self.every_n_epochs and (state.epoch + 1) % self.every_n_epochs == 0:
            self._save_checkpoint(state)

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._save_model_assets()

    def _save_model_assets(self):
        args: "Arguments" = self.trainer.args
        if args.train.global_rank == 0:
            save_model_assets(args.train.model_assets_dir, self.trainer.model_assets)
        dist.barrier()

    def _save_checkpoint(self, state: TrainerState):
        """Save model in HuggingFace format."""

        args: "Arguments" = self.trainer.args
        save_checkpoint_path = os.path.join(args.train.save_checkpoint_path, f"global_step_{state.global_step}")
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            super()._save_checkpoint(state)

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        if args.train.global_rank == 0:
            hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
            model_state_dict = ckpt_to_state_dict(
                save_checkpoint_path=save_checkpoint_path,
                ckpt_manager=args.train.ckpt_manager,
            )
            save_model_weights(hf_weights_path, model_state_dict, model_assets=self.trainer.model_assets)
            logger.info_rank0(f"Huggingface checkpoint saved at {hf_weights_path} successfully!")

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

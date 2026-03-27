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

import os
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from ...checkpoint import CheckpointerBase, build_checkpointer
from ...models import save_model_assets
from ...utils import helper
from ...utils.save_safetensor_utils import save_hf_safetensor
from .base import Callback, TrainerState


if TYPE_CHECKING:
    from ..base import BaseTrainer, VeOmniArguments


logger = helper.create_logger(__name__)


class CheckpointerCallback(Callback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.every_n_steps = args.train.checkpoint.save_steps
        self.every_n_epochs = args.train.checkpoint.save_epochs
        self._last_saved_step: int = -1
        self.trainer.checkpointer: CheckpointerBase = build_checkpointer(
            dist_backend=args.train.accelerator.fsdp_config.fsdp_mode, ckpt_manager=args.train.checkpoint.manager
        )

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

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint from path."""
        args: "VeOmniArguments" = self.trainer.args
        if args.train.checkpoint.load_path is None:
            return

        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        self.trainer.checkpointer.load(args.train.checkpoint.load_path, state)

        self.trainer.state.global_step = state["extra_state"]["global_step"]
        self.trainer.start_epoch = self.trainer.state.global_step // args.train_steps
        self.trainer.start_step = self.trainer.state.global_step % args.train_steps

        self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])

        # dataloader may only init on sp_rank_0 to save memory
        if (
            self.trainer.train_dataloader is not None
            and state["extra_state"].get("train_dataloader", None) is not None
        ):
            self.trainer.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "VeOmniArguments" = self.trainer.args

        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")

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
        self.trainer.checkpointer.save(save_checkpoint_path, ckpt_state, save_async=args.train.checkpoint.save_async)

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")


class HuggingfaceCkptCallback(CheckpointerCallback):
    def __init__(self, trainer: "BaseTrainer"):
        super().__init__(trainer)
        args: "VeOmniArguments" = self.trainer.args
        self.save_hf_weights = args.train.checkpoint.save_hf_weights
        self.every_n_steps = args.train.checkpoint.hf_save_steps
        self.every_n_epochs = args.train.checkpoint.hf_save_epochs

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

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        self._save_model_assets()

    def _save_model_assets(self):
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0:
            save_model_assets(args.train.checkpoint.model_assets_dir, self.trainer.model_assets)
        dist.barrier()

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        """Save model in HuggingFace format."""
        args: "VeOmniArguments" = self.trainer.args
        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            super()._save_checkpoint(state)

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        if stage == "train_end":
            del self.trainer.optimizer
            del self.trainer.lr_scheduler

        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            model_assets=self.trainer.model_assets,
            ckpt_manager=args.train.checkpoint.manager,
            train_architecture=args.train.train_architecture,
            output_dir=args.train.checkpoint.output_dir,
            save_checkpoint_path=save_checkpoint_path,
            model=self.trainer.model,
            fqn_to_index_mapping=args.model.fqn_to_index_mapping,
            is_rank_0=args.train.global_rank == 0,
        )

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step


class HFLoraCkptCallback(HuggingfaceCkptCallback):
    """Save LoRA HF weights once at train end."""

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        """Save LoRA checkpoint in HuggingFace format at train end."""
        args: "VeOmniArguments" = self.trainer.args
        save_checkpoint_path = os.path.join(args.train.checkpoint.output_dir, f"global_step_{state.global_step}")
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            CheckpointerCallback._save_checkpoint(self, state)

        if getattr(self.trainer.checkpointer, "save_future", None) is not None:  # async save
            self.trainer.checkpointer.save_future.result()

        from peft import get_peft_model_state_dict

        from veomni.checkpoint import ckpt_to_state_dict
        from veomni.models.module_utils import _save_state_dict

        model_state_dict = ckpt_to_state_dict(
            save_checkpoint_path=save_checkpoint_path,
            output_dir=args.train.checkpoint.output_dir,
            ckpt_manager=args.train.checkpoint.manager,
        )

        model_state_dict = get_peft_model_state_dict(self.trainer.model, model_state_dict)
        # save bf16 lora
        model_state_dict = {k: v.to(torch.bfloat16) for k, v in model_state_dict.items()}
        lora_adapter_save_path = os.path.join(save_checkpoint_path, "adapter_model.bin")
        os.makedirs(save_checkpoint_path, exist_ok=True)
        _save_state_dict(model_state_dict, lora_adapter_save_path, safe_serialization=False)
        self.trainer.model.peft_config["default"].save_pretrained(save_checkpoint_path)
        logger.info_rank0(f"Lora adapter saved at {save_checkpoint_path} successfully!")

        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step

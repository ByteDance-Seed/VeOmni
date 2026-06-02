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
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

from ...checkpoint import CheckpointerBase, build_checkpointer
from ...models import save_model_assets
from ...utils import helper
from ...utils.save_safetensor_utils import save_hf_safetensor, save_lora_adapter_with_dcp
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

    # ── Overridable seams ──────────────────────────────────────────────────────
    # Subclasses (e.g. the per-OmniModule callbacks) override these to retarget
    # the on-disk location and slim the non-model payload, while reusing the DCP
    # save / load / HF / LoRA machinery below.  Defaults reproduce the original
    # single-model behaviour verbatim, so every existing trainer is unaffected.

    def _save_dir(self, state: TrainerState) -> str:
        """Directory the DCP (and HF weights) are written to for this step."""
        return os.path.join(self.trainer.args.train.checkpoint.save_path, f"global_step_{state.global_step}")

    def _load_dir(self) -> Optional[str]:
        """Directory to resume the DCP from (``None`` → no resume)."""
        return self.trainer.args.train.checkpoint.load_path

    def _output_dir(self, state: TrainerState) -> str:
        """Output directory for HF / LoRA exports for this step."""
        return os.path.join(self.trainer.args.train.checkpoint.output_dir, f"global_step_{state.global_step}")

    def _extra_state(self, state: TrainerState) -> Dict[str, Any]:
        """Non-``model``/``optimizer`` payload to persist alongside the DCP.

        The default bundles per-model (``lr_scheduler``) **and** global (step /
        dataloader / environ-meter / rng) state.  Per-module subclasses override
        this to keep only the per-model bits; the global bits are saved once by
        the orchestrator.
        """
        return {
            "global_step": state.global_step,
            "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
            "train_dataloader": self.trainer.train_dataloader.state_dict()
            if self.trainer.train_dataloader is not None
            else {},
            "environ_meter": self.trainer.environ_meter.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
        }

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        """Restore the payload produced by :meth:`_extra_state` (inverse op)."""
        args: "VeOmniArguments" = self.trainer.args
        self.trainer.state.global_step = extra_state["global_step"]
        self.trainer.start_epoch = self.trainer.state.global_step // args.train_steps
        self.trainer.start_step = self.trainer.state.global_step % args.train_steps
        self.trainer.lr_scheduler.load_state_dict(extra_state["lr_scheduler"])
        # dataloader may only init on sp_rank_0 to save memory
        if self.trainer.train_dataloader is not None and extra_state.get("train_dataloader", None) is not None:
            self.trainer.train_dataloader.load_state_dict(extra_state["train_dataloader"])
        self.trainer.environ_meter.load_state_dict(extra_state["environ_meter"])
        torch.set_rng_state(extra_state["torch_rng_state"])
        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

    def _load_checkpoint(self):
        """Load checkpoint from path."""
        args: "VeOmniArguments" = self.trainer.args
        load_dir = self._load_dir()
        if load_dir is None:
            return

        state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }

        self.trainer.checkpointer.wait_for_pending_save()

        self.trainer.checkpointer.load(
            load_dir,
            state,
            trainable_only=bool(getattr(args.model, "lora_config", None)),
        )

        self._load_extra_state(state["extra_state"])

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "VeOmniArguments" = self.trainer.args

        save_checkpoint_path = self._save_dir(state)

        ckpt_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": self._extra_state(state),
        }

        # Free the training step's residual activations / autograd buffers
        # before DCP allocates NCCL collective buffers for the gather.
        # Mirrors the existing post-save ``empty_cache()`` below; without
        # this pre-save call the save can fight the training step for HBM
        # (observed as ``NCCL WARN Cuda failure 2 'out of memory'`` inside
        # dcp.save on Qwen3.5-35B-a3b VL h100x16). Cost: one ``cudaFree``
        # per ``save_steps``, well below noise.
        helper.empty_cache()

        self.trainer.checkpointer.save(
            save_checkpoint_path,
            ckpt_state,
            save_async=args.train.checkpoint.save_async,
            trainable_only=bool(getattr(args.model, "lora_config", None)),
        )

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

    def _model_assets_dir(self) -> str:
        """Directory the model assets (config / processor / tokenizer) are written to."""
        return self.trainer.args.train.checkpoint.model_assets_dir

    def _save_model_assets(self):
        args: "VeOmniArguments" = self.trainer.args
        if args.train.global_rank == 0:
            save_model_assets(self._model_assets_dir(), self.trainer.model_assets)
        dist.barrier()

    def _save_checkpoint(self, state: TrainerState, stage: str = "step_end"):
        """Save model in HuggingFace format."""
        args: "VeOmniArguments" = self.trainer.args
        save_checkpoint_path = self._save_dir(state)
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            super()._save_checkpoint(state)

        self.trainer.checkpointer.wait_for_pending_save()

        if stage == "train_end":
            self.trainer.optimizer = None
            self.trainer.lr_scheduler = None

        hf_weights_path = os.path.join(save_checkpoint_path, "hf_ckpt")
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            model_assets=self.trainer.model_assets,
            ckpt_manager=args.train.checkpoint.manager,
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
        save_checkpoint_path = self._save_dir(state)
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            CheckpointerCallback._save_checkpoint(self, state)

        self.trainer.checkpointer.wait_for_pending_save()

        if stage == "train_end":
            self.trainer.optimizer = None
            self.trainer.lr_scheduler = None

        lora_save_path = self._output_dir(state)
        save_lora_adapter_with_dcp(
            model=self.trainer.model,
            save_path=lora_save_path,
            adapter_name="default",
        )

        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step

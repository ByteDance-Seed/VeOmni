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
from ...checkpoint.checkpoint_manifest import (
    build_checkpoint_manifest,
    read_checkpoint_manifest,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from ...checkpoint.rng_state import restore_rng_state, snapshot_rng_state
from ...models import save_model_assets
from ...utils import helper
from ...utils.device import get_torch_device
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

    def _manifest_identity(self):
        """Return ``(extra_hashes, soft_fields)`` for the checkpoint manifest.

        Both hard-gated identities (component lifecycle policy, flow recipe) are
        model-config fields, so a model that declares neither keeps a generic manifest.
        """
        args: "VeOmniArguments" = self.trainer.args
        model_config = getattr(self.trainer, "model_config", None)
        extra = {}
        soft = {}

        component_policy = getattr(model_config, "component_policy", None)
        if component_policy:
            extra["component_policy"] = dict(component_policy)
        flow = getattr(model_config, "flow", None)
        if flow:
            extra["flow"] = dict(flow)

        seed = getattr(args.train, "seed", None)
        if seed is not None:
            soft["train_seed"] = seed
        return extra, soft

    def _assert_manifest_compatible(self, load_path: str) -> None:
        """Validate the saved manifest and agree across ranks BEFORE the DCP load.

        The all-reduce makes every rank raise identically on a hard mismatch, rather
        than one rank bailing out while the rest deadlock in the load collective. A
        missing manifest (pre-feature checkpoint) only warns.
        """
        saved = read_checkpoint_manifest(load_path)
        if saved is None:
            logger.warning_rank0(
                f"No checkpoint_manifest.json in {load_path}; skipping same-topology validation "
                "(checkpoint predates the manifest). Ensure the resume topology matches."
            )
            return

        extra, soft = self._manifest_identity()
        current = build_checkpoint_manifest(
            model_config=self.trainer.model_config,
            parallel_state=self.parallel_state,
            extra_hashes=extra,
            soft_fields=soft,
        )
        hard_reasons, soft_reasons = validate_checkpoint_manifest(saved, current)
        for reason in soft_reasons:
            logger.warning_rank0(f"Checkpoint manifest soft mismatch on resume: {reason}")

        local_ok = len(hard_reasons) == 0
        agreed_ok = local_ok
        if dist.is_available() and dist.is_initialized():
            flag = torch.tensor([1 if local_ok else 0], dtype=torch.int64, device=get_torch_device().current_device())
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            agreed_ok = bool(flag.item() > 0)
        if not agreed_ok:
            detail = "; ".join(hard_reasons) if hard_reasons else "another rank reported an incompatible manifest"
            raise ValueError(
                f"Incompatible checkpoint resume rejected by the same-topology manifest gate: {detail}. "
                "Resume under the exact saved mesh/model/policy, or start a fresh run."
            )

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

        self.trainer.checkpointer.wait_for_pending_save()

        # Must run before the DCP load collective: a topology/identity mismatch would
        # otherwise deadlock or reshard incorrectly mid-collective.
        self._assert_manifest_compatible(args.train.checkpoint.load_path)

        self.trainer.checkpointer.load(
            args.train.checkpoint.load_path,
            state,
            trainable_only=bool(getattr(args.model, "lora_config", None)),
            parallel_state=self.parallel_state,
        )

        self.trainer.state.global_step = state["extra_state"]["global_step"]
        self.trainer.start_epoch = self.trainer.state.global_step // args.train_steps
        self.trainer.start_step = self.trainer.state.global_step % args.train_steps

        self.trainer.lr_scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])

        channel_loss_state = state["extra_state"].get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        # dataloader may only init on sp_rank_0 to save memory
        if (
            self.trainer.train_dataloader is not None
            and state["extra_state"].get("train_dataloader", None) is not None
        ):
            self.trainer.train_dataloader.load_state_dict(state["extra_state"]["train_dataloader"])

        self.trainer.environ_meter.load_state_dict(state["extra_state"]["environ_meter"])
        # ``torch_rng_state`` is the legacy torch-CPU-only key, kept for checkpoints
        # written before the full snapshot.
        rng_state = state["extra_state"].get("rng_state")
        if rng_state is not None:
            restore_rng_state(rng_state)
        elif state["extra_state"].get("torch_rng_state") is not None:
            torch.set_rng_state(state["extra_state"]["torch_rng_state"])
        if self.trainer.start_step == 0:
            # If resume at the end of epoch, clear resume state and prefetch data
            iter(self.trainer.train_dataloader)

        # Free transient buffers from DCP materialization before the first train step.
        # Large MoE resumes are often near GPU capacity; leftover allocator fragments
        # after load can OOM the first NCCL collective (e.g. grad-norm all-reduce).
        helper.empty_cache()

        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {args.train.checkpoint.load_path} successfully!")

    def _save_checkpoint(self, state: TrainerState):
        """Save distributed checkpoint and optimizer state at each save_steps."""
        args: "VeOmniArguments" = self.trainer.args

        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")

        if hasattr(self.trainer, "data_iterator") and hasattr(self.trainer.data_iterator, "state_dict"):
            train_dataloader_state = self.trainer.data_iterator.state_dict()
        elif self.trainer.train_dataloader is not None:
            train_dataloader_state = self.trainer.train_dataloader.state_dict()
        else:
            train_dataloader_state = {}

        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        channel_loss_state = channel_loss_callback.state_dict() if channel_loss_callback is not None else {}

        ckpt_state = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {
                "global_step": state.global_step,
                "lr_scheduler": self.trainer.lr_scheduler.state_dict(),
                "train_dataloader": train_dataloader_state,
                "environ_meter": self.trainer.environ_meter.state_dict(),
                "channel_loss_callback": channel_loss_state,
                "rng_state": snapshot_rng_state(),
            },
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
            save_to_lowest_rank=args.train.checkpoint.dcp_save_to_lowest_rank,
            parallel_state=self.parallel_state,
        )

        # Empty cache and barrier
        helper.empty_cache()
        dist.barrier()

        if not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0:
            extra, soft = self._manifest_identity()
            manifest = build_checkpoint_manifest(
                model_config=self.trainer.model_config,
                parallel_state=self.parallel_state,
                extra_hashes=extra,
                soft_fields=soft,
            )
            write_checkpoint_manifest(save_checkpoint_path, manifest)

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
            parallel_state=self.parallel_state,
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
        save_checkpoint_path = os.path.join(args.train.checkpoint.save_path, f"global_step_{state.global_step}")
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            CheckpointerCallback._save_checkpoint(self, state)

        self.trainer.checkpointer.wait_for_pending_save()

        if stage == "train_end":
            self.trainer.optimizer = None
            self.trainer.lr_scheduler = None

        lora_save_path = os.path.join(args.train.checkpoint.output_dir, f"global_step_{state.global_step}")
        save_lora_adapter_with_dcp(
            model=self.trainer.model,
            save_path=lora_save_path,
            adapter_name="default",
        )

        helper.empty_cache()
        dist.barrier()

        self._last_saved_step = state.global_step

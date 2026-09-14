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

"""Checkpoint/resume for one trainer-owned model."""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch
import torch.distributed as dist

from ...checkpoint import CheckpointerBase, build_checkpointer
from ...distributed.parallel_state import get_parallel_state
from ...utils import helper


if TYPE_CHECKING:
    from ...arguments import CheckpointConfig
    from ...trainer.base import BaseTrainer
    from ...trainer.callbacks import TrainerState


logger = helper.create_logger(__name__)


class ModelCheckpointManager:
    """Own DCP / HF / LoRA save-load for the trainer's model.

    The trainer supplies the module, optimizer, scheduler and assets; this class
    owns the *ordering* around them — when to drain an in-flight async save,
    where the ``empty_cache`` and ``barrier`` calls go, and which directory each
    artifact lands in.

    That ordering is load-bearing rather than incidental. The two ``empty_cache``
    calls bracketing a DCP save keep the save from competing with the training
    step for HBM: without the pre-save one, DCP's NCCL gather buffers can fail to
    allocate (seen as ``NCCL WARN Cuda failure 2 'out of memory'`` inside
    ``dcp.save`` on a Qwen3.5-35B-a3b VL h100x16 run).

    On-disk layout for a single-model job::

        <save_path>/global_step_{N}/
        ├── __0_0.distcp …     # DCP shards {model, optimizer, extra_state}
        └── hf_ckpt/           # HF safetensors export

    A subclass managing one module of a multi-module model sets
    :attr:`checkpoint_subfolder` so every artifact nests one level deeper.
    """

    checkpoint_subfolder: str = ""

    def __init__(self, trainer: "BaseTrainer"):
        self.trainer = trainer
        self.config: "CheckpointConfig" = trainer.args.train.checkpoint
        self._last_saved_step: int = -1
        # Cached at construction, same as Callback.parallel_state: later save/load
        # must not depend on whichever mesh is ambient.
        self.parallel_state = get_parallel_state()
        self.checkpointer: CheckpointerBase = build_checkpointer(
            ckpt_manager=self.config.manager,
            dist_backend=trainer.args.model.accelerator.fsdp_config.fsdp_mode,
        )

    @property
    def last_saved_step(self) -> int:
        return self._last_saved_step

    @property
    def trainable_only(self) -> bool:
        return bool(self.trainer.args.model.lora_config)

    def _step_dir(self, root: str, state: "TrainerState") -> str:
        step_dir = os.path.join(root, f"global_step_{state.global_step}")
        return os.path.join(step_dir, self.checkpoint_subfolder) if self.checkpoint_subfolder else step_dir

    def save_dir(self, state: "TrainerState") -> str:
        """Where this step's DCP shards live."""
        return self._step_dir(self.config.save_path, state)

    def output_dir(self, state: "TrainerState") -> str:
        """Where user-facing exports (LoRA adapters) live."""
        return self._step_dir(self.config.output_dir, state)

    def hf_export_dir(self, state: "TrainerState") -> str:
        """Where this step's safetensors export lives."""
        return os.path.join(self.save_dir(state), "hf_ckpt")

    def load_dir(self) -> Optional[str]:
        load_path = self.config.load_path
        if load_path is None:
            return None
        return os.path.join(load_path, self.checkpoint_subfolder) if self.checkpoint_subfolder else load_path

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        """Model-bound state to store beside the weights."""
        lr_scheduler = self.trainer.lr_scheduler
        return {"lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict()}

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        lr_state = extra_state.get("lr_scheduler")
        lr_scheduler = self.trainer.lr_scheduler
        if lr_state is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_state)

        # Pre-split DCP extra_state also held the job cursor. New writes do not;
        # GlobalStateCallback owns that file. Restore the old blob so a mid-job
        # resume from a CheckpointerCallback checkpoint does not silently restart
        # at step 0 with restored weights.
        if "global_step" not in extra_state:
            return
        logger.warning_rank0(
            "DCP extra_state still contains job-level keys (global_step, dataloader, "
            "rng). Restoring them for compatibility with checkpoints written before "
            "GlobalStateCallback; new saves keep only lr_scheduler here."
        )
        self._restore_legacy_job_state(extra_state)

    def _restore_legacy_job_state(self, extra_state: Dict[str, Any]) -> None:
        args = self.trainer.args
        global_step = extra_state["global_step"]
        self.trainer.state.global_step = global_step
        self.trainer.start_epoch = global_step // args.train_steps
        self.trainer.start_step = global_step % args.train_steps

        channel_loss_state = extra_state.get("channel_loss_callback")
        channel_loss_callback = getattr(self.trainer, "channel_loss_callback", None)
        if channel_loss_state is not None and channel_loss_callback is not None:
            channel_loss_callback.load_state_dict(channel_loss_state)

        if self.trainer.train_dataloader is not None and extra_state.get("train_dataloader") is not None:
            self.trainer.train_dataloader.load_state_dict(extra_state["train_dataloader"])

        environ_meter = getattr(self.trainer, "environ_meter", None)
        if environ_meter is not None and extra_state.get("environ_meter") is not None:
            environ_meter.load_state_dict(extra_state["environ_meter"])

        rng_state = extra_state.get("torch_rng_state")
        if rng_state is not None:
            torch.set_rng_state(rng_state)
        if self.trainer.start_step == 0 and self.trainer.train_dataloader is not None:
            iter(self.trainer.train_dataloader)

    def wait_for_pending_save(self) -> None:
        self.checkpointer.wait_for_pending_save()

    def load(self) -> None:
        load_dir = self.load_dir()
        if load_dir is None:
            return

        self.wait_for_pending_save()
        state: Dict[str, Any] = {
            "model": self.trainer.model,
            "optimizer": self.trainer.optimizer,
            "extra_state": {},
        }
        self.checkpointer.load(
            load_dir,
            state,
            trainable_only=self.trainable_only,
            parallel_state=self.parallel_state,
        )
        self._load_extra_state(state["extra_state"])
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: "TrainerState") -> None:
        """Write model, optimizer and this model's extra state for ``state.global_step``.

        Only model-bound state goes in here. Job-level state — where the
        dataloader is, the rng — has its own writer.
        """
        extra_state = self._extra_state(state)

        helper.empty_cache()
        self.checkpointer.save(
            self.config.save_path,
            {"model": self.trainer.model, "optimizer": self.trainer.optimizer, "extra_state": extra_state},
            global_steps=state.global_step,
            save_async=self.config.save_async,
            trainable_only=self.trainable_only,
            save_to_lowest_rank=self.config.dcp_save_to_lowest_rank,
            parallel_state=self.parallel_state,
            stage_dir=self.config.stage_dir,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {self.save_dir(state)} successfully!")

    def _prepare_export(self, state: "TrainerState", stage: str) -> str:
        save_path = self.save_dir(state)
        if not os.path.exists(save_path):
            dist.barrier()
            self.save_dcp(state)

        self.wait_for_pending_save()

        if stage == "train_end":
            self.trainer.optimizer = None
            self.trainer.lr_scheduler = None

        return save_path

    def save_hf(self, state: "TrainerState", stage: str = "step_end") -> None:
        from ...utils.save_safetensor_utils import save_hf_safetensor

        save_path = self._prepare_export(state, stage)

        save_hf_safetensor(
            save_hf_safetensor_path=self.hf_export_dir(state),
            model_assets=self.trainer.model_assets,
            ckpt_manager=self.config.manager,
            output_dir=self.config.output_dir,
            save_checkpoint_path=save_path,
            model=self.trainer.model,
            fqn_to_index_mapping=self.trainer.args.model.fqn_to_index_mapping,
            is_rank_0=self.trainer.args.train.global_rank == 0,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def save_lora(self, state: "TrainerState", stage: str = "step_end", adapter_name: str = "default") -> None:
        from ...utils.save_safetensor_utils import save_lora_adapter_with_dcp

        self._prepare_export(state, stage)
        save_lora_adapter_with_dcp(
            model=self.trainer.model,
            save_path=self.output_dir(state),
            adapter_name=adapter_name,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        if self.trainable_only:
            self.save_lora(state, stage=stage)
        else:
            self.save_hf(state, stage=stage)


__all__ = ["ModelCheckpointManager"]

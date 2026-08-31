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

"""Checkpoint/resume for one :class:`~veomni.models.model_runtime.VeOmniModelRuntime`."""

import os
from typing import TYPE_CHECKING, Any, Dict, Optional

import torch.distributed as dist

from ..checkpoint import CheckpointerBase, build_checkpointer
from ..utils import helper


if TYPE_CHECKING:
    from ..arguments import CheckpointConfig
    from ..trainer.callbacks import TrainerState
    from .model_runtime import VeOmniModelRuntime


logger = helper.create_logger(__name__)


class ModelCheckpointManager:
    """Own DCP / HF / LoRA save-load for one model runtime.

    The runtime supplies the model, the optimizer and the ParallelState to write
    them under; this class owns the *ordering* around them — when to drain an
    in-flight async save, where the ``empty_cache`` and ``barrier`` calls go, and
    which directory each artifact lands in.

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

    #: Extra directory level for every artifact. Empty for a single-model job;
    #: a multi-module model sets it to the module name.
    checkpoint_subfolder: str = ""

    def __init__(self, runtime: "VeOmniModelRuntime", config: "CheckpointConfig"):
        """
        Args:
            runtime: The model whose weights and optimizer this manages.
            config: ``train.checkpoint`` — paths and save flags.
        """
        self.runtime = runtime
        self.config = config
        self._last_saved_step: int = -1
        self.checkpointer: CheckpointerBase = build_checkpointer(
            ckpt_manager=config.manager,
            dist_backend=runtime.args.accelerator.fsdp_config.fsdp_mode,
        )

    @property
    def last_saved_step(self) -> int:
        """Step of the most recent write by this manager, or ``-1`` if none."""
        return self._last_saved_step

    @property
    def trainable_only(self) -> bool:
        """LoRA runs checkpoint only the adapters; a full run checkpoints everything."""
        return bool(self.runtime.args.lora_config)

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
        """Where to resume from, or ``None`` when starting fresh."""
        load_path = self.config.load_path
        if load_path is None:
            return None
        return os.path.join(load_path, self.checkpoint_subfolder) if self.checkpoint_subfolder else load_path

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        """Model-bound state to store beside the weights."""
        lr_scheduler = self.runtime.lr_scheduler
        return {"lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict()}

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        """Restore the model-bound half of ``extra_state``; the caller takes the rest."""
        lr_state = extra_state.get("lr_scheduler")
        lr_scheduler = self.runtime.lr_scheduler
        if lr_state is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_state)

    def wait_for_pending_save(self) -> None:
        """Block until any in-flight async save completes."""
        self.checkpointer.wait_for_pending_save()

    def load(self) -> None:
        """Restore model, optimizer and this model's own extra state from ``load_path``."""
        load_dir = self.load_dir()
        if load_dir is None:
            return

        self.wait_for_pending_save()
        state: Dict[str, Any] = {
            "model": self.runtime.model,
            "optimizer": self.runtime.optimizer,
            "extra_state": {},
        }
        self.checkpointer.load(
            load_dir,
            state,
            trainable_only=self.trainable_only,
            parallel_state=self.runtime.parallel_state,
        )
        self._load_extra_state(state["extra_state"])
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: "TrainerState") -> None:
        """Write model, optimizer and this model's extra state for ``state.global_step``.

        Only model-bound state goes in here. Job-level state — where the
        dataloader is, the rng — has its own writer, because with several models
        in one job there is one such record but N of these checkpoints.
        """
        save_path = self.save_dir(state)
        extra_state = self._extra_state(state)

        helper.empty_cache()
        self.checkpointer.save(
            save_path,
            {"model": self.runtime.model, "optimizer": self.runtime.optimizer, "extra_state": extra_state},
            save_async=self.config.save_async,
            trainable_only=self.trainable_only,
            save_to_lowest_rank=self.config.dcp_save_to_lowest_rank,
            parallel_state=self.runtime.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {save_path} successfully!")

    def _prepare_export(self, state: "TrainerState", stage: str) -> str:
        """Guarantee a DCP checkpoint exists for this step, then quiesce for export.

        Both export formats read the DCP shards back, so a step that has not been
        saved yet is saved now. At ``train_end`` nothing will step the optimizer
        or scheduler again, so they are dropped to leave the export more HBM.
        """
        save_path = self.save_dir(state)
        if not os.path.exists(save_path):
            dist.barrier()
            self.save_dcp(state)

        self.wait_for_pending_save()

        if stage == "train_end":
            self.runtime.optimizer = None
            self.runtime.lr_scheduler = None

        return save_path

    def save_hf(self, state: "TrainerState", stage: str = "step_end") -> None:
        """Export the weights in HuggingFace safetensors layout."""
        from ..utils.save_safetensor_utils import save_hf_safetensor

        save_path = self._prepare_export(state, stage)

        save_hf_safetensor(
            save_hf_safetensor_path=self.hf_export_dir(state),
            model_assets=self.runtime.model_assets,
            ckpt_manager=self.config.manager,
            output_dir=self.config.output_dir,
            save_checkpoint_path=save_path,
            model=self.runtime.model,
            fqn_to_index_mapping=self.runtime.args.fqn_to_index_mapping,
            is_rank_0=self.runtime.parallel_state.global_rank == 0,
            parallel_state=self.runtime.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def save_lora(self, state: "TrainerState", stage: str = "step_end", adapter_name: str = "default") -> None:
        """Export the LoRA adapter in PEFT layout."""
        from ..utils.save_safetensor_utils import save_lora_adapter_with_dcp

        self._prepare_export(state, stage)

        save_lora_adapter_with_dcp(
            model=self.runtime.model,
            save_path=self.output_dir(state),
            adapter_name=adapter_name,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        """Export whichever format this model was trained in."""
        if self.trainable_only:
            self.save_lora(state, stage=stage)
        else:
            self.save_hf(state, stage=stage)


__all__ = ["ModelCheckpointManager"]

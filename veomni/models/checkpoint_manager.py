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

"""Checkpoint/resume for one :class:`~veomni.models.model_runtime.VeOmniModelRuntime`.

Two blobs, two owners:

* **lr_scheduler** — this model's scheduler. Passed to DCP like the optimizer;
  the checkpointer pickles ``state_dict`` as a single ``lr_scheduler.pt``.
* **global_state** — the job cursor (step, dataloader, rng, meters), written by
  :class:`~veomni.trainer.callbacks.global_state_callback.GlobalStateCallback`
  as ``trainer_state_rank_*.pt``.

On-disk layout: ``docs/usage/checkpoint.md``.
"""

import os
from typing import TYPE_CHECKING, Optional

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
        ├── __0_0.distcp …     # DCP shards {model, optimizer}
        ├── lr_scheduler.pt    # replicated scheduler
        ├── trainer_state_rank_{R}.pt  # job cursor (written by GlobalStateCallback)
        ├── adapter_config.json / adapter_model.safetensors  # LoRA export
        └── hf_ckpt/           # full-model HF safetensors export

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
        # This runtime's mesh, looked up by name at construction — not the
        # ambient get_parallel_state(). build_checkpoint() runs outside the
        # runtime's use_parallel_state scope, so ambient is still "base" while
        # a DPO policy or an Omni module is registered under its own name.
        self.parallel_state = runtime.parallel_state
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
        """Where this step's DCP shards (and LoRA adapter export) live."""
        return self._step_dir(self.config.save_path, state)

    def hf_export_dir(self, state: "TrainerState") -> str:
        """Where this step's safetensors export lives."""
        return os.path.join(self.save_dir(state), "hf_ckpt")

    def load_dir(self) -> Optional[str]:
        """Where to resume from, or ``None`` when starting fresh."""
        load_path = self.config.load_path
        if load_path is None:
            return None
        return os.path.join(load_path, self.checkpoint_subfolder) if self.checkpoint_subfolder else load_path

    def wait_for_pending_save(self) -> None:
        """Block until any in-flight async save completes."""
        self.checkpointer.wait_for_pending_save()

    def load(self) -> None:
        """Restore model, optimizer and lr_scheduler from ``load_path``."""
        load_dir = self.load_dir()
        if load_dir is None:
            return

        self.wait_for_pending_save()
        self.checkpointer.load(
            load_dir,
            {
                "model": self.runtime.model,
                "optimizer": self.runtime.optimizer,
                "lr_scheduler": self.runtime.lr_scheduler,
            },
            trainable_only=self.trainable_only,
            parallel_state=self.parallel_state,
        )
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: "TrainerState") -> None:
        """Write model, optimizer and lr_scheduler for ``state.global_step``.

        global_state — where the dataloader is, the rng — is a separate file,
        because with several models in one job there is one such record but N
        of these checkpoints.

        Staging keys off the run root plus ``global_steps``, not a path that
        already contains the step: folding the step in gives each save a fresh
        staging directory, and a kill mid-write strands a copy no later save
        clears.
        """
        helper.empty_cache()
        self.checkpointer.save(
            self.config.save_path,
            {
                "model": self.runtime.model,
                "optimizer": self.runtime.optimizer,
                "lr_scheduler": self.runtime.lr_scheduler,
            },
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
            is_rank_0=self.parallel_state.global_rank == 0,
            parallel_state=self.parallel_state,
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
            save_path=self.save_dir(state),
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

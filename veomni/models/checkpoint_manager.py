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

On-disk layout is owned by :mod:`veomni.checkpoint.layout` (see
``docs/usage/checkpoint.md``). This class only binds that layout to one model's
runtime — the module, optimizer, scheduler, assets, and the ParallelState they
were built under.

Two blobs, two owners:

* **lr_scheduler** — this model's scheduler. Passed to DCP like the optimizer;
  the checkpointer pickles ``state_dict`` as a single ``lr_scheduler.pt`` beside
  the two DCP directories.
* **global_state** — the job cursor (dataloader in ``loader/``; step, rng and
  meters in ``extra_state/``), written per rank by
  :class:`~veomni.trainer.callbacks.global_state_callback.GlobalStateCallback`.
"""

from typing import TYPE_CHECKING, Any, Dict, Optional

import torch.distributed as dist

from ..checkpoint import CheckpointerBase, build_checkpointer, layout
from ..utils import helper


if TYPE_CHECKING:
    from ..arguments import CheckpointConfig
    from ..trainer.callbacks import TrainerState
    from .model_runtime import VeOmniModelRuntime


logger = helper.create_logger(__name__)


class ModelCheckpointManager:
    """Own DCP / HF / LoRA save-load for one model runtime.

    The runtime supplies the module, optimizer, scheduler, assets and mesh; this
    class owns the *ordering* around them — when to drain an in-flight async
    save, where the ``empty_cache`` and ``barrier`` calls go, and which
    directory each artifact lands in.

    That ordering is load-bearing rather than incidental. The two ``empty_cache``
    calls bracketing a DCP save keep the save from competing with the training
    step for HBM: without the pre-save one, DCP's NCCL gather buffers can fail to
    allocate (seen as ``NCCL WARN Cuda failure 2 'out of memory'`` inside
    ``dcp.save`` on a Qwen3.5-35B-a3b VL h100x16 run).

    On-disk layout for a single-model job::

        <save_path>/global_step_{N}/
        ├── model/
        │   ├── ckpt/          # DCP shards: weights
        │   ├── optimizer/     # DCP shards: optimizer state
        │   └── lr_scheduler.pt
        ├── loader/            # dataloader cursor, per rank
        ├── extra_state/       # step, rng, meters, per rank
        ├── hf_ckpt/           # full-model HF safetensors export
        └── lora_ckpt/         # PEFT adapter export

    ``loader/`` and ``extra_state/`` are written by
    :class:`~veomni.trainer.callbacks.global_state_callback.GlobalStateCallback`,
    which also writes the step's ``checkpoint_manifest.json``.

    A subclass managing one module of a multi-module model sets
    :attr:`module_name`; every path below then nests one level deeper, and
    nothing else changes. Paths are never built here — they all come from
    :mod:`veomni.checkpoint.layout`, so a save and the load that follows it
    cannot drift apart.
    """

    module_name: str = ""

    def __init__(self, runtime: "VeOmniModelRuntime"):
        self.runtime = runtime
        self.config: "CheckpointConfig" = runtime.train_args.checkpoint
        self._last_saved_step: int = -1
        self.checkpointer: CheckpointerBase = build_checkpointer(
            ckpt_manager=self.config.manager,
            dist_backend=runtime.args.accelerator.fsdp_config.fsdp_mode,
        )

    @property
    def parallel_state(self):
        """This model's mesh, via the runtime's by-name registry lookup.

        Not the ambient ``get_parallel_state()``: ``build_checkpoint()`` runs
        outside the runtime's ``use_parallel_state`` scope, so ambient is still
        ``"base"`` while a DPO policy or an Omni module is registered under its
        own name. A property rather than a cached object, so a re-registered
        mesh is picked up the same way ``VeOmniModelRuntime.parallel_state`` is.
        """
        return self.runtime.parallel_state

    @property
    def last_saved_step(self) -> int:
        """Last step this run handed to the checkpointer, for same-step dedupe.

        Says nothing about what is on disk -- under ``save_async`` the shards are
        still being written when ``save_dcp`` returns. It answers the one
        question the filesystem cannot: whether *this* run wrote the step. A
        complete checkpoint left at the same step by an earlier run looks
        identical from the outside but holds different weights, so
        :meth:`_prepare_export` overwrites rather than trusting what it finds.
        """
        return self._last_saved_step

    @property
    def trainable_only(self) -> bool:
        return bool(self.runtime.args.lora_config)

    def step_dir(self, state: "TrainerState") -> str:
        """Root of this step's checkpoint, shared by every module of the job."""
        return layout.step_dir(self.config.save_path, state.global_step)

    def save_dir(self, state: "TrainerState") -> str:
        """Where this step's model state lives: weights, optimizer, scheduler."""
        return layout.model_dir(self.step_dir(state), self.module_name)

    def hf_export_dir(self, state: "TrainerState") -> str:
        """Where this step's full-model safetensors export lives."""
        return layout.hf_export_dir(self.step_dir(state), self.module_name)

    def lora_export_dir(self, state: "TrainerState") -> str:
        """Where this step's PEFT adapter export lives."""
        return layout.lora_export_dir(self.step_dir(state), self.module_name)

    def assets_dir(self) -> str:
        """Where this model's config/tokenizer/processor sidecars live.

        Once per run, at the output root — not inside a step. Nested under
        :attr:`module_name` so two modules cannot overwrite each other's
        ``config.json``.
        """
        return layout.assets_dir(self.config.model_assets_dir, self.module_name)

    def load_dir(self) -> Optional[str]:
        """Step directory to resume from.

        The module is not folded in here: the checkpointer takes it separately
        and resolves ``model/<module>/`` itself, so one path serves the whole job.
        """
        return self.config.load_path

    def wait_for_pending_save(self) -> None:
        """Block until the in-flight async save is on disk, if there is one."""
        self.checkpointer.wait_for_pending_save()

    def _extra_state(self, state: "TrainerState") -> Dict[str, Any]:
        """Model-bound state to store beside the weights.

        The lr scheduler plus whatever the runtime contributes via
        ``extra_state()`` (e.g. the DiT condition model's generator).
        """
        lr_scheduler = self.runtime.lr_scheduler
        extra_state = {
            "lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict(),
        }
        extra_state.update(self.runtime.extra_state())
        return extra_state

    def _load_extra_state(self, extra_state: Dict[str, Any]) -> None:
        lr_state = extra_state.get("lr_scheduler")
        lr_scheduler = self.runtime.lr_scheduler
        if lr_state is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_state)

        self.runtime.load_extra_state(extra_state)

    def load(self) -> None:
        """Restore model, optimizer and model-bound extra state from ``load_path``."""
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
            module=self.module_name,
            trainable_only=self.trainable_only,
            parallel_state=self.parallel_state,
        )
        self._load_extra_state(state["extra_state"])
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: "TrainerState") -> None:
        """Write model, optimizer and model-bound extra state for ``state.global_step``.

        Only model-bound state goes in here. Job-level state — where the
        dataloader is, the rng — has its own writer.
        """
        extra_state = self._extra_state(state)
        helper.empty_cache()
        self.checkpointer.save(
            self.config.save_path,
            {
                "model": self.runtime.model,
                "optimizer": self.runtime.optimizer,
                "extra_state": extra_state,
            },
            global_steps=state.global_step,
            module=self.module_name,
            save_async=self.config.save_async,
            trainable_only=self.trainable_only,
            save_to_lowest_rank=self.config.dcp_save_to_lowest_rank,
            parallel_state=self.parallel_state,
            stage_dir=self.config.stage_dir,
            save_timeout_seconds=self.config.save_timeout_seconds,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {self.save_dir(state)} successfully!")

    def _prepare_export(self, state: "TrainerState", stage: str) -> str:
        """Make sure this step's DCP exists, then return its weights directory.

        Returns the weights directory rather than ``save_dir`` because that is
        what the legacy (non-distributed) export path converts from, and it now
        holds the weights alone.

        The DCP written here can land on a step the save cadence never reaches.
        ``GlobalStateCallback`` finishes such a step off with the cursor files
        and the manifest, so it resumes like any other.
        """
        if self._last_saved_step != state.global_step:
            dist.barrier()
            self.save_dcp(state)

        self.wait_for_pending_save()

        if stage == "train_end":
            self.runtime.optimizer = None
            self.runtime.lr_scheduler = None

        return layout.weights_dir(self.step_dir(state), self.module_name)

    def save_hf(self, state: "TrainerState", stage: str = "step_end") -> None:
        from ..utils.save_safetensor_utils import save_hf_safetensor

        weights_path = self._prepare_export(state, stage)

        save_hf_safetensor(
            save_hf_safetensor_path=self.hf_export_dir(state),
            model_assets=self.runtime.model_assets,
            ckpt_manager=self.config.manager,
            output_dir=self.config.output_dir,
            save_checkpoint_path=weights_path,
            model=self.runtime.model,
            fqn_to_index_mapping=self.runtime.args.fqn_to_index_mapping,
            is_rank_0=self.parallel_state.global_rank == 0,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()

    def save_lora(self, state: "TrainerState", stage: str = "step_end", adapter_name: str = "default") -> None:
        from ..utils.save_safetensor_utils import save_lora_adapter_with_dcp

        self._prepare_export(state, stage)
        save_lora_adapter_with_dcp(
            model=self.runtime.model,
            save_path=self.lora_export_dir(state),
            adapter_name=adapter_name,
        )
        helper.empty_cache()
        dist.barrier()

    def save_hf_or_lora(self, state: "TrainerState", stage: str = "step_end") -> None:
        if self.trainable_only:
            self.save_lora(state, stage=stage)
        else:
            self.save_hf(state, stage=stage)


__all__ = ["ModelCheckpointManager"]

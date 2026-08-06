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

"""Per-module checkpoint/resume for :class:`~veomni.models.seed_omni.accelerator.module_runtime.ModuleRuntime`."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import torch.distributed as dist

from ....checkpoint import CheckpointerBase, build_checkpointer
from ....utils import helper, logging
from ....utils.save_safetensor_utils import save_hf_safetensor, save_lora_adapter_with_dcp
from ..accelerator.dispatch import unwrap_module_chain
from ..mixins.offline_encoding import OfflineEncodingMixin


if TYPE_CHECKING:
    from ....omni_arguments.arguments_types import OmniModuleRuntimeArguments
    from ....trainer.callbacks import TrainerState
    from ..accelerator.module_runtime import ModuleRuntime


logger = logging.get_logger(__name__)


class OmniModuleCheckpointManager:
    """Own DCP / HF / LoRA save-load for one :class:`ModuleRuntime`.

    On-disk layout::

        <save_path>/global_step_{N}/
        ├── <module_a>/        # DCP {model, optimizer, extra_state={lr_scheduler}} (+ hf export)
        ├── <module_b>/        # …
        └── trainer_state.pt   # global (orchestrator-owned)
    """

    def __init__(self, runtime: ModuleRuntime) -> None:
        self.runtime = runtime
        self.module_name = runtime.module_name
        self.checkpoint_subfolder = runtime.module_name
        args: OmniModuleRuntimeArguments = runtime.args
        ckpt = runtime.train.checkpoint
        self._last_saved_step: int = -1
        self.checkpointer: CheckpointerBase = build_checkpointer(
            dist_backend=args.accelerator.fsdp_config.fsdp_mode,
            ckpt_manager=ckpt.manager,
        )

    @property
    def last_saved_step(self) -> int:
        return self._last_saved_step

    @property
    def args(self) -> OmniModuleRuntimeArguments:
        return self.runtime.args

    @property
    def parallel_state(self):
        return self.runtime.parallel_state

    # ── Path helpers ──────────────────────────────────────────────────────────

    def _global_step_root(self, state: TrainerState) -> str:
        return os.path.join(self.runtime.train.checkpoint.save_path, f"global_step_{state.global_step}")

    def _hf_export_dir(self, state: TrainerState) -> str:
        return os.path.join(self._global_step_root(state), self.checkpoint_subfolder)

    def _module_subdir(self, root: str, state: TrainerState) -> str:
        return os.path.join(root, f"global_step_{state.global_step}", self.module_name)

    def _save_dir(self, state: TrainerState) -> str:
        return self._module_subdir(self.runtime.train.checkpoint.save_path, state)

    def _output_dir(self, state: TrainerState) -> str:
        return self._module_subdir(self.runtime.train.checkpoint.output_dir, state)

    def _load_dir(self) -> str | None:
        load_path = self.runtime.train.checkpoint.load_path
        return None if load_path is None else os.path.join(load_path, self.module_name)

    def _extra_state(self, state: TrainerState) -> dict[str, Any]:
        lr_scheduler = self.runtime.lr_scheduler
        return {"lr_scheduler": None if lr_scheduler is None else lr_scheduler.state_dict()}

    def _load_extra_state(self, extra_state: dict[str, Any]) -> None:
        lr_sd = extra_state.get("lr_scheduler")
        lr_scheduler = self.runtime.lr_scheduler
        if lr_sd is not None and lr_scheduler is not None:
            lr_scheduler.load_state_dict(lr_sd)

    def _offline_cache_model(self) -> OfflineEncodingMixin | None:
        model = unwrap_module_chain(self.runtime.model)
        if not isinstance(model, OfflineEncodingMixin) or model.cache_mode == "full":
            return None
        return model

    # ── Load ──────────────────────────────────────────────────────────────────

    def load(self) -> None:
        model = self._offline_cache_model()
        if model is not None:
            self._load_partial_dcp(model)
            return
        self._load_dcp()

    def _load_dcp(self) -> None:
        load_dir = self._load_dir()
        if load_dir is None:
            return

        state = {
            "model": self.runtime.model,
            "optimizer": self.runtime.optimizer,
            "extra_state": {},
        }
        self.checkpointer.wait_for_pending_save()
        self.checkpointer.load(
            load_dir,
            state,
            trainable_only=bool(self.args.lora_config),
            parallel_state=self.parallel_state,
        )
        self._load_extra_state(state["extra_state"])
        dist.barrier()
        logger.info_rank0(f"Load distributed checkpoint from {load_dir} successfully!")

    def _load_partial_dcp(self, model: OfflineEncodingMixin) -> None:
        load_dir = self._load_dir()
        if load_dir is None:
            return
        self.checkpointer.wait_for_pending_save()
        model.load_partial_dcp_checkpoint(load_dir, trainer=self.runtime)
        if dist.is_initialized():
            dist.barrier()
        logger.info_rank0(f"Load partial offline-cache checkpoint from {load_dir} successfully!")

    # ── Save (DCP / HF / LoRA) ────────────────────────────────────────────────

    def save_dcp(self, state: TrainerState) -> None:
        model = self._offline_cache_model()
        if model is not None:
            model.save_partial_dcp_checkpoint(self._save_dir(state), trainer=self.runtime, state=state)
            self._last_saved_step = state.global_step
            return

        args = self.args
        save_checkpoint_path = self._save_dir(state)
        ckpt_state = {
            "model": self.runtime.model,
            "optimizer": self.runtime.optimizer,
            "extra_state": self._extra_state(state),
        }
        helper.empty_cache()
        self.checkpointer.save(
            save_checkpoint_path,
            ckpt_state,
            save_async=self.runtime.train.checkpoint.save_async,
            trainable_only=bool(args.lora_config),
            save_to_lowest_rank=self.runtime.train.checkpoint.dcp_save_to_lowest_rank,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step
        logger.info_rank0(f"Distributed checkpoint saved at {save_checkpoint_path} successfully!")

    def save_hf(self, state: TrainerState) -> None:
        model = self._offline_cache_model()
        if model is not None:
            self._save_full_hf_offline_cache(model, state)
            return

        args = self.args
        save_checkpoint_path = self._save_dir(state)
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            self.save_dcp(state)

        self.checkpointer.wait_for_pending_save()

        if state.stage == "train_end":
            self.runtime.optimizer = None
            self.runtime.lr_scheduler = None

        hf_weights_path = self._hf_export_dir(state)
        save_hf_safetensor(
            save_hf_safetensor_path=hf_weights_path,
            model_assets=self.runtime.collect_hf_export_assets(),
            ckpt_manager=self.runtime.train.checkpoint.manager,
            output_dir=self.runtime.train.checkpoint.output_dir,
            save_checkpoint_path=save_checkpoint_path,
            model=self.runtime.model,
            fqn_to_index_mapping=args.fqn_to_index_mapping,
            is_rank_0=self.runtime.train.global_rank == 0,
            parallel_state=self.parallel_state,
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def save_lora(self, state: TrainerState) -> None:
        save_checkpoint_path = self._save_dir(state)
        if not os.path.exists(save_checkpoint_path):
            dist.barrier()
            self.save_dcp(state)

        self.checkpointer.wait_for_pending_save()

        if state.stage == "train_end":
            self.runtime.optimizer = None
            self.runtime.lr_scheduler = None

        save_lora_adapter_with_dcp(
            model=self.runtime.model,
            save_path=self._output_dir(state),
            adapter_name="default",
        )
        helper.empty_cache()
        dist.barrier()
        self._last_saved_step = state.global_step

    def _save_full_hf_offline_cache(self, model: OfflineEncodingMixin, state: TrainerState) -> None:
        hf_weights_path = self._hf_export_dir(state)
        if self.runtime.train.global_rank == 0:
            model.save_full_hf_checkpoint(
                hf_weights_path,
                source_path=self.args.model_path,
                trainer=self.runtime,
                state=state,
            )
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = state.global_step

    def save_hf_or_lora(self, state: TrainerState) -> None:
        if self.args.lora_config:
            self.save_lora(state)
        else:
            self.save_hf(state)


__all__ = ["OmniModuleCheckpointManager"]

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

from typing import TYPE_CHECKING

import torch.distributed as dist

from ....checkpoint import layout
from ....models.checkpoint_manager import ModelCheckpointManager
from ....utils import logging
from ..accelerator.dispatch import unwrap_module_chain
from ..mixins.offline_encoding_mixin import OfflineEncodingMixin


if TYPE_CHECKING:
    from ....arguments.omni_arguments_types import OmniModuleRuntimeArguments
    from ....trainer.callbacks import TrainerState
    from ..accelerator.module_runtime import ModuleRuntime


logger = logging.get_logger(__name__)


class OmniModuleCheckpointManager(ModelCheckpointManager):
    """Own DCP / HF / LoRA save-load for one :class:`ModuleRuntime`.

    Paths come from the base, which nests every artifact under
    :attr:`module_name` — ``model/<module>/`` for the resume tree,
    ``hf_ckpt/<module>/`` for the export. That is the whole reason
    :mod:`veomni.checkpoint.layout` takes a ``module`` argument, so this class
    builds no paths of its own.

    Three things do differ from a single-model job:

    * **Offline cache.** A module whose ``cache_mode`` is not ``full`` has its
      encoder output precomputed, so there is no ordinary DCP state to write and
      its HF artifact is merged from the frozen source rather than converted
      from shards.
    * **Export assets.** A module's config/tokenizer/processor are bound onto
      the model, not cached beside it, so they are read off the live model.
    * **Stage.** The orchestrator puts the save stage on ``state``, where the
      base takes it as an argument.
    """

    def __init__(self, runtime: ModuleRuntime) -> None:
        self.module_name = runtime.module_name
        super().__init__(runtime)

    @property
    def args(self) -> OmniModuleRuntimeArguments:
        return self.runtime.args

    @property
    def hf_export_assets(self) -> list:
        """Read off the live model — ``ModuleRuntime._build_model_assets`` binds
        this module's sidecars onto it instead of caching them on the runtime."""
        return self.runtime.collect_hf_export_assets()

    def _offline_cache_model(self) -> OfflineEncodingMixin | None:
        """The module's own model when its encoder output is precomputed, else None.

        ``full`` means nothing is cached, so such a module saves and loads like
        any other and this returns None.
        """
        model = unwrap_module_chain(self.runtime.model)
        if not isinstance(model, OfflineEncodingMixin) or model.cache_mode == "full":
            return None
        return model

    def load(self) -> None:
        model = self._offline_cache_model()
        if model is None:
            super().load()
            return

        load_dir = self.load_dir()
        if load_dir is None:
            return
        self.wait_for_pending_save()
        model.load_partial_dcp_checkpoint(layout.model_dir(load_dir, self.module_name), trainer=self.runtime)
        if dist.is_initialized():
            dist.barrier()
        logger.info_rank0(f"Load partial offline-cache checkpoint from {load_dir} successfully!")

    def save_dcp(self, state: TrainerState) -> None:
        model = self._offline_cache_model()
        if model is None:
            super().save_dcp(state)
            return

        model.save_partial_dcp_checkpoint(self.save_dir(state), trainer=self.runtime, state=state)
        self._last_saved_step = state.global_step

    def save_hf(self, state: TrainerState, stage: str = "step_end") -> None:
        model = self._offline_cache_model()
        if model is None:
            super().save_hf(state, stage=stage)
            return

        # Merged from the frozen source by the module itself: there are no shards
        # to convert, so this never goes through _prepare_export.
        if self.parallel_state.global_rank == 0:
            model.save_full_hf_checkpoint(
                self.hf_export_dir(state),
                source_path=self.args.model_path,
                trainer=self.runtime,
                state=state,
            )
        if dist.is_initialized():
            dist.barrier()
        self._last_saved_step = state.global_step

    def save_hf_or_lora(self, state: TrainerState, stage: str = "step_end") -> None:
        """Route by LoRA, with the stage taken from ``state``.

        ``ModuleRuntime.save_hf_or_lora`` drops the keyword, so the base's
        default would report a train-end export as ``step_end`` and keep the
        optimizer alive through it.
        """
        del stage
        super().save_hf_or_lora(state, stage=state.stage or "step_end")


__all__ = ["OmniModuleCheckpointManager"]

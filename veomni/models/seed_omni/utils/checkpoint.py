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

"""Per-module checkpoint/resume for :class:`~veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime.ModuleRuntime`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ....models.checkpoint_manager import ModelCheckpointManager
from ....utils import logging


if TYPE_CHECKING:
    from ....arguments.omni_arguments_types import OmniModuleRuntimeArguments
    from ....trainer.callbacks import TrainerState
    from ..accelerated.omni_module.omni_module_runtime import ModuleRuntime


logger = logging.get_logger(__name__)


class OmniModuleCheckpointManager(ModelCheckpointManager):
    """Own DCP / HF / LoRA save-load for one :class:`ModuleRuntime`.

    Paths come from the base, which nests every artifact under
    :attr:`module_name` — ``model/<module>/`` for the resume tree,
    ``hf_ckpt/<module>/`` for the export. That is the whole reason
    :mod:`veomni.checkpoint.layout` takes a ``module`` argument, so this class
    builds no paths of its own.

    Two things do differ from a single-model job:

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

    def save_hf_or_lora(self, state: TrainerState, stage: str = "step_end") -> None:
        """Route by LoRA, with the stage taken from ``state``.

        ``ModuleRuntime.save_hf_or_lora`` drops the keyword, so the base's
        default would report a train-end export as ``step_end`` and keep the
        optimizer alive through it.
        """
        del stage
        super().save_hf_or_lora(state, stage=state.stage or "step_end")


__all__ = ["OmniModuleCheckpointManager"]

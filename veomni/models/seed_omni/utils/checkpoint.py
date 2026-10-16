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

from typing import TYPE_CHECKING, Any

from ....checkpoint import layout
from ....models.checkpoint_manager import ModelCheckpointManager


if TYPE_CHECKING:
    from ....arguments.omni_arguments_types import OmniModuleRuntimeArguments
    from ....trainer.callbacks import TrainerState


class OmniModuleCheckpointManager(ModelCheckpointManager):
    """Own DCP / HF / LoRA save-load for one :class:`ModuleRuntime`.

    Every module shares the job's step directory, so each artifact nests one
    level deeper under :attr:`module_name` — ``model/<module>/`` for the resume
    tree, ``hf_ckpt/<module>/`` for the export, ``model_assets/<module>/`` for
    the sidecars, so two modules cannot overwrite each other's ``config.json``.
    """

    @property
    def module_name(self) -> str:
        return self.runtime.module_name

    @property
    def args(self) -> OmniModuleRuntimeArguments:
        return self.runtime.args

    def save_dir(self, state: TrainerState) -> str:
        return layout.model_dir(self.step_dir(state), self.module_name)

    def weights_dir(self, state: TrainerState) -> str:
        return layout.weights_dir(self.step_dir(state), self.module_name)

    def hf_export_dir(self, state: TrainerState) -> str:
        return layout.hf_export_dir(self.step_dir(state), self.module_name)

    def lora_export_dir(self, state: TrainerState) -> str:
        return layout.lora_export_dir(self.step_dir(state), self.module_name)

    def assets_dir(self) -> str:
        return layout.assets_dir(self.config.model_assets_dir, self.module_name)

    def _checkpointer_kwargs(self) -> dict[str, Any]:
        """The checkpointer resolves ``model/<module>/`` under the step itself."""
        return {**super()._checkpointer_kwargs(), "module": self.module_name}


__all__ = ["OmniModuleCheckpointManager"]

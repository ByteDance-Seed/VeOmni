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

from __future__ import annotations

from typing import TYPE_CHECKING

from ..base import Callback


if TYPE_CHECKING:
    from ...omni.omni_trainer import OmniTrainer


class OmniRootAssetsCallback(Callback):
    """Export the omni-root HF layout (config + graphs + module sidecars) once at train begin.

    Job-level state (step, dataloader cursor, RNG) is not omni-specific and is
    handled by the shared :class:`~veomni.trainer.callbacks.GlobalStateCallback`.
    """

    trainer: OmniTrainer

    def on_train_begin(self, state, **kwargs) -> None:
        self.trainer.save_model_assets()


__all__ = ["OmniRootAssetsCallback"]

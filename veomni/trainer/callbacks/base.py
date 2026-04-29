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

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List


if TYPE_CHECKING:
    from ..base import BaseTrainer


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0


class Callback:
    def __init__(self, trainer: "BaseTrainer") -> None:
        self.trainer = trainer

    def on_step_begin(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        pass

    def on_step_end(
        self, state: TrainerState, loss: float, loss_dict: Dict[str, float], grad_norm: float, **kwargs
    ) -> None:
        pass

    def on_epoch_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_epoch_end(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_train_end(self, state: TrainerState, **kwargs) -> None:
        pass

    # Sub-step lifecycle hooks. Defaults are no-ops so existing callbacks
    # do not need to be updated; consumers (e.g. DiagnosticsCallback) opt in.

    def on_data_load_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_data_load_end(self, state: TrainerState, micro_batches: List[Dict[str, Any]] = None, **kwargs) -> None:
        pass

    def on_forward_begin(self, state: TrainerState, micro_batch: Dict[str, Any] = None, **kwargs) -> None:
        pass

    def on_forward_end(self, state: TrainerState, micro_batch: Dict[str, Any] = None, **kwargs) -> None:
        pass

    def on_backward_begin(self, state: TrainerState, micro_batch: Dict[str, Any] = None, **kwargs) -> None:
        pass

    def on_backward_end(self, state: TrainerState, micro_batch: Dict[str, Any] = None, **kwargs) -> None:
        pass

    def on_optimizer_step_begin(self, state: TrainerState, **kwargs) -> None:
        pass

    def on_optimizer_step_end(self, state: TrainerState, grad_norm: float = None, **kwargs) -> None:
        pass

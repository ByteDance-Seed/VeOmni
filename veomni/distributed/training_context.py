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


from enum import Enum

from ..utils.singleton import Singleton


class TrainingStage(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"


class TrainingContext(metaclass=Singleton):
    _training_stage: TrainingStage = TrainingStage.FORWARD
    _layer_index: int = 0
    _model_depth: int = 0

    def set_training_stage(self, training_stage: TrainingStage) -> None:
        self._training_stage = training_stage

    def get_training_stage(self) -> TrainingStage:
        return self._training_stage

    def set_layer_index(self, layer_index: int) -> None:
        self._layer_index = layer_index

    def set_model_depth(self, model_depth: int) -> None:
        self._model_depth = model_depth

    def get_layer_index(self) -> int:
        return self._layer_index

    def get_model_depth(self) -> int:
        return self._model_depth
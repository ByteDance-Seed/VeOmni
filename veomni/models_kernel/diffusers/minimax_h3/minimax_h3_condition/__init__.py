# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.

"""Register the local MiniMax H3 condition config and model."""

from veomni.models_kernel.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("MiniMaxH3ConditionModel")
def register_minimax_h3_condition_config():
    from .configuration_minimax_h3_condition import MiniMaxH3ConditionModelConfig

    return MiniMaxH3ConditionModelConfig


@MODELING_REGISTRY.register("MiniMaxH3ConditionModel")
def register_minimax_h3_condition_modeling(_architecture: str | None = None):
    from .modeling_minimax_h3_condition import MiniMaxH3ConditionModel

    return MiniMaxH3ConditionModel

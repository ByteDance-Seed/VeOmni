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

"""Register the local LTX 2.3 condition config and model."""

from veomni.models.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("LTXVideoConditionModel")
def register_ltx_condition_config():
    from .configuration_ltx2_3_condition import LTXVideoConditionModelConfig

    return LTXVideoConditionModelConfig


@MODELING_REGISTRY.register("LTXVideoConditionModel")
def register_ltx_condition_modeling(_architecture: str | None = None):
    from .modeling_ltx2_3_condition import LTXVideoConditionModel

    return LTXVideoConditionModel

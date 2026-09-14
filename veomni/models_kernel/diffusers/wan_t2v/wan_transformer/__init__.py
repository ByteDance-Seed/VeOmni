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

"""Register the local Wan T2V transformer config and model."""

from veomni.models_kernel.registry import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("WanTransformer3DModel")
def register_wan_diffusers_transformer_config():
    from .configuration_wan_transformer import WanTransformer3DModelConfig

    return WanTransformer3DModelConfig


@MODELING_REGISTRY.register("WanTransformer3DModel")
def register_wan_diffusers_transformer_modeling(_architecture: str):
    from .modeling_wan_transformer import WanTransformer3DModel, apply_veomni_wan_transformer_patch

    apply_veomni_wan_transformer_patch()
    return WanTransformer3DModel

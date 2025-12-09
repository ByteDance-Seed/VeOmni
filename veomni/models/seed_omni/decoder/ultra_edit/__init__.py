# Copyright 2024-2025 The Black-forest-labs Authors. All rights reserved.
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
from ....loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("ultra_edit")
def register_janus_config():
    from .configuration_ultra_edit import UltraEditConfig

    return UltraEditConfig


@MODELING_REGISTRY.register("ultra_edit")
def register_ultra_edit_modeling(architecture: str):
    from .modeling_ultra_edit import UltraEdit

    return UltraEdit


@MODEL_PROCESSOR_REGISTRY.register("UltraEditProcessor")
def register_ultra_edit_processor():
    from .processing_ultra_edit import UltraEditProcessor

    return UltraEditProcessor

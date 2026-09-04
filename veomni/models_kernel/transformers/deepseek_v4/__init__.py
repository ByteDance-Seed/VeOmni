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

"""DeepSeek-V4 modeling that calls local VeomniKernel handles."""

# TODO: hook this module into the models_kernel registry fill. DeepseekV4Config
# must replace the upstream class on MODEL_CONFIG_REGISTRY so from_dict keeps
# dsa_indexer_loss / dsa_indexer_loss_coef. Importing this package is enough
# once that fill site exists; do not re-export the class from models_kernel.

from veomni.models_kernel.registry import MODEL_CONFIG_REGISTRY

from .configuration_deepseek_v4 import DeepseekV4Config


@MODEL_CONFIG_REGISTRY.register("deepseek_v4")
def register_deepseek_v4_config():
    return DeepseekV4Config


__all__ = ["DeepseekV4Config"]

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

from ..loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY
from .auto import SeedOmniConfig, SeedOmniModel, SeedOmniProcessor, build_omni_model, build_omni_processor
from .decoder import *
from .encoder import *
from .foundation import *


@MODEL_CONFIG_REGISTRY.register("seed_omni")
def register_seed_omni_config():
    from .configuration_seed_omni import SeedOmniConfig

    return SeedOmniConfig


@MODELING_REGISTRY.register("seed_omni")
def register_seed_omni_modeling(architecture: str):
    from .modeling_seed_omni import SeedOmniModel

    return SeedOmniModel


@MODEL_PROCESSOR_REGISTRY.register("SeedOmniProcessor")
def register_seed_omni_processor():
    from .processing_seed_omni import SeedOmniProcessor

    return SeedOmniProcessor


__all__ = [
    "build_omni_model",
    "build_omni_processor",
    "SeedOmniModel",
    "SeedOmniConfig",
    "SeedOmniProcessor",
]

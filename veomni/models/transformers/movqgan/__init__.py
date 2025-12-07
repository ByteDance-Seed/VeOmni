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
from ...loader import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("movqgan")
def register_movqgan_config():
    from .configuration_movqgan import MoVQGANConfig

    return MoVQGANConfig


@MODELING_REGISTRY.register("movqgan")
def register_movqgan_modeling(architecture: str):
    from .modeling_movqgan import MoVQGAN

    return MoVQGAN


@MODEL_PROCESSOR_REGISTRY.register("MoVQGANProcessor")
def register_movqgan_processor():
    from .processing_movqgan import MoVQGANProcessor

    return MoVQGANProcessor

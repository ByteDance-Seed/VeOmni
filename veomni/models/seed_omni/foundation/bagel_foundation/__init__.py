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
from ....loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("bagel_foundation")
def register_bagel_foundation_config():
    from transformers import AutoTokenizer, Qwen2TokenizerFast

    from .configuration_bagel_foundation import BagelFoundationConfig

    AutoTokenizer.register(BagelFoundationConfig, fast_tokenizer_class=Qwen2TokenizerFast)
    return BagelFoundationConfig


@MODELING_REGISTRY.register("bagel_foundation")
def register_bagel_foundation_modeling(architecture: str):
    from .modeling_bagel_foundation import BagelFoundationModel

    return BagelFoundationModel

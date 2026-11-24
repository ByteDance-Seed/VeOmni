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

"""LLaDA2.0-Uni image decoder OmniModule (ZImage DiT + Flux VAE, on-demand)."""

from ... import OMNI_ACCELERATED_MODEL_REGISTRY, OMNI_CONFIG_REGISTRY, OMNI_MODEL_REGISTRY


@OMNI_CONFIG_REGISTRY.register("llada2_uni_decoder")
def register_llada2_uni_decoder_config():
    from .configuration import LladaUniDecoderConfig

    return LladaUniDecoderConfig


@OMNI_MODEL_REGISTRY.register("llada2_uni_decoder")
def register_llada2_uni_decoder_model():
    from .modeling import LladaUniDecoder

    return LladaUniDecoder


@OMNI_ACCELERATED_MODEL_REGISTRY.register("llada2_uni_decoder")
def register_llada2_uni_decoder_accelerated_model():
    from .accelerated import LladaUniDecoderAccelerated

    return LladaUniDecoderAccelerated

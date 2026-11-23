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

"""LLaDA-MoE as a single SeedOmni module.

The sparse sibling of :mod:`..llada`. Same one-node graph, same sample-mean
objective, plus a router load-balancing term and an Expert Parallel plan.
``model_type`` is ``llada_moe_omni`` so the composed-model registries stay
disjoint from the vendored ``veomni.models.transformers.llada_moe`` entries.
"""

from .. import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("llada_moe_omni")
def register_llada_moe_omni_config():
    from .configuration import LladaMoeOmniConfig

    return LladaMoeOmniConfig


@OMNI_MODEL_REGISTRY.register("llada_moe_omni")
def register_llada_moe_omni_modeling():
    from .modeling import LladaMoeOmni

    return LladaMoeOmni


@OMNI_ACCELERATED_MODEL_REGISTRY.register("llada_moe_omni")
def register_llada_moe_omni_accelerated_model():
    from .accelerated import LladaMoeOmniAccelerated

    return LladaMoeOmniAccelerated


@OMNI_PROCESSOR_REGISTRY.register("llada_moe_omni")
def register_llada_moe_omni_processor():
    from .processing import LladaMoeOmniPreprocessor

    return LladaMoeOmniPreprocessor


# Side-effect only: register the HF layout so `OmniModel.from_pretrained` (and the
# trainer/inferencer that go through it) can read a
# `LLaDA-MoE-7B-A1B-Instruct-VeOmni` checkpoint in place. Imported last so the
# layout module's own imports of `.configuration` resolve against an
# already-populated package.
from . import hf_layout  # noqa: F401  E402

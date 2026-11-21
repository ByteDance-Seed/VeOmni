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

"""LLaDA as a single SeedOmni module.

The mask predictor supervises itself in place, so the graph is one node deep and
the module needs no sibling to hand anything to. ``model_type`` is
``llada_omni`` rather than ``llada`` so the composed-model registries stay
disjoint from the vendored ``veomni.models.transformers.llada`` entries that the
non-Omni trainer resolves.

This family currently covers LLaDA only -- the dense 8B mask predictor. The
``llada_moe`` / ``llada_v`` / ``llada2_uni`` variants each need their own module
shape (expert routing, a vision tower, an image tokenizer) and are not part of
this slice.
"""

from .. import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("llada_omni")
def register_llada_omni_config():
    from .configuration import LladaOmniConfig

    return LladaOmniConfig


@OMNI_MODEL_REGISTRY.register("llada_omni")
def register_llada_omni_modeling():
    from .modeling import LladaOmni

    return LladaOmni


@OMNI_ACCELERATED_MODEL_REGISTRY.register("llada_omni")
def register_llada_omni_accelerated_model():
    from .accelerated import LladaOmniAccelerated

    return LladaOmniAccelerated


@OMNI_PROCESSOR_REGISTRY.register("llada_omni")
def register_llada_omni_processor():
    from .processing import LladaOmniPreprocessor

    return LladaOmniPreprocessor


# Side-effect only: register the HF layout so `OmniModel.from_pretrained` (and the
# trainer/inferencer that go through it) can read a `LLaDA-8B-Instruct-veomni`
# checkpoint in place. Imported last so the layout module's own imports of
# `.configuration` resolve against an already-populated package.
from . import hf_layout  # noqa: F401  E402

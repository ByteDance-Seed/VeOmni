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

"""LLaDA-V as a single SeedOmni module.

The vision sibling of :mod:`..llada`. Same one-node graph and same sample-mean
objective; the SigLIP tower and projector run inside that node, scattering image
rows into the text embeddings before the backbone.
``model_type`` is ``llada_v_omni`` so the composed-model registries stay disjoint
from the vendored ``veomni.models.transformers.llada_v`` entries.
"""

from .. import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("llada_v_omni")
def register_llada_v_omni_config():
    from .configuration import LladaVOmniConfig

    return LladaVOmniConfig


@OMNI_MODEL_REGISTRY.register("llada_v_omni")
def register_llada_v_omni_modeling():
    from .modeling import LladaVOmni

    return LladaVOmni


@OMNI_ACCELERATED_MODEL_REGISTRY.register("llada_v_omni")
def register_llada_v_omni_accelerated_model():
    from .accelerated import LladaVOmniAccelerated

    return LladaVOmniAccelerated


@OMNI_PROCESSOR_REGISTRY.register("llada_v_omni")
def register_llada_v_omni_processor():
    from .processing import LladaVOmniPreprocessor

    return LladaVOmniPreprocessor


# Side-effect only: register the HF layout so `OmniModel.from_pretrained` (and the
# trainer/inferencer that go through it) can read a `LLaDA-V-VeOmni` checkpoint in
# place. Imported last so the layout module's own imports of `.configuration`
# resolve against an already-populated package.
from . import hf_layout  # noqa: F401  E402

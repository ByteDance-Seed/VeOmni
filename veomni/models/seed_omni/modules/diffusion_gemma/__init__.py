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

"""DiffusionGemma as a single SeedOmni module.

A block-diffusion language model: the encoder prefills the prompt and exports a KV
cache, and the decoder denoises a fixed-length canvas against it. Both passes run
inside one node, so the graph is one entry deep -- there is no cache to hand to a
sibling module.

``model_type`` is ``diffusion_gemma_omni`` so the composed-model registries stay
disjoint from the vendored ``veomni.models.transformers.diffusion_gemma`` entry.
"""

from .. import (
    OMNI_ACCELERATED_MODEL_REGISTRY,
    OMNI_CONFIG_REGISTRY,
    OMNI_MODEL_REGISTRY,
    OMNI_PROCESSOR_REGISTRY,
)


@OMNI_CONFIG_REGISTRY.register("diffusion_gemma_omni")
def register_diffusion_gemma_omni_config():
    from .configuration import DiffusionGemmaOmniConfig

    return DiffusionGemmaOmniConfig


@OMNI_MODEL_REGISTRY.register("diffusion_gemma_omni")
def register_diffusion_gemma_omni_modeling():
    from .modeling import DiffusionGemmaOmni

    return DiffusionGemmaOmni


@OMNI_ACCELERATED_MODEL_REGISTRY.register("diffusion_gemma_omni")
def register_diffusion_gemma_omni_accelerated_model():
    from .accelerated import DiffusionGemmaOmniAccelerated

    return DiffusionGemmaOmniAccelerated


@OMNI_PROCESSOR_REGISTRY.register("diffusion_gemma_omni")
def register_diffusion_gemma_omni_processor():
    from .processing import DiffusionGemmaOmniPreprocessor

    return DiffusionGemmaOmniPreprocessor


# Side-effect only: register the HF layout so `OmniModel.from_pretrained` (and the
# trainer/inferencer that go through it) can read a `diffusiongemma-26B-A4B-it`
# checkpoint in place. Imported last so the layout module's own imports of
# `.configuration` resolve against an already-populated package.
from . import hf_layout  # noqa: F401  E402

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
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("diffusion_gemma")
def register_diffusion_gemma_config():
    from .configuration_diffusion_gemma import DiffusionGemmaConfig

    return DiffusionGemmaConfig


@MODELING_REGISTRY.register("diffusion_gemma")
def register_diffusion_gemma_modeling(architecture: str):
    # DiffusionGemma is vendored rather than patchgen-generated: transformers 5.16.1
    # already ships the architecture, and this package overrides it with the canvas
    # training, packing and kernel paths. See ``modeling_diffusion_gemma``.
    from .gemma4_initialization import (
        convert_gemma4_fqn_to_index_mapping,
        create_gemma4_to_diffusion_gemma_checkpoint_tensor_converter,
        validate_gemma4_initialization_missing_parameters,
    )
    from .modeling_diffusion_gemma import DiffusionGemmaForBlockDiffusion, DiffusionGemmaModel

    DiffusionGemmaForBlockDiffusion._create_checkpoint_tensor_converter = staticmethod(
        create_gemma4_to_diffusion_gemma_checkpoint_tensor_converter
    )
    DiffusionGemmaForBlockDiffusion._convert_fqn_to_index_mapping = staticmethod(convert_gemma4_fqn_to_index_mapping)
    DiffusionGemmaForBlockDiffusion._validate_checkpoint_missing_parameters = (
        validate_gemma4_initialization_missing_parameters
    )

    if architecture.endswith("Model"):
        return DiffusionGemmaModel
    # The published checkpoint declares ``DiffusionGemmaForBlockDiffusion``; it is also the
    # only class carrying the lm_head and the training objective.
    return DiffusionGemmaForBlockDiffusion

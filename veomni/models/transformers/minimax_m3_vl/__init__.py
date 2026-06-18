# Copyright 2026 The MiniMax AI Team, HuggingFace Team, and the VeOmni Team. All rights reserved.
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
from ....utils.device import IS_NPU_AVAILABLE
from ....utils.import_utils import is_transformers_version_greater_or_equal_to
from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


_MINIMAX_M3_TRANSFORMERS_REQUIREMENT = (
    "MiniMax M3 VL modeling requires a local environment with transformers>=5.12.0, "
    "because earlier transformers releases do not ship transformers.models.minimax_m3_vl. "
    "VeOmni keeps the global transformers-stable pin unchanged; use the MiniMax example "
    "environment documented in docs/examples/minimax_m3_vl.md when training this model."
)


@MODEL_CONFIG_REGISTRY.register("minimax_m3_vl")
def register_minimax_m3_vl_config():
    from .configuration_minimax_m3_vl import MiniMaxM3VLConfig

    return MiniMaxM3VLConfig


@MODEL_CONFIG_REGISTRY.register("minimax_m3_vl_text")
def register_minimax_m3_vl_text_config():
    from .configuration_minimax_m3_vl import MiniMaxM3VLTextConfig

    return MiniMaxM3VLTextConfig


@MODEL_CONFIG_REGISTRY.register("minimax_m3_vl_vision")
def register_minimax_m3_vl_vision_config():
    from .configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig

    return MiniMaxM3VLVisionConfig


@MODELING_REGISTRY.register("minimax_m3_vl")
def register_minimax_m3_vl_modeling(architecture: str):
    from .checkpoint_tensor_converter import (
        convert_minimax_m3_vl_fqn_to_index_mapping,
        create_minimax_m3_vl_checkpoint_tensor_converter,
    )

    if not is_transformers_version_greater_or_equal_to("5.12.0"):
        raise RuntimeError(_MINIMAX_M3_TRANSFORMERS_REQUIREMENT)

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_minimax_m3_vl_npu import (
            MiniMaxM3SparseForConditionalGeneration,
            MiniMaxM3VLForCausalLM,
            MiniMaxM3VLModel,
            MiniMaxM3VLTextModel,
        )
    else:
        from .generated.patched_modeling_minimax_m3_vl_gpu import (
            MiniMaxM3SparseForConditionalGeneration,
            MiniMaxM3VLForCausalLM,
            MiniMaxM3VLModel,
            MiniMaxM3VLTextModel,
        )

    for model_cls in (
        MiniMaxM3SparseForConditionalGeneration,
        MiniMaxM3VLForCausalLM,
        MiniMaxM3VLModel,
        MiniMaxM3VLTextModel,
    ):
        model_cls._create_checkpoint_tensor_converter = staticmethod(create_minimax_m3_vl_checkpoint_tensor_converter)
        model_cls._convert_fqn_to_index_mapping = staticmethod(convert_minimax_m3_vl_fqn_to_index_mapping)

    if "ForCausalLM" in architecture and "ConditionalGeneration" not in architecture:
        return MiniMaxM3VLForCausalLM
    elif "TextModel" in architecture:
        return MiniMaxM3VLTextModel
    elif "Model" in architecture and "ForConditionalGeneration" not in architecture:
        return MiniMaxM3VLModel
    else:
        return MiniMaxM3SparseForConditionalGeneration

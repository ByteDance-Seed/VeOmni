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
from ...registry import MODEL_CONFIG_REGISTRY, MODEL_PROCESSOR_REGISTRY, MODELING_REGISTRY


# Registered so ``get_model_config`` swaps the checkpoint's ``auto_map`` remote
# config class for the transformers-native one the generated modeling targets.
@MODEL_CONFIG_REGISTRY.register("minimax_m3_vl")
def register_minimax_m3_vl_config():
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLConfig

    return MiniMaxM3VLConfig


# ``get_model_processor`` keys this registry on the class name ``AutoProcessor``
# resolved from the checkpoint. Public MiniMax M3 checkpoints ship their own
# ``processing_minimax.py`` (``MiniMaxVLProcessor``) alongside the upstream class
# name, and which one wins depends on the checkpoint's ``auto_map``; register
# both so VeOmni's processor is used either way. ``MiniMaxVLProcessor`` is generic,
# so the processor refuses checkpoints whose ``model_type`` is not ``minimax_m3_vl``.
@MODEL_PROCESSOR_REGISTRY.register("MiniMaxVLProcessor")
@MODEL_PROCESSOR_REGISTRY.register("MiniMaxM3VLProcessor")
def register_minimax_m3_vl_processor():
    from .processing_minimax_m3_vl import MiniMaxM3VLProcessor

    return MiniMaxM3VLProcessor


# Only the multimodal ``MiniMaxM3SparseForConditionalGeneration`` is supported:
# the checkpoint converter and parallel plan use its ``model.language_model.*``
# parameter names.
@MODELING_REGISTRY.register("minimax_m3_vl")
def register_minimax_m3_vl_modeling(architecture: str):
    """Register the generated GPU or NPU MiniMax model for the active accelerator backend."""
    from .checkpoint_tensor_converter import (
        convert_minimax_m3_vl_fqn_to_index_mapping,
        create_minimax_m3_vl_checkpoint_tensor_converter,
    )

    if IS_NPU_AVAILABLE:
        from .generated.patched_modeling_minimax_m3_vl_npu import MiniMaxM3SparseForConditionalGeneration
    else:
        from .generated.patched_modeling_minimax_m3_vl_gpu import MiniMaxM3SparseForConditionalGeneration

    if "ForConditionalGeneration" not in architecture:
        raise ValueError(
            f"MiniMax M3 VL supports only MiniMaxM3SparseForConditionalGeneration, got architecture {architecture!r}."
        )
    model_cls = MiniMaxM3SparseForConditionalGeneration
    model_cls._create_checkpoint_tensor_converter = staticmethod(create_minimax_m3_vl_checkpoint_tensor_converter)
    model_cls._convert_fqn_to_index_mapping = staticmethod(convert_minimax_m3_vl_fqn_to_index_mapping)
    return model_cls

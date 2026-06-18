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
"""
Patch configuration for MiniMax M3 VL NPU code generation.

This path intentionally mirrors the GPU patch set for the first migration
slice. It provides a checked-in NPU generated artifact and VeOmni registry
selection without claiming Ascend-specific kernel replacements before NPU
runtime evidence exists.

Regen command:
patchgen veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_npu_patch_gen_config -o veomni/models/transformers/minimax_m3_vl/generated --diff
"""

from veomni.models.transformers.minimax_m3_vl.minimax_m3_vl_gpu_patch_gen_config import (
    _grid_thw_to_list,
    collate_multimodal_metadata,
    minimax_m3_vl_3d_rotary_embedding_forward_patched,
    minimax_m3_vl_get_metadata_collate_func_patched,
    minimax_m3_vl_get_parallel_plan_patched,
    minimax_m3_vl_get_position_id_func_patched,
    minimax_m3_vl_model_forward_patched,
    minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    minimax_m3_vl_text_get_parallel_plan_patched,
    minimax_m3_vl_vision_dummy_forward_patched,
    minimax_m3_vl_vision_model_forward_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.minimax_m3_vl.modeling_minimax_m3_vl",
    target_file="patched_modeling_minimax_m3_vl_npu.py",
    description="MiniMax M3 VL with VeOmni parallel-plan hooks for NPU runtime selection",
    transformers_version="5.12.0",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.ops.dispatch", names=["OpSlot"])
config.add_post_import_block(
    """
veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
"""
)
config.add_helper(_grid_thw_to_list)
config.add_helper(collate_multimodal_metadata)

config.override_method(
    "MiniMaxM3VL3DRotaryEmbedding.forward",
    replacement=minimax_m3_vl_3d_rotary_embedding_forward_patched,
    description="Consume collator-precomputed MiniMax vision grid lists when available",
)
config.override_method(
    "MiniMaxM3VLVisionModel.forward",
    replacement=minimax_m3_vl_vision_model_forward_patched,
    description="Consume MiniMax collator-precomputed vision grid metadata",
)
config.override_method(
    "MiniMaxM3VLVisionModel.dummy_forward",
    replacement=minimax_m3_vl_vision_dummy_forward_patched,
    description="Provide MiniMax dummy vision forward for asymmetric FSDP batches",
)
config.override_method(
    "MiniMaxM3VLModel.forward",
    replacement=minimax_m3_vl_model_forward_patched,
    description="Add MiniMax VLM metadata fast path and FSDP dummy vision branch",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.forward",
    replacement=minimax_m3_vl_sparse_for_conditional_generation_forward_patched,
    description="Unpack VeOmni causal LM loss tuple and route fused loss kernels when selected",
)

config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_parallel_plan",
    replacement=minimax_m3_vl_get_parallel_plan_patched,
    description="Register MiniMax M3 VL expert parallel plan for the multimodal training path",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_position_id_func",
    replacement=minimax_m3_vl_get_position_id_func_patched,
    description="Use VeOmni's default 1-D packed-sequence position IDs for MiniMax M3 VL SFT data",
)
config.override_method(
    "MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func",
    replacement=minimax_m3_vl_get_metadata_collate_func_patched,
    description="Expose MiniMax CPU-side vision grid metadata derivation to the VeOmni collator",
)
config.override_method(
    "MiniMaxM3VLForCausalLM.get_parallel_plan",
    replacement=minimax_m3_vl_text_get_parallel_plan_patched,
    description="Register MiniMax M3 VL expert parallel plan for text-only reduced-layer smoke tests",
)

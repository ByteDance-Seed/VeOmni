# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# See the License for the specific language governing limitations
# under the License.
"""
Patch configuration for DeepseekV3 NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.deepseek_v3.deepseek_v3_npu_patch_gen_config -o veomni/models_kernel/transformers/deepseek_v3/generated --diff

Reuses the GPU structural patches. ``VeomniKernel`` reads NPU impl names from
the installed kernel config. No local ``triton_bmm`` on NPU.
"""

from veomni.models_kernel.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config import (
    PatchedDeepseekV3NaiveMoe,
    apply_rotary_pos_emb_patched,
    deepseek_v3_forcausallm_forward_patched,
    deepseek_v3_forcausallm_init_patched,
    deepseek_v3_get_parallel_plan_patched,
    deepseek_v3_mlp_forward_patched,
    deepseek_v3_mlp_init_patched,
    deepseek_v3_moe_forward_patched,
    deepseek_v3_rmsnorm_forward_patched,
    deepseek_v3_rmsnorm_init_patched,
    deepseek_v3_rotary_embedding_forward_patched,
    deepseek_v3_topk_router_forward_patched,
)
from veomni.models_kernel.transformers.deepseek_v3.deepseek_v3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.deepseek_v3.modeling_deepseek_v3",
    target_file="patched_modeling_deepseek_v3_npu.py",
    description="DeepseekV3 with VeomniKernel NPU replacements",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)

config.override_method(
    "DeepseekV3RMSNorm.__init__",
    replacement=deepseek_v3_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV3RMSNorm.forward",
    replacement=deepseek_v3_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "DeepseekV3RotaryEmbedding.forward",
    replacement=deepseek_v3_rotary_embedding_forward_patched,
    description="Use local triton_bmm for deterministic freqs when rotary impl is triton",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)
config.override_method(
    "DeepseekV3MLP.__init__",
    replacement=deepseek_v3_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniKernel",
)
config.override_method(
    "DeepseekV3MLP.forward",
    replacement=deepseek_v3_mlp_forward_patched,
    description="Always call the local swiglu_mlp VeomniKernel",
)
config.replace_class(
    "DeepseekV3NaiveMoe",
    replacement=PatchedDeepseekV3NaiveMoe,
    description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights",
)
config.override_method(
    "DeepseekV3TopkRouter.forward",
    replacement=deepseek_v3_topk_router_forward_patched,
    description="Disable autocast around fp32 router linear for VeRL actor/rollout parity",
)
config.override_method(
    "DeepseekV3MoE.forward",
    replacement=deepseek_v3_moe_forward_patched,
    description="Report top-k indices to the MoE load-balance monitor",
)
config.override_method(
    "DeepseekV3ForCausalLM.__init__",
    replacement=deepseek_v3_forcausallm_init_patched,
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
config.override_method(
    "DeepseekV3ForCausalLM.forward",
    replacement=deepseek_v3_forcausallm_forward_patched,
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)
config.override_method(
    "DeepseekV3ForCausalLM.get_parallel_plan",
    replacement=deepseek_v3_get_parallel_plan_patched,
    description="Register DeepseekV3 expert parallel plan for v5 generated modeling",
)

rotate_half = None  # noqa: E305

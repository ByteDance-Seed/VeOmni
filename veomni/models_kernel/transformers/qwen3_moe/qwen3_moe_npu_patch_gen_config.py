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
"""
Patch configuration for Qwen3Moe NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3_moe.qwen3_moe_npu_patch_gen_config -o veomni/models_kernel/transformers/qwen3_moe/generated --diff
"""

from veomni.models_kernel.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config import (
    PatchedQwen3MoeExperts,
    apply_rotary_pos_emb_patched,
    qwen3_moe_attention_forward_patched,
    qwen3_moe_forcausallm_forward_patched,
    qwen3_moe_forcausallm_init_patched,
    qwen3_moe_get_parallel_plan_patched,
    qwen3_moe_mlp_forward_patched,
    qwen3_moe_mlp_init_patched,
    qwen3_moe_model_forward_patched,
    qwen3_moe_rmsnorm_forward_patched,
    qwen3_moe_rmsnorm_init_patched,
    qwen3_moe_topk_router_forward_patched,
)
from veomni.models_kernel.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
    target_file="patched_modeling_qwen3_moe_npu.py",
    description="Qwen3Moe with VeOmni patches and VeomniKernel NPU replacements",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same SP helpers, same rot_pos_ids /
# async ulysses / get_position_id helpers).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
# Propagate the GPU config's dropped imports (e.g. ``Qwen3MoeCausalLMOutputWithPast``,
# now superseded by ``Qwen3MoeCausalLMOutputWithLogProbs`` for the FSDP2-safe
# pre-backward unshard hook on ``lm_head``).
config.drop_imported_names.update(gpu_config.drop_imported_names)


config.override_method(
    "Qwen3MoeRMSNorm.__init__",
    replacement=qwen3_moe_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3MoeRMSNorm.forward",
    replacement=qwen3_moe_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3MoeMLP.__init__",
    replacement=qwen3_moe_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniKernel",
)
config.override_method(
    "Qwen3MoeMLP.forward",
    replacement=qwen3_moe_mlp_forward_patched,
    description="Always call the local swiglu_mlp VeomniKernel",
)


config.replace_class(
    "Qwen3MoeExperts",
    replacement=PatchedQwen3MoeExperts,
    description="Always call moe_experts VeomniKernel on v5 gate_up_proj weights",
)


config.override_method(
    "Qwen3MoeTopKRouter.forward",
    replacement=qwen3_moe_topk_router_forward_patched,
    description=(
        "Return raw pre-softmax logits as `router_logits` so HF's "
        "`load_balancing_loss_func` (which applies softmax internally) "
        "stays consistent with the HF aux-loss baseline."
    ),
)


config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)

# Dummy reference resolved at codegen time from the generated module.
rotate_half = None  # noqa: E305


config.override_method(
    "Qwen3MoeModel.forward",
    replacement=qwen3_moe_model_forward_patched,
    description="Support SP in Qwen3MoeModel.forward",
)


config.override_method(
    "Qwen3MoeForCausalLM.__init__",
    replacement=qwen3_moe_forcausallm_init_patched,
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
config.override_method(
    "Qwen3MoeForCausalLM.forward",
    replacement=qwen3_moe_forcausallm_forward_patched,
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
)


config.override_method(
    "Qwen3MoeForCausalLM.get_parallel_plan",
    replacement=qwen3_moe_get_parallel_plan_patched,
    description="Register Qwen3Moe expert parallel plan for v5 generated modeling",
)

config.override_method(
    "Qwen3MoeAttention.forward",
    replacement=qwen3_moe_attention_forward_patched,
    description="Dispatch attention through the interned VeomniKernel",
)

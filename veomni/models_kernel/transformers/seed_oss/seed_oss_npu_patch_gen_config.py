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
# See the License for the specific language governing limitations
# under the License.
"""
Patch configuration for SeedOss NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.seed_oss.seed_oss_npu_patch_gen_config -o veomni/models_kernel/transformers/seed_oss/generated --diff

Mirrors the GPU consume. The models/ NPU path imported ``veomni.ops.kernels``
directly for RoPE / RMSNorm; those become local VeomniKernel calls.
"""

from veomni.models_kernel.transformers.seed_oss.seed_oss_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    seed_oss_attention_forward_patched,
    seed_oss_forcausallm_forward_patched,
    seed_oss_forcausallm_init_patched,
    seed_oss_mlp_forward_patched,
    seed_oss_mlp_init_patched,
    seed_oss_rmsnorm_forward_patched,
    seed_oss_rmsnorm_init_patched,
)
from veomni.models_kernel.transformers.seed_oss.seed_oss_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.seed_oss.modeling_seed_oss",
    target_file="patched_modeling_seed_oss_npu.py",
    description="SeedOss with VeomniKernel-based NPU kernel replacements",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)

config.override_method(
    "SeedOssRMSNorm.__init__",
    replacement=seed_oss_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "SeedOssRMSNorm.forward",
    replacement=seed_oss_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "SeedOssMLP.__init__",
    replacement=seed_oss_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniKernel",
)
config.override_method(
    "SeedOssMLP.forward",
    replacement=seed_oss_mlp_forward_patched,
    description="Always call the local swiglu_mlp VeomniKernel, then residual dropout",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)
config.override_method(
    "SeedOssForCausalLM.__init__",
    replacement=seed_oss_forcausallm_init_patched,
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
config.override_method(
    "SeedOssForCausalLM.forward",
    replacement=seed_oss_forcausallm_forward_patched,
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)

config.override_method(
    "SeedOssAttention.forward",
    replacement=seed_oss_attention_forward_patched,
    description="Dispatch attention through the interned VeomniKernel",
)

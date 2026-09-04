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
Patch configuration for GLM-MoE-DSA NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.glm_moe_dsa.glm_moe_dsa_npu_patch_gen_config -o veomni/models_kernel/transformers/glm_moe_dsa/generated --diff

CausalLM uses ``ForCausalLMLoss``. Indexer and attention stay on the
HuggingFace official modules.
"""

from veomni.models_kernel.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.models_kernel.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config import (
    glm_moe_dsa_forcausallm_forward_patched,
    glm_moe_dsa_forcausallm_init_patched,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_npu.py",
    description="GLM-MoE-DSA with VeomniKernel fused loss",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)

config.override_method(
    "GlmMoeDsaForCausalLM.__init__",
    replacement=glm_moe_dsa_forcausallm_init_patched,
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    replacement=glm_moe_dsa_forcausallm_forward_patched,
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)

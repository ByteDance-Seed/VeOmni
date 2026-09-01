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
Patch configuration for Qwen3 NPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3.qwen3_npu_patch_gen_config -o veomni/models_kernel/transformers/qwen3/generated --diff

This mirrors the GPU patch in
veomni/models_kernel/transformers/qwen3/qwen3_gpu_patch_gen_config.py.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_qwen3_npu.py".
"""

from veomni.models_kernel.transformers.qwen3.qwen3_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    qwen3_attention_forward_patched,
    qwen3_attention_init_patched,
    qwen3_forcausallm_forward_patched,
    qwen3_forcausallm_init_patched,
    qwen3_mlp_forward_patched,
    qwen3_mlp_init_patched,
    qwen3_rmsnorm_forward_patched,
    qwen3_rmsnorm_init_patched,
    qwen3_seq_cls_init_patched,
    qwen3forsequenceclassification_forward_patched,
)
from veomni.models_kernel.transformers.qwen3.qwen3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_npu.py",
    description="Qwen3 with VeomniKernel-based NPU kernel replacements",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)


config.override_method(
    "Qwen3RMSNorm.__init__",
    replacement=qwen3_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3RMSNorm.forward",
    replacement=qwen3_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniKernel",
)
config.override_method(
    "Qwen3MLP.__init__",
    replacement=qwen3_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniKernel",
)
config.override_method(
    "Qwen3MLP.forward",
    replacement=qwen3_mlp_forward_patched,
    description="Always call the local swiglu_mlp VeomniKernel",
)
config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="Always call rope full VeomniKernel",
)
config.override_method(
    "Qwen3Attention.__init__",
    replacement=qwen3_attention_init_patched,
    description="Construct local rope and attention VeomniKernels",
)
config.override_method(
    "Qwen3Attention.forward",
    replacement=qwen3_attention_forward_patched,
    description="Always call the local rope and attention VeomniKernels",
)
config.override_method(
    "Qwen3ForCausalLM.__init__",
    replacement=qwen3_forcausallm_init_patched,
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
config.override_method(
    "Qwen3ForCausalLM.forward",
    replacement=qwen3_forcausallm_forward_patched,
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)
config.override_method(
    "Qwen3ForSequenceClassification.__init__",
    replacement=qwen3_seq_cls_init_patched,
    description="Bind ForSequenceClassificationLoss to a local cross_entropy_loss VeomniKernel",
)
config.override_method(
    "Qwen3ForSequenceClassification.forward",
    replacement=qwen3forsequenceclassification_forward_patched,
    description="Always call self.loss_function (seq-cls helper + VeomniKernel)",
)

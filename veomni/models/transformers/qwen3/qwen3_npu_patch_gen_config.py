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
Patch configuration for Qwen3 NPU OpSlot-based kernel replacements.

Regen command:
patchgen veomni.models.transformers.qwen3.qwen3_npu_patch_gen_config -o veomni/models/transformers/qwen3/generated --diff

This mirrors the runtime GPU patch in
veomni/models/transformers/qwen3/qwen3_gpu_patch_gen_config.py.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_qwen3_npu.py".
"""

from veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config import (
    apply_rotary_pos_emb_patched,
    qwen3_forcausallm_forward_patched,
    qwen3_mlp_forward_patched,
    qwen3_rmsnorm_forward_patched,
    qwen3forsequenceclassification_forward_patched,
)
from veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_npu.py",
    description="Qwen3 with OpSlot-based NPU kernel replacements",
)

# Mirror additional imports + post-import helpers from the GPU config so the
# generated file is self-contained (same SP helpers, same rot_pos_ids /
# async ulysses / get_position_id helpers).
config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
# Propagate the GPU config's dropped imports (e.g. ``Qwen3CausalLMOutputWithPast``,
# now superseded by ``Qwen3CausalLMOutputWithLogProbs`` for the FSDP2-safe
# pre-backward unshard hook on ``lm_head``).
config.drop_imported_names.update(gpu_config.drop_imported_names)


# ── RMSNorm (OpSlot guard, functional NPU kernel) ──────────────────────────


config.override_method(
    "Qwen3RMSNorm.forward",
    replacement=qwen3_rmsnorm_forward_patched,
    description="OpSlot guard for NPU fused RMSNorm (standard formulation)",
)


# ── SwiGLU MLP (OpSlot guard, functional NPU kernel) ───────────────────────


config.override_method(
    "Qwen3MLP.forward",
    replacement=qwen3_mlp_forward_patched,
    description="OpSlot guard for NPU fused SwiGLU MLP",
)


# ── Rotary Positional Embedding (OpSlot guard) ───────────────────────────────


config.replace_function(
    "apply_rotary_pos_emb",
    replacement=apply_rotary_pos_emb_patched,
    description="OpSlot guard for NPU fused RoPE",
)


# ── Qwen3ForCausalLM.forward (fused cross-entropy via OpSlot) ────────────────


config.override_method(
    "Qwen3ForCausalLM.forward",
    replacement=qwen3_forcausallm_forward_patched,
    description="OpSlot guard for fused cross entropy in Qwen3ForCausalLM.forward",
)

# ── Qwen3ForSequenceClassification.forward (fused cross-entropy via OpSlot) ──


config.override_method(
    "Qwen3ForSequenceClassification.forward",
    replacement=qwen3forsequenceclassification_forward_patched,
    description="OpSlot guard for fused cross entropy in Qwen3ForSequenceClassification.forward",
)

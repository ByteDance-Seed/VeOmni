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
Patch configuration for GLM-MoE-DSA NPU VeomniOp replacements.

Regen command:
patchgen veomni.models.transformers.glm_moe_dsa.glm_moe_dsa_npu_patch_gen_config -o veomni/models/transformers/glm_moe_dsa/generated --diff

CausalLM uses ``ForCausalLMLoss``. Attention reuses the GPU
``dsa_attention`` / ``glm`` VeomniOp patches. Indexer binds
``rope`` / ``interleave`` and keeps the Hugging Face scoring path.
"""

import torch
import torch.nn.functional as F
from transformers.cache_utils import Cache

from veomni.models.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.models.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config import (
    glm_moe_dsa_attention_forward_patched,
    glm_moe_dsa_attention_init_patched,
    glm_moe_dsa_forcausallm_forward_patched,
    glm_moe_dsa_forcausallm_init_patched,
    glm_moe_dsa_get_parallel_plan_patched,
    glm_moe_dsa_mlp_forward_patched,
    glm_moe_dsa_mlp_init_patched,
    glm_moe_dsa_rmsnorm_forward_patched,
    glm_moe_dsa_rmsnorm_init_patched,
)
from veomni.ops import VeomniOp
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_npu.py",
    description="GLM-MoE-DSA with VeomniOp fused loss",
)

config.additional_imports.extend(gpu_config.additional_imports)
config.post_import_blocks.extend(gpu_config.post_import_blocks)
config.helpers.extend(gpu_config.helpers)
config.drop_imported_names.update(gpu_config.drop_imported_names)
config.exclude_from_output("apply_rotary_pos_emb_interleave", "use_kernel_forward_from_hub")


def glm_moe_dsa_npu_indexer_bind_rope(original_init, self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    self.veomni_rope = VeomniOp("rope", "interleave", "eager")


def glm_moe_dsa_npu_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    past_key_values: Cache | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings
    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

    k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

    q_rot, k_rot = self.veomni_rope(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

    if past_key_values is not None:
        k = past_key_values.update_indexer(k, self.layer_idx)

    scores = torch.matmul(q.float(), k.transpose(-1, -2).float().unsqueeze(1)) * self.softmax_scale
    scores = F.relu(scores)

    weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
    index_scores = torch.matmul(weights.unsqueeze(-2), scores).squeeze(-2)

    if attention_mask is not None:
        index_scores = index_scores + attention_mask
    else:
        key_positions = torch.arange(index_scores.shape[-1], device=index_scores.device)
        causal = key_positions[None, None, :] > position_ids[:, :, None]
        index_scores = index_scores.masked_fill(causal, float("-inf"))

    topk = min(self.index_topk, index_scores.shape[-1])
    return index_scores.topk(topk, dim=-1).indices.to(torch.int32)


config.override_method(
    "GlmMoeDsaRMSNorm.__init__",
    replacement=glm_moe_dsa_rmsnorm_init_patched,
    description="Construct a local rms_norm VeomniOp",
)
config.override_method(
    "GlmMoeDsaRMSNorm.forward",
    replacement=glm_moe_dsa_rmsnorm_forward_patched,
    description="Always call the local rms_norm VeomniOp",
)
config.override_method(
    "GlmMoeDsaMLP.__init__",
    replacement=glm_moe_dsa_mlp_init_patched,
    description="Construct a local swiglu_mlp VeomniOp",
)
config.override_method(
    "GlmMoeDsaMLP.forward",
    replacement=glm_moe_dsa_mlp_forward_patched,
    description="Call swiglu_mlp for silu/swish, otherwise self.act_fn",
)
config.modify_init(
    "GlmMoeDsaIndexer",
    replacement=glm_moe_dsa_npu_indexer_bind_rope,
    description="Bind instance-local interleave rope VeomniOp",
)
config.override_method(
    "GlmMoeDsaIndexer.forward",
    replacement=glm_moe_dsa_npu_indexer_forward_patched,
    description="Call interleave rope; keep Hugging Face indexer scoring",
)
config.override_method(
    "GlmMoeDsaAttention.__init__",
    replacement=glm_moe_dsa_attention_init_patched,
    description="Construct a local dsa_attention glm VeomniOp",
)
config.override_method(
    "GlmMoeDsaAttention.forward",
    replacement=glm_moe_dsa_attention_forward_patched,
    description="DSA consumes compressed K/V from past_key_values.update(), not module buffers",
)
config.override_method(
    "GlmMoeDsaForCausalLM.__init__",
    replacement=glm_moe_dsa_forcausallm_init_patched,
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniOp",
)
config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    replacement=glm_moe_dsa_forcausallm_forward_patched,
    description="Always call self.loss_function (ForCausalLMLoss + VeomniOp)",
)
config.override_method(
    "GlmMoeDsaForCausalLM.get_parallel_plan",
    replacement=glm_moe_dsa_get_parallel_plan_patched,
    description="Register GLM-MoE-DSA expert parallel plan for v5 generated modeling",
)

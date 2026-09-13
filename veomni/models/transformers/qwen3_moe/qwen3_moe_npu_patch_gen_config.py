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
Patch configuration for Qwen3Moe NPU/SP patched modeling generation.

Regen command:
patchgen veomni.models.transformers.qwen3_moe.qwen3_moe_npu_patch_gen_config -o veomni/models/transformers/qwen3_moe/generated --diff

This keeps only the needed v5 patches:
1. NPU replacements for rotary/rms_norm/mlp.
2. Fused loss path in Qwen3MoeForCausalLM.forward.
3. Register get_parallel_plan on Qwen3MoeForCausalLM.
"""

from transformers.modeling_layers import GradientCheckpointingLayer

from veomni.models.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config import (
    PatchedQwen3MoeExperts,
    apply_rotary_pos_emb_patched,
    qwen3_moe_forcausallm_forward_patched,
    qwen3_moe_get_parallel_plan_patched,
    qwen3_moe_mlp_forward_patched,
    qwen3_moe_model_forward_patched,
    qwen3_moe_rmsnorm_forward_patched,
    qwen3_moe_topk_router_forward_patched,
)
from veomni.models.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config import (
    config as gpu_config,
)
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
    target_file="patched_modeling_qwen3_moe_npu.py",
    description="Qwen3Moe with NPU replacements and VeOmni SP/fused loss patches",
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


# ── Decoder layer split for inter-layer replay ─────────────────────────────


@config.replace_class(
    "Qwen3MoeDecoderLayer",
    description="Expose attention and MLP phases for NPU inter-layer replay scheduling",
)
class PatchedQwen3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)
        self.input_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.hidden_size = config.hidden_size

    def forward_attention(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return residual + hidden_states

    def forward_mlp(self, hidden_states):
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        position_embeddings=None,
        **kwargs,
    ):
        hidden_states = self.forward_attention(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        return self.forward_mlp(hidden_states)


# ── RMSNorm (OpSlot guard, functional NPU kernel) ──────────────────────────


config.override_method(
    "Qwen3MoeRMSNorm.forward",
    replacement=qwen3_moe_rmsnorm_forward_patched,
    description="OpSlot guard for NPU fused RMSNorm (standard formulation)",
)

# ── SwiGLU MLP (OpSlot guard, functional NPU kernel) ───────────────────────


config.override_method(
    "Qwen3MoeMLP.forward",
    replacement=qwen3_moe_mlp_forward_patched,
    description="OpSlot guard for NPU fused SwiGLU MLP",
)


config.replace_class(
    "Qwen3MoeExperts",
    replacement=PatchedQwen3MoeExperts,
    description="Use v5 gate_up_proj expert weights and explicit VeOmni fused MoE path",
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
    description="OpSlot guard for NPU fused RoPE",
)

# Dummy reference resolved at codegen time from the generated module.
rotate_half = None  # noqa: E305


config.override_method(
    "Qwen3MoeModel.forward",
    replacement=qwen3_moe_model_forward_patched,
    description="Support SP in Qwen3MoeModel.forward",
)


config.override_method(
    "Qwen3MoeForCausalLM.forward",
    replacement=qwen3_moe_forcausallm_forward_patched,
    description="Support fused cross entropy path in Qwen3MoeForCausalLM.forward",
)


config.override_method(
    "Qwen3MoeForCausalLM.get_parallel_plan",
    replacement=qwen3_moe_get_parallel_plan_patched,
    description="Register Qwen3Moe expert parallel plan for v5 generated modeling",
)

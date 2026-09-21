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
Patch configuration for GPT-OSS GPU patched modeling generation.

Regen command:
patchgen veomni.models.transformers.gpt_oss.gpt_oss_gpu_patch_gen_config -o veomni/models/transformers/gpt_oss/generated --diff

Patches:
1. ``GptOssPreTrainedModel`` — extends HF's compatible flash-attention list
   with VeOmni's FA4+SP implementation name, preventing the Transformers init
   check from falling back to hub kernels.
2. ``GptOssExperts`` — drops the upstream hub expert decorator so VeOmni owns
   the MoE dispatch point. The exact eager GPT-OSS math is preserved, and a
   GPT-OSS-specific fused grouped-GEMM path is exposed through OpSlot.
3. ``GptOssMLP`` — drops the upstream ``MegaBlocksMoeMLP`` hub decorator so
   the patched ``GptOssExperts`` module is always the MoE implementation.
4. ``GptOssForCausalLM.get_parallel_plan`` — exposes the expert-parallel plan
   for GPT-OSS expert weights and biases.
5. ``GptOssForCausalLM.forward`` — OpSlot guard for fused cross entropy and
   VeOmni's ``MoeCausalLMOutputWithLogProbs`` output contract.

GPT-OSS attention already passes ``sliding_window`` and learnable sink
(``s_aux=self.sinks``) through Transformers' attention interface, so FA4 support
is provided by VeOmni's global attention registry patch.
"""

import torch
from torch import nn
from transformers import initialization as init
from transformers.cache_utils import Cache
from transformers.modeling_outputs import MoeModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import (
    GptOssAttention,
    GptOssDecoderLayer,
    GptOssTopKRouter,
    load_balancing_loss_func,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.output_capturing import OutputRecorder

from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs


config = PatchConfig(
    source_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_file="patched_modeling_gpt_oss_gpu.py",
    description="GPT-OSS with VeOmni FA4-compatible attention dispatch and fused-loss contract",
)

config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.drop_import_names("MoeCausalLMOutputWithPast")

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_moe_experts_forward = OpSlot("moe_experts", "gpt_oss")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_load_balancing_loss = OpSlot("load_balancing_loss", "standard")
    """
)


@config.replace_class(
    "GptOssPreTrainedModel",
    description="Allow VeOmni FA4 implementation names during Transformers attention backend validation",
)
@auto_docstring
class PatchedGptOssPreTrainedModel(PreTrainedModel):
    config: GptOssConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["GptOssDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "router_logits": OutputRecorder(GptOssTopKRouter, index=0),
        "hidden_states": GptOssDecoderLayer,
        "attentions": GptOssAttention,
    }
    _keep_in_fp32_modules = ["post_attention_layernorm", "input_layernorm", "norm"]
    _compatible_flash_implementations = [
        "kernels-community/vllm-flash-attn3",
        "flash_attention_4",
        "veomni_flash_attention_4_with_sp",
    ]

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, GptOssExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.zeros_(module.gate_up_proj_bias)
            init.normal_(module.down_proj, mean=0.0, std=std)
            init.zeros_(module.down_proj_bias)
        elif isinstance(module, GptOssAttention):
            init.normal_(module.sinks, mean=0.0, std=std)
        elif isinstance(module, GptOssTopKRouter):
            init.normal_(module.weight, mean=0.0, std=std)
            init.normal_(module.bias, mean=0.0, std=std)


@config.replace_class(
    "GptOssExperts",
    description="Drop upstream expert hub decorator and expose VeOmni MoE dispatch guard",
)
class PatchedGptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.intermediate_size, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0

    def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
        gate, up = gate_up[..., ::2], gate_up[..., 1::2]
        gate = gate.clamp(min=None, max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        glu = gate * torch.sigmoid(gate * self.alpha)
        gated_output = (up + 1) * glu
        return gated_output

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        next_states = torch.zeros_like(hidden_states, dtype=hidden_states.dtype, device=hidden_states.device)

        if veomni_moe_experts_forward.use_non_eager_impl:
            return veomni_moe_experts_forward(self, hidden_states, router_indices, routing_weights)
        if not veomni_moe_experts_forward.use_eager_impl:
            raise RuntimeError(
                "GPT-OSS MoE experts have no implementation bound. "
                "Set moe_implementation='eager' to use the HuggingFace reference path, "
                "or set moe_implementation to a supported fused backend such as 'fused_triton' or 'fused_quack'."
            )

        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
            gated_output = self._apply_gate(gate_up)
            out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
            weighted_output = out * routing_weights[token_idx, top_k_pos, None]
            next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

        return next_states


@config.replace_class(
    "GptOssMLP",
    description="Drop upstream MegaBlocks hub decorator and route through patched GptOssExperts",
)
class PatchedGptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores


@config.override_method(
    "GptOssForCausalLM.get_parallel_plan",
    description="Expose GPT-OSS expert-parallel plan",
)
def gpt_oss_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()


@config.override_method(
    "GptOssForCausalLM.forward",
    description="Support VeOmni fused cross entropy contract in GptOssForCausalLM.forward",
)
def gpt_oss_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithLogProbs:
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    outputs: MoeModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_router_logits=output_router_logits,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
    hidden_states = hidden_states[:, slice_indices, :]

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            loss, _, fused_linear_aux = self.loss_function(
                logits=None,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
    else:
        logits = self.lm_head(hidden_states)

    aux_loss = None
    if output_router_logits:
        if veomni_load_balancing_loss.use_non_eager_impl:
            aux_loss = veomni_load_balancing_loss(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        else:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits,
                self.num_experts,
                self.num_experts_per_tok,
                attention_mask,
            )
        if labels is not None and loss is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )

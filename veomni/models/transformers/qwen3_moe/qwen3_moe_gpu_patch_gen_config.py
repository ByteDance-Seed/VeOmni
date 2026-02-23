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
Patch configuration for Qwen3Moe GPU/SP patched modeling generation.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3_moe.qwen3_moe_gpu_patch_gen_config -o veomni/models/transformers/qwen3_moe/generated --diff

This keeps only the needed v5 patches:
1. Liger replacements for rotary/rms_norm/mlp.
2. SP slicing in Qwen3MoeModel.forward.
3. Fused loss path in Qwen3MoeForCausalLM.forward.
4. Register get_parallel_plan on Qwen3MoeForCausalLM.
"""

from typing import Optional

import torch
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import MoeCausalLMOutputWithPast, MoeModelOutputWithPast
from transformers.models.qwen3_moe.modeling_qwen3_moe import load_balancing_loss_func
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.ops import fused_moe_forward
from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external


config = PatchConfig(
    source_module="transformers.models.qwen3_moe.modeling_qwen3_moe",
    target_file="patched_modeling_qwen3_moe_gpu.py",
    description="Qwen3Moe with LigerKernel GPU replacements and VeOmni SP/fused loss patches",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.distributed.sequence_parallel", names=["slice_position_embedding"])
config.add_import("veomni.ops", names=["fused_moe_forward"])

config.patches.append(
    create_patch_from_external(
        target="Qwen3MoeRMSNorm",
        replacement_module="liger_kernel.transformers.rms_norm",
        replacement_name="LigerRMSNorm",
        description="Use LigerKernel RMSNorm",
    )
)

config.patches.append(
    create_patch_from_external(
        target="Qwen3MoeMLP",
        replacement_module="liger_kernel.transformers.swiglu",
        replacement_name="LigerSwiGLUMLP",
        description="Use LigerKernel SwiGLU MLP",
    )
)


@config.replace_class(
    "Qwen3MoeExperts", description="Use legacy gate/up/down expert weights and explicit VeOmni fused MoE path"
)
class PatchedQwen3MoeExperts(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.up_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = torch.nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]
        self._moe_implementation = getattr(config, "_moe_implementation", "eager")

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        if self._moe_implementation == "eager":
            with torch.no_grad():
                expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
                expert_mask = expert_mask.permute(2, 1, 0)
                expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

            for expert_idx in expert_hit:
                expert_idx = expert_idx[0]
                if expert_idx == self.num_experts:
                    continue
                top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
                current_state = hidden_states[token_idx]
                gate = torch.nn.functional.linear(current_state, self.gate_proj[expert_idx])
                up = torch.nn.functional.linear(current_state, self.up_proj[expert_idx])
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = torch.nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        elif self._moe_implementation == "fused":
            final_hidden_states = fused_moe_forward(
                module=self,
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(final_hidden_states.dtype),
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


@config.replace_function("apply_rotary_pos_emb", description="Use LigerKernel rotary embedding")
def apply_rotary_pos_emb_liger(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    return liger_rotary_pos_emb(
        q,
        k,
        cos,
        sin,
        position_ids=position_ids,
        unsqueeze_dim=unsqueeze_dim,
    )


@config.override_method("Qwen3MoeModel.forward", description="Support SP in Qwen3MoeModel.forward")
def qwen3_moe_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    mask_function = create_causal_mask if self.config.sliding_window is None else create_sliding_window_causal_mask
    causal_mask = mask_function(
        config=self.config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)

    # ============================== VeOmni SP Patch Start ==============================
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    # =============================== VeOmni SP Patch End ===============================

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@config.override_method(
    "Qwen3MoeForCausalLM.forward",
    description="Support fused cross entropy path in Qwen3MoeForCausalLM.forward",
)
def qwen3_moe_forcausal_lm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    output_router_logits: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithPast:
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
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    if labels is not None:
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


@config.override_method(
    "Qwen3MoeForCausalLM.get_parallel_plan",
    description="Register Qwen3Moe expert parallel plan for v5 generated modeling",
)
def qwen3_moe_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()

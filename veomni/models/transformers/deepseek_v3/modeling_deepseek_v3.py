from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
import transformers.models.deepseek_v3.modeling_deepseek_v3 as hf_deepseek_v3
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.deepseek_v3.modeling_deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3MLP,
    apply_rotary_pos_emb,
    apply_rotary_pos_emb_interleave,
    eager_attention_forward,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import slice_position_embedding
from ....ops import fused_moe_forward
from ....utils import logging
from ....utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ....utils.import_utils import (
    is_transformers_version_greater_or_equal_to,
)
from ...transformers.attention_utils import VARLEN_ATTENTION_TYPES


logger = logging.get_logger(__name__)


# ================================================================
# Patch: DeepseekV3TopkRouter, DeepseekV3NaiveMoe
# 1. Patch to merge ckpt and align with transformers v5, in case upgrade to v5.0.0 later.
#    https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/deepseek_v3/modeling_deepseek_v3.py
# 2. Support init weight function for experts and gate. Also will be
#    align with transformers v5.0.0, just temporary in transformers v4.57.3.
# 3. Add fused moe implementation with triton.
# ================================================================
def _init_weight(
    tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0, generator: torch.Generator | None = None
) -> torch.Tensor:
    if not getattr(tensor, "_is_hf_initialized", False):
        return torch.nn.init.normal_(tensor, mean=mean, std=std, generator=generator)
    return tensor


class PatchDeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits


class PatchDeepseekV3NaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.up_proj = nn.Parameter(torch.empty(self.num_experts, self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
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
                gate = nn.functional.linear(current_state, self.gate_proj[expert_idx])
                up = nn.functional.linear(current_state, self.up_proj[expert_idx])
                current_hidden_states = self.act_fn(gate) * up
                current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
                current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
                final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        elif self._moe_implementation == "fused":
            # cast top_k_weights to dtype of final_hidden_states
            top_k_weights = top_k_weights.to(final_hidden_states.dtype)

            final_hidden_states = fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights,
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )
        else:
            raise ValueError(f"Invalid moe implementation: {self._moe_implementation}")

        return final_hidden_states


class PatchDeepseekV3MoE(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = PatchDeepseekV3NaiveMoe(config)
        self.gate = PatchDeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

        _init_weight(self.experts.gate_proj)
        _init_weight(self.experts.up_proj)
        _init_weight(self.experts.down_proj)
        _init_weight(self.gate.weight)

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# ================================================================
# PATCH: DeepseekV3Attention.forward
# 1. Support for veomni attention implementation
# ================================================================
def deepseek_v3_attention_forward(
    self: DeepseekV3Attention,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[tuple[torch.Tensor]]]:
    batch_size, seq_length = hidden_states.shape[:-1]
    query_shape = (batch_size, seq_length, -1, self.qk_head_dim)
    key_shape = (batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)

    if self.q_lora_rank is None:
        q_states = self.q_proj(hidden_states)
    else:
        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
    q_states = q_states.view(query_shape).transpose(1, 2)
    q_pass, q_rot = torch.split(q_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_pass, k_rot = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)

    k_pass = self.kv_b_proj(self.kv_a_layernorm(k_pass)).view(key_shape).transpose(1, 2)
    k_pass, value_states = torch.split(k_pass, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

    k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    cos, sin = position_embeddings
    if self.config.rope_interleave:  # support using interleaved weights for efficiency
        q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin)
    else:
        q_rot, k_rot = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    k_rot = k_rot.expand(*k_pass.shape[:-1], -1)

    query_states = torch.cat((q_pass, q_rot), dim=-1)
    key_states = torch.cat((k_pass, k_rot), dim=-1)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # --- Patch.1 ---
    # Flash Attention requires Q/K and V to have the same head dimension on non-Hopper GPUs.
    # For DeepSeek V3 MLA architecture where qk_head_dim != v_head_dim, we pad V to match Q/K.
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])
    # --- Patch.1 ---

    attention_interface: Callable = eager_attention_forward
    if self.config._attn_implementation != "eager":
        attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # --- Patch.1 ---
    # Truncate the output back to the original v_head_dim after Flash Attention.
    if self.config._attn_implementation in VARLEN_ATTENTION_TYPES and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]
    # --- Patch.1 ---

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


# ================================================================
# PATCH: DeepseekV3Model.forward
# 1. Support SP
# ================================================================
def deepseek_v3_model_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = create_causal_mask(
        config=self.config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # --- Patch.1 ---
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    # --- Patch.1 ---

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ================================================================
# PATCH: DeepseekV3ForCausalLM.forward
# 1. Support use with fuse cross_entropy loss function.
# ================================================================
def deepseek_v3_forcausal_lm_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    r"""
    Example:

    ```python
    >>> from transformers import AutoTokenizer, DeepseekV3ForCausalLM

    >>> model = DeepseekV3ForCausalLM.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-deepseek_v3/DeepseekV3-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    # --- Patch.1 ---
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
    # --- Patch.1 ---

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def apply_veomni_deepseek_v3_patch():
    logger.info_rank0("Apply VeOmni patch to deepseek_v3.")

    hf_deepseek_v3.DeepseekV3MoE = PatchDeepseekV3MoE
    from .parallel_plan import get_parallel_plan

    hf_deepseek_v3.DeepseekV3ForCausalLM.get_parallel_plan = lambda self: get_parallel_plan()

    hf_deepseek_v3.DeepseekV3Attention.forward = deepseek_v3_attention_forward
    hf_deepseek_v3.DeepseekV3Model.forward = deepseek_v3_model_forward
    hf_deepseek_v3.DeepseekV3ForCausalLM.forward = deepseek_v3_forcausal_lm_forward

    if IS_CUDA_AVAILABLE:
        from .gpu_patch import apply_veomni_deepseek_v3_gpu_patch

        apply_veomni_deepseek_v3_gpu_patch()
    elif IS_NPU_AVAILABLE and is_transformers_version_greater_or_equal_to("4.50.4"):
        from .npu_patch import apply_deepseek_v3_npu_patch

        apply_deepseek_v3_npu_patch()
    else:
        logger.warning_rank0(
            "DeepseekV3ForCausalLM in VeOmni only support CUDA or NPU with transformers version >= 4.50.4."
        )

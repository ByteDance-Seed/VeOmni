# Copyright 2025 The Qwen team, Alibaba Group and the HuggingFace Inc. team. All rights reserved.
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

from typing import Optional, Union

import torch
import torch.nn.functional as F
import transformers.models.qwen3_moe.modeling_qwen3_moe as hf_qwen3_moe
from torch import nn
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoeModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
)

from ....distributed.parallel_state import get_parallel_state
from ....distributed.sequence_parallel import slice_position_embedding
from ....ops import fused_moe_forward
from ....utils import logging
from ....utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from ....utils.import_utils import (
    is_liger_kernel_available,
    is_transformers_version_greater_or_equal_to,
)


if is_liger_kernel_available():
    pass

logger = logging.get_logger(__name__)


# ================================================================
# PATCH: PatchQwen3MoeExperts, PatchQwen3MoeTopKRouter, PatchQwen3MoeSparseMoeBlock
# 1. Patch to merge ckpt and align with transformers v5, in case upgrade to v5.0.0 later.
#    https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
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


class PatchQwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
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


class PatchQwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_logits = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices


class PatchQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = PatchQwen3MoeExperts(config)
        self.gate = PatchQwen3MoeTopKRouter(config)

        # --- Patch.2 ---
        _init_weight(self.experts.gate_proj)
        _init_weight(self.experts.up_proj)
        _init_weight(self.experts.down_proj)

        _init_weight(self.gate.weight)
        # --- Patch.2 ---

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


# ================================================================
# PATCH: Qwen3MoeModel.forward
# 1. Support SP
# ================================================================
def qwen3_moe_model_forward(
    self: Qwen3MoeModel,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
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
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )

    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)
    # --- Patch.1 ---
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    # --- Patch.1 ---

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return MoeModelOutputWithPast(  # only diff with Mistral is the output type, we need MoE
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


# ================================================================
# PATCH: Qwen3MoeForCausalLM.forward
# 1. Support use with fuse cross_entropy loss function.
# ================================================================
def qwen3_moe_forcausal_lm_forward(
    self: Qwen3MoeForCausalLM,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: Union[int, torch.Tensor] = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> MoeCausalLMOutputWithPast:
    r"""
    labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
        config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
        (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Example:

    ```python
    >>> from transformers import AutoTokenizer, Qwen3MoeForCausalLM

    >>> model = Qwen3MoeForCausalLM.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-MoE-15B-A2B")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""

    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
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

    aux_loss = None
    if output_router_logits:
        aux_loss = hf_qwen3_moe.load_balancing_loss_func(
            outputs.router_logits,
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )


def apply_veomni_qwen3_moe_patch():
    logger.info_rank0("Apply VeOmni patch to qwen3_moe.")

    hf_qwen3_moe.Qwen3MoeSparseMoeBlock = PatchQwen3MoeSparseMoeBlock
    from .parallel_plan import get_parallel_plan

    hf_qwen3_moe.Qwen3MoeForCausalLM.get_parallel_plan = lambda self: get_parallel_plan()

    hf_qwen3_moe.Qwen3MoeModel.forward = qwen3_moe_model_forward
    hf_qwen3_moe.Qwen3MoeForCausalLM.forward = qwen3_moe_forcausal_lm_forward

    if IS_CUDA_AVAILABLE:
        from .gpu_patch import apply_veomni_qwen3_moe_gpu_patch

        apply_veomni_qwen3_moe_gpu_patch()
    elif IS_NPU_AVAILABLE and is_transformers_version_greater_or_equal_to("4.50.4"):
        from .npu_patch import apply_qwen3_moe_npu_patch

        apply_qwen3_moe_npu_patch()
    else:
        logger.warning_rank0(
            "Qwen3ForCausalLM in VeOmni only support CUDA or NPU with transformers version >= 4.50.4."
        )

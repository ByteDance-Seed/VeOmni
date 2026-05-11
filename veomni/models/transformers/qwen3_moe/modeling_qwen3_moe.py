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
from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM, Qwen3MoePreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    TransformersKwargs,
)

from ....ops import fused_moe_forward
from ....ops.dispatch import OpSlot
from ....utils import logging
from ....utils.model_outputs import MoeCausalLMOutputWithLogProbs


logger = logging.get_logger(__name__)


# Module-level OpSlot bound by `_bind_veomni_ops` in `auto.py` after the model
# is constructed. `use_non_eager_impl` flips to True when the user selects a
# fused MoE backend in `OpsImplementationConfig.moe_implementation`; the slot
# is not invoked directly because v4 storage (split gate_proj/up_proj/down_proj)
# does not match the universal `_make_moe_experts_adapter` signature.
veomni_moe_experts_forward = OpSlot("moe_experts", "standard")


# ================================================================
# PATCH: PatchQwen3MoeExperts, PatchQwen3MoeTopKRouter, PatchQwen3MoeSparseMoeBlock
# 1. Patch to merge ckpt and align with transformers v5, in case upgrade to v5.0.0 later.
#    https://github.com/huggingface/transformers/blob/v5.0.0/src/transformers/models/qwen3_moe/modeling_qwen3_moe.py
# 2. Support init weight function for experts and gate. Also will be
#    align with transformers v5.0.0, just temporary in transformers v4.57.3.
# 3. Add fused moe implementation with triton via OpSlot dispatch.
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)

        if veomni_moe_experts_forward.use_non_eager_impl:
            return fused_moe_forward(
                num_experts=self.num_experts,
                routing_weights=top_k_weights.to(final_hidden_states.dtype),
                selected_experts=top_k_index,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )

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

        return final_hidden_states


class PatchQwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.norm_topk_prob = config.norm_topk_prob
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

        # Always register the monitoring hook; it's a no-op when no monitor is active.
        from veomni.utils.moe_monitor import router_forward_hook

        self.register_forward_hook(router_forward_hook)

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        # Return raw pre-softmax logits as `router_logits`; HF's
        # `load_balancing_loss_func` applies softmax internally. The post-softmax
        # tensor is kept locally as `routing_weights` for top-k selection only.
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        routing_weights = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)  # (seq_len, top_k)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        # Cast top-k weights back to input dtype so the downstream expert
        # multiplication runs in bf16/fp16 rather than fp32.
        router_top_value = router_top_value.to(input_dtype)
        return router_logits, router_top_value, router_indices


class PatchQwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = PatchQwen3MoeExperts(config)
        self.gate = PatchQwen3MoeTopKRouter(config)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)
        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        final_hidden_states = self.experts(hidden_states_reshaped, selected_experts, routing_weights)
        return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


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
) -> MoeCausalLMOutputWithLogProbs:
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
    log_probs = None
    entropy = None
    if labels is not None:
        loss, logits, log_probs, entropy = self.loss_function(
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

    return MoeCausalLMOutputWithLogProbs(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
        log_probs=log_probs,
        entropy=entropy,
    )


# ================================================================
# PATCH: Qwen3MoePreTrainedModel._init_weights
# Wrap (not replace) HF's _init_weights so standard nn.Linear / nn.Embedding /
# RMSNorm modules keep HF's init, and PatchQwen3MoeExperts / PatchQwen3MoeTopKRouter
# (whose params are not handled by HF's _init_weights) get N(0, initializer_range).
# ================================================================
# Snapshot HF's `_init_weights` at module import; the wrapper always wraps this
# fixed reference so repeated `apply_veomni_qwen3_moe_patch()` calls (e.g.
# multiple `get_model_class()` in one process) stay idempotent.
_HF_QWEN3_MOE_INIT_WEIGHTS = hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights


def _make_qwen3_moe_init_weights(orig_init_weights):
    @torch.no_grad()
    def qwen3_moe_pretrained_model_init_weights(self: Qwen3MoePreTrainedModel, module):
        orig_init_weights(self, module)
        if isinstance(module, PatchQwen3MoeExperts):
            _init_weight(module.gate_proj, mean=0.0, std=self.config.initializer_range)
            _init_weight(module.up_proj, mean=0.0, std=self.config.initializer_range)
            _init_weight(module.down_proj, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, PatchQwen3MoeTopKRouter):
            _init_weight(module.weight, mean=0.0, std=self.config.initializer_range)

    return qwen3_moe_pretrained_model_init_weights


def apply_veomni_qwen3_moe_patch():
    logger.info_rank0("Apply VeOmni patch to qwen3_moe.")

    hf_qwen3_moe.Qwen3MoeSparseMoeBlock = PatchQwen3MoeSparseMoeBlock
    from .parallel_plan import get_parallel_plan

    hf_qwen3_moe.Qwen3MoeForCausalLM.get_parallel_plan = lambda self: get_parallel_plan(use_gate_up_proj=False)

    hf_qwen3_moe.Qwen3MoeForCausalLM.forward = qwen3_moe_forcausal_lm_forward
    # Wrap the import-time snapshot, not the live attribute, so this is idempotent.
    hf_qwen3_moe.Qwen3MoePreTrainedModel._init_weights = _make_qwen3_moe_init_weights(_HF_QWEN3_MOE_INIT_WEIGHTS)

    # Mirror the OpSlot onto the HF module so `_bind_veomni_ops` (which walks
    # `sys.modules[model.__class__.__module__]`) can find it. On transformers v4
    # the model class still resolves to the upstream HF module, even though its
    # SparseMoeBlock has been monkey-patched to our Patch* classes. The mirrored
    # attribute is the same Python object — binding mutates state seen by the
    # closure inside PatchQwen3MoeExperts.forward.
    hf_qwen3_moe.veomni_moe_experts_forward = veomni_moe_experts_forward

    from .device_patch import apply_veomni_qwen3_moe_device_patch

    apply_veomni_qwen3_moe_device_patch()

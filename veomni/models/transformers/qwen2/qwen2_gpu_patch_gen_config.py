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
Patch configuration for Qwen2 GPU LigerKernel replacements.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen2.qwen2_gpu_patch_gen_config -o veomni/models/transformers/qwen2/generated

This mirrors the current runtime Qwen2 patches while moving the v5 path to an
explicit patched modeling module.
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import CausalLMOutputWithLogProbs  # noqa: F401  re-emitted into generated file


config = PatchConfig(
    source_module="transformers.models.qwen2.modeling_qwen2",
    target_file="patched_modeling_qwen2_gpu.py",
    description="Qwen2 with OpSlot-based GPU kernel replacements",
)

# Surface ``CausalLMOutputWithLogProbs`` in the generated file so the patched
# ``forward`` can return per-token log-probs in the unified output dataclass.
config.add_import("veomni.utils.model_outputs", names=["CausalLMOutputWithLogProbs"])

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    """
)


# ── RMSNorm (OpSlot guard, functional Liger kernel) ──────────────────────────


@config.override_method(
    "Qwen2RMSNorm.forward",
    description="OpSlot guard for Liger fused RMSNorm (standard formulation)",
)
def qwen2_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Modification: OpSlot guard — use fused RMSNorm kernel when bound.
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    # Original HF code below, unchanged.
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


# ── SwiGLU MLP (OpSlot guard, functional Liger kernel) ───────────────────────


@config.override_method(
    "Qwen2MLP.forward",
    description="OpSlot guard for Liger fused SwiGLU MLP",
)
def qwen2_mlp_forward_patched(self, x):
    # Modification: OpSlot guard — use fused SwiGLU kernel when bound.
    if veomni_swiglu_mlp.use_non_eager_impl:
        return veomni_swiglu_mlp(self, x)
    # Original HF code below, unchanged.
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


# ── Rotary Positional Embedding (OpSlot guard) ───────────────────────────────


@config.replace_function("apply_rotary_pos_emb", description="OpSlot guard for Liger fused RoPE")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids: Optional[torch.Tensor] = None,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Modification: OpSlot guard — use fused RoPE kernel when bound.
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, position_ids=position_ids, unsqueeze_dim=unsqueeze_dim)
    # Original HF code below, unchanged.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# Dummy reference resolved at codegen time from the generated module.
rotate_half = None  # noqa: E305


@config.override_method("Qwen2Model.forward", description="Support SP in Qwen2Model.forward")
def qwen2_model_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for decoder_layer in self.layers[: self.config.num_hidden_layers]:
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[decoder_layer.attention_type],
            position_embeddings=position_embeddings,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values if use_cache else None,
    )


@config.override_method(
    "Qwen2ForCausalLM.forward",
    description="Support fused cross entropy path in Qwen2ForCausalLM.forward",
)
def qwen2forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    cache_position: torch.LongTensor | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    outputs = self.model(
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
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    log_probs = None
    entropy = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, log_probs, entropy = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, log_probs, entropy = self.loss_function(
                logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs
            )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        log_probs=log_probs,
        entropy=entropy,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

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
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_npu_patch_gen_config -o veomni/models/transformers/qwen3/generated --diffs

This mirrors the runtime NPU patch in
veomni/models/transformers/qwen3/npu_patch.py.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_qwen3_npu.py".
"""

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_npu.py",
    description="Qwen3 with OpSlot-based NPU kernel replacements",
)

config.add_import("transformers.modeling_outputs", names=["SequenceClassifierOutputWithPast"])

config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # These are bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot
    veomni_rms_norm = OpSlot("rms_norm", "standard")
    veomni_apply_rotary_pos_emb = OpSlot("rotary_pos_emb", "full")
    veomni_swiglu_mlp = OpSlot("swiglu_mlp", "standard")
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_seq_cls_loss = OpSlot("cross_entropy_loss", "seq_cls")
    """
)


# ── RMSNorm (OpSlot guard, functional NPU kernel) ──────────────────────────


@config.override_method(
    "Qwen3RMSNorm.forward",
    description="OpSlot guard for NPU fused RMSNorm (standard formulation)",
)
def qwen3_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    # Modification: OpSlot guard — use fused RMSNorm kernel when bound.
    if veomni_rms_norm.use_non_eager_impl:
        return veomni_rms_norm(hidden_states, self.weight, self.variance_epsilon)
    # Original HF code below, unchanged.
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    return self.weight * hidden_states.to(input_dtype)


# ── SwiGLU MLP (OpSlot guard, functional NPU kernel) ───────────────────────


@config.override_method(
    "Qwen3MLP.forward",
    description="OpSlot guard for NPU fused SwiGLU MLP",
)
def qwen3_mlp_forward_patched(self, x):
    # Modification: OpSlot guard — use fused SwiGLU kernel when bound.
    if veomni_swiglu_mlp.use_non_eager_impl:
        return veomni_swiglu_mlp(self, x)
    # Original HF code below, unchanged.
    down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
    return down_proj


# ── Rotary Positional Embedding (OpSlot guard) ───────────────────────────────


@config.replace_function("apply_rotary_pos_emb", description="OpSlot guard for NPU fused RoPE")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Modification: OpSlot guard — use fused RoPE kernel when bound.
    if veomni_apply_rotary_pos_emb.use_non_eager_impl:
        return veomni_apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)
    # Original HF code below, unchanged.
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# ── Qwen3ForCausalLM.forward (fused cross-entropy via OpSlot) ────────────────


@config.override_method(
    "Qwen3ForCausalLM.forward",
    description="OpSlot guard for fused cross entropy in Qwen3ForCausalLM.forward",
)
def qwen3_forcausallm_forward_patched(
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
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _ = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


# ── Qwen3ForSequenceClassification.forward (fused cross-entropy via OpSlot) ──


@config.override_method(
    "Qwen3ForSequenceClassification.forward",
    description="OpSlot guard for fused cross entropy in Qwen3ForSequenceClassification.forward",
)
def qwen3forsequenceclassification_forward_patched(
    self,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    labels=None,
    use_cache=None,
    cache_position=None,
    **kwargs,
):
    outputs = self.model(
        input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        cache_position=cache_position,
        **kwargs,
    )
    hidden_states = outputs.last_hidden_state

    loss = None
    logits = None
    if labels is not None:
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_seq_cls_loss.use_non_eager_impl:
            loss, logits = veomni_seq_cls_loss(
                logits=logits,
                labels=labels,
                num_labels=self.num_labels,
                hidden_states=hidden_states,
                weights=self.score.weight,
                **kwargs,
            )
        else:
            logits = self.score(hidden_states)
            loss, _ = self.loss_function(logits=logits, labels=labels, num_labels=self.num_labels, **kwargs)
    else:
        logits = self.score(hidden_states)

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

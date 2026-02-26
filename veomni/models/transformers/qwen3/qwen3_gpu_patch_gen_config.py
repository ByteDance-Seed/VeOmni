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
Patch configuration for Qwen3 GPU LigerKernel replacements.

Regen command:
python -m veomni.patchgen.run_codegen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config -o veomni/models/transformers/qwen3/generated

This mirrors the runtime GPU patch in
veomni/models/transformers/qwen3/gpu_patch.py.

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_qwen3_gpu.py".
"""

from typing import Optional

import torch
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import slice_position_embedding
from veomni.patchgen.patch_spec import PatchConfig, create_patch_from_external


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_gpu.py",
    description="Qwen3 with LigerKernel GPU replacements",
)

config.add_import("veomni.distributed.parallel_state", names=["get_parallel_state"])
config.add_import("veomni.distributed.sequence_parallel", names=["slice_position_embedding"])
config.add_import("transformers.modeling_outputs", names=["SequenceClassifierOutputWithPast"])

config.patches.append(
    create_patch_from_external(
        target="Qwen3RMSNorm",
        replacement_module="liger_kernel.transformers.rms_norm",
        replacement_name="LigerRMSNorm",
        description="Use LigerKernel RMSNorm",
    )
)

config.patches.append(
    create_patch_from_external(
        target="Qwen3MLP",
        replacement_module="liger_kernel.transformers.swiglu",
        replacement_name="LigerSwiGLUMLP",
        description="Use LigerKernel SwiGLU MLP",
    )
)


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


@config.override_method("Qwen3Model.forward", description="Support SP in Qwen3Model.forward")
def qwen3_model_forward_patched(
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

    # It may already have been prepared by e.g. `generate`
    if not isinstance(causal_mask_mapping := attention_mask, dict):
        # Prepare mask arguments
        mask_kwargs = {
            "config": self.config,
            "inputs_embeds": inputs_embeds,
            "attention_mask": attention_mask,
            "cache_position": cache_position,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
        }
        # Create the masks
        causal_mask_mapping = {
            "full_attention": create_causal_mask(**mask_kwargs),
        }
        # The sliding window alternating layers are not always activated depending on the config
        if self.has_sliding_layers:
            causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

    hidden_states = inputs_embeds

    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # ============================== VeOmni SP Patch Start ==============================
    sp_group = get_parallel_state().sp_group if get_parallel_state().sp_enabled else None
    position_embeddings = slice_position_embedding(position_embeddings, dim=1, sp_group=sp_group)
    # =============================== VeOmni SP Patch End ===============================

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
    "Qwen3ForCausalLM.forward",
    description="Support fused cross entropy path in Qwen3ForCausalLM.forward",
)
def qwen3forcausallm_forward_patched(
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

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@config.override_method(
    "Qwen3ForSequenceClassification.forward",
    description="Support SP in Qwen3ForSequenceClassification.forward",
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
        loss, logits = self.loss_function(
            logits=logits,
            labels=labels,
            num_labels=self.num_labels,
            hidden_states=hidden_states,
            weights=self.score.weight,
            **kwargs,
        )
    else:
        logits = self.score(hidden_states)

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

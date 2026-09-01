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
Patch configuration for the text-only Gemma 3 VeomniKernel consume.

Regen command:
patchgen veomni.models_kernel.transformers.gemma3.gemma3_gpu_patch_gen_config -o veomni/models_kernel/transformers/gemma3/generated --diff

TextModel mask prep uses ``veomni.kernels.mask``. Multimodal
``Gemma3Model.forward`` keeps HuggingFace ``create_causal_mask``.
"""

from functools import partial

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.kernels.mask import causal_mask, packed_causal_mask, sliding_window_mask
from veomni.models_kernel.utils.kernel_utils import attention_kernel, resolve_kernel_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (  # noqa: F401  re-emitted into generated file
    CausalLMOutputWithLogProbs,
    FusedLinearAuxOutput,
    FusedLinearAuxOutputMixin,
)


config = PatchConfig(
    source_module="transformers.models.gemma3.modeling_gemma3",
    target_file="patched_modeling_gemma3_gpu.py",
    description="Gemma 3 text model with VeomniKernel fused-loss integration",
)

config.add_import("functools", names=["partial"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["attention_kernel", "resolve_kernel_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)
config.add_import(
    "veomni.kernels.mask",
    names=["causal_mask", "packed_causal_mask", "sliding_window_mask"],
)
apply_rotary_pos_emb = None  # noqa: E305  resolved from the generated modeling file
_bidirectional_window_overlay = None  # noqa: E305  resolved from the generated modeling file


@config.override_method(
    "Gemma3TextModel.forward",
    description="Build full / sliding masks through veomni.kernels.mask",
)
def gemma3_textmodel_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    use_cache: bool | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> BaseModelOutputWithPast:
    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if use_cache and past_key_values is None:
        past_key_values = DynamicCache(config=self.config)

    if position_ids is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
        position_ids = position_ids.unsqueeze(0)

    if not isinstance(causal_mask_mapping := attention_mask, dict):
        impl = resolve_kernel_impl("attn_implementation")
        q_len = inputs_embeds.shape[1]
        past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
        kv_len = q_len + past_seen
        mask_kwargs: dict = {
            "impl": impl,
            "device": inputs_embeds.device,
            "batch_size": inputs_embeds.shape[0],
            "dtype": inputs_embeds.dtype,
        }
        if attention_mask is not None:
            mask_kwargs["attention_mask"] = attention_mask
        sliding_kwargs = dict(mask_kwargs)
        if self.config.use_bidirectional_attention:
            mask_kwargs["or_mask_function"] = lambda *args: torch.tensor(True, dtype=torch.bool)
            sliding_kwargs["or_mask_function"] = _bidirectional_window_overlay(self.config.sliding_window)
        cu_seq_lens_q = kwargs.get("cu_seq_lens_q")
        if cu_seq_lens_q is not None:
            causal_mask_mapping = {
                "full_attention": packed_causal_mask(q_len, kv_len, cu_seqlens=cu_seq_lens_q, **mask_kwargs),
                "sliding_attention": sliding_window_mask(
                    q_len,
                    kv_len,
                    sliding_window=self.config.sliding_window,
                    cu_seqlens=cu_seq_lens_q,
                    **sliding_kwargs,
                ),
            }
        else:
            causal_mask_mapping = {
                "full_attention": causal_mask(q_len, kv_len, **mask_kwargs),
                "sliding_attention": sliding_window_mask(
                    q_len,
                    kv_len,
                    sliding_window=self.config.sliding_window,
                    **sliding_kwargs,
                ),
            }

    hidden_states = inputs_embeds
    position_embeddings = {}
    for layer_type in set(self.config.layer_types):
        position_embeddings[layer_type] = self.rotary_emb(hidden_states, position_ids, layer_type)

    for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
        hidden_states = decoder_layer(
            hidden_states,
            attention_mask=causal_mask_mapping[self.config.layer_types[i]],
            position_embeddings=position_embeddings[self.config.layer_types[i]],
            position_ids=position_ids,
            past_key_values=past_key_values,
            **kwargs,
        )

    hidden_states = self.norm(hidden_states)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
    )


@config.override_method(
    "Gemma3ForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
def gemma3_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = Gemma3TextModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.post_init()


@config.override_method(
    "Gemma3ForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)
def gemma3_forcausallm_forward_patched(
    self,
    input_ids: torch.LongTensor | None = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    past_key_values: Cache | None = None,
    inputs_embeds: torch.FloatTensor | None = None,
    labels: torch.LongTensor | None = None,
    use_cache: bool | None = None,
    logits_to_keep: int | torch.Tensor = 0,
    **kwargs: Unpack[TransformersKwargs],
) -> CausalLMOutputWithPast:
    outputs: BaseModelOutputWithPast = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        **kwargs,
    )

    hidden_states = outputs.last_hidden_state
    slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep

    loss = None
    logits = None
    fused_linear_aux = None
    if labels is not None:
        if self.config.final_logit_softcapping is not None:
            logits = self.lm_head(hidden_states)
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.vocab_size,
                **kwargs,
            )
            if fused_linear_aux is not None:
                logits = None
        else:
            loss, logits, fused_linear_aux = self.loss_function(
                logits=None,
                labels=labels,
                vocab_size=self.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        if self.config.final_logit_softcapping is not None:
            logits = logits / self.config.final_logit_softcapping
            logits = torch.tanh(logits)
            logits = logits * self.config.final_logit_softcapping

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@config.override_method(
    "Gemma3Attention.forward",
    description="Dispatch attention through the interned VeomniKernel",
)
def gemma3_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    query_states = self.q_norm(query_states)
    key_states = self.k_norm(key_states)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attn_output, attn_weights = attention_kernel()(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=self.attention_dropout if self.training else 0.0,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

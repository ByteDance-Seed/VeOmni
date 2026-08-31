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
# See the License for the specific language governing limitations
# under the License.
"""
Patch configuration for Qwen3 GPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.qwen3.qwen3_gpu_patch_gen_config -o veomni/models_kernel/transformers/qwen3/generated --diff

This file itself is not runnable. It's used to generate the runnable explicitly patched modeling file
"generated/patched_modeling_qwen3_gpu.py".
"""

from functools import partial

import torch
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import linear_bias, resolve_kernel_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss, ForSequenceClassificationLoss
from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.qwen3.modeling_qwen3",
    target_file="patched_modeling_qwen3_gpu.py",
    description="Qwen3 with VeomniKernel-based GPU kernel replacements",
)

config.add_import("functools", names=["partial"])
config.add_import("transformers.modeling_outputs", names=["SequenceClassifierOutputWithPast"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["linear_bias", "resolve_kernel_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss", "ForSequenceClassificationLoss"],
)


@config.override_method(
    "Qwen3RMSNorm.__init__",
    description="Construct a local rms_norm VeomniKernel",
)
def qwen3_rmsnorm_init_patched(self, hidden_size, eps: float = 1e-6) -> None:
    nn.Module.__init__(self)
    self.weight = nn.Parameter(torch.ones(hidden_size))
    self.variance_epsilon = eps
    self.veomni_rms_norm = VeomniKernel("rms_norm", "standard", resolve_kernel_impl("rms_norm_implementation"))


@config.override_method(
    "Qwen3RMSNorm.forward",
    description="Always call the local rms_norm VeomniKernel",
)
def qwen3_rmsnorm_forward_patched(self, hidden_states: torch.Tensor) -> torch.Tensor:
    return self.veomni_rms_norm(hidden_states, self.weight, eps=self.variance_epsilon)


@config.override_method(
    "Qwen3MLP.__init__",
    description="Construct a local swiglu_mlp VeomniKernel",
)
def qwen3_mlp_init_patched(self, config):
    nn.Module.__init__(self)
    self.config = config
    self.hidden_size = config.hidden_size
    self.intermediate_size = config.intermediate_size
    self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
    self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
    self.act_fn = ACT2FN[config.hidden_act]
    self.veomni_swiglu_mlp = VeomniKernel("swiglu_mlp", "standard", resolve_kernel_impl("swiglu_mlp_implementation"))


@config.override_method(
    "Qwen3MLP.forward",
    description="Always call the local swiglu_mlp VeomniKernel",
)
def qwen3_mlp_forward_patched(self, x):
    return self.veomni_swiglu_mlp(
        x,
        self.gate_proj.weight,
        linear_bias(self.gate_proj),
        self.up_proj.weight,
        linear_bias(self.up_proj),
        self.down_proj.weight,
        linear_bias(self.down_proj),
    )


@config.replace_function("apply_rotary_pos_emb", description="Always call rope full VeomniKernel")
def apply_rotary_pos_emb_patched(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    rope = VeomniKernel("rope", "full", resolve_kernel_impl("rotary_pos_emb_implementation"))
    return rope(q, k, cos, sin, unsqueeze_dim=unsqueeze_dim)


@config.override_method(
    "Qwen3Attention.__init__",
    description="Construct a local rope full VeomniKernel",
)
def qwen3_attention_init_patched(self, config, layer_idx: int):
    nn.Module.__init__(self)
    self.layer_type = config.layer_types[layer_idx] if hasattr(config, "layer_types") else None
    self.config = config
    self.layer_idx = layer_idx
    self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.scaling = self.head_dim**-0.5
    self.attention_dropout = config.attention_dropout
    self.is_causal = True

    self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias)
    self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias)
    self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias)
    self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
    self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None
    self.veomni_rope = VeomniKernel("rope", "full", resolve_kernel_impl("rotary_pos_emb_implementation"))


@config.override_method(
    "Qwen3Attention.forward",
    description="Always call the local rope VeomniKernel",
)
def qwen3_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = self.veomni_rope(query_states, key_states, cos, sin)

    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@config.override_method(
    "Qwen3ForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
def qwen3_forcausallm_init_patched(self, config):
    super(Qwen3ForCausalLM, self).__init__(config)
    self.model = Qwen3Model(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.post_init()


@config.override_method(
    "Qwen3ForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
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
    fused_linear_aux = None
    if labels is not None:
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
            labels=labels,
            vocab_size=self.config.vocab_size,
            hidden_states=hidden_states,
            weights=self.lm_head.weight,
            **kwargs,
        )
    else:
        logits = self.lm_head(hidden_states[:, slice_indices, :])

    return CausalLMOutputWithLogProbs(
        loss=loss,
        logits=logits,
        fused_linear_aux=fused_linear_aux,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


@config.override_method(
    "Qwen3ForSequenceClassification.__init__",
    description="Bind ForSequenceClassificationLoss to a local cross_entropy_loss VeomniKernel",
)
def qwen3_seq_cls_init_patched(self, config):
    super(Qwen3ForSequenceClassification, self).__init__(config)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForSequenceClassificationLoss, kernel=self.veomni_ce)


@config.override_method(
    "Qwen3ForSequenceClassification.forward",
    description="Always call self.loss_function (seq-cls helper + VeomniKernel)",
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
    logits = self.score(hidden_states)

    loss = None
    if labels is not None:
        loss, _, _ = self.loss_function(
            logits=None,
            labels=labels,
            num_labels=self.num_labels,
            hidden_states=hidden_states,
            weights=self.score.weight,
            **kwargs,
        )

    return SequenceClassifierOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

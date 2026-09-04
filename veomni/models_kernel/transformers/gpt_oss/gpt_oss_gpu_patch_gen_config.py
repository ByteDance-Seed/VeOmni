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
Patch configuration for GPT-OSS GPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.gpt_oss.gpt_oss_gpu_patch_gen_config -o veomni/models_kernel/transformers/gpt_oss/generated --diff

FA4 allowlist, hub-decorator drop, and EP plan stay. MoE, CE, and
load-balancing loss call local VeomniKernel.
"""

from functools import partial

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
)
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, auto_docstring
from transformers.utils.output_capturing import OutputRecorder

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import attention_kernel, resolve_kernel_impl, resolve_moe_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import MoeCausalLMOutputWithLogProbs


config = PatchConfig(
    source_module="transformers.models.gpt_oss.modeling_gpt_oss",
    target_file="patched_modeling_gpt_oss_gpu.py",
    description="GPT-OSS with VeOmni FA4-compatible attention dispatch and VeomniKernel replacements",
)

config.add_import("functools", names=["partial"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "MoeCausalLMOutputWithLogProbs"],
)
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["attention_kernel", "resolve_kernel_impl", "resolve_moe_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)
apply_rotary_pos_emb = None  # noqa: E305  resolved from the generated modeling file


config.drop_import_names("MoeCausalLMOutputWithPast")


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
        "veomni_flash_attention_4",
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
    description="Always call moe_experts gpt_oss VeomniKernel",
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
        self.veomni_moe = VeomniKernel("moe_experts", "gpt_oss", resolve_moe_impl())

    def forward(self, hidden_states: torch.Tensor, router_indices=None, routing_weights=None) -> torch.Tensor:
        return self.veomni_moe(
            hidden_states,
            routing_weights,
            router_indices,
            self.gate_up_proj,
            self.gate_up_proj_bias,
            self.down_proj,
            self.down_proj_bias,
            num_experts=self.num_experts,
            alpha=self.alpha,
            limit=self.limit,
        )


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
    "GptOssForCausalLM.__init__",
    description="Bind ForCausalLMLoss and load_balancing_loss VeomniKernels",
)
def gpt_oss_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = GptOssModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    self.router_aux_loss_coef = config.router_aux_loss_coef
    self.num_experts = config.num_local_experts
    self.num_experts_per_tok = config.num_experts_per_tok
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.veomni_lb = VeomniKernel(
        "load_balancing_loss",
        "standard",
        resolve_kernel_impl("load_balancing_loss_implementation"),
    )
    self.post_init()


@config.override_method(
    "GptOssForCausalLM.forward",
    description="Always call ForCausalLMLoss and load_balancing_loss VeomniKernels",
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
        loss, logits, fused_linear_aux = self.loss_function(
            logits=logits,
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
        router_logits = outputs.router_logits
        if router_logits is None or not isinstance(router_logits, tuple):
            aux_loss = 0
        else:
            gate = torch.cat([layer.reshape(-1, layer.shape[-1]) for layer in router_logits], dim=0)
            mask = attention_mask if isinstance(attention_mask, torch.Tensor) else gate.new_empty(0)
            aux_loss = self.veomni_lb(gate, mask, top_k=self.num_experts_per_tok)
        if labels is not None and isinstance(aux_loss, torch.Tensor):
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


@config.override_method(
    "GptOssAttention.forward",
    description="Dispatch attention through the interned VeomniKernel",
)
def gpt_oss_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    **kwargs: Unpack[TransformersKwargs],
) -> tuple[torch.Tensor, torch.Tensor]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

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
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        sliding_window=self.sliding_window,
        s_aux=self.sinks,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights

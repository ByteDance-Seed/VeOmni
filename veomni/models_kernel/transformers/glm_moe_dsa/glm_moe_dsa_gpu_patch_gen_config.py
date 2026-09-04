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
Patch configuration for GLM-MoE-DSA GPU VeomniKernel replacements.

Regen command:
patchgen veomni.models_kernel.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config -o veomni/models_kernel/transformers/glm_moe_dsa/generated --diff

Indexer and attention always call ``dsa_indexer`` / ``dsa_attention``
``glm``. CausalLM uses ``ForCausalLMLoss``.
"""

from functools import partial

import torch
from torch import nn
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.kernels import VeomniKernel
from veomni.models_kernel.utils.kernel_utils import resolve_kernel_impl
from veomni.models_kernel.utils.loss_utils import ForCausalLMLoss
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (  # noqa: F401  re-emitted into generated file
    CausalLMOutputWithLogProbs,
    FusedLinearAuxOutput,
    FusedLinearAuxOutputMixin,
)


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_gpu.py",
    description="GLM-MoE-DSA with VeomniKernel DSA indexer / attention and fused loss",
)

config.add_import("functools", names=["partial"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)
config.add_import("veomni.kernels", names=["VeomniKernel"])
config.add_import(
    "veomni.models_kernel.utils.kernel_utils",
    names=["resolve_kernel_impl"],
)
config.add_import(
    "veomni.models_kernel.utils.loss_utils",
    names=["ForCausalLMLoss"],
)
apply_rotary_pos_emb = None  # noqa: E305  resolved from the generated modeling file


@config.override_method(
    "GlmMoeDsaIndexer.__init__",
    description="Construct a local dsa_indexer glm VeomniKernel",
)
def glm_moe_dsa_indexer_init_patched(self, config: "GlmMoeDsaConfig", layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx

    self.hidden_size: int = config.hidden_size
    self.n_heads: int = config.index_n_heads
    self.head_dim: int = config.index_head_dim
    self.qk_rope_head_dim: int = config.qk_rope_head_dim
    self.index_topk: int = config.index_topk
    self.q_lora_rank: int = config.q_lora_rank

    self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
    self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
    self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
    self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
    self.softmax_scale = self.head_dim**-0.5
    self.register_buffer("_cached_keys", None, persistent=False)
    self.veomni_dsa_indexer = VeomniKernel(
        "dsa_indexer",
        "glm",
        resolve_kernel_impl("dsa_indexer_implementation"),
    )


@config.override_method(
    "GlmMoeDsaIndexer.forward",
    description="Always call the local dsa_indexer glm VeomniKernel",
)
def glm_moe_dsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    use_cache: bool = False,
) -> torch.LongTensor:
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_pe, q_nope], dim=-1)

    k = self.k_norm(self.wk(hidden_states))
    k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
    k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
    k = torch.cat([k_pe, k_nope], dim=-1)

    if seq_len > 1:
        self._cached_keys = None

    if use_cache:
        if self._cached_keys is not None:
            k_cached = torch.cat([self._cached_keys, k], dim=1)
        else:
            k_cached = k
        self._cached_keys = k_cached
    else:
        k_cached = k

    weights = self.weights_proj(hidden_states).float() * (self.n_heads**-0.5)
    return self.veomni_dsa_indexer(
        q,
        k_cached,
        weights,
        self.index_topk,
        ratio=1,
        qhead_per_kv_head=self.n_heads,
        sm_scale=self.softmax_scale,
        attention_mask=attention_mask,
    )


@config.override_method(
    "GlmMoeDsaAttention.__init__",
    description="Construct a local dsa_attention glm VeomniKernel",
)
def glm_moe_dsa_attention_init_patched(self, config: GlmMoeDsaConfig, layer_idx: int):
    nn.Module.__init__(self)
    self.config = config
    self.layer_idx = layer_idx
    self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
    self.attention_dropout = config.attention_dropout
    self.num_heads = config.num_attention_heads

    self.q_lora_rank = config.q_lora_rank
    self.qk_rope_head_dim = config.qk_rope_head_dim
    self.kv_lora_rank = config.kv_lora_rank
    self.v_head_dim = config.v_head_dim
    self.qk_nope_head_dim = config.qk_nope_head_dim
    self.qk_head_dim = config.qk_head_dim

    self.is_causal = True

    if self.q_lora_rank is None:
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
    else:
        self.q_a_proj = nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank)
        self.q_b_proj = nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

    self.kv_a_proj_with_mqa = nn.Linear(
        config.hidden_size,
        self.kv_lora_rank + self.qk_rope_head_dim,
        bias=config.attention_bias,
    )
    self.kv_a_layernorm = GlmMoeDsaRMSNorm(self.kv_lora_rank)
    self.kv_b_proj = nn.Linear(
        self.kv_lora_rank,
        self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
        bias=False,
    )
    self.o_proj = nn.Linear(
        self.num_heads * self.v_head_dim,
        config.hidden_size,
        bias=config.attention_bias,
    )
    self.scaling = self.qk_head_dim ** (-0.5)
    self.indexer = GlmMoeDsaIndexer(config, layer_idx)
    self.skip_topk = config.indexer_types[layer_idx] == "shared"
    self.next_skip_topk = (
        config.indexer_types[layer_idx + 1] == "shared" if layer_idx < len(config.indexer_types) - 1 else False
    )
    self.register_buffer("_cached_k_pe", None, persistent=False)
    self.register_buffer("_cached_kv", None, persistent=False)
    self.veomni_dsa_attention = VeomniKernel(
        "dsa_attention",
        "glm",
        resolve_kernel_impl("dsa_attention_implementation"),
    )


@config.override_method(
    "GlmMoeDsaAttention.forward",
    description="Always call the local dsa_attention glm VeomniKernel",
)
def glm_moe_dsa_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))
        query_states = self.q_b_proj(q_resid)
    query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)

    kv_expanded = self.kv_b_proj(k_compressed)
    kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
    k_pe_mqa = k_pe
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)

    key_states = torch.cat([k_nope, k_pe], dim=-1)
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    if not self.skip_topk or prev_topk_indices is None:
        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            indexer_mask,
            use_cache=past_key_values is not None,
        )
    else:
        topk_indices = prev_topk_indices

    kv_b_weight = self.kv_b_proj.weight.contiguous().view(
        self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
        self.kv_lora_rank,
    )
    k_nope_weight = kv_b_weight[:, : self.qk_nope_head_dim, :]
    value_weight = kv_b_weight[:, self.qk_nope_head_dim :, :]
    q_nope_absorbed = torch.einsum("bhsd,hdr->bshr", q_nope, k_nope_weight).contiguous()
    k_pe_kernel = k_pe_mqa.transpose(1, 2).contiguous()
    kv_cache = k_compressed.unsqueeze(2).contiguous()
    if past_key_values is not None:
        if seq_length > 1:
            self._cached_k_pe = None
            self._cached_kv = None
        if self._cached_k_pe is not None:
            k_pe_kernel = torch.cat([self._cached_k_pe, k_pe_kernel], dim=1)
            kv_cache = torch.cat([self._cached_kv, kv_cache], dim=1)
        self._cached_k_pe = k_pe_kernel
        self._cached_kv = kv_cache

    compressed_attn_output = self.veomni_dsa_attention(
        q_pe.transpose(1, 2).contiguous(),
        k_pe_kernel,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        softmax_scale=self.scaling,
        attention_mask=attention_mask,
    )
    attn_output = torch.einsum("bshr,hvr->bshv", compressed_attn_output, value_weight)
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None, topk_indices if self.next_skip_topk else None


@config.override_method(
    "GlmMoeDsaForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniKernel",
)
def glm_moe_dsa_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = GlmMoeDsaModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_kernel_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniKernel("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, kernel=self.veomni_ce)
    self.post_init()


@config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniKernel)",
)
def glm_moe_dsa_forcausallm_forward_patched(
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
            logits=None,
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

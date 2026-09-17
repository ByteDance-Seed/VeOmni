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
Patch configuration for GLM-MoE-DSA GPU VeomniOp replacements.

Regen command:
patchgen veomni.models.transformers.glm_moe_dsa.glm_moe_dsa_gpu_patch_gen_config -o veomni/models/transformers/glm_moe_dsa/generated --diff

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

from veomni.models.loss_utils import ForCausalLMLoss
from veomni.ops import VeomniOp
from veomni.ops.config import resolve_op_impl
from veomni.ops.kernels.dsa.mask import copy_dsa_mask_provenance, translate_fused_dsa_mask
from veomni.patchgen.patch_spec import PatchConfig
from veomni.utils.model_outputs import (  # noqa: F401  re-emitted into generated file
    CausalLMOutputWithLogProbs,
    FusedLinearAuxOutput,
    FusedLinearAuxOutputMixin,
)


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_gpu.py",
    description="GLM-MoE-DSA with VeomniOp DSA indexer / attention and fused loss",
)

config.add_import("functools", names=["partial"])
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)
config.add_import("veomni.ops", names=["VeomniOp"])
config.add_import(
    "veomni.ops.config",
    names=["resolve_op_impl"],
)
config.add_import(
    "veomni.models.loss_utils",
    names=["ForCausalLMLoss"],
)
config.add_import(
    "veomni.ops.kernels.dsa.mask",
    names=["copy_dsa_mask_provenance", "create_standard_causal_mask", "translate_fused_dsa_mask"],
)
# Model.forward still calls create_causal_mask; bind it to the provenance
# wrapper so a no-padding HF causal mask can be dropped without a host scan.
config.drop_import_names("create_causal_mask")
config.add_post_import_block("create_causal_mask = create_standard_causal_mask")
config.exclude_from_output("apply_rotary_pos_emb_interleave")
yarn_apply_mscale = None
GlmMoeDsaRMSNorm = None


@config.override_method(
    "GlmMoeDsaIndexer.__init__",
    description="Construct a local dsa_indexer glm VeomniOp",
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
    self.veomni_rope = VeomniOp("rope", "interleave", "eager")
    self.veomni_dsa_indexer = VeomniOp(
        "dsa_indexer",
        "glm",
        resolve_op_impl("dsa_indexer_implementation"),
    )


@config.override_method(
    "GlmMoeDsaIndexer.forward",
    description="Always call the local dsa_indexer glm VeomniOp",
)
def glm_moe_dsa_indexer_forward_patched(
    self,
    hidden_states: torch.Tensor,
    q_resid: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    position_ids: torch.Tensor,
    past_key_values: Cache | None = None,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape
    cos, sin = position_embeddings

    q = self.wq_b(q_resid)
    q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
    q_rot, q_pass = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

    k = self.k_norm(self.wk(hidden_states)).unsqueeze(2)
    k_rot, k_pass = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)

    q_rot, k_rot = self.veomni_rope(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    q = torch.cat([q_rot, q_pass], dim=-1)
    k = torch.cat([k_rot, k_pass], dim=-1).squeeze(2)

    if past_key_values is not None:
        k = past_key_values.update_indexer(k, self.layer_idx)

    weights = self.weights_proj(hidden_states.to(self.weights_proj.weight.dtype)).float() * (self.n_heads**-0.5)
    kv_len = k.shape[1]
    attention_mask = translate_fused_dsa_mask(
        attention_mask,
        q_len=seq_len,
        kv_len=kv_len,
        fused=self.veomni_dsa_indexer.impl != "eager",
        what="cuDNN GLM sparse-attention indexer",
    )
    return self.veomni_dsa_indexer(
        q,
        k,
        weights.to(q.dtype),
        self.index_topk,
        ratio=1,
        qhead_per_kv_head=self.n_heads,
        sm_scale=self.softmax_scale,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=past_key_values is not None,
    ).to(torch.int32)


@config.override_method(
    "GlmMoeDsaAttention.__init__",
    description="Construct a local dsa_attention glm VeomniOp",
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

    self.q_proj = (
        nn.Linear(config.hidden_size, self.num_heads * self.qk_head_dim, bias=False)
        if self.q_lora_rank is None
        else None
    )
    self.q_a_proj = (
        nn.Linear(config.hidden_size, config.q_lora_rank, bias=config.attention_bias)
        if self.q_lora_rank is not None
        else None
    )
    self.q_a_layernorm = GlmMoeDsaRMSNorm(config.q_lora_rank) if self.q_lora_rank is not None else None
    self.q_b_proj = (
        nn.Linear(config.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        if self.q_lora_rank is not None
        else None
    )

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
    self.scaling = yarn_apply_mscale(config.rope_parameters, self.qk_head_dim ** (-0.5))
    self.skip_topk = config.indexer_types[layer_idx] == "shared"
    self.indexer = None if self.skip_topk else GlmMoeDsaIndexer(config, layer_idx)
    self.veomni_rope = VeomniOp("rope", "interleave", "eager")
    self.veomni_dsa_attention = VeomniOp(
        "dsa_attention",
        "glm",
        resolve_op_impl("dsa_attention_implementation"),
    )


@config.override_method(
    "GlmMoeDsaAttention.forward",
    description="DSA consumes compressed K/V from past_key_values.update(), not module buffers",
)
def glm_moe_dsa_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    position_ids: torch.Tensor | None = None,
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

    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)
    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)

    q_pe, k_pe = self.veomni_rope(q_pe, k_pe, cos, sin)

    # DSA consumes MQA compressed latents. Keep them on the shared Cache object
    # (BHSD, concat on seq) instead of module buffers so chunked prefill,
    # independent requests, and reorder_cache share one lifecycle.
    k_pe_states = k_pe
    kv_states = k_compressed.unsqueeze(1)
    if past_key_values is not None:
        k_pe_states, kv_states = past_key_values.update(k_pe_states, kv_states, self.layer_idx)

    if self.indexer is not None:
        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1)
            if attention_mask is not None
            else None
        )
        indexer_mask = copy_dsa_mask_provenance(attention_mask, indexer_mask)
        topk_indices = self.indexer(
            hidden_states,
            q_resid,
            position_embeddings,
            indexer_mask,
            position_ids,
            past_key_values=past_key_values,
        )
    else:
        if prev_topk_indices is None:
            raise ValueError("Shared DSA layers require top-k indices from a previous full indexer layer.")
        topk_indices = prev_topk_indices

    kv_b_weight = self.kv_b_proj.weight.contiguous().view(
        self.num_heads,
        self.qk_nope_head_dim + self.v_head_dim,
        self.kv_lora_rank,
    )
    k_nope_weight = kv_b_weight[:, : self.qk_nope_head_dim, :]
    value_weight = kv_b_weight[:, self.qk_nope_head_dim :, :]
    q_nope_absorbed = torch.einsum("bhsd,hdr->bshr", q_nope, k_nope_weight).contiguous()
    k_pe_kernel = k_pe_states.transpose(1, 2).contiguous()
    kv_cache = kv_states.transpose(1, 2).contiguous()
    output_attentions = bool(kwargs.get("output_attentions", False)) or bool(
        getattr(self.config, "output_attentions", False)
    )
    fused_attention = self.veomni_dsa_attention.impl != "eager"
    if fused_attention and output_attentions:
        raise ValueError(
            "flashmla_cudnn GLM sparse attention does not support output_attentions=True; "
            "use the eager implementation."
        )
    attention_mask = translate_fused_dsa_mask(
        attention_mask,
        q_len=seq_length,
        kv_len=k_pe_kernel.shape[1],
        fused=fused_attention,
        what="flashmla_cudnn GLM sparse attention",
    )
    attention_dropout = 0.0 if not self.training else self.attention_dropout
    attn_result = self.veomni_dsa_attention(
        q_pe.transpose(1, 2).contiguous(),
        k_pe_kernel,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        softmax_scale=self.scaling,
        attention_mask=attention_mask,
        use_cache=past_key_values is not None,
        training=self.training,
        attention_dropout=attention_dropout,
        return_attn_weights=output_attentions,
    )
    if output_attentions:
        compressed_attn_output, attn_weights = attn_result
    else:
        compressed_attn_output = attn_result
        attn_weights = None
    attn_output = torch.einsum("bshr,hvr->bshv", compressed_attn_output, value_weight)
    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, topk_indices


@config.override_method(
    "GlmMoeDsaForCausalLM.__init__",
    description="Bind ForCausalLMLoss to a local cross_entropy_loss VeomniOp",
)
def glm_moe_dsa_forcausallm_init_patched(self, config):
    super().__init__(config)
    self.model = GlmMoeDsaModel(config)
    self.vocab_size = config.vocab_size
    self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
    impl = resolve_op_impl("cross_entropy_loss_implementation", npu_as="chunk_loss")
    self.veomni_ce = VeomniOp("cross_entropy_loss", "standard", impl)
    self.loss_function = partial(ForCausalLMLoss, op=self.veomni_ce)
    self.post_init()


@config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    description="Always call self.loss_function (ForCausalLMLoss + VeomniOp)",
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
    r"""
    cache_position (`torch.LongTensor` of shape `(sequence_length)`, *optional*):
        Indices depicting the position of input tokens in the sequence. This is
        retained explicitly for callers that pass it positionally.
    """
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


@config.override_method(
    "GlmMoeDsaForCausalLM.get_parallel_plan",
    description="Register GLM-MoE-DSA expert parallel plan for v5 generated modeling",
)
def glm_moe_dsa_get_parallel_plan_patched(self):
    from ..parallel_plan import get_parallel_plan as _get_parallel_plan

    return _get_parallel_plan()

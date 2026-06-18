import torch
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs

from veomni.patchgen.patch_spec import PatchConfig


config = PatchConfig(
    source_module="transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    target_file="patched_modeling_glm_moe_dsa_gpu.py",
    description="GLM-5 with GPU replacements",
)

# Surface ``CausalLMOutputWithLogProbs`` so the patched ``forward`` can
# return per-token log-probs in the unified output dataclass.
config.add_import(
    "veomni.utils.model_outputs",
    names=["FusedLinearAuxOutput", "FusedLinearAuxOutputMixin", "CausalLMOutputWithLogProbs"],
)

# TODO: glm_moe_dsa GPU and NPU configs are currently full copies of each
# other. Consider consolidating (NPU config imports shared patch functions
# from this module) once NPU-specific divergence is clearer.
config.add_post_import_block(
    """
    # ── OpSlot declarations ──────────────────────────────────────────────────
    # Bound at model-build time by _bind_veomni_ops() in auto.py.
    from veomni.ops.dispatch import OpSlot, OpsConfigSlot
    veomni_causal_lm_loss = OpSlot("cross_entropy_loss", "causal")
    veomni_dsa_indexer_backend = OpsConfigSlot("dsa_indexer_backend")
    veomni_dsa_attention_backend = OpsConfigSlot("dsa_attention_backend")
    """
)


@config.override_method(
    "GlmMoeDsaIndexer.forward",
    description="Use cuDNN Frontend DSA indexer kernels when supported",
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
    # In the model path, create_causal_mask may return None for the no-padding
    # causal case and let the attention backend use is_causal internally.
    if attention_mask is None and not use_cache:
        has_standard_causal_mask = True
    elif (
        attention_mask is not None
        and attention_mask.dim() == 3
        and not use_cache
        and attention_mask.shape[-2:]
        == (
            seq_len,
            k_cached.shape[1],
        )
    ):
        q_positions = torch.arange(seq_len, device=attention_mask.device)[:, None]
        k_positions = torch.arange(k_cached.shape[1], device=attention_mask.device)[None, :]
        expected_masked = k_positions > q_positions + k_cached.shape[1] - seq_len
        has_standard_causal_mask = bool(
            torch.equal(attention_mask < 0, expected_masked.unsqueeze(0).expand_as(attention_mask))
        )
    else:
        has_standard_causal_mask = False

    indexer_backend = veomni_dsa_indexer_backend.value
    if indexer_backend not in ("eager", "cudnn"):
        raise ValueError(f"Unknown dsa_indexer_backend={indexer_backend!r}; expected 'eager' or 'cudnn'")
    if indexer_backend == "cudnn":
        from veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn import indexer_select_topk

        qhead_per_kv_head = self.n_heads
        unsupported_reasons = []
        if not hidden_states.is_cuda:
            unsupported_reasons.append("hidden_states must be CUDA")
        if q.dtype not in (torch.bfloat16, torch.float16):
            unsupported_reasons.append(f"q dtype must be bf16/fp16, got {q.dtype}")
        if k_cached.dtype not in (torch.bfloat16, torch.float16):
            unsupported_reasons.append(f"k dtype must be bf16/fp16, got {k_cached.dtype}")
        if weights.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            unsupported_reasons.append(f"weights dtype must be bf16/fp16/fp32, got {weights.dtype}")
        if not has_standard_causal_mask:
            unsupported_reasons.append("only the standard causal mask is supported")
        if qhead_per_kv_head not in (32, 64):
            unsupported_reasons.append(f"qhead_per_kv_head must be 32 or 64, got {qhead_per_kv_head}")
        if unsupported_reasons:
            raise ValueError("dsa_indexer_backend='cudnn' is not supported: " + "; ".join(unsupported_reasons))
        return indexer_select_topk(
            q,
            k_cached,
            weights.to(q.dtype),
            self.index_topk,
            ratio=1,
            qhead_per_kv_head=qhead_per_kv_head,
            sm_scale=self.softmax_scale,
        )

    scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

    if attention_mask is not None:
        index_scores = index_scores + attention_mask

    total_len = index_scores.shape[-1]
    topk = min(self.index_topk, total_len)
    return index_scores.topk(topk, dim=-1).indices


@config.override_method(
    "GlmMoeDsaAttention.forward",
    description="Use FlashMLA sparse prefill forward with cuDNN FE DSA backward when supported",
)
def glm_moe_dsa_attention_forward_patched(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor | None,
    past_key_values: Cache | None = None,
    prev_topk_indices: torch.Tensor | None = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    batch_size, seq_length = hidden_states.shape[:-1]
    cos, sin = position_embeddings

    # ===== Query path =====
    if self.q_lora_rank is None:
        query_states = self.q_proj(hidden_states)
        q_resid = None
    else:
        q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
        query_states = self.q_b_proj(q_resid)
    query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
    # Split nope/rope, apply RoPE, recombine — layout: [B, H, S, D]
    q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)  # BHSD format

    # ===== KV path =====
    compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
    k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    k_compressed = self.kv_a_layernorm(k_compressed)  # [B, S, kv_rank]

    # Expand KV through kv_b_proj
    kv_expanded = self.kv_b_proj(k_compressed)  # [B, S, H * (nope_D + v_D)]
    kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_nope = k_nope.transpose(1, 2)  # [B, H, S, nope_D]
    value_states = value_states.transpose(1, 2)  # [B, H, S, v_D]

    # RoPE on k_pe (single-head rope stream)
    k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)  # [B, 1, S, rope_D]
    k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)  # BHSD format
    k_pe_mqa = k_pe
    k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)  # [B, H, S, rope_D]

    # Assemble full Q and K
    query_states = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, S, qk_head_dim]
    key_states = torch.cat([k_nope, k_pe], dim=-1)  # [B, H, S, qk_head_dim]

    # Cache update
    if past_key_values is not None:
        key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

    # ===== Indexer (DSA sparse mask) =====
    # attention_mask is [B, 1, S, T] (4D) for eager and (2D) otherwise but indexer works with [B, S, T] (3D)
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
        )  # [B, S, topk]
    else:
        topk_indices = prev_topk_indices  # [B, S, topk]

    attention_backend = veomni_dsa_attention_backend.value
    if attention_backend not in ("eager", "flashmla_cudnn"):
        raise ValueError(f"Unknown dsa_attention_backend={attention_backend!r}; expected 'eager' or 'flashmla_cudnn'")
    if attention_backend == "flashmla_cudnn":
        from veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn import (
            check_flash_mla_sparse_forward_compatible,
            flash_mla_sparse_attention_with_cudnn_backward,
        )

        if not hidden_states.is_cuda:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' requires CUDA hidden_states")
        if past_key_values is not None:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' does not support KV cache")
        if self.training and self.attention_dropout != 0:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' requires attention_dropout=0")

        if not self.kv_b_proj.weight.is_contiguous():
            raise ValueError("dsa_attention_backend='flashmla_cudnn' requires contiguous kv_b_proj.weight")
        kv_b_weight = self.kv_b_proj.weight.view(
            self.num_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_nope_weight = kv_b_weight[:, : self.qk_nope_head_dim, :]
        value_weight = kv_b_weight[:, self.qk_nope_head_dim :, :]
        q_nope_absorbed = torch.einsum("bhsd,hdr->bshr", q_nope, k_nope_weight)
        compatible, reason = check_flash_mla_sparse_forward_compatible(
            q_pe.transpose(1, 2),
            k_pe_mqa.transpose(1, 2),
            k_compressed.unsqueeze(2),
            q_nope_absorbed,
            topk_indices,
        )
        if not compatible:
            raise ValueError("dsa_attention_backend='flashmla_cudnn' is not supported: " + reason)
        compressed_attn_output = flash_mla_sparse_attention_with_cudnn_backward(
            q_pe.transpose(1, 2).contiguous(),
            k_pe_mqa.transpose(1, 2).contiguous(),
            k_compressed.unsqueeze(2).contiguous(),
            q_nope_absorbed.contiguous(),
            topk_indices,
            softmax_scale=self.scaling,
        )
        attn_output = torch.einsum("bshr,hvr->bshv", compressed_attn_output, value_weight)
        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, None, topk_indices if self.next_skip_topk else None

    # Build combined DSA + causal mask: -inf everywhere except selected top-k positions
    total_len = key_states.shape[2]
    index_mask = torch.full(
        (batch_size, seq_length, total_len),
        float("-inf"),
        device=hidden_states.device,
        dtype=query_states.dtype,
    )
    index_mask.scatter_(-1, topk_indices, 0.0)  # [B, S, T]
    index_mask = index_mask.unsqueeze(1)  # [B, 1, S, T]
    if attention_mask is not None and attention_mask.dim() == 4:
        causal_mask = attention_mask[..., :total_len]
        combined_mask = index_mask + causal_mask
    else:
        combined_mask = (
            attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
            if attention_mask is not None
            else index_mask
        )

    # Flash attention head_dim padding (qk_head_dim != v_head_dim)
    if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
        value_states = F.pad(value_states, [0, self.qk_head_dim - self.v_head_dim])

    attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation, eager_attention_forward
    )

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        combined_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        indices=topk_indices,  # flash_mla_with_kvcache
        **kwargs,
    )

    if is_flash_attention_requested(self.config) and self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output[:, :, :, : self.v_head_dim]

    attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights, topk_indices if self.next_skip_topk else None


@config.override_method(
    "GlmMoeDsaForCausalLM.forward",
    description="Support fused cross entropy path in GlmMoeDsaForCausalLM.forward",
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
        # Modification: OpSlot guard for cross-entropy loss.
        if veomni_causal_lm_loss.use_non_eager_impl:
            loss, logits, fused_linear_aux = veomni_causal_lm_loss(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
        else:
            logits = self.lm_head(hidden_states)
            loss, _, fused_linear_aux = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                hidden_states=hidden_states,
                weights=self.lm_head.weight,
                **kwargs,
            )
            if fused_linear_aux is not None:
                # fused_linear_aux path empties loss/logits slots; clear the local 3D
                # logits so output mirrors the OpSlot branch's contract.
                logits = None
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

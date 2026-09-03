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

"""DSA registry, eager vs HuggingFace, then TileLang vs that eager."""

from __future__ import annotations

import importlib
import importlib.util
import sys

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from transformers.models.deepseek_v4.modeling_deepseek_v4 import eager_attention_forward

from tests.kernels.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.kernels import KERNEL_REGISTRY, resolve_kernel
from veomni.utils.device import IS_CUDA_AVAILABLE, get_gpu_compute_capability


# Installed functions: eager_attention_forward / DeepseekV4Indexer.forward
# in transformers 5.9.0.
# https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/deepseek_v4/modeling_deepseek_v4.py
#
# GLM indexer / attention eager path:
# https://github.com/huggingface/transformers/blob/v5.9.0/src/transformers/models/glm_moe_dsa/modeling_glm_moe_dsa.py

_TILELANG_AVAILABLE = (
    IS_CUDA_AVAILABLE and get_gpu_compute_capability() >= 90 and importlib.util.find_spec("tilelang") is not None
)


def _clone(*tensors: Tensor) -> tuple[Tensor, ...]:
    """Detach and turn ``requires_grad`` on for a fresh backward."""
    return tuple(t.detach().requires_grad_(True) for t in tensors)


def _cosine(actual: Tensor, expected: Tensor) -> float:
    """Cosine similarity of two flattened tensors."""
    return F.cosine_similarity(actual.float().flatten(), expected.float().flatten(), dim=0).item()


class _HFAttentionModule(nn.Module):
    """HF ``eager_attention_forward`` needs ``sinks`` and ``num_key_value_groups``."""

    def __init__(self, sinks: Tensor, num_key_value_groups: int) -> None:
        super().__init__()
        self.sinks = sinks
        self.num_key_value_groups = num_key_value_groups


def _official_topk_additive_mask(topk_idxs: Tensor, kv_len: int, dtype: torch.dtype) -> Tensor:
    """Official HF additive mask: 0 at selected keys, ``finfo.min`` elsewhere.

    ``-1`` is not a selected key. Write selected positions directly so a
    sentinel cannot overwrite key 0 the way ``scatter_(safe, valid)`` does.
    """
    batch, q_len, _ = topk_idxs.shape
    min_value = torch.finfo(dtype).min
    mask = torch.full((batch, 1, q_len, kv_len), min_value, dtype=dtype, device=topk_idxs.device)
    valid = (topk_idxs >= 0) & (topk_idxs < kv_len)
    if valid.any():
        batch_i, query_i, slot_i = valid.nonzero(as_tuple=True)
        mask[batch_i, 0, query_i, topk_idxs[batch_i, query_i, slot_i].long()] = 0
    return mask


def _hf_dsv4_indexer_scores(
    q_bshd: Tensor,
    compressed_kv: Tensor,
    weights: Tensor,
    compress_ratio: int,
    topk: int,
) -> tuple[Tensor, Tensor]:
    """Oracle from ``DeepseekV4Indexer.forward`` (transformers v5.9.0)."""
    scores = torch.matmul(q_bshd.float(), compressed_kv.transpose(-1, -2).float().unsqueeze(1))
    scores = F.relu(scores) * (q_bshd.shape[-1] ** -0.5)
    index_scores = (scores * weights.float().unsqueeze(-1)).sum(dim=2)
    batch, seq_len, _ = index_scores.shape
    compressed_len = compressed_kv.shape[1]
    position_ids = torch.arange(seq_len, device=q_bshd.device).unsqueeze(0).expand(batch, -1)
    causal_threshold = (position_ids + 1) // compress_ratio
    if compressed_len > 0:
        entry_indices = torch.arange(compressed_len, device=index_scores.device)
        future_mask = entry_indices.view(1, 1, -1) >= causal_threshold.unsqueeze(-1)
        index_scores = index_scores.masked_fill(future_mask, float("-inf"))
        top_k = min(topk, compressed_len)
        top_k_indices = index_scores.topk(top_k, dim=-1).indices
        invalid = top_k_indices >= causal_threshold.unsqueeze(-1)
        top_k_indices = torch.where(invalid, torch.full_like(top_k_indices, -1), top_k_indices)
        return index_scores, top_k_indices
    top_k_indices = index_scores.topk(min(topk, max(compressed_len, 1)), dim=-1).indices
    return index_scores, top_k_indices


def _hf_glm_indexer_indices(q: Tensor, k: Tensor, w: Tensor, top_k: int, sm_scale: float) -> Tensor:
    """Oracle from ``GlmMoeDsaIndexer.forward`` (transformers glm_moe_dsa)."""
    scores = torch.einsum("bshd,btd->bsht", q.float(), k.float()) * sm_scale
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, w.float())
    return index_scores.topk(min(top_k, index_scores.shape[-1]), dim=-1).indices.to(torch.long)


def test_dsa_package_does_not_import_tilelang_eagerly():
    """Registering DSA must not import TileLang or FlashMLA."""
    importlib.import_module("veomni.kernels._kernels.dsa")
    assert "veomni.kernels._kernels.dsa.vendor.tilelang_sparse_mla" not in sys.modules
    assert "veomni.kernels._kernels.dsa.vendor.tilelang_indexer" not in sys.modules
    assert "veomni.kernels._kernels.dsa.vendor.flashmla_cudnn" not in sys.modules


def test_dsa_rows_are_registered():
    """Both kernels expose eager plus the fused impl names."""
    assert KERNEL_REGISTRY.list_registered("dsa_attention", "deepseek_v4") == ["eager", "tilelang"]
    assert KERNEL_REGISTRY.list_registered("dsa_attention", "glm") == ["eager", "flashmla_cudnn"]
    assert KERNEL_REGISTRY.list_registered("dsa_indexer", "deepseek_v4") == ["eager", "tilelang"]
    assert KERNEL_REGISTRY.list_registered("dsa_indexer", "glm") == ["eager", "cudnn"]
    for kernel, variant in (
        ("dsa_attention", "deepseek_v4"),
        ("dsa_attention", "glm"),
        ("dsa_indexer", "deepseek_v4"),
        ("dsa_indexer", "glm"),
    ):
        assert "eager" in KERNEL_REGISTRY.list_available(kernel, variant)


def test_dsa_attention_deepseek_v4_eager_matches_hf():
    """Eager sparse MQA matches HF ``eager_attention_forward`` plus sink."""
    torch.manual_seed(0)
    batch, seq_len, heads, dim, kv_len, topk = 2, 4, 4, 8, 6, 3
    q = torch.randn(batch, seq_len, heads, dim)
    kv = torch.randn(batch, kv_len, dim)
    sink = torch.randn(heads)
    indices = torch.randint(kv_len, (batch, seq_len, topk), dtype=torch.int32)
    indices[..., 0] = 0
    indices[..., -1] = -1
    scale = dim**-0.5

    q_h, kv_h, sink_h = _clone(q, kv, sink)
    query = q_h.transpose(1, 2).contiguous()
    key = kv_h.unsqueeze(1).contiguous()
    mask = _official_topk_additive_mask(indices, kv_len, query.dtype)
    hf_out, _ = eager_attention_forward(
        _HFAttentionModule(sink_h, heads),
        query,
        key,
        key,
        mask,
        scale,
        dropout=0.0,
    )

    q_e, kv_e, sink_e = _clone(q, kv, sink)
    ours = resolve_kernel("dsa_attention", "deepseek_v4", "eager").wrapper(q_e, kv_e, sink_e, indices, sm_scale=scale)
    assert torch.allclose(ours, hf_out, atol=EAGER_ATOL, rtol=EAGER_RTOL)

    go = torch.randn_like(ours)
    hf_out.backward(go)
    ours.backward(go)
    assert torch.allclose(q_e.grad, q_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(kv_e.grad, kv_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)
    assert torch.allclose(sink_e.grad, sink_h.grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL)


def test_dsa_attention_deepseek_v4_eager_matches_official_hf_sliding_mask():
    """Modeling mask→topk must match official HF additive sliding-window attention.

    Query 0 only keeps key 0. The conversion fills the rest with ``-1``, which
    is the overwrite case the eager mask helper has to survive.
    """
    from transformers.masking_utils import create_sliding_window_causal_mask
    from transformers.models.deepseek_v4.configuration_deepseek_v4 import DeepseekV4Config

    torch.manual_seed(5)
    batch, seq_len, heads, dim = 2, 8, 2, 16
    config = DeepseekV4Config(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=1,
        head_dim=dim,
        attn_implementation="eager",
    )
    embeds = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    mask = create_sliding_window_causal_mask(
        config=config,
        inputs_embeds=embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    q = torch.randn(batch, heads, seq_len, dim)
    kv = torch.randn(batch, 1, seq_len, dim)
    sinks = torch.randn(heads)
    scale = dim**-0.5
    hf_out, _ = eager_attention_forward(
        _HFAttentionModule(sinks, heads),
        q,
        kv,
        kv,
        mask,
        scale,
        dropout=0.0,
    )

    allowed = mask[:, 0] if mask.dtype == torch.bool else mask[:, 0] >= 0
    _, topk_indices = allowed.to(torch.int8).topk(seq_len, dim=-1, sorted=False)
    selected_valid = allowed.gather(-1, topk_indices)
    topk_indices = topk_indices.to(torch.int32).masked_fill(~selected_valid, -1).contiguous()
    assert (topk_indices[0, 0] == 0).any()
    assert (topk_indices[0, 0] == -1).any()

    ours = resolve_kernel("dsa_attention", "deepseek_v4", "eager").wrapper(
        q.transpose(1, 2).contiguous(),
        kv[:, 0].contiguous(),
        sinks,
        topk_indices,
        sm_scale=scale,
    )
    torch.testing.assert_close(ours, hf_out, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_dsa_indexer_deepseek_v4_eager_matches_hf():
    """Eager indexer matches ``DeepseekV4Indexer.forward`` scores and top-k."""
    torch.manual_seed(1)
    seq_len, batch, heads, dim, compress, topk = 8, 2, 4, 16, 2, 3
    q_sbhd = torch.randn(seq_len, batch, heads, dim)
    k_tbd = torch.randn(seq_len // compress, batch, dim)
    weights = torch.randn(seq_len, batch, heads) * 0.01
    q_bshd = q_sbhd.permute(1, 0, 2, 3).contiguous()
    compressed_kv = k_tbd.transpose(0, 1).contiguous()
    weights_bsh = weights.permute(1, 0, 2).contiguous()
    softmax_scale = dim**-0.5

    hf_scores, hf_indices = _hf_dsv4_indexer_scores(q_bshd, compressed_kv, weights_bsh, compress, topk)
    ours_scores, ours_indices = resolve_kernel("dsa_indexer", "deepseek_v4", "eager").wrapper(
        q_sbhd,
        k_tbd,
        weights * softmax_scale,
        compress,
        topk,
    )
    torch.testing.assert_close(ours_indices, hf_indices.to(torch.int32))
    valid = hf_indices >= 0
    safe = hf_indices.clamp(min=0).long()
    hf_topk_scores = torch.gather(hf_scores, dim=-1, index=safe)
    hf_topk_scores = torch.where(valid, hf_topk_scores, float("-inf"))
    torch.testing.assert_close(ours_scores, hf_topk_scores, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_dsa_indexer_glm_eager_matches_hf():
    """Eager GLM indexer matches ``GlmMoeDsaIndexer.forward`` top-k."""
    torch.manual_seed(3)
    batch, seq_len, heads, dim, kv_len, topk = 2, 6, 4, 8, 6, 2
    q = torch.randn(batch, seq_len, heads, dim)
    k = torch.randn(batch, kv_len, dim)
    w = torch.randn(batch, seq_len, heads)
    sm_scale = 0.5
    hf = _hf_glm_indexer_indices(q, k, w, topk, sm_scale)
    ours = resolve_kernel("dsa_indexer", "glm", "eager").wrapper(q, k, w, topk, ratio=1, sm_scale=sm_scale)
    torch.testing.assert_close(ours, hf)


def test_dsa_attention_glm_eager_matches_hf_mask_path():
    """Eager GLM attention matches the official top-k mask softmax path."""
    torch.manual_seed(2)
    batch, seq_len, heads, d_pe, d_nope, kv_len, topk = 2, 4, 2, 8, 16, 6, 3
    q_pe = torch.randn(batch, seq_len, heads, d_pe)
    q_nope = torch.randn(batch, seq_len, heads, d_nope)
    k_pe = torch.randn(batch, kv_len, 1, d_pe)
    kv_cache = torch.randn(batch, kv_len, 1, d_nope)
    indices = torch.randint(kv_len, (batch, seq_len, topk), dtype=torch.int32)
    scale = 0.1
    query = torch.cat((q_nope, q_pe), dim=-1)
    key = torch.cat((kv_cache.squeeze(2), k_pe.squeeze(2)), dim=-1)
    value = kv_cache.squeeze(2)
    # Official GlmMoeDsaAttention.forward: fill -inf, scatter 0 at top-k.
    index_mask = torch.full((batch, seq_len, kv_len), float("-inf"), dtype=query.dtype)
    index_mask.scatter_(-1, indices.long(), 0.0)
    index_mask = index_mask.unsqueeze(1)
    query_h = query.transpose(1, 2)
    key_h = key.unsqueeze(1).expand(-1, heads, -1, -1)
    value_h = value.unsqueeze(1).expand(-1, heads, -1, -1)
    attn_weights = torch.matmul(query_h, key_h.transpose(2, 3)) * scale + index_mask
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_h.dtype)
    hf_out = torch.matmul(attn_weights, value_h).transpose(1, 2).contiguous()

    ours = resolve_kernel("dsa_attention", "glm", "eager").wrapper(
        q_pe, k_pe, kv_cache, q_nope, indices, softmax_scale=scale
    )
    assert torch.allclose(ours, hf_out, atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_dsa_attention_glm_eager_matches_official_hf_causal_plus_scatter():
    """GLM kernel matches official scatter top-k plus official causal additive mask."""
    from transformers.masking_utils import create_causal_mask
    from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import GlmMoeDsaConfig

    torch.manual_seed(6)
    batch, seq_len, heads, d_pe, d_nope, topk = 2, 8, 2, 8, 16, 4
    config = GlmMoeDsaConfig(
        vocab_size=32,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=heads,
        num_key_value_heads=heads,
        qk_rope_head_dim=d_pe,
        qk_nope_head_dim=d_nope,
        v_head_dim=d_nope,
        attn_implementation="eager",
    )
    embeds = torch.randn(batch, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0)
    causal = create_causal_mask(
        config=config,
        inputs_embeds=embeds,
        attention_mask=None,
        past_key_values=None,
        position_ids=position_ids,
    )
    q_pe = torch.randn(batch, seq_len, heads, d_pe)
    q_nope = torch.randn(batch, seq_len, heads, d_nope)
    k_pe = torch.randn(batch, seq_len, 1, d_pe)
    kv_cache = torch.randn(batch, seq_len, 1, d_nope)
    indices = torch.randint(seq_len, (batch, seq_len, topk), dtype=torch.int32)
    indices[..., 0] = 0
    scale = 0.1
    query = torch.cat((q_nope, q_pe), dim=-1)
    key = torch.cat((kv_cache.squeeze(2), k_pe.squeeze(2)), dim=-1)
    value = kv_cache.squeeze(2)
    index_mask = torch.full((batch, seq_len, seq_len), float("-inf"), dtype=query.dtype)
    index_mask.scatter_(-1, indices.long(), 0.0)
    combined = index_mask.unsqueeze(1) + causal[..., :seq_len]
    query_h = query.transpose(1, 2)
    key_h = key.unsqueeze(1).expand(-1, heads, -1, -1)
    value_h = value.unsqueeze(1).expand(-1, heads, -1, -1)
    attn_weights = torch.matmul(query_h, key_h.transpose(2, 3)) * scale + combined
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(value_h.dtype)
    hf_out = torch.matmul(attn_weights, value_h).transpose(1, 2).contiguous()

    ours = resolve_kernel("dsa_attention", "glm", "eager").wrapper(
        q_pe,
        k_pe,
        kv_cache,
        q_nope,
        indices,
        softmax_scale=scale,
        attention_mask=causal,
    )
    torch.testing.assert_close(ours, hf_out, atol=EAGER_ATOL, rtol=EAGER_RTOL)


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="DeepSeek V4 TileLang requires SM90+ NVIDIA CUDA")
def test_dsa_attention_tilelang_matches_eager():
    """TileLang sparse MQA matches the HF-aligned eager row."""
    torch.manual_seed(4)
    device = torch.device("cuda")
    batch, seq_len, heads, dim, kv_len, topk = 1, 32, 8, 512, 48, 64
    q = torch.randn(batch, seq_len, heads, dim, device=device, dtype=torch.bfloat16)
    kv = torch.randn(batch, kv_len, dim, device=device, dtype=torch.bfloat16)
    sink = torch.randn(heads, device=device)
    indices = torch.randint(kv_len, (batch, seq_len, topk), device=device, dtype=torch.int32)
    indices[..., -1] = -1
    scale = dim**-0.5
    q_e, kv_e, sink_e = _clone(q, kv, sink)
    q_t, kv_t, sink_t = _clone(q, kv, sink)
    eager = resolve_kernel("dsa_attention", "deepseek_v4", "eager").wrapper
    fused = resolve_kernel("dsa_attention", "deepseek_v4", "tilelang").wrapper
    expected = eager(q_e, kv_e, sink_e, indices, sm_scale=scale)
    actual = fused(q_t, kv_t, sink_t, indices, sm_scale=scale)
    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)
    grad = torch.randn_like(actual)
    expected.backward(grad)
    actual.backward(grad)
    for actual_grad, expected_grad in zip((q_t.grad, kv_t.grad, sink_t.grad), (q_e.grad, kv_e.grad, sink_e.grad)):
        assert actual_grad is not None and expected_grad is not None
        assert _cosine(actual_grad, expected_grad) > 0.95
    # dAttnSink is accumulated by an atomic under a replicated T.Parallel loop, so a
    # lost replication guard would scale it by the warp count -- which cosine, being
    # scale-invariant, cannot see.
    torch.testing.assert_close(sink_t.grad, sink_e.grad, rtol=2e-2, atol=2e-2)


@pytest.mark.skipif(not _TILELANG_AVAILABLE, reason="DeepSeek V4 TileLang requires SM90+ NVIDIA CUDA")
def test_dsa_indexer_tilelang_matches_eager():
    """TileLang indexer matches the HF-aligned eager row."""
    torch.manual_seed(5)
    device = torch.device("cuda")
    seq_len, batch, heads, dim, compress, topk = 64, 2, 8, 128, 4, 7
    q = torch.randn(seq_len, batch, heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(seq_len // compress, batch, dim, device=device, dtype=torch.bfloat16)
    weights = torch.randn(seq_len, batch, heads, device=device) * 0.01
    q_e, k_e, w_e = _clone(q, k, weights)
    q_t, k_t, w_t = _clone(q, k, weights)
    eager = resolve_kernel("dsa_indexer", "deepseek_v4", "eager").wrapper
    fused = resolve_kernel("dsa_indexer", "deepseek_v4", "tilelang").wrapper
    expected_scores, expected_indices = eager(q_e, k_e, w_e, compress, topk)
    actual_scores, actual_indices = fused(q_t, k_t, w_t, compress, topk)
    torch.testing.assert_close(actual_scores, expected_scores, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_indices, expected_indices)
    valid = actual_indices >= 0
    grad = torch.randn_like(actual_scores).masked_fill(~valid, 0)
    expected_scores.backward(grad)
    actual_scores.backward(grad)
    for actual_grad, expected_grad in zip((q_t.grad, k_t.grad, w_t.grad), (q_e.grad, k_e.grad, w_e.grad)):
        assert actual_grad is not None and expected_grad is not None
        assert _cosine(actual_grad, expected_grad) > 0.95

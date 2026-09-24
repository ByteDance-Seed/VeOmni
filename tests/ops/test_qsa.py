"""Unit tests for the Qwen4-Exp QSA indexer."""

from __future__ import annotations

import importlib

import pytest
import torch

from veomni.ops import kernel_registry
from veomni.ops.kernel_registry import KERNEL_REGISTRY
from veomni.ops.kernels.qsa.eager import qsa_indexer_forward_eager
from veomni.ops.kernels.qsa.triton import qsa_indexer_forward_triton


class _RMSNorm:
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        self.weight = torch.randn(dim) * 0.02
        self.eps = eps

    def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
        normalized = hidden_states.float() * torch.rsqrt(
            hidden_states.float().square().mean(-1, keepdim=True) + self.eps
        )
        return (normalized * (1.0 + self.weight.float())).to(hidden_states.dtype)


class _QSAIndexer:
    def __init__(self, hidden_size: int, num_heads: int, head_dim: int) -> None:
        self.index_n_heads = num_heads
        self.index_kv_heads = 1
        self.index_head_dim = head_dim
        self.token_budget = 6
        self.compress_ratio = 2
        self.layer_idx = 0
        self.index_qk_proj = torch.nn.Linear(
            hidden_size,
            (num_heads + 1) * head_dim,
            bias=False,
        )
        self.q_layernorm = _RMSNorm(head_dim)
        self.k_layernorm = _RMSNorm(head_dim)


def _block_diagonal_mask(lengths: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
    total_length = sum(lengths)
    mask = torch.zeros((1, 1, total_length, total_length), dtype=torch.bool)
    start = 0
    for length in lengths:
        end = start + length
        mask[..., start:end, start:end] = torch.ones((length, length), dtype=torch.bool).tril()
        start = end
    if dtype == torch.bool:
        return mask
    if not torch.tensor(0, dtype=dtype).is_floating_point():
        raise ValueError("Test mask dtype must be bool or floating point")
    return torch.where(mask, torch.zeros((), dtype=dtype), torch.finfo(dtype).min)


@pytest.mark.parametrize("mask_dtype", [torch.bool, torch.float32])
def test_qsa_triton_packed_fallback_matches_eager(mask_dtype: torch.dtype) -> None:
    """Verify the exact fallback used by CPU CI; the fused path needs NPU."""
    torch.manual_seed(0)
    lengths = (3, 4)
    total_length = sum(lengths)
    hidden_size = 12
    num_heads = 2
    head_dim = 16
    rotary_dim = 8
    indexer = _QSAIndexer(hidden_size, num_heads, head_dim)
    hidden_states = torch.randn(1, total_length, hidden_size, dtype=torch.float32)
    cos = torch.randn(1, total_length, rotary_dim, dtype=torch.float32)
    sin = torch.randn(1, total_length, rotary_dim, dtype=torch.float32)
    attention_mask = _block_diagonal_mask(lengths, mask_dtype)
    cu_seq_lens_q = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)

    expected = qsa_indexer_forward_eager(
        indexer,
        hidden_states,
        (cos, sin),
        attention_mask,
        None,
        cu_seq_lens_q,
    )
    actual = qsa_indexer_forward_triton(
        indexer,
        hidden_states,
        (cos, sin),
        attention_mask,
        None,
        cu_seq_lens_q,
    )

    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    if actual.dtype == torch.bool:
        assert torch.equal(actual, expected)
    else:
        torch.testing.assert_close(actual, expected)


def test_qsa_packed_metadata_avoids_per_layer_tensor_sync() -> None:
    torch.manual_seed(0)
    lengths = (3, 4)
    total_length = sum(lengths)
    indexer = _QSAIndexer(12, 2, 16)
    hidden_states = torch.randn(1, total_length, 12, dtype=torch.float32)
    position_embeddings = (torch.randn(1, total_length, 8), torch.randn(1, total_length, 8))
    attention_mask = _block_diagonal_mask(lengths, torch.bool)

    expected = qsa_indexer_forward_eager(
        indexer,
        hidden_states,
        position_embeddings,
        attention_mask,
        None,
        torch.tensor([0, 3, 7], dtype=torch.int32),
    )
    actual = qsa_indexer_forward_triton(
        indexer,
        hidden_states,
        position_embeddings,
        attention_mask,
        None,
        object(),
        qsa_packed_seq_lens=lengths,
    )

    assert torch.equal(actual, expected)


def test_qsa_packed_rejects_cache_state() -> None:
    indexer = _QSAIndexer(12, 2, 16)
    hidden_states = torch.randn(1, 5, 12)
    position_embeddings = (torch.randn(1, 5, 8), torch.randn(1, 5, 8))
    attention_mask = _block_diagonal_mask((5,), torch.bool)
    cu_seq_lens_q = torch.tensor([0, 5], dtype=torch.int32)

    with pytest.raises(ValueError, match="does not support cache state"):
        qsa_indexer_forward_triton(
            indexer,
            hidden_states,
            position_embeddings,
            attention_mask,
            object(),
            cu_seq_lens_q,
        )


def test_qsa_kernel_registry_exposes_triton_backend() -> None:
    assert "triton" in KERNEL_REGISTRY.list_available("qsa_indexer", "standard")


@pytest.mark.parametrize("cuda_available", [False, True])
def test_qsa_triton_registry_rejects_non_npu(monkeypatch, cuda_available) -> None:
    monkeypatch.setattr(kernel_registry, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(kernel_registry, "IS_CUDA_AVAILABLE", cuda_available)
    monkeypatch.setattr(kernel_registry, "get_gpu_compute_capability", lambda: 90)
    with pytest.raises(RuntimeError, match="requires device_type='npu'"):
        KERNEL_REGISTRY.resolve("qsa_indexer", "standard", "triton")


def test_qsa_model_dispatch_is_npu_only() -> None:
    prefix = "veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_"
    gpu_model = importlib.import_module(prefix + "gpu")
    npu_model = importlib.import_module(prefix + "npu")
    assert not hasattr(gpu_model, "veomni_qsa_indexer")
    assert hasattr(npu_model, "veomni_qsa_indexer")


def test_qsa_block_diagonal_mask_has_no_cross_segment_visibility() -> None:
    mask = _block_diagonal_mask((3, 4), torch.bool)
    assert not mask[0, 0, 3:, :3].any()
    assert not mask[0, 0, :3, 3:].any()
    assert mask[0, 0, 2, 2]
    assert not mask[0, 0, 2, 3]

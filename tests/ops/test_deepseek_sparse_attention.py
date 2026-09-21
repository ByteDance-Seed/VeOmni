import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


dsa = pytest.importorskip("veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn")


def test_deepseek_sparse_attention_is_not_eagerly_imported_by_kernels():
    sys.modules.pop("veomni.ops.kernels.deepseek_sparse_attention", None)
    sys.modules.pop("veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn", None)

    importlib.import_module("veomni.ops.kernels")

    assert "veomni.ops.kernels.deepseek_sparse_attention" not in sys.modules
    assert "veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn" not in sys.modules


def test_deepseek_sparse_attention_package_does_not_import_flashmla_cudnn_backend():
    sys.modules.pop("veomni.ops.kernels.deepseek_sparse_attention", None)
    sys.modules.pop("veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn", None)

    importlib.import_module("veomni.ops.kernels.deepseek_sparse_attention")

    assert "veomni.ops.kernels.deepseek_sparse_attention.flashmla_cudnn" not in sys.modules


def test_indexer_select_topk_uses_cudnn_score_wrapper(monkeypatch):
    scores = torch.tensor([[[0.1, 0.7, 0.2], [0.9, 0.0, 0.3]]], dtype=torch.float32)

    def fake_indexer_forward(q, k, w, *, ratio, qhead_per_kv_head, sm_scale):
        assert q.shape == (1, 2, 32, 128)
        assert k.shape == (1, 3, 1, 128)
        assert w.shape == (1, 2, 32)
        assert ratio == 1
        assert qhead_per_kv_head == 32
        assert sm_scale == 0.5
        return {"scores": scores}

    monkeypatch.setattr(dsa, "DSA", SimpleNamespace(indexer_forward_wrapper=fake_indexer_forward))

    indices = dsa.indexer_select_topk(
        torch.empty(1, 2, 32, 128, dtype=torch.bfloat16),
        torch.empty(1, 3, 128, dtype=torch.bfloat16),
        torch.empty(1, 2, 32, dtype=torch.bfloat16),
        2,
        ratio=1,
        qhead_per_kv_head=32,
        sm_scale=0.5,
    )

    assert indices.dtype == torch.long
    assert indices.tolist() == [[[1, 2], [0, 2]]]


def test_sparse_attention_backward_flattens_batched_inputs(monkeypatch):
    q = torch.empty(2, 3, 4, 5, dtype=torch.bfloat16)
    kv = torch.empty(2, 7, 5, dtype=torch.bfloat16)
    out = torch.empty(2, 3, 4, 5, dtype=torch.bfloat16)
    dout = torch.empty_like(out)
    lse = torch.empty(2, 3, 4, dtype=torch.float32)
    attn_sink = torch.empty(4, dtype=torch.float32)
    topk_indices = torch.tensor(
        [
            [[0, 2], [1, 3], [2, 4]],
            [[0, 6], [3, 4], [5, 6]],
        ],
        dtype=torch.long,
    )
    topk_length = torch.tensor([[2, 1, 2], [2, 2, 1]], dtype=torch.int64)

    def fake_sparse_attention_backward(q_flat, kv_flat, out_flat, dout_flat, lse_flat, sink, topk_flat, **kwargs):
        assert q_flat.shape == (6, 4, 5)
        assert kv_flat.shape == (14, 5)
        assert out_flat.shape == (6, 4, 5)
        assert dout_flat.shape == (6, 4, 5)
        assert lse_flat.shape == (6, 4)
        assert sink is attn_sink
        assert topk_flat.dtype == torch.int32
        assert topk_flat.tolist() == [[0, 2], [1, 3], [2, 4], [7, 13], [10, 11], [12, 13]]
        assert kwargs["softmax_scale"] == 0.25
        assert kwargs["topk_length"].dtype == torch.int32
        assert kwargs["topk_length"].tolist() == [2, 1, 2, 2, 2, 1]
        return {
            "dq": torch.ones_like(q_flat),
            "dkv": torch.ones_like(kv_flat),
            "d_sink": torch.ones_like(sink),
        }

    monkeypatch.setattr(dsa, "DSA", SimpleNamespace(sparse_attention_backward_wrapper=fake_sparse_attention_backward))

    result = dsa.sparse_attention_backward(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_indices,
        softmax_scale=0.25,
        topk_length=topk_length,
    )

    assert result["dq"].shape == q.shape
    assert result["dkv"].shape == kv.shape
    assert result["d_sink"].shape == attn_sink.shape


def test_flash_mla_sparse_forward_returns_lse(monkeypatch):
    q_pe = torch.empty(1, 2, 128, 64, dtype=torch.bfloat16)
    k_pe = torch.empty(1, 4, 1, 64, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16)
    q_nope = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16)
    gather = torch.zeros(1, 2, 128, dtype=torch.int32)
    expected_out = torch.zeros(1, 2, 128, 512, dtype=torch.bfloat16)
    expected_lse = torch.arange(256, dtype=torch.float32).reshape(1, 2, 128)

    def fake_flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v):
        assert q.shape == (2, 128, 576)
        assert kv.shape == (4, 1, 576)
        assert indices.shape == (2, 1, 128)
        assert indices.dtype == torch.int32
        assert sm_scale == 0.25
        assert d_v == 512
        return (
            expected_out.reshape(2, 128, 512),
            torch.empty(2, 128, dtype=torch.float32),
            expected_lse.reshape(2, 128),
        )

    monkeypatch.setattr(dsa, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)

    result = dsa.flash_mla_sparse_forward(
        q_pe,
        k_pe,
        kv_cache,
        q_nope,
        gather,
        softmax_scale=0.25,
    )

    assert torch.equal(result["out"], expected_out)
    assert torch.equal(result["lse"], expected_lse)


def test_flash_mla_sparse_forward_uses_imported_flash_mla_symbol(monkeypatch):
    q_pe = torch.empty(1, 2, 128, 64, dtype=torch.bfloat16)
    k_pe = torch.empty(1, 4, 1, 64, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16)
    q_nope = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16)
    gather = torch.zeros(1, 2, 128, dtype=torch.int32)

    def fake_flash_mla_sparse_fwd(q, kv, indices, sm_scale, d_v):
        return torch.empty(2, 128, 512, dtype=q.dtype), torch.empty(2, 128), torch.empty(2, 128)

    monkeypatch.setattr(dsa, "flash_mla_sparse_fwd", fake_flash_mla_sparse_fwd)

    result = dsa.flash_mla_sparse_forward(q_pe, k_pe, kv_cache, q_nope, gather)
    assert set(result) == {"out", "lse"}


def test_flash_mla_sparse_forward_compatibility_rejects_unaligned_topk():
    q_pe = torch.empty(1, 2, 128, 64, dtype=torch.bfloat16)
    k_pe = torch.empty(1, 4, 1, 64, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16)
    q_nope = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16)
    gather = torch.zeros(1, 2, 64, dtype=torch.int32)

    compatible, reason = dsa.check_flash_mla_sparse_forward_compatible(q_pe, k_pe, kv_cache, q_nope, gather)

    assert not compatible
    assert "multiple of 128" in reason


def test_flash_mla_sparse_forward_compatibility_rejects_unsupported_packed_dim():
    q_pe = torch.empty(1, 2, 128, 32, dtype=torch.bfloat16)
    k_pe = torch.empty(1, 4, 1, 32, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16)
    q_nope = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16)
    gather = torch.zeros(1, 2, 128, dtype=torch.int32)

    compatible, reason = dsa.check_flash_mla_sparse_forward_compatible(q_pe, k_pe, kv_cache, q_nope, gather)

    assert not compatible
    assert "packed q/k dim 576" in reason


def test_flash_mla_sparse_forward_compatibility_rejects_sink():
    q_pe = torch.empty(1, 2, 128, 64, dtype=torch.bfloat16)
    k_pe = torch.empty(1, 4, 1, 64, dtype=torch.bfloat16)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16)
    q_nope = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16)
    gather = torch.zeros(1, 2, 128, dtype=torch.int32)
    sink = torch.zeros(128, dtype=torch.bfloat16)

    compatible, reason = dsa.check_flash_mla_sparse_forward_compatible(q_pe, k_pe, kv_cache, q_nope, gather, sink)

    assert not compatible
    assert "learnable_sink" in reason


def test_pack_flash_mla_tensors_for_sparse_backward():
    q_pe = torch.full((1, 2, 128, 64), 2.0, dtype=torch.bfloat16)
    k_pe = torch.full((1, 4, 1, 64), 4.0, dtype=torch.bfloat16)
    kv_cache = torch.full((1, 4, 1, 512), 3.0, dtype=torch.bfloat16)
    q_nope = torch.full((1, 2, 128, 512), 1.0, dtype=torch.bfloat16)

    packed = dsa.pack_flash_mla_tensors_for_sparse_backward(q_pe, k_pe, kv_cache, q_nope)

    assert packed["q"].shape == (1, 2, 128, 576)
    assert packed["kv"].shape == (1, 4, 576)
    assert torch.equal(packed["q"][..., :512], q_nope)
    assert torch.equal(packed["q"][..., 512:], q_pe)
    assert torch.equal(packed["kv"][..., :512], kv_cache.squeeze(2))
    assert torch.equal(packed["kv"][..., 512:], k_pe.squeeze(2))


def test_flash_mla_sparse_attention_with_cudnn_backward_splits_gradients(monkeypatch):
    q_pe = torch.empty(1, 2, 128, 64, dtype=torch.bfloat16, requires_grad=True)
    k_pe = torch.empty(1, 4, 1, 64, dtype=torch.bfloat16, requires_grad=True)
    kv_cache = torch.empty(1, 4, 1, 512, dtype=torch.bfloat16, requires_grad=True)
    q_nope_absorbed = torch.empty(1, 2, 128, 512, dtype=torch.bfloat16, requires_grad=True)
    topk_indices = torch.zeros(1, 2, 128, dtype=torch.long)
    out = torch.ones(1, 2, 128, 512, dtype=torch.bfloat16)
    lse = torch.ones(1, 2, 128, dtype=torch.float32)

    def fake_flash_mla_sparse_forward(q_pe_arg, k_pe_arg, kv_cache_arg, q_nope_arg, topk_arg, *, softmax_scale):
        assert q_pe_arg is q_pe
        assert k_pe_arg is k_pe
        assert kv_cache_arg is kv_cache
        assert q_nope_arg is q_nope_absorbed
        assert topk_arg.dtype == torch.int32
        assert softmax_scale == 0.25
        return {"out": out, "lse": lse}

    def fake_sparse_attention_backward(q, kv, out_arg, dout, lse_arg, attn_sink, topk_arg, *, softmax_scale):
        assert q.shape == (1, 2, 128, 576)
        assert kv.shape == (1, 4, 576)
        assert out_arg.shape == out.shape
        assert torch.equal(out_arg, out)
        assert dout.shape == out.shape
        assert lse_arg.shape == lse.shape
        assert torch.equal(lse_arg, lse)
        assert torch.isneginf(attn_sink).all()
        assert topk_arg is topk_indices
        assert softmax_scale == 0.25
        dq_nope = torch.full((1, 2, 128, 512), 1.0, dtype=torch.bfloat16)
        dq_pe = torch.full((1, 2, 128, 64), 2.0, dtype=torch.bfloat16)
        dkv_cache = torch.full((1, 4, 512), 3.0, dtype=torch.bfloat16)
        dk_pe = torch.full((1, 4, 64), 4.0, dtype=torch.bfloat16)
        return {
            "dq": torch.cat((dq_nope, dq_pe), dim=-1),
            "dkv": torch.cat((dkv_cache, dk_pe), dim=-1),
            "d_sink": torch.zeros_like(attn_sink),
        }

    monkeypatch.setattr(dsa, "flash_mla_sparse_forward", fake_flash_mla_sparse_forward)
    monkeypatch.setattr(dsa, "sparse_attention_backward", fake_sparse_attention_backward)

    result = dsa.flash_mla_sparse_attention_with_cudnn_backward(
        q_pe,
        k_pe,
        kv_cache,
        q_nope_absorbed,
        topk_indices,
        softmax_scale=0.25,
    )
    result.float().sum().backward()

    assert torch.equal(q_nope_absorbed.grad, torch.full_like(q_nope_absorbed, 1.0))
    assert torch.equal(q_pe.grad, torch.full_like(q_pe, 2.0))
    assert torch.equal(kv_cache.grad, torch.full_like(kv_cache, 3.0))
    assert torch.equal(k_pe.grad, torch.full_like(k_pe, 4.0))


def test_sparse_attention_backward_compatibility_rejects_expanded_kv_layout():
    q = torch.empty(2, 3, 4, 5, dtype=torch.bfloat16)
    expanded_key = torch.empty(2, 4, 7, 5, dtype=torch.bfloat16)
    out = torch.empty(2, 3, 4, 5, dtype=torch.bfloat16)
    dout = torch.empty_like(out)
    lse = torch.empty(2, 3, 4, dtype=torch.float32)
    attn_sink = torch.empty(4, dtype=torch.float32)
    topk_indices = torch.zeros(2, 3, 2, dtype=torch.long)

    compatible, reason = dsa.check_sparse_attention_backward_compatible(
        q,
        expanded_key,
        out,
        dout,
        lse,
        attn_sink,
        topk_indices,
    )

    assert not compatible
    assert "unified K=V" in reason


def test_sparse_attention_backward_compatibility_rejects_split_value_dim():
    q = torch.empty(2, 3, 4, 5, dtype=torch.bfloat16)
    kv = torch.empty(2, 7, 5, dtype=torch.bfloat16)
    out = torch.empty(2, 3, 4, 6, dtype=torch.bfloat16)
    dout = torch.empty_like(out)
    lse = torch.empty(2, 3, 4, dtype=torch.float32)
    attn_sink = torch.empty(4, dtype=torch.float32)
    topk_indices = torch.zeros(2, 3, 2, dtype=torch.long)

    compatible, reason = dsa.check_sparse_attention_backward_compatible(
        q,
        kv,
        out,
        dout,
        lse,
        attn_sink,
        topk_indices,
    )

    assert not compatible
    assert "value dim" in reason

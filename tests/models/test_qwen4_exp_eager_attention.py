# Copyright 2026 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

from types import SimpleNamespace

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import eager_attention_forward as hf_eager_attention_forward

from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_gpu import (
    eager_attention_forward,
)


def test_qwen4_exp_eager_attention_preserves_standard_fallback():
    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(2, 4, 3, 5)
    key = torch.randn(2, 2, 3, 5)
    value = torch.randn(2, 2, 3, 5)
    attention_mask = torch.triu(torch.full((2, 1, 3, 3), float("-inf")), diagonal=1)

    expected = hf_eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)
    actual = eager_attention_forward(module, query, key, value, attention_mask, scaling=5**-0.5)

    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_qwen4_exp_eager_attention_accepts_compact_selection():
    torch.manual_seed(0)
    module = SimpleNamespace(num_key_value_groups=2, training=False)
    query = torch.randn(1, 4, 3, 5)
    key = torch.randn(1, 2, 3, 5)
    value = torch.randn(1, 2, 3, 5)
    selected_indices = torch.tensor([[[0, -1], [0, 1], [1, 2]]], dtype=torch.int32)
    attention_mask = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    attention_mask[:, :, 1, 0] = False

    output, attention_weights = eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask,
        scaling=5**-0.5,
        selected_indices=selected_indices,
    )

    repeated_key = key.repeat_interleave(2, dim=1)
    repeated_value = value.repeat_interleave(2, dim=1)
    allowed = torch.zeros(1, 3, 4, dtype=torch.bool)
    allowed.scatter_(-1, torch.where(selected_indices >= 0, selected_indices, 3).long(), True)
    allowed = allowed[..., :3][:, None] & attention_mask
    scores = torch.matmul(query, repeated_key.transpose(2, 3)) * 5**-0.5
    probabilities = torch.softmax(scores.masked_fill(~allowed, torch.finfo(scores.dtype).min), dim=-1)
    expected = torch.matmul(probabilities.masked_fill(~allowed, 0), repeated_value).transpose(1, 2)

    torch.testing.assert_close(output, expected)
    assert attention_weights is None


def test_qwen4_exp_qsa_sdpa_gradients_match_dense_reference():
    """Exercise GQA, duplicate/invalid selections, bias and nondefault scaling."""
    torch.manual_seed(12)
    module = SimpleNamespace(training=True)
    q = torch.randn(2, 3, 17, 8, requires_grad=True)
    k = torch.randn(2, 1, 17, 8, requires_grad=True)
    v = torch.randn(2, 1, 17, 8, requires_grad=True)
    indices = torch.randint(0, 17, (2, 17, 6))
    indices[:, :, 0] = 0
    indices[:, :, 1] = -1
    indices[:, :, 2] = indices[:, :, 3]
    allowed = torch.zeros(2, 17, 18, dtype=torch.bool)
    allowed.scatter_(-1, indices.masked_fill(indices < 0, 17), True)
    allowed = allowed[:, None, :, :17]
    bias = torch.randn(2, 1, 17, 17) * 0.1
    scores = q @ k.repeat_interleave(3, dim=1).transpose(-1, -2) * 0.23 + bias
    ref = scores.masked_fill(~allowed, float("-inf")).softmax(-1) @ v.repeat_interleave(3, dim=1)
    ref = ref.transpose(1, 2).contiguous()
    actual, weights = eager_attention_forward(
        module, q, k, v, bias, scaling=0.23, selected_indices=indices, query_chunk_size=4
    )
    upstream = torch.randn_like(ref)
    expected_grads = torch.autograd.grad(ref, (q, k, v), upstream)
    actual_grads = torch.autograd.grad(actual, (q, k, v), upstream)
    torch.testing.assert_close(actual, ref, atol=2e-6, rtol=2e-6)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-6, rtol=2e-6)
    assert weights is None


def test_qwen4_exp_qsa_sdpa_empty_rows_and_packed_boundaries():
    torch.manual_seed(7)
    module = SimpleNamespace(training=True)
    q = torch.randn(1, 3, 6, 8, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 1, 6, 8, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 1, 6, 8, dtype=torch.bfloat16, requires_grad=True)
    indices = torch.arange(6).expand(1, 6, 6).clone()
    indices[:, 0] = -1
    segments = torch.tensor([0, 0, 0, 1, 1, 1])
    allowed = (segments[:, None] == segments[None, :]) & torch.ones(6, 6, dtype=torch.bool).tril()
    allowed[2] = False
    bias = torch.zeros(1, 1, 6, 6, dtype=torch.float32).masked_fill(~allowed, float("-inf"))
    result, _ = eager_attention_forward(
        module, q, k, v, bias, scaling=0.23, selected_indices=indices, query_chunk_size=4
    )
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result[:, [0, 2]]) == 0
    altered_v = v.detach().clone()
    altered_v[:, :, :3] += 100
    altered, _ = eager_attention_forward(
        module, q, k, altered_v, bias, scaling=0.23, selected_indices=indices, query_chunk_size=4
    )
    torch.testing.assert_close(result[:, 3:], altered[:, 3:])
    for grad in torch.autograd.grad(result.float().sum(), (q, k, v)):
        assert torch.isfinite(grad).all()


def test_qwen4_exp_qsa_chunk_checkpoint_does_not_save_dense_masks():
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(9)
    q = torch.randn(1, 3, 19, 8, requires_grad=True)
    k = torch.randn(1, 1, 19, 8, requires_grad=True)
    v = torch.randn(1, 1, 19, 8, requires_grad=True)
    indices = torch.arange(19).expand(1, 19, 19)
    saved = []

    def pack(tensor):
        saved.append((tensor.dtype, tuple(tensor.shape)))
        return tensor

    def run(q, k, v):
        return eager_attention_forward(
            SimpleNamespace(training=True), q, k, v, None, scaling=0.23, selected_indices=indices, query_chunk_size=4
        )[0]

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        result = run(q, k, v)
    assert not any(dtype == torch.bool for dtype, _ in saved)
    result.sum().backward()
    # The trainer already checkpoints layers: nested chunk recomputation must
    # preserve gradients and remain valid under an outer checkpoint too.
    expected = [t.grad.clone() for t in (q, k, v)]
    for t in (q, k, v):
        t.grad = None
    checkpoint(run, q, k, v, use_reentrant=False).sum().backward()
    for t, grad in zip((q, k, v), expected):
        torch.testing.assert_close(t.grad, grad)


def test_qwen4_exp_qsa_chunk_broadcast_key_mask():
    q = torch.randn(1, 3, 7, 8, requires_grad=True)
    k = torch.randn(1, 1, 7, 8, requires_grad=True)
    v = torch.randn(1, 1, 7, 8, requires_grad=True)
    indices = torch.arange(7).expand(1, 7, 7)
    mask = torch.tensor([True, True, False, True, False, True, False])
    args = (SimpleNamespace(training=True), q, k, v)
    actual = eager_attention_forward(*args, mask, scaling=0.2, selected_indices=indices, query_chunk_size=3)[0]
    expected = eager_attention_forward(
        *args, mask[None, None, None], scaling=0.2, selected_indices=indices, query_chunk_size=7
    )[0]
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual.sum(), (q, k, v)), torch.autograd.grad(expected.sum(), (q, k, v))):
        torch.testing.assert_close(a, b)

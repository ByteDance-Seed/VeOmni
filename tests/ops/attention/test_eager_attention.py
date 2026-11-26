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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure-PyTorch eager attention adapter and mask builder contracts."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers.masking_utils import (
    ALL_MASK_ATTENTION_FUNCTIONS,
    and_masks,
    bidirectional_mask_function,
    causal_mask_function,
    sliding_window_overlay,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ops.attention.attention_cases import clone_qkv, dense_mask, math_sdpa_reference
from tests.ops.attention.utils import UlyssesHelperRecorder
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.arguments import OpsImplementationConfig
from veomni.models.utils.mask_utils import _resolve_skip_ulysses
from veomni.ops import VeomniOp
from veomni.ops.kernels.attention import ulysses as ulysses_backend
from veomni.ops.kernels.attention.mask import eager as eager_mask_backend
from veomni.ops.kernels.attention.standard import eager as eager_backend
from veomni.ops.mask import causal_mask, eager_attention_mask_builder, packed_causal_mask


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _eager_forward(entrypoint: str):
    if entrypoint == "adapter":
        return eager_backend.eager_attention_forward
    return VeomniOp("attention", "standard", entrypoint)


@pytest.mark.parametrize("entrypoint", ("adapter", "veomni_eager", "eager"))
@pytest.mark.parametrize("mask_case", ("causal", "full", "2d_mask"))
@pytest.mark.parametrize("mask_format", ("bool", "additive"))
def test_eager_attention_matches_math_reference(entrypoint: str, mask_case: str, mask_format: str):
    """Forward outputs and Q/K/V gradients match MATH SDPA for bool and additive masks."""
    sequence_length = 24
    query_heads, kv_heads, head_dim, value_dim = 4, 2, 16, 12
    generator = torch.Generator().manual_seed(11)
    query = torch.randn(2, query_heads, sequence_length, head_dim, generator=generator)
    key = torch.randn(2, kv_heads, sequence_length, head_dim, generator=generator)
    value = torch.randn(2, kv_heads, sequence_length, value_dim, generator=generator)
    output_gradient = torch.randn(2, sequence_length, query_heads, value_dim, generator=generator)
    scaling = head_dim**-0.5

    dense = dense_mask(mask_case, sequence_length, "cpu")
    mask = dense if mask_format == "bool" else eager_mask_backend._to_eager_additive(dense, query.dtype)

    ref_qkv = clone_qkv(query, key, value)
    reference_output, _ = math_sdpa_reference(*ref_qkv, dense, scaling=scaling)
    reference_gradients = torch.autograd.grad(reference_output, ref_qkv, output_gradient)

    # Deliberately omit ``num_key_value_groups`` to verify shape-derived GQA expansion.
    module = SimpleNamespace(is_causal=False, training=False)
    actual_qkv = clone_qkv(query, key, value)
    actual_output, attn_weights = _eager_forward(entrypoint)(
        module,
        *actual_qkv,
        attention_mask=mask,
        dropout=0.0,
        scaling=scaling,
    )
    actual_gradients = torch.autograd.grad(actual_output, actual_qkv, output_gradient)

    assert attn_weights is not None
    assert attn_weights.shape == (2, query_heads, sequence_length, sequence_length)
    torch.testing.assert_close(actual_output, reference_output, rtol=EAGER_RTOL, atol=EAGER_ATOL)
    for actual_grad, ref_grad in zip(actual_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=EAGER_GRAD_RTOL, atol=EAGER_GRAD_ATOL)


@pytest.mark.parametrize("entrypoint", ("adapter", "veomni_eager", "eager"))
@pytest.mark.parametrize(
    ("q_len", "kv_len", "is_causal", "sliding_window"),
    (
        pytest.param(8, 8, True, None, id="square-causal"),
        pytest.param(8, 8, False, None, id="square-bidirectional"),
        pytest.param(1, 8, True, None, id="decode-causal"),
        pytest.param(3, 8, True, None, id="cached-prefill-causal"),
        pytest.param(8, 8, True, 3, id="causal-sliding-window"),
        pytest.param(1, 8, True, 3, id="decode-sliding-window"),
        pytest.param(8, 8, False, 3, id="bidirectional-sliding-window"),
    ),
)
def test_eager_attention_unmasked_visibility_patterns(
    entrypoint: str,
    q_len: int,
    kv_len: int,
    is_causal: bool,
    sliding_window: int | None,
):
    """Unmasked eager calls apply causal and sliding-window visibility from args/module."""
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(1, 4, q_len, 8, generator=generator)
    key = torch.randn(1, 2, kv_len, 8, generator=generator)
    value = torch.randn(1, 2, kv_len, 8, generator=generator)
    grad_out = torch.randn(1, q_len, 4, 8, generator=generator)

    q_pos = torch.arange(q_len)[:, None] + (kv_len - q_len)
    k_pos = torch.arange(kv_len)[None, :]
    if is_causal:
        expected_visible = q_pos >= k_pos
        if sliding_window is not None:
            expected_visible = expected_visible & ((q_pos - k_pos) < sliding_window)
    else:
        expected_visible = (
            torch.ones(q_len, kv_len, dtype=torch.bool)
            if sliding_window is None
            else (q_pos - k_pos).abs() <= sliding_window
        )

    ref_qkv = clone_qkv(query, key, value)
    ref_out, _ = math_sdpa_reference(*ref_qkv, expected_visible[None, None], scaling=8**-0.5)
    ref_grads = torch.autograd.grad(ref_out, ref_qkv, grad_out)

    actual_qkv = clone_qkv(query, key, value)
    module = SimpleNamespace(is_causal=is_causal, training=False)
    actual_out, attn_weights = _eager_forward(entrypoint)(
        module,
        *actual_qkv,
        attention_mask=None,
        dropout=0.0,
        scaling=None,
        sliding_window=sliding_window,
    )
    actual_grads = torch.autograd.grad(actual_out, actual_qkv, grad_out)

    assert attn_weights is not None
    torch.testing.assert_close(actual_out, ref_out, rtol=EAGER_RTOL, atol=EAGER_ATOL)
    for actual_grad, ref_grad in zip(actual_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=EAGER_GRAD_RTOL, atol=EAGER_GRAD_ATOL)


def test_eager_attention_zeroes_fully_blocked_rows():
    """Boolean and ``-inf`` rows with no visible key get zero probability and stay finite."""
    query = torch.zeros(1, 1, 2, 4)
    key = torch.zeros(1, 1, 2, 4)
    value = torch.tensor([1.0, 3.0]).view(1, 1, 2, 1)
    module = SimpleNamespace(is_causal=False, training=False)
    bool_mask = torch.tensor([[[[False, False], [True, False]]]])
    additive = torch.zeros(1, 1, 2, 2)
    additive[..., 0, :] = float("-inf")
    additive[..., 1, 1] = float("-inf")

    for mask in (bool_mask, additive):
        out, weights = eager_backend.eager_attention_forward(module, query, key, value, mask, dropout=0.0)
        assert weights is not None
        assert torch.isfinite(out).all()
        assert torch.isfinite(weights).all()
        torch.testing.assert_close(out[0, 0, 0], torch.zeros(1))
        torch.testing.assert_close(out[0, 1, 0], torch.ones(1))
        torch.testing.assert_close(weights[0, 0, 0], torch.zeros(2))
        torch.testing.assert_close(weights[0, 0, 1], torch.tensor([1.0, 0.0]))

    value_grad = value.detach().requires_grad_(True)
    grad_out, _ = eager_backend.eager_attention_forward(module, query, key, value_grad, additive, dropout=0.0)
    grad_out.sum().backward()
    assert value_grad.grad is not None
    assert torch.isfinite(value_grad.grad).all()


def test_unmasked_bidirectional_sliding_window_matches_explicit_mask():
    """Unmasked bidirectional windows include the boundary token, matching the mask builder."""
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(1, 2, 5, 8, generator=generator)
    key = torch.randn(1, 2, 5, 8, generator=generator)
    value = torch.randn(1, 2, 5, 8, generator=generator)
    module = SimpleNamespace(is_causal=False, training=False)
    sliding_window = 1

    unmasked, _ = eager_backend.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=None,
        sliding_window=sliding_window,
    )
    explicit = eager_attention_mask_builder(
        1,
        5,
        5,
        device=query.device,
        dtype=query.dtype,
        mask_function=bidirectional_mask_function,
        sliding_window=sliding_window,
    )
    assert explicit is not None
    masked, _ = eager_backend.eager_attention_forward(module, query, key, value, explicit)
    torch.testing.assert_close(unmasked, masked, rtol=EAGER_RTOL, atol=EAGER_ATOL)

    positions = torch.arange(5)
    distance = (positions[:, None] - positions[None, :]).abs()
    assert torch.all(explicit[0, 0][distance == sliding_window] == 0)
    assert torch.all(explicit[0, 0][distance > sliding_window] == torch.finfo(explicit.dtype).min)


def test_eager_attention_slices_wider_mask_and_supports_position_bias():
    """Masks and position biases wider than kv_len are sliced to key length."""
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(1, 2, 4, 8, generator=generator)
    key = torch.randn(1, 2, 4, 8, generator=generator)
    value = torch.randn(1, 2, 4, 8, generator=generator)
    wide_mask = torch.zeros(1, 1, 4, 6)
    wide_mask[:, :, :, 3:] = torch.finfo(query.dtype).min
    wide_bias = torch.randn(1, 2, 4, 6, generator=generator)

    module = SimpleNamespace(is_causal=False, training=False)
    actual, _ = eager_backend.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=wide_mask,
        position_bias=wide_bias,
    )
    expected, _ = eager_backend.eager_attention_forward(
        module,
        query,
        key,
        value,
        attention_mask=wide_mask[..., :4] + wide_bias[..., :4],
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_eager_attention_softcap_matches_gemma3_formula():
    """Softcap applies tanh(scores / softcap) * softcap before adding the mask."""
    generator = torch.Generator().manual_seed(51)
    query = torch.randn(1, 4, 6, 8, generator=generator)
    key = torch.randn(1, 2, 6, 8, generator=generator)
    value = torch.randn(1, 2, 6, 8, generator=generator)
    mask = causal_mask(6, 6, impl="eager", device="cpu")
    grad_out = torch.randn(1, 6, 4, 8, generator=generator)
    scaling = 8**-0.5
    softcap = 20.0

    ref_q, ref_k, ref_v = clone_qkv(query, key, value)
    k_rep = ref_k.repeat_interleave(2, dim=1)
    v_rep = ref_v.repeat_interleave(2, dim=1)
    scores = torch.matmul(ref_q, k_rep.transpose(2, 3)) * scaling
    scores = torch.tanh(scores / softcap) * softcap
    scores = scores + mask
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(ref_q.dtype)
    ref_out = torch.matmul(weights, v_rep).transpose(1, 2).contiguous()
    ref_grads = torch.autograd.grad(ref_out, (ref_q, ref_k, ref_v), grad_out)

    act_q, act_k, act_v = clone_qkv(query, key, value)
    module = SimpleNamespace(is_causal=True, training=False)
    act_out, act_weights = VeomniOp("attention", "standard", "veomni_eager")(
        module,
        act_q,
        act_k,
        act_v,
        mask,
        scaling=scaling,
        softcap=softcap,
    )
    act_grads = torch.autograd.grad(act_out, (act_q, act_k, act_v), grad_out)

    torch.testing.assert_close(act_out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(act_weights, weights, rtol=0, atol=0)
    for actual_grad, ref_grad in zip(act_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=0, atol=0)


def test_eager_attention_s_aux_matches_gpt_oss_formula():
    """Attention sinks (``s_aux``) match GPT-OSS eager softmax semantics bitwise."""
    generator = torch.Generator().manual_seed(61)
    query = torch.randn(2, 4, 6, 8, generator=generator)
    key = torch.randn(2, 2, 6, 8, generator=generator)
    value = torch.randn(2, 2, 6, 8, generator=generator)
    sinks = torch.randn(4, generator=generator)
    mask = causal_mask(6, 6, impl="eager", device="cpu", batch_size=2)
    grad_out = torch.randn(2, 6, 4, 8, generator=generator)
    scaling = 8**-0.5

    ref_q, ref_k, ref_v = clone_qkv(query, key, value)
    ref_sinks = sinks.detach().clone().requires_grad_(True)
    k_rep = ref_k.repeat_interleave(2, dim=1)
    v_rep = ref_v.repeat_interleave(2, dim=1)
    scores = torch.matmul(ref_q, k_rep.transpose(2, 3)) * scaling + mask
    sink_col = ref_sinks.reshape(1, -1, 1, 1).expand(2, -1, 6, -1)
    combined = torch.cat([scores, sink_col], dim=-1)
    combined = combined - combined.max(dim=-1, keepdim=True).values
    probs = F.softmax(combined, dim=-1, dtype=combined.dtype)[..., :-1].to(ref_v.dtype)
    ref_out = torch.matmul(probs, v_rep).transpose(1, 2).contiguous()
    ref_grads = torch.autograd.grad(ref_out, (ref_q, ref_k, ref_v, ref_sinks), grad_out)

    act_q, act_k, act_v = clone_qkv(query, key, value)
    act_sinks = sinks.detach().clone().requires_grad_(True)
    module = SimpleNamespace(is_causal=True, training=False)
    act_out, act_weights = VeomniOp("attention", "standard", "veomni_eager")(
        module,
        act_q,
        act_k,
        act_v,
        mask,
        scaling=scaling,
        s_aux=act_sinks,
    )
    act_grads = torch.autograd.grad(act_out, (act_q, act_k, act_v, act_sinks), grad_out)

    torch.testing.assert_close(act_out, ref_out, rtol=0, atol=0)
    torch.testing.assert_close(act_weights, probs, rtol=0, atol=0)
    for actual_grad, ref_grad in zip(act_grads, ref_grads, strict=True):
        torch.testing.assert_close(actual_grad, ref_grad, rtol=0, atol=0)


@pytest.mark.parametrize("entrypoint", ("adapter", "veomni_eager", "eager"))
@pytest.mark.parametrize("is_causal", (False, True))
def test_eager_attention_splits_unmasked_packed_cu_seqlens(entrypoint: str, is_causal: bool):
    """Unmasked batch-1 packed calls split per segment for both causal and bidirectional eager."""
    cu_seqlens = torch.tensor([0, 3, 5, 9], dtype=torch.int32)
    lengths = [3, 2, 4]
    generator = torch.Generator().manual_seed(71)
    query = torch.randn(1, 4, 9, 8, generator=generator)
    key = torch.randn(1, 2, 9, 8, generator=generator)
    value = torch.randn(1, 2, 9, 8, generator=generator)
    grad_out = torch.randn(1, 9, 4, 8, generator=generator)
    scaling = 8**-0.5

    module = SimpleNamespace(is_causal=is_causal, training=False)
    ref_q, ref_k, ref_v = clone_qkv(query, key, value)
    ref_chunks = [
        eager_backend._eager_attention_core(
            module,
            q_chunk,
            k_chunk,
            v_chunk,
            None,
            scaling=scaling,
            is_causal=is_causal,
        )[0]
        for q_chunk, k_chunk, v_chunk in zip(
            torch.split(ref_q, lengths, dim=2),
            torch.split(ref_k, lengths, dim=2),
            torch.split(ref_v, lengths, dim=2),
            strict=True,
        )
    ]
    expected = torch.cat(ref_chunks, dim=1)
    expected_grads = torch.autograd.grad(expected, (ref_q, ref_k, ref_v), grad_out)

    act_q, act_k, act_v = clone_qkv(query, key, value)
    actual, attn_weights = _eager_forward(entrypoint)(
        module,
        act_q,
        act_k,
        act_v,
        None,
        scaling=scaling,
        is_causal=is_causal,
        cu_seq_lens_q=cu_seqlens,
        cu_seq_lens_k=cu_seqlens,
        max_length_q=4,
        max_length_k=4,
    )
    actual_grads = torch.autograd.grad(actual, (act_q, act_k, act_v), grad_out)

    assert attn_weights is None
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


def test_eager_attention_accepts_single_segment_and_dense_packed_masks():
    """Single-segment metadata is dropped; dense 4D packed masks run without splitting."""
    query = torch.randn(1, 2, 4, 8)
    module = SimpleNamespace(is_causal=True, training=False)
    cu_single = torch.tensor([0, 4], dtype=torch.int32)

    plain, _ = eager_backend.eager_attention_forward(module, query, query, query, None)
    with_meta, _ = eager_backend.eager_attention_forward(
        module,
        query,
        query,
        query,
        None,
        cu_seq_lens_q=cu_single,
        cu_seq_lens_k=cu_single,
        max_length_q=4,
        max_length_k=4,
    )
    torch.testing.assert_close(with_meta, plain, rtol=0, atol=0)

    cu_packed = torch.tensor([0, 2, 4], dtype=torch.int32)
    mask = packed_causal_mask(4, 4, impl="veomni_eager", device="cpu", cu_seqlens=cu_packed)
    masked_out, weights = eager_backend.eager_attention_forward(
        module,
        query,
        query,
        query,
        mask,
        cu_seq_lens_q=cu_packed,
        cu_seq_lens_k=cu_packed,
        max_length_q=2,
        max_length_k=2,
    )
    assert weights is not None
    torch.testing.assert_close(masked_out[0, 2], query[0, :, 2], rtol=EAGER_RTOL, atol=EAGER_ATOL)


def test_eager_attention_rejects_invalid_inputs():
    """Zero dims, bad GQA ratio, mismatched KV lengths, and bad s_aux fail closed."""
    module = SimpleNamespace(is_causal=False, training=False)
    query = torch.randn(1, 4, 4, 8)

    with pytest.raises(ValueError, match="zero dimensions"):
        eager_backend.eager_attention_forward(module, torch.randn(1, 0, 4, 8), query, query, None)

    with pytest.raises(ValueError, match="divisible by matching key/value heads"):
        eager_backend.eager_attention_forward(module, query, torch.randn(1, 3, 4, 8), torch.randn(1, 3, 4, 8), None)

    with pytest.raises(ValueError, match="matching key and value sequence lengths"):
        eager_backend.eager_attention_forward(module, query, torch.randn(1, 2, 4, 8), torch.randn(1, 2, 5, 8), None)

    with pytest.raises(ValueError, match="s_aux"):
        eager_backend.eager_attention_forward(module, query, query, query, None, s_aux=torch.zeros(3))


def test_raw_eager_rejects_active_ulysses_unless_skipped(monkeypatch):
    """Raw ``eager`` rejects active Ulysses unless ``skip_ulysses=True``."""
    monkeypatch.setattr(
        ulysses_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_size=2, async_enabled=False),
    )
    module = SimpleNamespace(is_causal=False, training=False)
    query = torch.randn(1, 4, 4, 8)

    with pytest.raises(ValueError, match="does not perform Ulysses sequence-parallel exchanges"):
        _eager_forward("eager")(module, query, query, query, None)

    out, weights = _eager_forward("eager")(module, query, query, query, None, skip_ulysses=True)
    assert out.shape == (1, 4, 4, 8)
    assert weights is not None


@pytest.mark.parametrize("entrypoint", ("adapter", "veomni_eager"))
def test_veomni_eager_delegates_active_ulysses_and_slices_head_tensors(monkeypatch, entrypoint: str):
    """``veomni_eager`` delegates active Ulysses exchanges and slices global head-indexed tensors."""
    group = object()
    recorder = UlyssesHelperRecorder()
    captured = {}

    def fake_core(module, query, key, value, attention_mask, **kwargs):
        recorder.calls.append(("backend", query, key, value))
        captured["query"] = query
        captured["mask"] = attention_mask
        captured["position_bias"] = kwargs.get("position_bias")
        captured["s_aux"] = kwargs.get("s_aux")
        return query.transpose(1, 2), torch.zeros(query.shape[0], query.shape[1], query.shape[2], key.shape[2])

    monkeypatch.setattr(
        eager_backend,
        "get_parallel_state",
        lambda: SimpleNamespace(ulysses_group=group, ulysses_size=2, async_enabled=False),
    )
    monkeypatch.setattr(eager_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(eager_backend, "prepare_ulysses_qkv", recorder.prepare)
    monkeypatch.setattr(eager_backend, "_eager_attention_core", fake_core)
    monkeypatch.setattr(eager_backend, "restore_ulysses_output", recorder.restore)
    monkeypatch.setattr(eager_backend.dist, "get_rank", lambda _group: 1)
    monkeypatch.setattr(ulysses_backend.dist, "get_rank", lambda _group: 1)

    query = torch.randn(1, 4, 8, 8)
    mask = torch.arange(4, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 8, 8)
    bias = torch.arange(10, 14, dtype=torch.float32).view(1, 4, 1, 1).expand(1, 4, 8, 8)
    sinks = torch.tensor([0.1, 0.2, 0.3, 0.4])

    output, weights = _eager_forward(entrypoint)(
        SimpleNamespace(is_causal=True, training=False),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=mask,
        position_bias=bias,
        s_aux=sinks,
    )

    assert [call[0] for call in recorder.calls] == ["prepare", "backend", "restore"]
    assert recorder.calls[0][1].shape == (1, 8, 4, 8)
    assert recorder.calls[0][4:] == (group, 2)
    assert output.shape == (1, 8, 2, 8)
    assert weights is None
    torch.testing.assert_close(captured["mask"], mask[:, 2:4])
    torch.testing.assert_close(captured["position_bias"], bias[:, 2:4])
    torch.testing.assert_close(captured["s_aux"], sinks[2:4])


def test_veomni_eager_skip_ulysses_skips_exchange(monkeypatch):
    """``skip_ulysses=True`` keeps ``veomni_eager`` local and returns attention weights."""
    monkeypatch.setattr(eager_backend, "should_apply_ulysses", lambda *, skip_ulysses=False: not skip_ulysses)
    monkeypatch.setattr(
        eager_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("skip_ulysses must not exchange QKV"),
    )
    query = torch.randn(1, 4, 6, 8)
    out, weights = _eager_forward("veomni_eager")(
        SimpleNamespace(is_causal=True, training=False),
        query,
        query[:, :2],
        query[:, :2],
        attention_mask=None,
        skip_ulysses=True,
    )
    assert out.shape == (1, 6, 4, 8)
    assert weights is not None


def test_eager_normalization_and_mask_utils_ulysses_routing(monkeypatch):
    """``eager`` normalizes to ``veomni_eager`` only under ``MODELING_BACKEND=veomni``."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    assert OpsImplementationConfig(attn_implementation="eager").attn_implementation == "veomni_eager"

    monkeypatch.setenv("MODELING_BACKEND", "hf")
    assert OpsImplementationConfig(attn_implementation="eager").attn_implementation == "eager"

    assert _resolve_skip_ulysses("eager", {}) is True
    assert _resolve_skip_ulysses("veomni_eager", {}) is False
    assert _resolve_skip_ulysses("veomni_eager", {"skip_ulysses": True}) is True


def test_eager_attention_mask_builder_aligns_with_hf_eager_mask():
    """veomni_eager mask builder matches Transformers' eager mask and rejects unisolated packing."""
    hf_causal = ALL_MASK_ATTENTION_FUNCTIONS["eager"](batch_size=1, q_length=4, kv_length=4, device="cpu")
    ours_causal = eager_attention_mask_builder(1, 4, 4, device="cpu")
    assert ours_causal is not None
    torch.testing.assert_close(ours_causal, hf_causal, rtol=0, atol=0)

    hf_sliding = ALL_MASK_ATTENTION_FUNCTIONS["eager"](
        batch_size=1,
        q_length=4,
        kv_length=4,
        device="cpu",
        mask_function=and_masks(causal_mask_function, sliding_window_overlay(2)),
    )
    ours_sliding = eager_attention_mask_builder(1, 4, 4, device="cpu", sliding_window=2)
    assert ours_sliding is not None
    torch.testing.assert_close(ours_sliding, hf_sliding, rtol=0, atol=0)

    hf_bidirectional = ALL_MASK_ATTENTION_FUNCTIONS["eager"](
        batch_size=1,
        q_length=4,
        kv_length=4,
        device="cpu",
        mask_function=bidirectional_mask_function,
    )
    ours_bidirectional = eager_attention_mask_builder(
        1,
        4,
        4,
        device="cpu",
        mask_function=bidirectional_mask_function,
        allow_is_bidirectional_skip=True,
    )
    assert ours_bidirectional is not None
    assert hf_bidirectional is not None
    torch.testing.assert_close(ours_bidirectional, hf_bidirectional, rtol=0, atol=0)
    assert torch.count_nonzero(ours_bidirectional) == 0

    # Single-segment metadata is dropped; multi-segment metadata raises.
    single = eager_attention_mask_builder(1, 4, 4, device="cpu", cu_seq_lens_q=torch.tensor([0, 4]))
    assert single is not None
    torch.testing.assert_close(single, hf_causal, rtol=0, atol=0)

    with pytest.raises(ValueError, match="several segments per row"):
        eager_attention_mask_builder(1, 4, 4, device="cpu", cu_seq_lens_q=torch.tensor([0, 2, 4]))

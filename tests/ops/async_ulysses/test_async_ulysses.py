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
# See the License for the specific language governing permissions and
# limitations under the License.

"""Single-process coverage for async Ulysses compound kernels.

Collectives are mocked as identity. Sequential references do the same math
without overlapping the next linear with an in-flight all-to-all.
"""

from __future__ import annotations

import inspect

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from veomni.ops import VeomniOp
from veomni.ops.registry import OpEntry, SavedState
from veomni.utils.device import IS_CUDA_AVAILABLE


_EAGER_ROWS = (
    ("async_ulysses_qkv", "standard"),
    ("async_ulysses_qkv", "dit"),
    ("async_ulysses_o", "standard"),
    ("async_ulysses_o", "dit"),
)


def _eager_module(op: str, variant: str):
    """Return the impl module that registered this row's raw pair."""
    module = inspect.getmodule(VeomniOp(op, variant, "eager").entry.backward)
    assert module is not None
    return module


def _mock_identity_comm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_all_to_all(tensor: Tensor, **kwargs):
        return (lambda: tensor) if kwargs.get("async_op") else tensor

    def fake_pad(tensor: Tensor, dim, **kwargs):
        return tensor

    def fake_unpad(tensor: Tensor, dim, size, **kwargs):
        return tensor

    for op, variant in _EAGER_ROWS:
        module = _eager_module(op, variant)
        monkeypatch.setattr(module, "all_to_all_tensor", fake_all_to_all)
        monkeypatch.setattr(module, "padding_tensor_for_seqeunce_parallel", fake_pad)
        monkeypatch.setattr(module, "unpadding_tensor_for_seqeunce_parallel", fake_unpad)
        if hasattr(module, "get_ulysses_sequence_parallel_world_size"):
            monkeypatch.setattr(module, "get_ulysses_sequence_parallel_world_size", lambda: 1)
        if hasattr(module, "get_ulysses_sequence_parallel_group"):
            monkeypatch.setattr(module, "get_ulysses_sequence_parallel_group", lambda: object())


def _leaf(*tensors: Tensor) -> list[Tensor]:
    return [tensor.detach().clone().requires_grad_(tensor.requires_grad) for tensor in tensors]


def _qkv_weights(
    hidden: int,
    query_size: int,
    key_value_size: int,
    *,
    dtype: torch.dtype = torch.float64,
    bias: bool = True,
    requires_grad: bool = True,
) -> tuple[Tensor, ...]:
    q_weight = torch.randn(query_size, hidden, dtype=dtype, requires_grad=requires_grad)
    k_weight = torch.randn(key_value_size, hidden, dtype=dtype, requires_grad=requires_grad)
    v_weight = torch.randn(key_value_size, hidden, dtype=dtype, requires_grad=requires_grad)
    if not bias:
        return q_weight, None, k_weight, None, v_weight, None
    q_bias = torch.randn(query_size, dtype=dtype, requires_grad=requires_grad)
    k_bias = torch.randn(key_value_size, dtype=dtype, requires_grad=requires_grad)
    v_bias = torch.randn(key_value_size, dtype=dtype, requires_grad=requires_grad)
    return q_weight, q_bias, k_weight, k_bias, v_weight, v_bias


def _sequential_standard_qkv(
    hidden: Tensor,
    q_weight: Tensor,
    q_bias: Tensor | None,
    k_weight: Tensor,
    k_bias: Tensor | None,
    v_weight: Tensor,
    v_bias: Tensor | None,
    norm_q_weight: Tensor | None,
    norm_k_weight: Tensor | None,
    *,
    head_dim: int,
    norm_type: str | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    batch = hidden.shape[0]
    num_q = q_weight.shape[0] // head_dim
    num_kv = k_weight.shape[0] // head_dim
    query = F.linear(hidden, q_weight, q_bias).view(batch, -1, num_q, head_dim)
    key = F.linear(hidden, k_weight, k_bias).view(batch, -1, num_kv, head_dim)
    value = F.linear(hidden, v_weight, v_bias).view(batch, -1, num_kv, head_dim)
    if norm_type == "rmsnorm":
        rms = VeomniOp("rms_norm", "standard")
        query = rms(query, norm_q_weight, eps=eps)
        key = rms(key, norm_k_weight, eps=eps)
    return query, key, value


def _sequential_dit_qkv(
    hidden: Tensor,
    q_weight: Tensor,
    q_bias: Tensor | None,
    k_weight: Tensor,
    k_bias: Tensor | None,
    v_weight: Tensor,
    v_bias: Tensor | None,
    norm_q_weight: Tensor | None,
    norm_k_weight: Tensor | None,
    *,
    norm_type: str | None,
    eps: float,
) -> tuple[Tensor, Tensor, Tensor]:
    query = F.linear(hidden, q_weight, q_bias)
    key = F.linear(hidden, k_weight, k_bias)
    value = F.linear(hidden, v_weight, v_bias)
    if norm_type == "rmsnorm":
        rms = VeomniOp("rms_norm", "standard")
        query = rms(query, norm_q_weight, eps=eps)
        key = rms(key, norm_k_weight, eps=eps)
    return query, key, value


def test_standard_qkv_rejects_nondivisible_kv_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reject KV-head layouts that cannot scatter evenly across Ulysses ranks."""
    module = _eager_module("async_ulysses_qkv", "standard")
    monkeypatch.setattr(module, "get_ulysses_sequence_parallel_world_size", lambda: 4)

    def unexpected_collective(*args, **kwargs):
        pytest.fail("head validation must run before the first collective")

    monkeypatch.setattr(module, "all_to_all_tensor", unexpected_collective)
    head_dim = 2
    hidden_states = torch.randn(1, 2, 8)
    weights = _qkv_weights(8, 12 * head_dim, 6 * head_dim, bias=False, requires_grad=False)

    with pytest.raises(
        ValueError,
        match=r"num_key_value_heads \(6\) must be divisible by ulysses_size \(4\)",
    ):
        VeomniOp("async_ulysses_qkv", "standard")(
            hidden_states,
            *weights,
            None,
            None,
            None,
            None,
            seq_dimension=1,
            head_dimension=2,
            unpadded_dim_size=2,
            head_dim=head_dim,
            group=object(),
        )


@pytest.mark.parametrize("norm_type", [None, "rmsnorm"])
def test_standard_qkv_matches_sequential(monkeypatch: pytest.MonkeyPatch, norm_type: str | None) -> None:
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6101)
    batch, seq, hidden, head_dim = 2, 3, 20, 5
    query_size, key_value_size = 20, 10
    eps = 1e-6
    hidden_states = torch.randn(batch, seq, hidden, dtype=torch.float64, requires_grad=True)
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = _qkv_weights(hidden, query_size, key_value_size)
    norm_q = torch.randn(head_dim, dtype=torch.float64, requires_grad=True) if norm_type else None
    norm_k = torch.randn(head_dim, dtype=torch.float64, requires_grad=True) if norm_type else None

    kernel_in = _leaf(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias)
    kernel_norms = _leaf(*(tensor for tensor in (norm_q, norm_k) if tensor is not None))
    seq_in = _leaf(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias)
    seq_norms = _leaf(*(tensor for tensor in (norm_q, norm_k) if tensor is not None))

    kernel_q, kernel_k, kernel_v = VeomniOp("async_ulysses_qkv", "standard")(
        *kernel_in,
        kernel_norms[0] if kernel_norms else None,
        None,
        kernel_norms[1] if kernel_norms else None,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        head_dim=head_dim,
        group=object(),
        norm_type=norm_type,
        normalized_shape=head_dim if norm_type else None,
        eps=eps if norm_type else None,
    )
    seq_q, seq_k, seq_v = _sequential_standard_qkv(
        *seq_in,
        seq_norms[0] if seq_norms else None,
        seq_norms[1] if seq_norms else None,
        head_dim=head_dim,
        norm_type=norm_type,
        eps=eps,
    )
    torch.testing.assert_close(kernel_q, seq_q)
    torch.testing.assert_close(kernel_k, seq_k)
    torch.testing.assert_close(kernel_v, seq_v)

    (kernel_q.sum() + kernel_k.sum() + kernel_v.sum()).backward()
    (seq_q.sum() + seq_k.sum() + seq_v.sum()).backward()
    for actual, expected in zip(kernel_in + kernel_norms, seq_in + seq_norms, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


def test_standard_qkv_repeated_kv_heads_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    """Exercise repeated-KV reduction through the public compound op."""
    _mock_identity_comm(monkeypatch)
    module = _eager_module("async_ulysses_qkv", "standard")
    monkeypatch.setattr(module, "get_ulysses_sequence_parallel_world_size", lambda: 4)
    torch.manual_seed(6105)
    batch, seq, hidden, head_dim = 2, 3, 20, 5
    inputs = (
        torch.randn(batch, seq, hidden, dtype=torch.float64, requires_grad=True),
        *_qkv_weights(hidden, 4 * head_dim, 2 * head_dim),
    )
    actual_inputs = _leaf(*inputs)
    expected_inputs = _leaf(*inputs)

    actual = VeomniOp("async_ulysses_qkv", "standard")(
        *actual_inputs,
        None,
        None,
        None,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        head_dim=head_dim,
        group=object(),
    )
    hidden_e, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = expected_inputs
    expected = (
        F.linear(hidden_e, q_weight, q_bias).view(batch, seq, 4, head_dim),
        F.linear(hidden_e, k_weight, k_bias).view(batch, seq, 2, head_dim).repeat_interleave(2, dim=2),
        F.linear(hidden_e, v_weight, v_bias).view(batch, seq, 2, head_dim).repeat_interleave(2, dim=2),
    )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)

    grad_outputs = tuple(torch.randn_like(output) for output in actual)
    torch.autograd.backward(actual, grad_outputs)
    torch.autograd.backward(expected, grad_outputs)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


@pytest.mark.parametrize("norm_type", [None, "rmsnorm"])
def test_dit_qkv_matches_sequential(monkeypatch: pytest.MonkeyPatch, norm_type: str | None) -> None:
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6102)
    batch, seq, hidden = 2, 3, 16
    eps = 1e-6
    hidden_states = torch.randn(batch, seq, hidden, dtype=torch.float64, requires_grad=True)
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = _qkv_weights(hidden, hidden, hidden)
    norm_q = torch.randn(hidden, dtype=torch.float64, requires_grad=True) if norm_type else None
    norm_k = torch.randn(hidden, dtype=torch.float64, requires_grad=True) if norm_type else None

    kernel_in = _leaf(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias)
    kernel_norms = _leaf(*(tensor for tensor in (norm_q, norm_k) if tensor is not None))
    seq_in = _leaf(hidden_states, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias)
    seq_norms = _leaf(*(tensor for tensor in (norm_q, norm_k) if tensor is not None))

    kernel_q, kernel_k, kernel_v = VeomniOp("async_ulysses_qkv", "dit")(
        *kernel_in,
        kernel_norms[0] if kernel_norms else None,
        None,
        kernel_norms[1] if kernel_norms else None,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
        norm_type=norm_type,
        normalized_shape=hidden if norm_type else None,
        eps=eps if norm_type else None,
    )
    seq_q, seq_k, seq_v = _sequential_dit_qkv(
        *seq_in,
        seq_norms[0] if seq_norms else None,
        seq_norms[1] if seq_norms else None,
        norm_type=norm_type,
        eps=eps,
    )
    torch.testing.assert_close(kernel_q, seq_q)
    torch.testing.assert_close(kernel_k, seq_k)
    torch.testing.assert_close(kernel_v, seq_v)

    (kernel_q.sum() + kernel_k.sum() + kernel_v.sum()).backward()
    (seq_q.sum() + seq_k.sum() + seq_v.sum()).backward()
    for actual, expected in zip(kernel_in + kernel_norms, seq_in + seq_norms, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize("variant", ("standard", "dit"))
def test_qkv_empty_sequence_matches_sequential(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, hidden, head_dim = 2, 0, 20, 5
    key_value_size = 10 if variant == "standard" else hidden
    inputs = (
        torch.empty(batch, seq, hidden, dtype=torch.float64, requires_grad=True),
        *_qkv_weights(hidden, hidden, key_value_size),
    )
    actual_inputs = _leaf(*inputs)
    expected_inputs = _leaf(*inputs)

    actual = VeomniOp("async_ulysses_qkv", variant)(
        *actual_inputs,
        None,
        None,
        None,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        head_dim=head_dim,
        group=object(),
    )
    hidden_e, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = expected_inputs
    expected = (
        F.linear(hidden_e, q_weight, q_bias),
        F.linear(hidden_e, k_weight, k_bias),
        F.linear(hidden_e, v_weight, v_bias),
    )
    if variant == "standard":
        expected = (
            expected[0].reshape(batch, seq, hidden // head_dim, head_dim),
            expected[1].reshape(batch, seq, key_value_size // head_dim, head_dim),
            expected[2].reshape(batch, seq, key_value_size // head_dim, head_dim),
        )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)

    sum(output.sum() for output in actual).backward()
    sum(output.sum() for output in expected).backward()
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="async Ulysses LayerNorm backward requires CUDA")
@pytest.mark.parametrize("variant", ("standard", "dit"))
def test_qkv_layer_norm_matches_sequential(monkeypatch: pytest.MonkeyPatch, variant: str) -> None:
    pytest.importorskip("fused_layer_norm_cuda")
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6106)
    batch, seq, hidden, head_dim = 2, 3, 16, 4
    norm_size = head_dim if variant == "standard" else hidden
    weights = tuple(tensor.to("cuda") for tensor in _qkv_weights(hidden, hidden, hidden, dtype=torch.float32))
    inputs = (
        torch.randn(batch, seq, hidden, device="cuda", dtype=torch.float32),
        *weights,
        torch.randn(norm_size, device="cuda"),
        torch.randn(norm_size, device="cuda"),
        torch.randn(norm_size, device="cuda"),
        torch.randn(norm_size, device="cuda"),
    )
    actual_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in inputs]

    actual = VeomniOp("async_ulysses_qkv", variant)(
        *actual_inputs[:7],
        *actual_inputs[7:],
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        head_dim=head_dim,
        group=object(),
        norm_type="layernorm",
        normalized_shape=norm_size,
        eps=1e-5,
    )
    hidden_e, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = expected_inputs[:7]
    norm_q_weight, norm_q_bias, norm_k_weight, norm_k_bias = expected_inputs[7:]
    query = F.linear(hidden_e, q_weight, q_bias)
    key = F.linear(hidden_e, k_weight, k_bias)
    value = F.linear(hidden_e, v_weight, v_bias)
    if variant == "standard":
        query = query.reshape(batch, seq, hidden // head_dim, head_dim)
        key = key.reshape(batch, seq, hidden // head_dim, head_dim)
        value = value.reshape(batch, seq, hidden // head_dim, head_dim)
    expected = (
        F.layer_norm(query, (norm_size,), norm_q_weight, norm_q_bias, 1e-5),
        F.layer_norm(key, (norm_size,), norm_k_weight, norm_k_bias, 1e-5),
        value,
    )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output, atol=2e-6, rtol=2e-5)

    grad_outputs = tuple(torch.randn_like(output) for output in actual)
    torch.autograd.backward(actual, grad_outputs)
    torch.autograd.backward(expected, grad_outputs)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad, atol=2e-6, rtol=2e-5)


def test_standard_o_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6103)
    batch, seq, heads, head_dim, out_dim = 2, 3, 4, 5, 7
    hidden = torch.randn(batch, seq, heads, head_dim, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(out_dim, heads * head_dim, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(out_dim, dtype=torch.float64, requires_grad=True)

    k_hidden, k_weight, k_bias = _leaf(hidden, weight, bias)
    s_hidden, s_weight, s_bias = _leaf(hidden, weight, bias)
    actual = VeomniOp("async_ulysses_o", "standard")(
        k_hidden,
        k_weight,
        k_bias,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
    )
    expected = F.linear(s_hidden.view(batch, seq, -1), s_weight, s_bias)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(k_hidden.grad, s_hidden.grad)
    torch.testing.assert_close(k_weight.grad, s_weight.grad)
    torch.testing.assert_close(k_bias.grad, s_bias.grad)


def test_dit_o_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6104)
    batch, seq, hidden, out_dim = 2, 3, 16, 16
    hidden_states = torch.randn(batch, seq, hidden, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(out_dim, hidden, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(out_dim, dtype=torch.float64, requires_grad=True)

    k_hidden, k_weight, k_bias = _leaf(hidden_states, weight, bias)
    s_hidden, s_weight, s_bias = _leaf(hidden_states, weight, bias)
    actual = VeomniOp("async_ulysses_o", "dit")(
        k_hidden,
        k_weight,
        k_bias,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
    )
    expected = F.linear(s_hidden, s_weight, s_bias)
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    expected.sum().backward()
    torch.testing.assert_close(k_hidden.grad, s_hidden.grad)
    torch.testing.assert_close(k_weight.grad, s_weight.grad)
    torch.testing.assert_close(k_bias.grad, s_bias.grad)


@pytest.mark.parametrize("variant", ("standard", "dit"))
def test_output_projection_empty_sequence_matches_sequential(
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, hidden, out_dim = 2, 0, 20, 7
    hidden_states = (
        torch.empty(batch, seq, 4, 5, dtype=torch.float64)
        if variant == "standard"
        else torch.empty(batch, seq, hidden, dtype=torch.float64)
    )
    weight = torch.randn(out_dim, hidden, dtype=torch.float64)
    bias = torch.randn(out_dim, dtype=torch.float64)
    actual_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (hidden_states, weight, bias)]
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (hidden_states, weight, bias)]

    actual = VeomniOp("async_ulysses_o", variant)(
        *actual_inputs,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
    )
    hidden_e = expected_inputs[0].flatten(start_dim=2) if variant == "standard" else expected_inputs[0]
    expected = F.linear(hidden_e, expected_inputs[1], expected_inputs[2])
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    expected.sum().backward()
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)


def test_output_projection_weight_bias_grad_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, heads, head_dim, out_dim = 2, 3, 4, 5, 7
    hidden = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    weight = torch.randn(out_dim, heads * head_dim, requires_grad=True)
    bias = torch.randn(out_dim, requires_grad=True)
    VeomniOp("async_ulysses_o", "standard")(
        hidden,
        weight,
        bias,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
    ).sum().backward()
    assert weight.grad is not None
    assert weight.grad.shape == weight.shape
    assert bias.grad is not None
    assert bias.grad.shape == bias.shape


def test_output_projection_bias_grad_when_weight_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, heads, head_dim, out_dim = 2, 3, 4, 5, 7
    hidden = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    weight = torch.randn(out_dim, heads * head_dim)
    bias = torch.randn(out_dim, requires_grad=True)
    VeomniOp("async_ulysses_o", "standard")(
        hidden,
        weight,
        bias,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        group=object(),
    ).sum().backward()
    assert bias.grad is not None
    assert bias.grad.shape == bias.shape


def test_qkv_projection_bias_grad_when_weights_frozen(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, hidden, head_dim = 2, 3, 20, 5
    hidden_states = torch.randn(batch, seq, hidden, requires_grad=True)
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = _qkv_weights(
        hidden, 20, 10, dtype=torch.float32, requires_grad=False
    )
    q_bias.requires_grad_(True)
    k_bias.requires_grad_(True)
    v_bias.requires_grad_(True)
    query, key, value = VeomniOp("async_ulysses_qkv", "standard")(
        hidden_states,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        None,
        None,
        None,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=seq,
        head_dim=head_dim,
        group=object(),
    )
    (query.sum() + key.sum() + value.sum()).backward()
    for bias in (q_bias, k_bias, v_bias):
        assert bias.grad is not None
        assert bias.grad.shape == bias.shape


def test_nested_rms_handle_is_used(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)

    def dummy_forward(hidden: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
        return hidden * 2, SavedState((hidden, weight), eps)

    def dummy_backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
        return grad_output * 2, None

    dummy = OpEntry(
        op="dummy_rms",
        variant="standard",
        impl="eager",
        description="Test RMSNorm",
        forward=dummy_forward,
        backward=dummy_backward,
    )
    hidden = torch.randn(2, 3, 16)
    weights = _qkv_weights(16, 16, 16, dtype=torch.float32)
    actual_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (hidden, *weights)]
    expected_inputs = [tensor.detach().clone().requires_grad_(True) for tensor in (hidden, *weights)]
    norm_q = torch.ones(16)
    norm_k = torch.ones(16)
    actual = VeomniOp("async_ulysses_qkv", "dit")(
        *actual_inputs,
        norm_q,
        None,
        norm_k,
        None,
        seq_dimension=1,
        head_dimension=2,
        unpadded_dim_size=3,
        group=object(),
        norm_type="rmsnorm",
        normalized_shape=16,
        eps=1e-6,
        rms_norm=dummy,
    )
    hidden_e, q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = expected_inputs
    expected = (
        F.linear(hidden_e, q_weight, q_bias) * 2,
        F.linear(hidden_e, k_weight, k_bias) * 2,
        F.linear(hidden_e, v_weight, v_bias),
    )
    for actual_output, expected_output in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_output, expected_output)

    grad_outputs = tuple(torch.randn_like(output) for output in actual)
    torch.autograd.backward(actual, grad_outputs)
    torch.autograd.backward(expected, grad_outputs)
    for actual_input, expected_input in zip(actual_inputs, expected_inputs, strict=True):
        torch.testing.assert_close(actual_input.grad, expected_input.grad)

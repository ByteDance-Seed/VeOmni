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

import importlib

import pytest
import torch
import torch.nn.functional as F
from torch import Tensor

from veomni.kernels import VeomniKernel, resolve_kernel
from veomni.kernels.registry import KernelEntry, SavedState


_EAGER_MODULES = (
    "veomni.kernels.async_ulysses.qkv_proj.standard.eager",
    "veomni.kernels.async_ulysses.qkv_proj.dit.eager",
    "veomni.kernels.async_ulysses.o_proj.standard.eager",
    "veomni.kernels.async_ulysses.o_proj.dit.eager",
)


def _mock_identity_comm(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_all_to_all(tensor: Tensor, **kwargs):
        return (lambda: tensor) if kwargs.get("async_op") else tensor

    def fake_pad(tensor: Tensor, dim, **kwargs):
        return tensor

    def fake_unpad(tensor: Tensor, dim, size, **kwargs):
        return tensor

    for name in _EAGER_MODULES:
        module = importlib.import_module(name)
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
        rms = VeomniKernel("rms_norm", "standard")
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
        rms = VeomniKernel("rms_norm", "standard")
        query = rms(query, norm_q_weight, eps=eps)
        key = rms(key, norm_k_weight, eps=eps)
    return query, key, value


def test_registered_eager_rows() -> None:
    for kernel, variant in (
        ("async_ulysses_qkv", "standard"),
        ("async_ulysses_qkv", "dit"),
        ("async_ulysses_o", "standard"),
        ("async_ulysses_o", "dit"),
    ):
        entry = resolve_kernel(kernel, variant, "eager")
        assert entry.forward is not None
        assert entry.backward is not None
    with pytest.raises(KeyError):
        resolve_kernel("async_ulysses_qkv", "bagel", "eager")


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

    kernel_q, kernel_k, kernel_v = VeomniKernel("async_ulysses_qkv", "standard")(
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

    kernel_q, kernel_k, kernel_v = VeomniKernel("async_ulysses_qkv", "dit")(
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


def test_standard_o_matches_sequential(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    torch.manual_seed(6103)
    batch, seq, heads, head_dim, out_dim = 2, 3, 4, 5, 7
    hidden = torch.randn(batch, seq, heads, head_dim, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(out_dim, heads * head_dim, dtype=torch.float64, requires_grad=True)
    bias = torch.randn(out_dim, dtype=torch.float64, requires_grad=True)

    k_hidden, k_weight, k_bias = _leaf(hidden, weight, bias)
    s_hidden, s_weight, s_bias = _leaf(hidden, weight, bias)
    actual = VeomniKernel("async_ulysses_o", "standard")(
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
    actual = VeomniKernel("async_ulysses_o", "dit")(
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


def test_output_projection_weight_bias_grad_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    _mock_identity_comm(monkeypatch)
    batch, seq, heads, head_dim, out_dim = 2, 3, 4, 5, 7
    hidden = torch.randn(batch, seq, heads, head_dim, requires_grad=True)
    weight = torch.randn(out_dim, heads * head_dim, requires_grad=True)
    bias = torch.randn(out_dim, requires_grad=True)
    VeomniKernel("async_ulysses_o", "standard")(
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
    VeomniKernel("async_ulysses_o", "standard")(
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
    query, key, value = VeomniKernel("async_ulysses_qkv", "standard")(
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
    seen: list[str] = []

    def dummy_forward(hidden: Tensor, weight: Tensor, *, eps: float) -> tuple[Tensor, SavedState]:
        seen.append("fwd")
        return hidden * 2, SavedState((hidden, weight), eps)

    def dummy_backward(grad_output: Tensor, saved: SavedState) -> tuple[Tensor | None, ...]:
        seen.append("bwd")
        return grad_output * 2, None

    dummy = KernelEntry(
        kernel="dummy_rms",
        variant="standard",
        impl="eager",
        forward=dummy_forward,
        backward=dummy_backward,
    )
    hidden = torch.randn(2, 3, 16, requires_grad=True)
    q_weight, q_bias, k_weight, k_bias, v_weight, v_bias = _qkv_weights(16, 16, 16, dtype=torch.float32)
    norm_q = torch.ones(16)
    norm_k = torch.ones(16)
    query, key, value = VeomniKernel("async_ulysses_qkv", "dit")(
        hidden,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
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
    (query + key + value).sum().backward()
    assert seen == ["fwd", "fwd", "bwd", "bwd"]

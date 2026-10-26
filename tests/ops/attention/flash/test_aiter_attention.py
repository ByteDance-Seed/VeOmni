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

"""AITER's FlashAttention calling-convention adapter."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest
import torch

from veomni.ops import install as install_backend
from veomni.ops.kernels.attention.standard import aiter as aiter_backend


AITER_IMPL = "veomni_flash_attention_aiter"


def _install_fake_aiter(monkeypatch: pytest.MonkeyPatch) -> dict[str, dict]:
    calls: dict[str, dict] = {}

    def _result(q, kwargs):
        if kwargs.get("return_lse") or kwargs.get("return_attn_probs"):
            return torch.zeros_like(q), None
        return torch.zeros_like(q)

    def flash_attn_func(q, k, v, **kwargs):
        calls["dense"] = kwargs
        return _result(q, kwargs)

    def flash_attn_varlen_func(q, k, v, **kwargs):
        calls["varlen"] = kwargs
        return _result(q, kwargs)

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func),
    )
    return calls


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (None, (-1, -1, 0)),
        ((-1, -1), (-1, -1, 0)),
        ((8, 0), (8, 0, 0)),
        ((8, 0, 4), (8, 0, 4)),
    ],
)
def test_window_size_translation(window_size, expected):
    assert aiter_backend.aiter_window_size(window_size) == expected


@pytest.mark.parametrize("window_size", [(8,), (8, 0, 4, 1)])
def test_window_size_rejects_unexpected_width(window_size):
    with pytest.raises(ValueError, match="window_size"):
        aiter_backend.aiter_window_size(window_size)


def test_dense_adapter_forwards_aiter_arguments(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(1, 3, 2, 4, requires_grad=True)

    output = kernels.flash_attn_func(
        q,
        q,
        q,
        dropout_p=0.1,
        softmax_scale=0.5,
        causal=True,
        window_size=(8, 0),
        deterministic=True,
        return_attn_probs=True,
    )

    assert isinstance(output, tuple)
    assert calls["dense"] == {
        "dropout_p": 0.1,
        "softmax_scale": 0.5,
        "causal": True,
        "window_size": (8, 0, 0),
        "deterministic": True,
        "return_lse": True,
        "return_attn_probs": True,
    }


def test_no_grad_adapter_requests_no_lse_and_returns_tensor(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(1, 3, 2, 4)

    with torch.no_grad():
        output = kernels.flash_attn_func(q, q, q)

    assert torch.is_tensor(output)
    assert calls["dense"]["return_lse"] is False


def test_dense_adapter_rejects_softcap(monkeypatch: pytest.MonkeyPatch):
    _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(1, 3, 2, 4)

    with pytest.raises(ValueError, match="softcap on the dense"):
        kernels.flash_attn_func(q, q, q, softcap=2.0)


@pytest.mark.parametrize("path", ("dense", "varlen"))
@pytest.mark.parametrize("sink_kwarg", ("s_aux", "learnable_sink"))
def test_adapter_rejects_attention_sinks(monkeypatch: pytest.MonkeyPatch, path: str, sink_kwarg: str):
    _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(1, 3, 2, 4)
    fn = kernels.flash_attn_func if path == "dense" else kernels.flash_attn_varlen_func

    with pytest.raises(ValueError, match="attention sinks"):
        fn(q, q, q, **{sink_kwarg: torch.zeros(2)})


def test_varlen_adapter_rejects_softcap_under_autograd(monkeypatch: pytest.MonkeyPatch):
    _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(3, 2, 4, requires_grad=True)

    with pytest.raises(ValueError, match="softcap under autograd"):
        kernels.flash_attn_varlen_func(q, q, q, softcap=0.5)


def test_varlen_inference_maps_softcap(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(3, 2, 4)

    with torch.no_grad():
        output = kernels.flash_attn_varlen_func(q, q, q, softcap=0.5, window_size=None)

    assert torch.is_tensor(output)
    assert calls["varlen"]["logits_soft_cap"] == 0.5
    assert calls["varlen"]["window_size"] == (-1, -1, 0)
    assert calls["varlen"]["return_lse"] is False


def test_grad_mode_without_input_grad_allows_inference_softcap(monkeypatch: pytest.MonkeyPatch):
    calls = _install_fake_aiter(monkeypatch)
    kernels = install_backend._load_veomni_local_flash_kernel(AITER_IMPL)
    q = torch.randn(3, 2, 4)

    output = kernels.flash_attn_varlen_func(q, q, q, softcap=0.5)

    assert torch.is_tensor(output)
    assert calls["varlen"]["logits_soft_cap"] == 0.5
    assert calls["varlen"]["return_lse"] is False


def test_loader_reports_missing_aiter_entry_points(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace())

    with pytest.raises(RuntimeError, match="does not expose"):
        install_backend._load_veomni_local_flash_kernel(AITER_IMPL)

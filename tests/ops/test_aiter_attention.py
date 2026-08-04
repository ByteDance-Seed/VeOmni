"""Dispatch/adapter tests for the ``aiter`` attention backend.

``aiter`` kernels only run on AMD ROCm, so these tests deliberately exercise the
wiring rather than the kernels: the registry entry, the argument translation, and
the shim that adapts aiter's calling convention to the one Transformers'
``_flash_attention_forward`` expects. A fake ``aiter`` module is injected so the
whole file runs on any accelerator (including the CUDA CI runners).
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import aiter as veomni_aiter
from veomni.ops.kernels.attention import flash as veomni_flash


AITER_IMPL = "veomni_flash_attention_aiter_with_sp"


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"_attn_implementation": AITER_IMPL})()
        self.is_causal = True
        self.layer_idx = 1
        self.proj = nn.Linear(4, 4)


def _install_fake_aiter(monkeypatch):
    """Register a fake ``aiter`` module and return the recorded call kwargs."""
    calls = {}

    def flash_attn_func(q, k, v, **kwargs):
        calls["dense"] = kwargs
        return (torch.zeros_like(q), None)

    def flash_attn_varlen_func(q, k, v, **kwargs):
        calls["varlen"] = kwargs
        return (torch.zeros_like(q), None)

    monkeypatch.setitem(
        sys.modules,
        "aiter",
        SimpleNamespace(flash_attn_func=flash_attn_func, flash_attn_varlen_func=flash_attn_varlen_func),
    )
    return calls


def test_aiter_is_registered_as_veomni_custom_attention():
    assert veomni_flash._VEOMNI_FLASH_ATTN_IMPL_MAPPING[AITER_IMPL] == "aiter"
    assert veomni_flash._is_veomni_custom_flash_attention(AITER_IMPL)


def test_aiter_is_routed_to_the_flash_backend_by_the_fused_dispatcher():
    """``fused_attention_forward`` picks a backend from ``_ATTENTION_FORWARD_DISPATCH``;
    aiter must land on the flash implementation rather than flex or an unknown-key error."""
    assert veomni_attention._ATTENTION_FORWARD_DISPATCH[AITER_IMPL] is veomni_flash.flash_attention_forward


def test_veomni_backend_rewrites_aiter_to_the_sp_implementation(monkeypatch):
    """Under the veomni modeling backend the user-facing name ``aiter`` must be
    swapped for the SP-aware implementation, like fa2/fa3/fa4 are."""
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    config = OpsImplementationConfig(attn_implementation="aiter")
    assert config.attn_implementation == AITER_IMPL


@pytest.mark.parametrize(
    ("window_size", "expected"),
    [
        (None, (-1, -1, 0)),  # full attention
        ((-1, -1), (-1, -1, 0)),
        ((8, 0), (8, 0, 0)),  # flash-attn 2-tuple gains aiter's sink slot
        ((8, 0, 4), (8, 0, 4)),  # already a 3-tuple: passed through
    ],
)
def test_window_size_is_translated_to_aiter_three_tuple(window_size, expected):
    assert veomni_aiter.aiter_window_size(window_size) == expected


def test_dense_path_requests_lse_and_translates_window(monkeypatch):
    """aiter's forward asserts return_lse whenever autograd is enabled."""
    calls = _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    q = torch.randn(1, 3, 2, 4)
    kernels.flash_attn_func(q, q, q, causal=True, window_size=(8, 0), softmax_scale=0.5)

    assert calls["dense"]["return_lse"] is True
    assert calls["dense"]["window_size"] == (8, 0, 0)
    assert calls["dense"]["causal"] is True
    assert calls["dense"]["softmax_scale"] == 0.5


@pytest.mark.parametrize("path", ["dense", "varlen"])
def test_lse_is_not_requested_without_autograd(monkeypatch, path):
    """Under no_grad aiter needs no log-sum-exp, and asking for it makes the kernel
    write a buffer Transformers immediately discards."""
    calls = _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    q = torch.randn(6, 2, 4)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    with torch.no_grad():
        if path == "dense":
            kernels.flash_attn_func(q.unsqueeze(0), q.unsqueeze(0), q.unsqueeze(0))
        else:
            kernels.flash_attn_varlen_func(
                q, q, q, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=3, max_seqlen_k=3
            )

    assert calls[path]["return_lse"] is False


def test_dense_path_rejects_softcap_instead_of_ignoring_it(monkeypatch):
    """aiter.flash_attn_func has no softcap argument; silently dropping it would
    change the model's maths, so the shim must fail loudly."""
    _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    q = torch.randn(1, 3, 2, 4)
    with pytest.raises(ValueError, match="softcap"):
        kernels.flash_attn_func(q, q, q, softcap=30.0)


def test_varlen_path_maps_softcap_to_logits_soft_cap(monkeypatch):
    calls = _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    q = torch.randn(6, 2, 4)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    kernels.flash_attn_varlen_func(
        q,
        q,
        q,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=3,
        max_seqlen_k=3,
        softcap=30.0,
        window_size=(8, 0),
    )

    varlen = calls["varlen"]
    assert varlen["logits_soft_cap"] == 30.0
    assert "softcap" not in varlen
    assert varlen["return_lse"] is True
    assert varlen["window_size"] == (8, 0, 0)
    assert varlen["max_seqlen_q"] == 3


def test_attention_forward_dispatches_under_the_aiter_implementation(monkeypatch):
    """The generic entry point must forward aiter's implementation name through,
    the same contract the fa2/fa3/fa4 backends rely on."""
    captured = {}

    def fake_flash_attention_forward(query, key, value, attention_mask, **kwargs):
        captured.update(kwargs)
        return torch.zeros_like(query)

    monkeypatch.setattr(veomni_flash, "_flash_attention_forward", fake_flash_attention_forward)

    module = _FakeAttentionModule()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 1, 3, 4)
    value = torch.randn(1, 1, 3, 4)

    output, attn_weights = veomni_flash.flash_attention_forward(
        module, query, key, value, attention_mask=None, scaling=0.5
    )

    assert output.shape == (1, 3, 2, 4)
    assert attn_weights is None
    assert captured["attn_implementation"] == AITER_IMPL


def test_missing_aiter_raises_actionable_import_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "aiter", None)
    with pytest.raises(ImportError, match="aiter"):
        veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

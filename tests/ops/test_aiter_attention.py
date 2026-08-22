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
    """Register a fake ``aiter`` module and return the recorded call kwargs.

    The fakes mirror the real return contract: a bare tensor unless ``return_lse``
    (or ``return_attn_probs``) is requested, a tuple otherwise. That is what makes the
    ``no_grad`` path exercise the same shape Transformers has to unwrap in production.
    """
    calls = {}

    def _result(q, kwargs):
        if kwargs.get("return_lse") or kwargs.get("return_attn_probs"):
            return (torch.zeros_like(q), None)
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


@pytest.mark.parametrize("window_size", [(8,), (8, 0, 4, 1)])
def test_unexpected_window_size_width_is_rejected(window_size):
    """A width other than 2 or 3 is a wiring bug; passing it through would surface as a
    confusing kernel-level failure instead."""
    with pytest.raises(ValueError, match="window_size"):
        veomni_aiter.aiter_window_size(window_size)


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


@pytest.mark.parametrize("path", ["dense", "varlen"])
@pytest.mark.parametrize("sink_kwarg", ["s_aux", "learnable_sink"])
def test_attention_sinks_are_rejected_instead_of_dropped(monkeypatch, path, sink_kwarg):
    """Transformers' sink forwarding is all-or-nothing: with neither name declared it
    drops them silently. Declaring both and raising is what makes a sink model fail
    loudly rather than train with different maths (DeepSeek-V4 always passes s_aux)."""
    _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    q = torch.randn(6, 2, 4)
    cu_seqlens = torch.tensor([0, 3, 6], dtype=torch.int32)
    sinks = {sink_kwarg: torch.zeros(2)}

    with pytest.raises(ValueError, match="attention sinks"):
        if path == "dense":
            kernels.flash_attn_func(q.unsqueeze(0), q.unsqueeze(0), q.unsqueeze(0), **sinks)
        else:
            kernels.flash_attn_varlen_func(
                q, q, q, cu_seqlens_q=cu_seqlens, cu_seqlens_k=cu_seqlens, max_seqlen_q=3, max_seqlen_k=3, **sinks
            )


def test_transformers_kwarg_selection_matches_the_shim_signature(monkeypatch):
    """Transformers decides which optional kwargs to forward by introspecting the varlen
    shim, so that mapping is the real contract this backend has to satisfy. Pinning it
    means a signature change cannot quietly stop a kwarg from being forwarded."""
    from transformers.modeling_flash_attention_utils import _lazy_define_process_function

    _install_fake_aiter(monkeypatch)
    kernels = veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

    supports_mapping = _lazy_define_process_function(kernels.flash_attn_varlen_func).keywords["supports_mapping"]

    for kwarg in ("dropout_p", "window_size", "deterministic", "softcap", "max_seqlen_q", "max_seqlen_k"):
        assert supports_mapping[kwarg], f"Transformers would stop forwarding {kwarg}"
    # Declared so the sinks reach the shim and get rejected rather than dropped.
    assert supports_mapping["s_aux"]
    assert supports_mapping["learnable_sink"]


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


@pytest.mark.parametrize("grad_enabled", [True, False])
def test_forward_unwraps_both_return_shapes_end_to_end(monkeypatch, grad_enabled):
    """Drives the real Transformers unwrap with the fake kernels behind it.

    Under autograd aiter returns ``(out, lse)``; under ``no_grad`` it returns a bare
    tensor. Transformers guards each call site with ``isinstance(out, tuple)``, and this
    pins that contract so a change there fails here instead of failing an eval run.
    """
    _install_fake_aiter(monkeypatch)
    monkeypatch.setattr(veomni_flash, "get_parallel_state", lambda: SimpleNamespace(ulysses_enabled=False))
    veomni_flash.patch_transformers_hub_kernel_loader_for_veomni()

    module = _FakeAttentionModule()
    query = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
    key = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)
    value = torch.randn(1, 1, 3, 4, dtype=torch.bfloat16)

    with torch.set_grad_enabled(grad_enabled):
        output, attn_weights = veomni_flash.flash_attention_forward(
            module, query, key, value, attention_mask=None, scaling=0.5
        )

    # A tuple leaking through would make this a 2-element sequence, not a tensor.
    assert isinstance(output, torch.Tensor)
    assert output.shape == (1, 3, 2, 4)
    assert attn_weights is None


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


def test_aiter_without_ck_kernels_fails_at_load_not_at_forward(monkeypatch):
    """aiter imports on a non-ROCm host but exports only its Triton ops, so the flash
    entry points are simply absent. Verified against the published ROCm image run without
    `--device /dev/kfd`. Catching it here beats an AttributeError mid-forward."""
    monkeypatch.setitem(sys.modules, "aiter", SimpleNamespace())

    with pytest.raises(RuntimeError, match="does not expose"):
        veomni_flash._load_veomni_local_flash_kernel(AITER_IMPL)

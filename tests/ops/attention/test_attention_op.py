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

"""Install and registry contract for ``veomni_*`` attention names."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from transformers.integrations.flex_attention import flex_attention_forward as hf_flex_attention_forward
from transformers.integrations.sdpa_attention import sdpa_attention_forward as hf_sdpa_attention_forward
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from veomni.ops import OP_REGISTRY, VeomniOp
from veomni.ops import registry as op_registry
from veomni.ops.install import apply_veomni_attention_patch
from veomni.ops.kernels.attention import lookup
from veomni.ops.kernels.attention import ulysses as ulysses_backend
from veomni.ops.kernels.attention.standard.flash import flash_attention_forward
from veomni.ops.kernels.attention.standard.flex import flex_attention_forward
from veomni.ops.kernels.attention.standard.magi import magi_attention_forward
from veomni.ops.kernels.attention.standard.sage import sage_attention_forward
from veomni.ops.kernels.attention.standard.sdpa import sdpa_attention_forward
from veomni.ops.kernels.attention.ulysses import should_apply_ulysses
from veomni.ops.platform import (
    ANY_DEVICE,
    NVIDIA_SM90_PLUS,
    ROCM_GPU,
    GpuKernelRequirement,
    MluKernelRequirement,
    NpuKernelRequirement,
    NvidiaGpuPlatform,
)


_VEOMNI_FORWARDS = {
    "veomni_flash_attention_2": flash_attention_forward,
    "veomni_flash_attention_3": flash_attention_forward,
    "veomni_flash_attention_4": flash_attention_forward,
    "veomni_flex_attention": flex_attention_forward,
    "veomni_magi_attention": magi_attention_forward,
    "veomni_sage_attention": sage_attention_forward,
    "veomni_sdpa": sdpa_attention_forward,
}

_STANDARD_IMPLS = (
    "eager",
    "sdpa",
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "flex_attention",
    "magi_attention",
    "native-sparse",
    "veomni_flash_attention_2",
    "veomni_flash_attention_3",
    "veomni_flash_attention_4",
    "veomni_flex_attention",
    "veomni_magi_attention",
    "veomni_sage_attention",
    "veomni_sdpa",
)

_PUBLIC_PARAMETERS = (
    "module",
    "query",
    "key",
    "value",
    "attention_mask",
    "dropout",
    "scaling",
    "sliding_window",
    "softcap",
    "kwargs",
)

_ANY_DEVICE_IMPLS = (
    "eager",
    "sdpa",
    "flex_attention",
    "native-sparse",
    "veomni_flex_attention",
    "veomni_sdpa",
)
_FA2_IMPLS = ("flash_attention_2", "veomni_flash_attention_2")
_FA3_IMPLS = ("flash_attention_3", "veomni_flash_attention_3")
_FA4_IMPLS = ("flash_attention_4", "veomni_flash_attention_4")
_MAGI_IMPLS = ("magi_attention", "veomni_magi_attention")


def test_standard_rows_are_registered():
    assert OP_REGISTRY.list_registered("attention", "standard") == list(_STANDARD_IMPLS)
    expected_keys = {("attention", "standard", impl, ANY_DEVICE) for impl in _ANY_DEVICE_IMPLS}
    expected_keys.update(("attention", "standard", impl, "cuda") for impl in (*_FA2_IMPLS, *_FA3_IMPLS))
    expected_keys.update(("attention", "standard", impl, "cuda") for impl in (*_FA4_IMPLS, *_MAGI_IMPLS))
    expected_keys.add(("attention", "standard", "veomni_sage_attention", "cuda"))
    expected_keys.update(("attention", "standard", impl, "npu") for impl in _FA2_IMPLS)
    expected_keys.update(("attention", "standard", impl, "mlu") for impl in _FA2_IMPLS)
    assert {key for key in OP_REGISTRY._entries if key[0] == "attention"} == expected_keys

    for impl in _STANDARD_IMPLS:
        entries = [entry for entry in OP_REGISTRY.list_entries("attention", "standard") if entry.impl == impl]
        assert entries
        assert all(entry.wrapper is entries[0].wrapper for entry in entries)


def test_registered_attention_rows_share_public_signature_contract():
    for impl in _STANDARD_IMPLS:
        wrapper = next(
            entry.wrapper for entry in OP_REGISTRY.list_entries("attention", "standard") if entry.impl == impl
        )
        assert wrapper is not None
        parameters = inspect.signature(wrapper).parameters
        assert tuple(parameters) == _PUBLIC_PARAMETERS
        assert parameters["attention_mask"].default is inspect.Parameter.empty
        assert parameters["dropout"].default == 0.0
        assert parameters["scaling"].default is None
        assert parameters["sliding_window"].default is None
        assert parameters["softcap"].default is None
        assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_veomni_hf_attention_adapters_share_extended_signature_contract():
    for forward in set(_VEOMNI_FORWARDS.values()):
        parameters = inspect.signature(forward).parameters
        assert tuple(parameters) == (*_PUBLIC_PARAMETERS[:-1], "skip_ulysses", "kwargs")
        assert parameters["skip_ulysses"].default is False
        assert parameters["kwargs"].kind is inspect.Parameter.VAR_KEYWORD


def test_veomni_names_register_on_hf_dict_without_overwriting_stock():
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward
    assert ALL_ATTENTION_FUNCTIONS["flex_attention"] is hf_flex_attention_forward
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is hf_sdpa_attention_forward
    assert "veomni_flash_attention_2" in OP_REGISTRY.list_registered("attention", "standard")


def test_cpu_availability_excludes_accelerator_attention(monkeypatch):
    monkeypatch.setattr(op_registry, "get_device_type", lambda: "cpu")

    assert OP_REGISTRY.list_available("attention", "standard") == list(_ANY_DEVICE_IMPLS)
    for impl in (*_FA2_IMPLS, *_FA3_IMPLS, *_FA4_IMPLS, *_MAGI_IMPLS, "veomni_sage_attention"):
        with pytest.raises(RuntimeError, match="not registered for device 'cpu'"):
            OP_REGISTRY.resolve("attention", "standard", impl)


def test_attention_rows_declare_platform_and_package_requirements():
    sm80_plus = NvidiaGpuPlatform(min_cc=80)
    expected = (
        (_FA2_IMPLS, (sm80_plus, ROCM_GPU), ("flash_attn",)),
        (_FA3_IMPLS, (NVIDIA_SM90_PLUS,), ("flash_attn_interface",)),
        (_FA4_IMPLS, (NVIDIA_SM90_PLUS,), ("flash_attn.cute",)),
        (_MAGI_IMPLS, (NVIDIA_SM90_PLUS,), ("magi_attention",)),
        (("veomni_sage_attention",), (sm80_plus,), ("sageattention",)),
    )

    for impls, platforms, requires in expected:
        for impl in impls:
            entry = OP_REGISTRY._entries[("attention", "standard", impl, "cuda")]
            assert isinstance(entry.requirement, GpuKernelRequirement)
            assert entry.requirement.platforms == platforms
            assert entry.requires == requires

    for impl in _FA2_IMPLS:
        assert isinstance(
            OP_REGISTRY._entries[("attention", "standard", impl, "npu")].requirement, NpuKernelRequirement
        )
        mlu_entry = OP_REGISTRY._entries[("attention", "standard", impl, "mlu")]
        assert isinstance(mlu_entry.requirement, MluKernelRequirement)
        assert mlu_entry.requires == ("flash_attn",)


def test_attention_packages_participate_in_gpu_availability(monkeypatch):
    monkeypatch.setattr(op_registry, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(NvidiaGpuPlatform, "matches", lambda self: True)
    monkeypatch.setattr(op_registry, "is_package_available", lambda _package: False)

    assert OP_REGISTRY.list_available("attention", "standard") == list(_ANY_DEVICE_IMPLS)


def test_magi_short_name_uses_installed_veomni_interface(monkeypatch):
    captured = {}

    def replacement(module, query, key, value, attention_mask, **kwargs):
        captured.update(module=module, query=query, kwargs=kwargs)
        return query.transpose(1, 2), "magi-metadata"

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "veomni_magi_attention", replacement)
    wrapper = OP_REGISTRY._entries[("attention", "standard", "magi_attention", "cuda")].wrapper
    module = SimpleNamespace(is_causal=True)
    query = torch.randn(2, 4, 3, 8)

    output, metadata = wrapper(module, query, query, query, None, scaling=0.5)

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["kwargs"]["scaling"] == 0.5
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert metadata == "magi-metadata"


def test_apply_veomni_attention_patch_is_idempotent():
    apply_veomni_attention_patch()
    apply_veomni_attention_patch()
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward


def test_apply_ops_patch_registers_attention_names():
    from veomni.ops import apply_ops_patch

    apply_ops_patch()
    for name, forward in _VEOMNI_FORWARDS.items():
        assert ALL_ATTENTION_FUNCTIONS[name] is forward


def test_apply_ops_patch_skips_hf_backend(monkeypatch):
    from veomni.ops import install

    calls = []
    monkeypatch.setattr(install, "get_env", lambda _name: "hf")
    monkeypatch.setattr(install, "apply_veomni_attention_patch", lambda: calls.append(True))

    install.apply_ops_patch()

    assert calls == []


def eager_attention_forward(module, query, key, value, attention_mask, **kwargs):
    del module, key, value, attention_mask, kwargs
    return query.transpose(1, 2), "eager-local"


class _EagerAttentionModule(torch.nn.Module):
    """Lookup resolves ``eager_attention_forward`` from this test module."""


def test_lookup_eager_uses_module_local_forward():
    module = _EagerAttentionModule()
    query = torch.randn(2, 4, 3, 8)
    output, metadata = lookup("eager")(module, query, query, query, None, dropout=0.0, scaling=0.5)
    torch.testing.assert_close(output, query.transpose(1, 2))
    assert metadata == "eager-local"

    op = VeomniOp("attention", "standard", "eager")
    op_output, op_metadata = op(module, query, query, query, None, dropout=0.0, scaling=0.5)
    torch.testing.assert_close(op_output, query.transpose(1, 2))
    assert op_metadata == "eager-local"


def test_lookup_dispatches_through_hf_dict(monkeypatch):
    captured = {}

    def replacement(module, query, key, value, attention_mask, **kwargs):
        captured.update(module=module, query=query, kwargs=kwargs)
        return query.transpose(1, 2) + 1, "attention-metadata"

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "veomni_sdpa", replacement)
    module = SimpleNamespace(is_causal=True)
    query = torch.randn(2, 4, 3, 8)
    output, metadata = lookup("veomni_sdpa")(module, query, query, query, None, dropout=0.0, scaling=0.5)

    assert captured["module"] is module
    assert captured["query"] is query
    assert captured["kwargs"]["scaling"] == 0.5
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert metadata == "attention-metadata"


def test_should_apply_ulysses_follows_parallel_state(monkeypatch):
    def _set_state(*, ulysses_size: int, async_enabled: bool) -> None:
        monkeypatch.setattr(
            ulysses_backend,
            "get_parallel_state",
            lambda: SimpleNamespace(ulysses_size=ulysses_size, async_enabled=async_enabled),
        )

    _set_state(ulysses_size=1, async_enabled=False)
    assert not should_apply_ulysses()
    _set_state(ulysses_size=2, async_enabled=False)
    assert should_apply_ulysses()
    assert not should_apply_ulysses(skip_ulysses=True)
    _set_state(ulysses_size=2, async_enabled=True)
    assert not should_apply_ulysses()


def test_ulysses_helpers_preserve_layout(monkeypatch):
    exchanges = []

    def fake_gather_seq(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("prepare", tensor.shape, seq_dim, head_dim, group))
        return tensor

    def fake_gather_heads(tensor, *, seq_dim, head_dim, group):
        exchanges.append(("restore", tensor.shape, seq_dim, head_dim, group))
        return tensor

    monkeypatch.setattr(ulysses_backend, "gather_seq_scatter_heads", fake_gather_seq)
    monkeypatch.setattr(ulysses_backend, "gather_heads_scatter_seq", fake_gather_heads)
    group = object()
    query = torch.randn(2, 5, 4, 8)
    key = torch.randn(2, 5, 1, 8)
    value = torch.randn(2, 5, 1, 8)

    prepared_query, prepared_key, prepared_value, query_heads = ulysses_backend.prepare_ulysses_qkv(
        query, key, value, group=group, ulysses_size=2
    )
    restored = ulysses_backend.restore_ulysses_output(prepared_query[:, :, :2], group=group)

    assert query_heads == 4
    torch.testing.assert_close(prepared_key, key.repeat_interleave(2, dim=2))
    torch.testing.assert_close(prepared_value, value.repeat_interleave(2, dim=2))
    assert [item[0] for item in exchanges] == ["prepare", "prepare", "prepare", "restore"]
    assert restored.shape == (2, 5, 2, 8)


def test_ulysses_helpers_reject_nondivisible_key_value_heads():
    query = torch.randn(1, 5, 8, 8)
    key = torch.randn(1, 5, 3, 8)
    value = torch.randn(1, 5, 3, 8)
    with pytest.raises(AssertionError, match="must be divisible"):
        ulysses_backend.prepare_ulysses_qkv(query, key, value, group=object(), ulysses_size=4)


def test_ulysses_head_auxiliary_slices_global_vector_by_rank(monkeypatch):
    monkeypatch.setattr(ulysses_backend.dist, "get_rank", lambda group: 1)
    sliced = ulysses_backend.slice_ulysses_head_auxiliary(
        torch.arange(4),
        query_head_count=4,
        local_query_head_count=2,
        group=object(),
    )
    torch.testing.assert_close(sliced, torch.tensor([2, 3]))

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

import gc
import importlib.util
import json
import os
import statistics
import sys
import time
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask
from transformers.masking_utils import bidirectional_mask_function, causal_mask_function

from veomni.ops import build_ALL_OPS
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import flex as flex_backend
from veomni.ops.kernels.attention import magi as magi_backend
from veomni.ops.kernels.attention.magi import _fa4 as magi_fa4_backend
from veomni.ops.kernels.attention.magi import mask as magi_mask_backend
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    empty_cache,
    get_device_type,
    get_torch_device,
    synchronize,
)


class _FakeAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation="veomni_magi_attention_with_sp")


def _mask(
    sequence_length: int,
    *,
    attn_type_map: torch.Tensor | None = None,
) -> magi_backend.MagiAttentionMask:
    ranges = torch.tensor([[0, sequence_length]], dtype=torch.int32)
    return magi_backend.MagiAttentionMask(
        q_ranges=ranges,
        k_ranges=ranges.clone(),
        attn_type_map=attn_type_map,
    )


def _cp1_state(*, ulysses_enabled: bool = False):
    return SimpleNamespace(
        cp_size=1,
        ulysses_enabled=ulysses_enabled,
        ulysses_group=object() if ulysses_enabled else None,
        ulysses_size=2 if ulysses_enabled else 1,
    )


def _set_fake_cutlass_backend(monkeypatch, *, available: bool) -> dict[str, object] | None:
    fake_package = ModuleType("flash_attn_cute")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "flash_attn_cute", fake_package)
    monkeypatch.delitem(sys.modules, "flash_attn_cute.ffa_fa3", raising=False)
    monkeypatch.delitem(sys.modules, "flash_attn_cute.ffa_fa3.flash_attn_config", raising=False)

    if available:
        fake_cutlass_package = ModuleType("flash_attn_cute.ffa_fa3")
        fake_cutlass_package.__path__ = []
        fake_interface = ModuleType("flash_attn_cute.ffa_fa3.flash_attn_interface")
        fake_interface._flash_attn_forward = lambda: None
        fake_interface._flash_attn_backward = lambda: None
        fake_config = ModuleType("flash_attn_cute.ffa_fa3.flash_attn_config")
        build_flags = {
            "FLASHATTENTION_DISABLE_BACKWARD": False,
            "FLASHATTENTION_DISABLE_SM90": False,
            "FLASHATTENTION_DISABLE_ARBITRARY": False,
            "FLASHATTENTION_DISABLE_FP16": True,
            "FLASHATTENTION_DISABLE_HDIM64": True,
            "FLASHATTENTION_DISABLE_HDIM96": True,
            "FLASHATTENTION_DISABLE_HDIM128": False,
            "FLASHATTENTION_DISABLE_HDIM192": True,
            "FLASHATTENTION_DISABLE_HDIM256": True,
            "FLASHATTENTION_DISABLE_SOFTCAP": True,
            "FLASHATTENTION_NUM_FUNC": [1],
        }
        fake_config.CONFIG = {"build_flags": build_flags}
        fake_cutlass_package.flash_attn_interface = fake_interface
        fake_cutlass_package.flash_attn_config = fake_config
        monkeypatch.setitem(sys.modules, "flash_attn_cute.ffa_fa3", fake_cutlass_package)
        monkeypatch.setitem(sys.modules, "flash_attn_cute.ffa_fa3.flash_attn_config", fake_config)
        return build_flags

    return None


def _set_fake_cuda_runtime(
    monkeypatch,
    *,
    current_stack_size: int,
    get_error: int = 0,
    set_error: int = 0,
    configured_stack_size: int | None = None,
) -> list[tuple[object, int]]:
    stack_limit = object()
    set_calls = []
    state = {"stack_size": current_stack_size}
    runtime = SimpleNamespace(
        cudaLimit=SimpleNamespace(cudaLimitStackSize=stack_limit),
        cudaError_t=SimpleNamespace(cudaSuccess=0),
    )
    runtime.cudaDeviceGetLimit = lambda limit: (get_error, state["stack_size"])

    def set_limit(limit, value):
        set_calls.append((limit, value))
        if set_error == 0:
            state["stack_size"] = value if configured_stack_size is None else configured_stack_size
        return (set_error,)

    runtime.cudaDeviceSetLimit = set_limit
    cuda_package = ModuleType("cuda")
    cuda_package.__path__ = []
    bindings_package = ModuleType("cuda.bindings")
    bindings_package.runtime = runtime
    cuda_package.bindings = bindings_package
    monkeypatch.setitem(sys.modules, "cuda", cuda_package)
    monkeypatch.setitem(sys.modules, "cuda.bindings", bindings_package)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return set_calls


def test_magi_attention_module_slot_preserves_public_ffa_contract(monkeypatch):
    captured = {}

    def fake_backend(query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs):
        captured.update(
            query=query,
            key=key,
            value=value,
            q_ranges=q_ranges,
            k_ranges=k_ranges,
            attn_type_map=attn_type_map,
            kwargs=kwargs,
        )
        output = query + 1
        meta = SimpleNamespace(lse=torch.ones(query.shape[:2]))
        return output, meta

    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", fake_backend)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)
    attn_type_map = torch.tensor([1], dtype=torch.int32)
    attention_mask = _mask(8, attn_type_map=attn_type_map)

    output, lse = veomni_attention.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        attention_mask,
        scaling=0.25,
        softcap=30.0,
    )

    assert captured["query"].shape == (8, 4, 16)
    assert captured["key"].shape == (8, 2, 16)
    assert captured["value"].shape == (8, 2, 16)
    assert captured["q_ranges"] is attention_mask.q_ranges
    assert captured["k_ranges"] is attention_mask.k_ranges
    assert captured["attn_type_map"] is attn_type_map
    assert captured["kwargs"] == {"softmax_scale": 0.25, "softcap": 30.0}
    torch.testing.assert_close(output, query.transpose(1, 2) + 1)
    assert lse.shape == (1, 4, 8)
    assert dict(build_ALL_OPS())["_magi_attention_forward"] is fake_backend


def test_default_magi_backend_lazily_calls_package_fa4_backend(monkeypatch):
    captured = {}
    fa4_attn_arg = object()

    def fake_get_attn_arg(query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim=None):
        captured["metadata_inputs"] = (query, key, q_ranges, k_ranges, attn_type_map, metadata_head_dim)
        return fa4_attn_arg

    def fake_apply(*args):
        captured["apply_args"] = args
        return "output", "lse"

    class FakeAttnForwardMeta:
        def __init__(self, *, lse, max_logits):
            self.lse = lse
            self.max_logits = max_logits

    fake_package = ModuleType("magi_attention")
    fake_package.__path__ = []
    fake_api = ModuleType("magi_attention.api")
    fake_api.AttnForwardMeta = FakeAttnForwardMeta
    monkeypatch.setitem(sys.modules, "magi_attention", fake_package)
    monkeypatch.setitem(sys.modules, "magi_attention.api", fake_api)
    monkeypatch.setattr(magi_fa4_backend, "_get_or_prepare_fa4_attn_arg", fake_get_attn_arg)
    monkeypatch.setattr(magi_fa4_backend._MagiFA4Function, "apply", fake_apply)
    monkeypatch.setattr(
        magi_fa4_backend,
        "_prepare_default_magi_kernel",
        lambda device: (magi_fa4_backend._MAGI_KERNEL_CUTE_JIT, None),
    )
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    value = torch.randn(8, 2, 16)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    result = magi_fa4_backend._default_magi_attention_forward(
        query,
        key,
        value,
        ranges,
        ranges,
        None,
        softmax_scale=0.25,
        softcap=30.0,
    )

    assert result[0] == "output"
    assert result[1].lse == "lse"
    assert result[1].max_logits is None
    assert captured["metadata_inputs"][0] is query
    assert captured["metadata_inputs"][1] is key
    assert captured["metadata_inputs"][2] is ranges
    assert captured["metadata_inputs"][3] is ranges
    assert captured["metadata_inputs"][4] is None
    assert captured["metadata_inputs"][5] is None
    assert captured["apply_args"][0] is query
    assert captured["apply_args"][1] is key
    assert captured["apply_args"][2] is value
    assert captured["apply_args"][3] is ranges
    assert captured["apply_args"][4] is ranges
    assert captured["apply_args"][5:] == (None, 0.25, 30.0, fa4_attn_arg)


def test_default_magi_backend_cold_import_uses_query_device(monkeypatch):
    active_devices = []

    @contextmanager
    def fake_device(device):
        active_devices.append(device)
        yield
        active_devices.pop()

    class DeviceAwareApi(ModuleType):
        def __getattr__(self, name):
            if name == "AttnForwardMeta":
                assert active_devices == [torch.device("cuda:1")]
                return lambda **kwargs: SimpleNamespace(**kwargs)
            raise AttributeError(name)

    fake_package = ModuleType("magi_attention")
    fake_package.__path__ = []
    fake_api = DeviceAwareApi("magi_attention.api")
    monkeypatch.setitem(sys.modules, "magi_attention", fake_package)
    monkeypatch.setitem(sys.modules, "magi_attention.api", fake_api)
    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(
        magi_fa4_backend,
        "_prepare_default_magi_kernel",
        lambda device: (magi_fa4_backend._MAGI_KERNEL_CUTE_JIT, None),
    )
    monkeypatch.setattr(magi_fa4_backend, "_get_or_prepare_fa4_attn_arg", lambda *args: object())
    monkeypatch.setattr(magi_fa4_backend._MagiFA4Function, "apply", lambda *args: ("output", "lse"))
    query = SimpleNamespace(device=torch.device("cuda:1"))
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    output, meta = magi_fa4_backend._default_magi_attention_forward(
        query,
        object(),
        object(),
        ranges,
        ranges,
        None,
        softmax_scale=None,
        softcap=0.0,
    )

    assert output == "output"
    assert meta.lse == "lse"
    assert active_devices == []


def test_magi_fa4_metadata_cache_reuses_only_matching_inputs(monkeypatch):
    built_args = []

    def fake_build(*args):
        built_arg = object()
        built_args.append((args, built_arg))
        return built_arg

    monkeypatch.setattr(magi_fa4_backend, "_prepare_fa4_attn_arg", fake_build)
    monkeypatch.setattr(magi_fa4_backend, "_fa4_cache_entry", None)
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)
    arguments = (query, key, ranges, ranges, None)

    first = magi_fa4_backend._get_or_prepare_fa4_attn_arg(*arguments)
    second = magi_fa4_backend._get_or_prepare_fa4_attn_arg(*arguments)

    ranges[0, 1] = 7
    after_mutation = magi_fa4_backend._get_or_prepare_fa4_attn_arg(*arguments)
    repeated_after_mutation = magi_fa4_backend._get_or_prepare_fa4_attn_arg(*arguments)

    shorter_query = query[:7]
    after_shape_change = magi_fa4_backend._get_or_prepare_fa4_attn_arg(
        shorter_query,
        key,
        ranges,
        ranges,
        None,
    )

    assert first is second
    assert after_mutation is repeated_after_mutation
    assert first is not after_mutation
    assert after_mutation is not after_shape_change
    assert len(built_args) == 3
    assert magi_fa4_backend._fa4_cache_entry is not None
    assert magi_fa4_backend._fa4_cache_entry.metadata_tensors[0] is ranges
    assert magi_fa4_backend._fa4_cache_entry.metadata_tensors[1] is ranges
    assert magi_fa4_backend._fa4_cache_entry.metadata_tensors[2] is None


def test_magi_fa4_metadata_cache_disables_reuse_without_version_counters(monkeypatch):
    built_args = []

    def fake_build(*args):
        built_arg = object()
        built_args.append(built_arg)
        return built_arg

    monkeypatch.setattr(magi_fa4_backend, "_prepare_fa4_attn_arg", fake_build)
    monkeypatch.setattr(magi_fa4_backend, "_fa4_cache_entry", None)
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    with torch.inference_mode():
        ranges = torch.tensor([[0, 8]], dtype=torch.int32)
        first = magi_fa4_backend._get_or_prepare_fa4_attn_arg(query, key, ranges, ranges, None)
        second = magi_fa4_backend._get_or_prepare_fa4_attn_arg(query, key, ranges, ranges, None)

    assert first is not second
    assert len(built_args) == 2
    assert magi_fa4_backend._fa4_cache_entry is None


def test_magi_fa4_metadata_preparation_uses_query_device(monkeypatch):
    active_devices = []

    @contextmanager
    def fake_device(device):
        active_devices.append(device)
        yield
        active_devices.pop()

    class FakeAttnRanges:
        @staticmethod
        def from_ranges(ranges):
            return ranges

    class FakeFA4AttnArg:
        def __init__(self, **kwargs):
            assert active_devices == [torch.device("cuda:1")]
            self.kwargs = kwargs

    fake_common = ModuleType("magi_attention.common")
    fake_common.__path__ = []
    fake_ranges = ModuleType("magi_attention.common.ranges")
    fake_ranges.AttnRanges = FakeAttnRanges
    fake_meta = ModuleType("magi_attention.meta")
    fake_meta.__path__ = []
    fake_collection = ModuleType("magi_attention.meta.collection")
    fake_collection.__path__ = []
    fake_calc_meta = ModuleType("magi_attention.meta.collection.calc_meta")
    fake_calc_meta.FA4AttnArg = FakeFA4AttnArg
    monkeypatch.setitem(sys.modules, "magi_attention.common", fake_common)
    monkeypatch.setitem(sys.modules, "magi_attention.common.ranges", fake_ranges)
    monkeypatch.setitem(sys.modules, "magi_attention.meta", fake_meta)
    monkeypatch.setitem(sys.modules, "magi_attention.meta.collection", fake_collection)
    monkeypatch.setitem(sys.modules, "magi_attention.meta.collection.calc_meta", fake_calc_meta)
    monkeypatch.setattr(torch.cuda, "device", fake_device)

    query = SimpleNamespace(device=torch.device("cuda:1"), shape=(8, 4, 16))
    key = SimpleNamespace(shape=(8, 2, 16))
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    attn_arg = magi_fa4_backend._prepare_fa4_attn_arg(query, key, ranges, ranges, None)

    assert isinstance(attn_arg, FakeFA4AttnArg)
    assert active_devices == []


def test_magi_fa4_explicit_arg_autograd(monkeypatch):
    captured = {}

    def fake_fa4_fwd(*, q, k, v, attn_arg, **kwargs):
        captured["fwd_attn_arg"] = attn_arg
        return q + k + v, q.float().sum(dim=-1)

    def fake_fa4_bwd(*, do, attn_arg, **kwargs):
        captured["bwd_attn_arg"] = attn_arg
        return do, do, do, None

    fake_functional = ModuleType("magi_attention.functional")
    fake_functional.__path__ = []
    fake_fa4 = ModuleType("magi_attention.functional.fa4")
    fake_fa4.fa4_fwd = fake_fa4_fwd
    fake_fa4.fa4_bwd = fake_fa4_bwd
    monkeypatch.setitem(sys.modules, "magi_attention.functional", fake_functional)
    monkeypatch.setitem(sys.modules, "magi_attention.functional.fa4", fake_fa4)

    query = torch.randn(8, 2, 16, requires_grad=True)
    key = torch.randn(8, 2, 16, requires_grad=True)
    value = torch.randn(8, 2, 16, requires_grad=True)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)
    fa4_attn_arg = object()

    output, _ = magi_fa4_backend._MagiFA4Function.apply(
        query,
        key,
        value,
        ranges,
        ranges,
        None,
        None,
        0.0,
        fa4_attn_arg,
    )
    output.sum().backward()

    assert captured == {"fwd_attn_arg": fa4_attn_arg, "bwd_attn_arg": fa4_attn_arg}
    for tensor in (query, key, value):
        torch.testing.assert_close(tensor.grad, torch.ones_like(tensor))


def test_magi_fa4_autograd_detects_range_mutation(monkeypatch):
    fake_functional = ModuleType("magi_attention.functional")
    fake_functional.__path__ = []
    fake_fa4 = ModuleType("magi_attention.functional.fa4")
    fake_fa4.fa4_fwd = lambda *, q, **kwargs: (q.clone(), q.float().sum(dim=-1))
    fake_fa4.fa4_bwd = lambda *, do, **kwargs: (do, do, do, None)
    monkeypatch.setitem(sys.modules, "magi_attention.functional", fake_functional)
    monkeypatch.setitem(sys.modules, "magi_attention.functional.fa4", fake_fa4)

    query = torch.randn(8, 2, 16, requires_grad=True)
    key = torch.randn(8, 2, 16, requires_grad=True)
    value = torch.randn(8, 2, 16, requires_grad=True)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)
    output, _ = magi_fa4_backend._MagiFA4Function.apply(
        query,
        key,
        value,
        ranges,
        ranges,
        None,
        None,
        0.0,
        object(),
    )

    ranges[0, 1] = 7
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        output.sum().backward()


@pytest.mark.parametrize(
    ("compute_capability", "expected_mode"),
    [
        (80, magi_fa4_backend._MAGI_KERNEL_UNSUPPORTED),
        (89, magi_fa4_backend._MAGI_KERNEL_UNSUPPORTED),
        (90, magi_fa4_backend._MAGI_KERNEL_CUTLASS),
        (99, magi_fa4_backend._MAGI_KERNEL_CUTLASS),
        (100, magi_fa4_backend._MAGI_KERNEL_CUTE_JIT),
    ],
)
def test_magi_kernel_mode_follows_query_device(monkeypatch, compute_capability, expected_mode):
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: compute_capability)

    assert magi_fa4_backend._get_magi_kernel_mode(torch.device("cuda")) == expected_mode


def test_magi_sm90_reports_cutlass_installer(monkeypatch):
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(ImportError, match=r"install_magi_sm90\.sh"):
        magi_fa4_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_sm100_plus_does_not_require_cutlass_backend(monkeypatch):
    calls = 0

    def fake_compute_capability(device):
        nonlocal calls
        calls += 1
        return 100

    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", fake_compute_capability)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()

    magi_fa4_backend._prepare_default_magi_kernel(torch.device("cuda"))
    magi_fa4_backend._prepare_default_magi_kernel(torch.device("cuda"))

    assert calls == 1
    assert magi_fa4_backend._prepare_default_magi_kernel.cache_info().currsize == 1


def test_magi_sm90_prepares_cutlass_device_once(monkeypatch):
    prepared = []
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=True)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()
    monkeypatch.setattr(
        magi_fa4_backend,
        "_install_magi_tile_size_compatibility",
        lambda: prepared.append("tile-size"),
    )
    monkeypatch.setattr(
        magi_fa4_backend,
        "_ensure_magi_cutlass_stack_size",
        lambda device: prepared.append(device),
    )

    for device in (torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:1")):
        kernel_mode, build_flags = magi_fa4_backend._prepare_default_magi_kernel(device)
        assert kernel_mode == magi_fa4_backend._MAGI_KERNEL_CUTLASS
        assert build_flags is not None

    assert prepared == ["tile-size", torch.device("cuda:0"), "tile-size", torch.device("cuda:1")]
    assert magi_fa4_backend._prepare_default_magi_kernel.cache_info().currsize == 2


@pytest.mark.parametrize(
    ("query", "softcap", "expected_message"),
    [
        (torch.empty(1, 1, 128, dtype=torch.float16), 0.0, "does not include FP16"),
        (torch.empty(1, 1, 129, dtype=torch.bfloat16), 0.0, "head_dim=129"),
        (torch.empty(1, 1, 128, dtype=torch.bfloat16), 30.0, "does not include softcap"),
    ],
)
def test_magi_sm90_rejects_inputs_excluded_from_cutlass_build(monkeypatch, query, softcap, expected_message):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None

    with pytest.raises((TypeError, ValueError), match=expected_message):
        magi_fa4_backend._validate_magi_cutlass_inputs(query, query, softcap, build_flags)


def test_magi_sm90_accepts_bf16_inputs_in_compiled_head_dim_bucket(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 64, dtype=torch.bfloat16)

    assert magi_fa4_backend._validate_magi_cutlass_inputs(query, query, 0.0, build_flags) == 128


@pytest.mark.parametrize(
    "build_flag",
    [
        "FLASHATTENTION_DISABLE_BACKWARD",
        "FLASHATTENTION_DISABLE_SM90",
        "FLASHATTENTION_DISABLE_ARBITRARY",
    ],
)
def test_magi_sm90_rejects_incompatible_cutlass_build(monkeypatch, build_flag):
    _set_fake_cutlass_backend(monkeypatch, available=True)
    config = sys.modules["flash_attn_cute.ffa_fa3.flash_attn_config"].CONFIG
    del config["build_flags"][build_flag]
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: 90)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(RuntimeError, match=build_flag):
        magi_fa4_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_sm90_rejects_different_query_value_head_dims(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    value = torch.empty(1, 1, 129, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same head dimension"):
        magi_fa4_backend._validate_magi_cutlass_inputs(query, value, 0.0, build_flags)


@pytest.mark.parametrize(
    ("current_stack_size", "expected_set_calls"),
    [
        (1024, 1),
        (magi_fa4_backend._MAGI_CUTLASS_STACK_SIZE, 0),
    ],
)
def test_magi_sm90_configures_cutlass_stack_size(monkeypatch, current_stack_size, expected_set_calls):
    set_calls = _set_fake_cuda_runtime(monkeypatch, current_stack_size=current_stack_size)

    magi_fa4_backend._ensure_magi_cutlass_stack_size(torch.device("cuda:1"))

    assert len(set_calls) == expected_set_calls
    if expected_set_calls:
        assert set_calls[0][1] == magi_fa4_backend._MAGI_CUTLASS_STACK_SIZE


def test_magi_sm90_rejects_clamped_cutlass_stack_size(monkeypatch):
    _set_fake_cuda_runtime(
        monkeypatch,
        current_stack_size=1024,
        configured_stack_size=4096,
    )

    with pytest.raises(RuntimeError, match="Failed to verify.*configured=4096"):
        magi_fa4_backend._ensure_magi_cutlass_stack_size(torch.device("cuda"))


@pytest.mark.parametrize(
    ("get_error", "set_error", "expected_message"),
    [
        (1, 0, "Failed to query"),
        (0, 1, "Failed to configure"),
    ],
)
def test_magi_sm90_reports_cutlass_stack_limit_errors(monkeypatch, get_error, set_error, expected_message):
    _set_fake_cuda_runtime(
        monkeypatch,
        current_stack_size=1024,
        get_error=get_error,
        set_error=set_error,
    )

    with pytest.raises(RuntimeError, match=expected_message):
        magi_fa4_backend._ensure_magi_cutlass_stack_size(torch.device("cuda"))


@pytest.mark.parametrize(
    ("device", "compute_capability", "expected_hardware"),
    [
        (torch.device("cpu"), 0, "cpu"),
        (torch.device("cuda"), 80, "SM80"),
    ],
)
def test_magi_unsupported_mode_fails_before_backend_call(
    monkeypatch,
    device,
    compute_capability,
    expected_hardware,
):
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: compute_capability)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()
    with pytest.raises(RuntimeError, match=rf"does not support {expected_hardware}"):
        magi_fa4_backend._prepare_default_magi_kernel(device)


def test_create_magi_mask_builds_standard_causal_and_bidirectional_ranges(monkeypatch):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state())

    causal_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=8,
        kv_length=8,
        device="cpu",
    )
    bidirectional_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=8,
        kv_length=8,
        mask_function=bidirectional_mask_function,
        device="cpu",
    )

    expected_ranges = torch.tensor([[0, 8]], dtype=torch.int32)
    torch.testing.assert_close(causal_mask.q_ranges, expected_ranges)
    torch.testing.assert_close(causal_mask.k_ranges, expected_ranges)
    torch.testing.assert_close(causal_mask.attn_type_map, torch.ones(1, dtype=torch.int32))
    torch.testing.assert_close(bidirectional_mask.q_ranges, expected_ranges)
    torch.testing.assert_close(bidirectional_mask.k_ranges, expected_ranges)
    assert bidirectional_mask.attn_type_map is None


def test_create_magi_mask_builds_packed_causal_ranges(monkeypatch):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state())
    cu_seq_lens = torch.tensor([0, 3, 8], dtype=torch.int32)

    attention_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=8,
        kv_length=8,
        attention_mask=torch.ones(1, 8, dtype=torch.bool),
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens.clone(),
        device="cpu",
    )

    expected_ranges = torch.tensor([[0, 3], [3, 8]], dtype=torch.int32)
    torch.testing.assert_close(attention_mask.q_ranges, expected_ranges)
    torch.testing.assert_close(attention_mask.k_ranges, expected_ranges)
    torch.testing.assert_close(attention_mask.attn_type_map, torch.ones(2, dtype=torch.int32))


def test_create_magi_mask_preserves_explicit_mixed_ranges(monkeypatch):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state())
    q_ranges = torch.tensor([[0, 4], [4, 8], [4, 8]], dtype=torch.int32)
    k_ranges = torch.tensor([[0, 4], [0, 4], [4, 8]], dtype=torch.int32)
    attn_type_map = torch.tensor([1, 0, 0], dtype=torch.int32)

    attention_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=8,
        kv_length=8,
        q_ranges=q_ranges,
        k_ranges=k_ranges,
        attn_type_map=attn_type_map,
        device="cpu",
    )

    torch.testing.assert_close(attention_mask.q_ranges, q_ranges)
    torch.testing.assert_close(attention_mask.k_ranges, k_ranges)
    torch.testing.assert_close(attention_mask.attn_type_map, attn_type_map)


@pytest.mark.parametrize("metadata_kind", ["cu_seqlens", "explicit_ranges"])
def test_create_magi_mask_preserves_strict_int32_metadata_contract(monkeypatch, metadata_kind):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state())
    kwargs = {}
    if metadata_kind == "cu_seqlens":
        kwargs["cu_seq_lens_q"] = torch.tensor([0, 8], dtype=torch.int64)
        kwargs["cu_seq_lens_k"] = torch.tensor([0, 8], dtype=torch.int64)
    else:
        kwargs["q_ranges"] = torch.tensor([[0, 8]], dtype=torch.int64)
        kwargs["k_ranges"] = torch.tensor([[0, 8]], dtype=torch.int64)

    with pytest.raises(TypeError, match="dtype torch.int32"):
        magi_backend.create_magi_mask(
            batch_size=1,
            q_length=8,
            kv_length=8,
            device="cpu",
            **kwargs,
        )


def test_create_magi_mask_uses_post_ulysses_sequence_length(monkeypatch):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state(ulysses_enabled=True))
    cu_seq_lens = torch.tensor([0, 3, 8], dtype=torch.int32)

    attention_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=4,
        kv_length=4,
        attention_mask=torch.ones(1, 8, dtype=torch.bool),
        cu_seq_lens_q=cu_seq_lens,
        cu_seq_lens_k=cu_seq_lens.clone(),
        device="cpu",
    )

    expected_ranges = torch.tensor([[0, 3], [3, 8]], dtype=torch.int32)
    torch.testing.assert_close(attention_mask.q_ranges, expected_ranges)
    torch.testing.assert_close(attention_mask.k_ranges, expected_ranges)

    local_attention_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=4,
        kv_length=4,
        skip_ulysses=True,
        device="cpu",
    )
    local_ranges = torch.tensor([[0, 4]], dtype=torch.int32)
    torch.testing.assert_close(local_attention_mask.q_ranges, local_ranges)
    torch.testing.assert_close(local_attention_mask.k_ranges, local_ranges)


def test_magi_attention_rejects_global_ranges_when_ulysses_is_skipped(monkeypatch):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state(ulysses_enabled=True))
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state(ulysses_enabled=True))
    attention_mask = magi_backend.create_magi_mask(
        batch_size=1,
        q_length=4,
        kv_length=4,
        device="cpu",
    )
    query = torch.randn(1, 4, 4, 16)

    with pytest.raises(ValueError, match="post-exchange query length \\(4\\)"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            attention_mask,
            skip_ulysses=True,
        )


@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"batch_size": 2}, "physical batch size 1"),
        ({"q_offset": 1}, "does not support KV-cache offsets"),
        ({"mask_function": lambda *args: True}, "canonical causal or bidirectional"),
        (
            {
                "attention_mask": torch.tensor([[True, False]]),
                "cu_seq_lens_q": torch.tensor([0, 2], dtype=torch.int32),
                "cu_seq_lens_k": torch.tensor([0, 2], dtype=torch.int32),
            },
            "requires an all-valid 2D attention mask",
        ),
    ],
)
def test_create_magi_mask_rejects_unsupported_registry_inputs(monkeypatch, override, expected_message):
    monkeypatch.setattr(magi_mask_backend, "get_parallel_state", lambda: _cp1_state())
    arguments = {
        "batch_size": 1,
        "q_length": 2,
        "kv_length": 2,
        "mask_function": causal_mask_function,
        "device": "cpu",
        **override,
    }

    with pytest.raises(ValueError, match=expected_message):
        magi_backend.create_magi_mask(**arguments)


@pytest.mark.parametrize(
    ("overrides", "error_type", "expected_message"),
    [
        ({"q_ranges": [[0, 8]]}, TypeError, "must be a torch.Tensor"),
        ({"q_ranges": torch.tensor([[0, 8]], dtype=torch.int64)}, TypeError, "dtype torch.int32"),
        (
            {"q_ranges": torch.tensor([0, 8], dtype=torch.int32)},
            ValueError,
            "shape \\[num_ranges, 2\\]",
        ),
        ({"q_ranges": torch.empty(0, 2, dtype=torch.int32)}, ValueError, "at least one range"),
        ({"attn_type_map": torch.tensor([0], dtype=torch.int64)}, TypeError, "dtype torch.int32"),
        (
            {"attn_type_map": torch.tensor([[0]], dtype=torch.int32)},
            ValueError,
            "shape \\[num_ranges\\]",
        ),
        (
            {
                "q_ranges": torch.tensor([[0, 4], [4, 8]], dtype=torch.int32),
                "k_ranges": torch.tensor([[0, 4], [4, 8]], dtype=torch.int32),
                "attn_type_map": torch.arange(4, dtype=torch.int32)[::2],
            },
            ValueError,
            "attn_type_map must be contiguous",
        ),
        (
            {"q_ranges": torch.tensor([[-1, 8]], dtype=torch.int32)},
            ValueError,
            "q_ranges must contain non-empty half-open ranges with non-negative starts",
        ),
        (
            {"k_ranges": torch.tensor([[4, 4]], dtype=torch.int32)},
            ValueError,
            "k_ranges must contain non-empty half-open ranges with non-negative starts",
        ),
        (
            {"attn_type_map": torch.tensor([4], dtype=torch.int32)},
            ValueError,
            "attn_type_map values must be in \\[0, 3\\]",
        ),
    ],
)
def test_magi_attention_mask_rejects_invalid_metadata(overrides, error_type, expected_message):
    values = {
        "q_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "k_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "attn_type_map": None,
        **overrides,
    }

    with pytest.raises(error_type, match=expected_message):
        magi_backend.MagiAttentionMask(**values)


@pytest.mark.parametrize(
    ("override", "expected_message"),
    [
        ({"attention_mask": None}, "requires a MagiAttentionMask"),
        ({"dropout": 0.1}, "does not support attention dropout"),
        ({"sliding_window": 4}, "encode visibility in MagiAttentionMask"),
    ],
)
def test_magi_attention_rejects_unsupported_features(monkeypatch, override, expected_message):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    query = torch.randn(1, 4, 8, 16)
    arguments = {"attention_mask": _mask(8), **override}

    with pytest.raises((TypeError, ValueError), match=expected_message):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            **arguments,
        )


def test_magi_attention_rejects_batch_and_cp_greater_than_one(monkeypatch):
    query = torch.randn(2, 4, 8, 16)
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())

    with pytest.raises(ValueError, match="requires batch size 1"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _mask(8),
        )

    query = query[:1]
    state = _cp1_state()
    state.cp_size = 2
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: state)
    with pytest.raises(ValueError, match="supports cp_size == 1"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            query,
            query,
            _mask(8),
        )


def test_magi_attention_rejects_invalid_gqa_before_ulysses(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state(ulysses_enabled=True))
    monkeypatch.setattr(
        magi_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("invalid GQA must fail before Ulysses collectives"),
    )
    query = torch.randn(1, 6, 8, 16)
    key = torch.randn(1, 4, 8, 16)
    value = torch.randn(1, 4, 8, 16)

    with pytest.raises(ValueError, match="GQA requires query heads"):
        magi_backend.magi_attention_forward(
            _FakeAttentionModule(),
            query,
            key,
            value,
            _mask(8),
        )


def test_magi_attention_delegates_active_ulysses_to_shared_helpers(monkeypatch):
    state = _cp1_state(ulysses_enabled=True)
    calls = []

    def fake_prepare(query, key, value, *, group, ulysses_size):
        calls.append(("prepare", query, key, value, group, ulysses_size))
        return (
            torch.randn(1, 16, 2, 16),
            torch.randn(1, 16, 1, 16),
            torch.randn(1, 16, 1, 16),
            4,
        )

    def fake_backend(query, key, value, q_ranges, k_ranges, attn_type_map, **kwargs):
        calls.append(("backend", query, key, value, q_ranges, k_ranges, attn_type_map, kwargs))
        return query, SimpleNamespace(lse=torch.ones(query.shape[:2]))

    def fake_restore(output, *, group):
        calls.append(("restore", output, group))
        return output[:, :8].repeat_interleave(2, dim=2)

    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: state)
    monkeypatch.setattr(magi_backend, "prepare_ulysses_qkv", fake_prepare)
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", fake_backend)
    monkeypatch.setattr(magi_backend, "restore_ulysses_output", fake_restore)
    query = torch.randn(1, 4, 8, 16)
    key = torch.randn(1, 2, 8, 16)
    value = torch.randn(1, 2, 8, 16)

    output, lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _mask(16),
    )

    assert [call[0] for call in calls] == ["prepare", "backend", "restore", "restore"]
    assert calls[0][1].shape == (1, 8, 4, 16)
    assert calls[0][2].shape == (1, 8, 2, 16)
    assert calls[0][4:] == (state.ulysses_group, 2)
    assert calls[1][1].shape == (16, 2, 16)
    assert calls[1][4].shape == (1, 2)
    assert calls[2][1].shape == (1, 16, 2, 16)
    assert calls[3][1].shape == (1, 16, 2, 1)
    assert output.shape == (1, 8, 4, 16)
    assert lse.shape == (1, 4, 8)


def test_magi_attention_skip_ulysses_uses_local_sequence_mask(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state(ulysses_enabled=True))
    monkeypatch.setattr(
        magi_backend,
        "prepare_ulysses_qkv",
        lambda *args, **kwargs: pytest.fail("prepare_ulysses_qkv must not be called"),
    )
    monkeypatch.setattr(
        magi_backend,
        "_magi_attention_forward",
        lambda query, key, value, *args, **kwargs: (
            query,
            SimpleNamespace(lse=torch.ones(query.shape[:2])),
        ),
    )
    query = torch.randn(1, 4, 8, 16)

    output, lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        query,
        query,
        _mask(8),
        skip_ulysses=True,
    )

    assert output.shape == (1, 8, 4, 16)
    assert lse.shape == (1, 4, 8)


_NUMERICAL_MASK_CASES = ("causal", "full", "bagel_mixed")
_NUMERICAL_QUERY_HEADS = 4
_NUMERICAL_KV_HEADS = 2
_NUMERICAL_HEAD_DIM = 64
_NUMERICAL_SEQUENCE_LENGTH = 128

_BAGEL_QUERY_HEADS = 28
_BAGEL_KV_HEADS = 4
_BAGEL_HEAD_DIM = 128
_BAGEL_SEQUENCE_LENGTH = 4096

_PROFILE_MASK_CASES = ("causal", "bagel_mixed")
_PROFILE_SEQUENCE_LENGTHS = (4096, 8192, 20000)
_PROFILE_WARMUP_ITERATIONS = 5
_PROFILE_ITERATIONS = 20
_RUN_PROFILE = os.environ.get("RUN_MAGI_ATTENTION_PROFILE") == "1"


def _is_magi_ffa_available() -> bool:
    if not IS_CUDA_AVAILABLE or importlib.util.find_spec("magi_attention") is None:
        return False

    kernel_mode = magi_fa4_backend._get_magi_kernel_mode(torch.device(get_device_type()))
    if kernel_mode == magi_fa4_backend._MAGI_KERNEL_CUTE_JIT:
        return importlib.util.find_spec("flash_attn_cute") is not None
    if kernel_mode != magi_fa4_backend._MAGI_KERNEL_CUTLASS:
        return False

    try:
        from flash_attn_cute.ffa_fa3 import flash_attn_interface
    except (ImportError, OSError, RuntimeError):
        return False

    return all(
        callable(getattr(flash_attn_interface, name, None)) for name in ("_flash_attn_forward", "_flash_attn_backward")
    )


_MAGI_FFA_AVAILABLE = _is_magi_ffa_available()
_MAGI_FFA_REASON = (
    "MagiAttention numerical tests require a supported NVIDIA GPU with its CUTLASS overlay or CUTE DSL/JIT backend"
)


class _NumericalAttentionModule(nn.Module):
    def __init__(self, implementation: str):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=implementation)
        self.layer_idx = 0


def _build_span_splits(sequence_length: int) -> list[int]:
    quarter = sequence_length // 4
    return [quarter, quarter, quarter, sequence_length - 3 * quarter]


def _build_bagel_dense_mask(sequence_length: int, device: torch.device) -> torch.Tensor:
    modes = ("causal", "noise", "full", "causal")
    visible = torch.zeros((sequence_length, sequence_length), device=device, dtype=torch.bool)
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(_build_span_splits(sequence_length), modes, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            visible[span_start:span_end, clean_start:clean_end] = True
        if mode == "causal":
            visible[span_start:span_end, span_start:span_end].fill_(True).tril_()
        else:
            visible[span_start:span_end, span_start:span_end] = True
        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end
    return visible.unsqueeze(0).unsqueeze(0).contiguous()


def _build_dense_mask(mask_case: str, sequence_length: int, device: torch.device) -> torch.Tensor:
    if mask_case == "causal":
        return torch.ones(
            (1, 1, sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        ).tril_()
    if mask_case == "full":
        return torch.ones(
            (1, 1, sequence_length, sequence_length),
            device=device,
            dtype=torch.bool,
        )
    if mask_case == "bagel_mixed":
        return _build_bagel_dense_mask(sequence_length, device)
    raise ValueError(f"Unsupported mask case: {mask_case}")


def _build_magi_mask(
    mask_case: str,
    sequence_length: int,
    device: torch.device,
) -> magi_backend.MagiAttentionMask:
    if mask_case in {"causal", "full"}:
        ranges = torch.tensor([[0, sequence_length]], device=device, dtype=torch.int32)
        attn_type_map = torch.tensor([1], device=device, dtype=torch.int32) if mask_case == "causal" else None
        return magi_backend.MagiAttentionMask(ranges, ranges.clone(), attn_type_map)

    if mask_case != "bagel_mixed":
        raise ValueError(f"Unsupported mask case: {mask_case}")

    modes = ("causal", "noise", "full", "causal")
    q_ranges: list[list[int]] = []
    k_ranges: list[list[int]] = []
    attn_types: list[int] = []
    clean_spans: list[tuple[int, int]] = []
    span_start = 0
    for length, mode in zip(_build_span_splits(sequence_length), modes, strict=True):
        span_end = span_start + length
        for clean_start, clean_end in clean_spans:
            q_ranges.append([span_start, span_end])
            k_ranges.append([clean_start, clean_end])
            attn_types.append(0)

        q_ranges.append([span_start, span_end])
        k_ranges.append([span_start, span_end])
        attn_types.append(1 if mode == "causal" else 0)

        if mode != "noise":
            clean_spans.append((span_start, span_end))
        span_start = span_end

    return magi_backend.MagiAttentionMask(
        q_ranges=torch.tensor(q_ranges, device=device, dtype=torch.int32),
        k_ranges=torch.tensor(k_ranges, device=device, dtype=torch.int32),
        attn_type_map=torch.tensor(attn_types, device=device, dtype=torch.int32),
    )


def _build_profile_flex_mask(mask_case: str, sequence_length: int, device: torch.device):
    if mask_case == "causal":

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            return query_idx >= key_idx

    elif mask_case == "bagel_mixed":
        if sequence_length % 4 != 0:
            raise ValueError("The BAGEL-like profile requires a sequence length divisible by four.")
        quarter = sequence_length // 4

        def mask_mod(batch_idx, head_idx, query_idx, key_idx):
            first = (query_idx < quarter) & (key_idx <= query_idx)
            noise = (query_idx >= quarter) & (query_idx < 2 * quarter) & (key_idx < 2 * quarter)
            full = (
                (query_idx >= 2 * quarter)
                & (query_idx < 3 * quarter)
                & ((key_idx < quarter) | ((key_idx >= 2 * quarter) & (key_idx < 3 * quarter)))
            )
            last = (query_idx >= 3 * quarter) & (
                (key_idx < quarter)
                | ((key_idx >= 2 * quarter) & (key_idx < 3 * quarter))
                | ((key_idx >= 3 * quarter) & (key_idx <= query_idx))
            )
            return first | noise | full | last

    else:
        raise ValueError(f"Unsupported profile mask case: {mask_case}")

    return create_block_mask(
        mask_mod,
        B=None,
        H=None,
        Q_LEN=sequence_length,
        KV_LEN=sequence_length,
        device=device,
        BLOCK_SIZE=128,
    )


def _materialize_magi_mask(
    attention_mask: magi_backend.MagiAttentionMask,
    sequence_length: int,
) -> torch.Tensor:
    visible = torch.zeros((sequence_length, sequence_length), dtype=torch.bool)
    attn_types = (
        torch.zeros(attention_mask.q_ranges.shape[0], dtype=torch.int32)
        if attention_mask.attn_type_map is None
        else attention_mask.attn_type_map.cpu()
    )
    for q_range, k_range, attn_type in zip(
        attention_mask.q_ranges.cpu(),
        attention_mask.k_ranges.cpu(),
        attn_types,
        strict=True,
    ):
        q_start, q_end = q_range.tolist()
        k_start, k_end = k_range.tolist()
        slice_mask = torch.ones((q_end - q_start, k_end - k_start), dtype=torch.bool)
        if attn_type.item() == 1:
            slice_mask.tril_()
        visible[q_start:q_end, k_start:k_end] |= slice_mask
    return visible.unsqueeze(0).unsqueeze(0)


def _clone_qkv(qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    return tuple(tensor.detach().clone().requires_grad_(True) for tensor in qkv)


def _math_sdpa_reference(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    dense_mask: torch.Tensor,
    *,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=dense_mask,
            dropout_p=0.0,
            scale=scaling,
            enable_gqa=True,
        ).transpose(1, 2)

    repeat_count = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(repeat_count, dim=1)
    logits = torch.einsum("bhqd,bhkd->bhqk", query.float(), expanded_key.float()) * scaling
    lse = torch.logsumexp(logits.masked_fill(~dense_mask, -torch.inf), dim=-1)
    return output, lse


def _assert_magi_matches_reference(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    gradients: tuple[torch.Tensor, ...],
    reference_output: torch.Tensor,
    reference_lse: torch.Tensor,
    reference_gradients: tuple[torch.Tensor, ...],
    *,
    dtype: torch.dtype,
) -> None:
    assert torch.isfinite(output).all()
    torch.testing.assert_close(
        output,
        reference_output,
        rtol=3e-2,
        atol=3e-2,
        msg=lambda message: f"MagiAttention output: {message}",
    )

    if lse is not None:
        assert torch.isfinite(lse).all()
        torch.testing.assert_close(
            lse.float(),
            reference_lse.float(),
            rtol=5e-3,
            atol=3e-2,
            msg=lambda message: f"MagiAttention LSE: {message}",
        )

    gradient_atol = 8e-2 if dtype == torch.bfloat16 else 5e-2
    for name, gradient, reference_gradient in zip(
        ("query", "key", "value"),
        gradients,
        reference_gradients,
        strict=True,
    ):
        assert torch.isfinite(gradient).all()
        assert torch.isfinite(reference_gradient).all()
        torch.testing.assert_close(
            gradient,
            reference_gradient,
            rtol=8e-2,
            atol=gradient_atol,
            msg=lambda message, tensor_name=name: f"MagiAttention {tensor_name} gradient: {message}",
        )


@pytest.mark.parametrize("mask_case", _PROFILE_MASK_CASES)
def test_attention_benchmark_masks_match_dense_visibility(mask_case):
    device = torch.device("cpu")
    dense_mask = _build_dense_mask(mask_case, _NUMERICAL_SEQUENCE_LENGTH, device)
    magi_mask = _build_magi_mask(mask_case, _NUMERICAL_SEQUENCE_LENGTH, device)
    flex_mask = _build_profile_flex_mask(mask_case, _NUMERICAL_SEQUENCE_LENGTH, device)
    query_idx = torch.arange(_NUMERICAL_SEQUENCE_LENGTH)[:, None]
    key_idx = torch.arange(_NUMERICAL_SEQUENCE_LENGTH)[None, :]

    materialized_magi_mask = _materialize_magi_mask(magi_mask, _NUMERICAL_SEQUENCE_LENGTH)
    materialized_flex_mask = flex_mask.mask_mod(0, 0, query_idx, key_idx)

    assert torch.equal(materialized_magi_mask, dense_mask)
    assert torch.equal(materialized_flex_mask, dense_mask[0, 0])


def _assert_magi_attention_matches_dense_reference(
    mask_case: str,
    *,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    seed: int,
) -> None:
    device = torch.device(get_device_type())
    generator = torch.Generator(device=device).manual_seed(seed)
    qkv = (
        torch.randn(
            (1, query_heads, sequence_length, head_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        torch.randn(
            (1, kv_heads, sequence_length, head_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
        torch.randn(
            (1, kv_heads, sequence_length, head_dim),
            device=device,
            dtype=dtype,
            generator=generator,
        ),
    )
    output_gradient = torch.randn(
        (1, sequence_length, query_heads, head_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    scaling = head_dim**-0.5
    dense_mask = _build_dense_mask(mask_case, sequence_length, device)
    magi_mask = _build_magi_mask(mask_case, sequence_length, device)

    reference_qkv = _clone_qkv(qkv)
    reference_output, reference_lse = _math_sdpa_reference(
        *reference_qkv,
        dense_mask,
        scaling=scaling,
    )
    reference_gradients = torch.autograd.grad(reference_output, reference_qkv, output_gradient)

    magi_qkv = _clone_qkv(qkv)
    magi_output, magi_lse = magi_backend.magi_attention_forward(
        _NumericalAttentionModule("veomni_magi_attention_with_sp").to(device),
        *magi_qkv,
        magi_mask,
        scaling=scaling,
    )
    magi_gradients = torch.autograd.grad(magi_output, magi_qkv, output_gradient)
    _assert_magi_matches_reference(
        magi_output,
        magi_lse,
        magi_gradients,
        reference_output,
        reference_lse,
        reference_gradients,
        dtype=dtype,
    )


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("mask_case", _NUMERICAL_MASK_CASES)
def test_magi_attention_matches_dense_reference(monkeypatch, mask_case, dtype):
    device = torch.device(get_device_type())
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    kernel_mode, build_flags = magi_fa4_backend._prepare_default_magi_kernel(device)
    if (
        dtype == torch.float16
        and kernel_mode == magi_fa4_backend._MAGI_KERNEL_CUTLASS
        and build_flags is not None
        and build_flags.get("FLASHATTENTION_DISABLE_FP16", False)
    ):
        pytest.skip("The installed CUTLASS overlay does not include FP16 kernels.")

    _assert_magi_attention_matches_dense_reference(
        mask_case,
        sequence_length=_NUMERICAL_SEQUENCE_LENGTH,
        query_heads=_NUMERICAL_QUERY_HEADS,
        kv_heads=_NUMERICAL_KV_HEADS,
        head_dim=_NUMERICAL_HEAD_DIM,
        dtype=dtype,
        seed=9300 + _NUMERICAL_MASK_CASES.index(mask_case) * 10 + int(dtype == torch.bfloat16),
    )


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
def test_magi_attention_matches_dense_reference_at_bagel_shape(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    _assert_magi_attention_matches_dense_reference(
        "bagel_mixed",
        sequence_length=_BAGEL_SEQUENCE_LENGTH,
        query_heads=_BAGEL_QUERY_HEADS,
        kv_heads=_BAGEL_KV_HEADS,
        head_dim=_BAGEL_HEAD_DIM,
        dtype=torch.bfloat16,
        seed=9052,
    )


def _measure_wall(callable_):
    synchronize()
    started = time.perf_counter()
    result = callable_()
    synchronize()
    return (time.perf_counter() - started) * 1000, result


def _profile_outputs_are_finite(
    output: torch.Tensor,
    lse: torch.Tensor | None,
    gradients: tuple[torch.Tensor, ...] | None = None,
) -> bool:
    finite = bool(torch.isfinite(output).all().item()) and all(
        bool(torch.isfinite(gradient).all().item()) for gradient in (() if gradients is None else gradients)
    )
    if lse is not None:
        finite = finite and bool(torch.isfinite(lse).all().item())
    return finite


def _profile_full_iteration(forward, qkv, output_gradient):
    output, lse = forward()
    gradients = torch.autograd.grad(output, qkv, output_gradient)
    return output, lse, gradients


def _profile_attention_backend(
    mask_case: str,
    sequence_length: int,
    backend: str,
) -> dict[str, object]:
    device_api = get_torch_device()
    gc.collect()
    empty_cache()

    device = torch.device(get_device_type())
    dtype = torch.bfloat16
    module = _NumericalAttentionModule("veomni_magi_attention_with_sp").to(device=device, dtype=dtype).train()
    generator = torch.Generator(device=device).manual_seed(12002 + sequence_length)
    query, key, value = (
        torch.randn(
            (1, heads, sequence_length, _BAGEL_HEAD_DIM),
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        for heads in (_BAGEL_QUERY_HEADS, _BAGEL_KV_HEADS, _BAGEL_KV_HEADS)
    )
    output_gradient = torch.randn(
        (1, sequence_length, _BAGEL_QUERY_HEADS, _BAGEL_HEAD_DIM),
        device=device,
        dtype=dtype,
        generator=generator,
    )

    if backend == "efficient_attention":
        attention_mask = None if mask_case == "causal" else _build_dense_mask(mask_case, sequence_length, device)
        is_causal = mask_case == "causal"
        mask_kind = "implicit_causal" if is_causal else "dense_bool"

        def forward():
            repeat_count = query.shape[1] // key.shape[1]
            expanded_key = key.repeat_interleave(repeat_count, dim=1)
            expanded_value = value.repeat_interleave(repeat_count, dim=1)
            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                output = F.scaled_dot_product_attention(
                    query,
                    expanded_key,
                    expanded_value,
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=_BAGEL_HEAD_DIM**-0.5,
                ).transpose(1, 2)
            return output, None

    elif backend == "flex_attention":
        attention_mask = _build_profile_flex_mask(mask_case, sequence_length, device)
        mask_kind = "native_BlockMask"

        def forward():
            return flex_backend.flex_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=_BAGEL_HEAD_DIM**-0.5,
                kernel_options={"BACKEND": "TRITON"},
            )

    elif backend == "magi_attention":
        attention_mask = _build_magi_mask(mask_case, sequence_length, device)
        mask_kind = "MagiAttentionMask"

        def forward():
            return magi_backend.magi_attention_forward(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling=_BAGEL_HEAD_DIM**-0.5,
            )

    else:
        raise ValueError(f"Unsupported profiling backend: {backend}")

    qkv = (query, key, value)
    if backend == "magi_attention":
        magi_fa4_backend._fa4_cache_entry = None

    device_api.reset_peak_memory_stats()
    compile_init_ms, result = _measure_wall(lambda: _profile_full_iteration(forward, qkv, output_gradient))
    output, lse, gradients = result
    all_finite = _profile_outputs_are_finite(output, lse, gradients)
    first_iteration_peak_allocated_gib = device_api.max_memory_allocated() / 1024**3
    del result, output, lse, gradients

    warmup_times = []
    for _ in range(_PROFILE_WARMUP_ITERATIONS):
        elapsed_ms, result = _measure_wall(lambda: _profile_full_iteration(forward, qkv, output_gradient))
        output, lse, gradients = result
        warmup_times.append(elapsed_ms)
        all_finite = all_finite and _profile_outputs_are_finite(output, lse, gradients)
        del result, output, lse, gradients

    forward_times = []
    for _ in range(_PROFILE_ITERATIONS):
        elapsed_ms, result = _measure_wall(forward)
        output, lse = result
        forward_times.append(elapsed_ms)
        all_finite = all_finite and _profile_outputs_are_finite(output, lse)
        del result, output, lse

    backward_times = []
    for _ in range(_PROFILE_ITERATIONS):
        output, lse = forward()
        synchronize()
        elapsed_ms, gradients = _measure_wall(lambda output=output: torch.autograd.grad(output, qkv, output_gradient))
        backward_times.append(elapsed_ms)
        all_finite = all_finite and _profile_outputs_are_finite(output, lse, gradients)
        del output, lse, gradients

    total_times = []
    for _ in range(_PROFILE_ITERATIONS):
        elapsed_ms, result = _measure_wall(lambda: _profile_full_iteration(forward, qkv, output_gradient))
        output, lse, gradients = result
        total_times.append(elapsed_ms)
        all_finite = all_finite and _profile_outputs_are_finite(output, lse, gradients)
        del result, output, lse, gradients

    gc.collect()
    empty_cache()
    baseline_bytes = device_api.memory_allocated()
    device_api.reset_peak_memory_stats()
    output, lse, gradients = _profile_full_iteration(forward, qkv, output_gradient)
    synchronize()
    peak_bytes = device_api.max_memory_allocated()
    all_finite = all_finite and _profile_outputs_are_finite(output, lse, gradients)
    del output, lse, gradients

    return {
        "backend": backend,
        "mask": mask_kind,
        "compile_init_excluded_ms": compile_init_ms,
        "first_iteration_peak_allocated_gib": first_iteration_peak_allocated_gib,
        "warmup_iterations": _PROFILE_WARMUP_ITERATIONS,
        "warmup_times": warmup_times,
        "measured_iterations": _PROFILE_ITERATIONS,
        "forward_times_ms": forward_times,
        "backward_times_ms": backward_times,
        "total_times_ms": total_times,
        "forward_median_ms": statistics.median(forward_times),
        "backward_median_ms": statistics.median(backward_times),
        "total_median_ms": statistics.median(total_times),
        "peak_allocated_gib": peak_bytes / 1024**3,
        "incremental_peak_gib": (peak_bytes - baseline_bytes) / 1024**3,
        "all_outputs_and_gradients_finite": all_finite,
    }


def _profile_speedup(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, float]:
    return {
        metric: baseline[f"{metric}_median_ms"] / candidate[f"{metric}_median_ms"]
        for metric in ("forward", "backward", "total")
    }


@pytest.mark.skipif(not _MAGI_FFA_AVAILABLE, reason=_MAGI_FFA_REASON)
@pytest.mark.skipif(
    not _RUN_PROFILE,
    reason="Set RUN_MAGI_ATTENTION_PROFILE=1 to run the causal and BAGEL-like CUDA profiles",
)
@pytest.mark.benchmark
@pytest.mark.parametrize("mask_case", _PROFILE_MASK_CASES)
@pytest.mark.parametrize("sequence_length", _PROFILE_SEQUENCE_LENGTHS)
def test_attention_profiles_efficient_sdpa_flex_and_magi(monkeypatch, mask_case, sequence_length):
    """Reproduce the PR's causal and BAGEL-like attention measurements."""
    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(flex_backend, "get_parallel_state", lambda: _cp1_state())
    device_api = get_torch_device()
    try:
        efficient_result = _profile_attention_backend(mask_case, sequence_length, "efficient_attention")
        flex_result = _profile_attention_backend(mask_case, sequence_length, "flex_attention")
        magi_result = _profile_attention_backend(mask_case, sequence_length, "magi_attention")
    except device_api.OutOfMemoryError as error:
        free_bytes, total_bytes = device_api.mem_get_info()
        pytest.fail(
            json.dumps(
                {
                    "mask_case": mask_case,
                    "sequence_length": sequence_length,
                    "error": str(error),
                    "allocated_gib": device_api.memory_allocated() / 1024**3,
                    "reserved_gib": device_api.memory_reserved() / 1024**3,
                    "free_gib": free_bytes / 1024**3,
                    "total_gib": total_bytes / 1024**3,
                },
                indent=2,
            )
        )

    result = {
        "mask_case": mask_case,
        "sequence_length": sequence_length,
        "dtype": str(torch.bfloat16),
        "batch_size": 1,
        "query_heads": _BAGEL_QUERY_HEADS,
        "kv_heads": _BAGEL_KV_HEADS,
        "head_dim": _BAGEL_HEAD_DIM,
        "efficient_attention": efficient_result,
        "flex_attention": flex_result,
        "magi_attention": magi_result,
        "flex_speedup_vs_efficient": _profile_speedup(efficient_result, flex_result),
        "magi_speedup_vs_efficient": _profile_speedup(efficient_result, magi_result),
        "magi_speedup_vs_flex": _profile_speedup(flex_result, magi_result),
    }
    print(f"Attention profile:\n{json.dumps(result, indent=2)}")

    assert efficient_result["all_outputs_and_gradients_finite"]
    assert flex_result["all_outputs_and_gradients_finite"]
    assert magi_result["all_outputs_and_gradients_finite"]

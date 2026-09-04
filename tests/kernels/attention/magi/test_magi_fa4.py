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

"""Magi FA4 CUDA internals: backend resolve, CUTLASS setup, metadata cache, autograd."""

from __future__ import annotations

import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch

from veomni.kernels._kernels.attention.standard import magi as magi_backend
from veomni.kernels._kernels.attention.standard.magi import _fa4_cuda as magi_fa4
from veomni.kernels._kernels.attention.standard.magi import _kernel as magi_kernel
from veomni.kernels._kernels.attention.standard.magi import _metadata as magi_metadata


@pytest.fixture(autouse=True)
def _isolate_magi_fa4_caches(monkeypatch):
    magi_kernel.prepare_kernel.cache_clear()
    monkeypatch.setattr(magi_metadata, "_cache_entry", None)
    yield
    magi_kernel.prepare_kernel.cache_clear()


def _set_fake_cutlass_backend(monkeypatch, *, available: bool) -> dict[str, object] | None:
    fake_package = ModuleType("flash_attn_cute")
    fake_package.__path__ = []
    monkeypatch.setitem(sys.modules, "flash_attn_cute", fake_package)
    monkeypatch.delitem(sys.modules, "flash_attn_cute.ffa_fa3", raising=False)
    monkeypatch.delitem(sys.modules, "flash_attn_cute.ffa_fa3.flash_attn_config", raising=False)

    if not available:
        return None

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


def _set_fake_cuda_runtime(
    monkeypatch,
    *,
    current_stack_size: int,
    get_error: int = 0,
    set_error: int = 0,
    configured_stack_size: int | None = None,
) -> list[tuple[object, int]]:
    import sys

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


def test_default_magi_backend_resolves_cuda_platform(monkeypatch):
    cuda_device_type = magi_kernel.CUDA_DEVICE_TYPE
    monkeypatch.setattr(magi_backend, "IS_CUDA_AVAILABLE", True)
    monkeypatch.setattr(magi_backend, "get_device_type", lambda: cuda_device_type)

    assert magi_backend._resolve_magi_backend(torch.device(cuda_device_type)) is magi_fa4._fa4_cuda_attention_forward


@pytest.mark.parametrize(
    ("device", "active_device_type", "expected_message"),
    [
        (
            torch.device("cpu"),
            magi_kernel.CUDA_DEVICE_TYPE,
            f"received tensors on cpu.*active device type is {magi_kernel.CUDA_DEVICE_TYPE}",
        ),
        (torch.device("cpu"), "cpu", "does not yet provide a CPU backend"),
    ],
)
def test_default_magi_backend_rejects_unsupported_platform(monkeypatch, device, active_device_type, expected_message):
    monkeypatch.setattr(magi_backend, "IS_CUDA_AVAILABLE", active_device_type == "cuda")
    monkeypatch.setattr(magi_backend, "get_device_type", lambda: active_device_type)

    with pytest.raises(RuntimeError, match=expected_message):
        magi_backend._resolve_magi_backend(device)


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
    monkeypatch.setattr(magi_metadata, "get_or_prepare_attn_arg", fake_get_attn_arg)
    monkeypatch.setattr(magi_fa4, "get_or_prepare_attn_arg", fake_get_attn_arg)
    monkeypatch.setattr(magi_fa4._MagiFA4Function, "apply", fake_apply)
    monkeypatch.setattr(magi_fa4, "prepare_kernel", lambda device: (magi_kernel.KERNEL_CUTE_JIT, None))
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    value = torch.randn(8, 2, 16)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    result = magi_fa4._fa4_cuda_attention_forward(
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


def test_default_magi_backend_attn_forward_meta_import_skips_query_device(monkeypatch):
    """AttnForwardMeta is a Python container. Kernels import it outside cuda_device_context.

    Leftover ops wrapped this import in ``torch.cuda.device(query.device)``.
    Device-sensitive Magi imports stay in ``_MagiFA4Function`` and ``_prepare_attn_arg``.
    """
    active_devices = []

    @contextmanager
    def fake_device(device):
        active_devices.append(device)
        yield
        active_devices.pop()

    class DeviceAwareApi(ModuleType):
        def __getattr__(self, name):
            if name == "AttnForwardMeta":
                assert active_devices == []
                return lambda **kwargs: SimpleNamespace(**kwargs)
            raise AttributeError(name)

    fake_package = ModuleType("magi_attention")
    fake_package.__path__ = []
    fake_api = DeviceAwareApi("magi_attention.api")
    monkeypatch.setitem(sys.modules, "magi_attention", fake_package)
    monkeypatch.setitem(sys.modules, "magi_attention.api", fake_api)
    monkeypatch.setattr(torch.cuda, "device", fake_device)
    monkeypatch.setattr(magi_fa4, "prepare_kernel", lambda device: (magi_kernel.KERNEL_CUTE_JIT, None))
    monkeypatch.setattr(magi_fa4, "get_or_prepare_attn_arg", lambda *args: object())
    monkeypatch.setattr(magi_fa4._MagiFA4Function, "apply", lambda *args: ("output", "lse"))
    query = SimpleNamespace(device=torch.device("cuda:1"))
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    output, meta = magi_fa4._fa4_cuda_attention_forward(
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

    monkeypatch.setattr(magi_metadata, "_prepare_attn_arg", fake_build)
    monkeypatch.setattr(magi_metadata, "_cache_entry", None)
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    first = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)
    second = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)

    ranges[0, 1] = 7
    after_mutation = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)
    repeated_after_mutation = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)

    shorter_query = query[:7]
    after_shape_change = magi_metadata.get_or_prepare_attn_arg(shorter_query, key, ranges, ranges, None)

    assert first is second
    assert after_mutation is repeated_after_mutation
    assert first is not after_mutation
    assert after_mutation is not after_shape_change
    assert len(built_args) == 3
    assert magi_metadata._cache_entry is not None


def test_magi_fa4_metadata_cache_disables_reuse_without_version_counters(monkeypatch):
    built_args = []

    def fake_build(*args):
        built_arg = object()
        built_args.append(built_arg)
        return built_arg

    monkeypatch.setattr(magi_metadata, "_prepare_attn_arg", fake_build)
    monkeypatch.setattr(magi_metadata, "_cache_entry", None)
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    with torch.inference_mode():
        ranges = torch.tensor([[0, 8]], dtype=torch.int32)
        first = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)
        second = magi_metadata.get_or_prepare_attn_arg(query, key, ranges, ranges, None)

    assert first is not second
    assert len(built_args) == 2
    assert magi_metadata._cache_entry is None


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

    attn_arg = magi_metadata._prepare_attn_arg(query, key, ranges, ranges, None, 16)

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

    output, lse = magi_fa4._MagiFA4Function.apply(
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
    assert output.requires_grad
    assert not lse.requires_grad
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
    output, _ = magi_fa4._MagiFA4Function.apply(
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
        (80, magi_kernel.KERNEL_UNSUPPORTED),
        (89, magi_kernel.KERNEL_UNSUPPORTED),
        (90, magi_kernel.KERNEL_CUTLASS),
        (99, magi_kernel.KERNEL_CUTLASS),
        (100, magi_kernel.KERNEL_CUTE_JIT),
    ],
)
def test_magi_kernel_mode_follows_query_device(monkeypatch, compute_capability, expected_mode):
    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", lambda device: compute_capability)

    assert magi_kernel.get_kernel_mode(torch.device("cuda")) == expected_mode


def test_magi_sm90_reports_cutlass_installer(monkeypatch):
    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=False)

    with pytest.raises(ImportError, match=r"install_magi_sm90\.sh"):
        magi_kernel.prepare_kernel(torch.device("cuda"))


def test_magi_sm100_plus_does_not_require_cutlass_backend(monkeypatch):
    calls = 0

    def fake_compute_capability(device):
        nonlocal calls
        calls += 1
        return 100

    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", fake_compute_capability)
    _set_fake_cutlass_backend(monkeypatch, available=False)

    magi_kernel.prepare_kernel(torch.device("cuda"))
    magi_kernel.prepare_kernel(torch.device("cuda"))

    assert calls == 1
    assert magi_kernel.prepare_kernel.cache_info().currsize == 1


def test_magi_sm90_prepares_cutlass_device_once(monkeypatch):
    prepared = []
    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=True)
    monkeypatch.setattr(magi_kernel, "_install_tile_size_compatibility", lambda: prepared.append("tile-size"))
    monkeypatch.setattr(magi_kernel, "_ensure_cutlass_stack_size", lambda device: prepared.append(device))

    for device in (torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:1")):
        kernel_mode, build_flags = magi_kernel.prepare_kernel(device)
        assert kernel_mode == magi_kernel.KERNEL_CUTLASS
        assert build_flags is not None

    assert prepared == ["tile-size", torch.device("cuda:0"), "tile-size", torch.device("cuda:1")]
    assert magi_kernel.prepare_kernel.cache_info().currsize == 2


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
        magi_kernel.validate_cutlass_inputs(query, query, softcap, build_flags)


def test_magi_sm90_accepts_bf16_inputs_in_compiled_head_dim_bucket(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 64, dtype=torch.bfloat16)

    assert magi_kernel.validate_cutlass_inputs(query, query, 0.0, build_flags) == 128


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
    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", lambda device: 90)

    with pytest.raises(RuntimeError, match=build_flag):
        magi_kernel.prepare_kernel(torch.device("cuda"))


def test_magi_sm90_rejects_different_query_value_head_dims(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    value = torch.empty(1, 1, 129, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same head dimension"):
        magi_kernel.validate_cutlass_inputs(query, value, 0.0, build_flags)


@pytest.mark.parametrize(
    ("current_stack_size", "expected_set_calls"),
    [
        (1024, 1),
        (magi_kernel._CUTLASS_STACK_SIZE, 0),
    ],
)
def test_magi_sm90_configures_cutlass_stack_size(monkeypatch, current_stack_size, expected_set_calls):
    set_calls = _set_fake_cuda_runtime(monkeypatch, current_stack_size=current_stack_size)

    magi_kernel._ensure_cutlass_stack_size(torch.device("cuda:1"))

    assert len(set_calls) == expected_set_calls
    if expected_set_calls:
        assert set_calls[0][1] == magi_kernel._CUTLASS_STACK_SIZE


def test_magi_sm90_rejects_clamped_cutlass_stack_size(monkeypatch):
    _set_fake_cuda_runtime(monkeypatch, current_stack_size=1024, configured_stack_size=4096)

    with pytest.raises(RuntimeError, match="Failed to verify.*configured=4096"):
        magi_kernel._ensure_cutlass_stack_size(torch.device("cuda"))


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
        magi_kernel._ensure_cutlass_stack_size(torch.device("cuda"))


@pytest.mark.parametrize(
    ("device", "compute_capability", "expected_hardware"),
    [
        (torch.device("cpu"), 0, "cpu"),
        (torch.device("cuda"), 80, "SM80"),
    ],
)
def test_magi_unsupported_mode_fails_before_backend_call(monkeypatch, device, compute_capability, expected_hardware):
    monkeypatch.setattr(magi_kernel, "get_gpu_compute_capability", lambda device: compute_capability)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    with pytest.raises(RuntimeError, match=rf"does not support {expected_hardware}"):
        magi_kernel.prepare_kernel(device)

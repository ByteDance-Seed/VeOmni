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

import sys
from contextlib import contextmanager, nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.masking_utils import bidirectional_mask_function, causal_mask_function

from veomni.ops import build_ALL_OPS
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import magi as magi_backend
from veomni.ops.kernels.attention.magi import _fa4 as magi_fa4_backend
from veomni.ops.kernels.attention.magi import mask as magi_mask_backend


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


def test_magi_attention_preserves_backend_autograd(monkeypatch):
    def differentiable_backend(query, key, value, *args, **kwargs):
        repeat_count = query.shape[1] // key.shape[1]
        output = query + key.repeat_interleave(repeat_count, dim=1)
        output = output + value.repeat_interleave(repeat_count, dim=1)
        return output, SimpleNamespace(lse=query.float().sum(dim=-1))

    monkeypatch.setattr(magi_backend, "get_parallel_state", lambda: _cp1_state())
    monkeypatch.setattr(magi_backend, "_magi_attention_forward", differentiable_backend)
    query = torch.randn(1, 4, 8, 16, requires_grad=True)
    key = torch.randn(1, 2, 8, 16, requires_grad=True)
    value = torch.randn(1, 2, 8, 16, requires_grad=True)

    output, lse = magi_backend.magi_attention_forward(
        _FakeAttentionModule(),
        query,
        key,
        value,
        _mask(8),
    )
    (output.square().mean() + lse.square().mean()).backward()

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()


def test_default_magi_backend_lazily_calls_package_fa4_backend(monkeypatch):
    captured = {}
    fa4_attn_arg = object()

    def fake_get_attn_arg(query, key, q_ranges, k_ranges, attn_type_map):
        captured["metadata_inputs"] = (query, key, q_ranges, k_ranges, attn_type_map)
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
        (75, magi_fa4_backend._MAGI_KERNEL_UNSUPPORTED),
        (80, magi_fa4_backend._MAGI_KERNEL_UNSUPPORTED),
        (89, magi_fa4_backend._MAGI_KERNEL_UNSUPPORTED),
        (90, magi_fa4_backend._MAGI_KERNEL_CUTLASS),
        (100, magi_fa4_backend._MAGI_KERNEL_CUTE_JIT),
        (120, magi_fa4_backend._MAGI_KERNEL_CUTE_JIT),
    ],
)
def test_magi_kernel_mode_follows_query_device(monkeypatch, compute_capability, expected_mode):
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: compute_capability)

    assert magi_fa4_backend._get_magi_kernel_mode(torch.device("cuda")) == expected_mode


@pytest.mark.parametrize("compute_capability", [75, 80, 89])
def test_magi_rejects_pre_sm90_gpus(monkeypatch, compute_capability):
    monkeypatch.setattr(magi_fa4_backend, "get_gpu_compute_capability", lambda device: compute_capability)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(RuntimeError, match=rf"does not support SM{compute_capability}.*SM90.*SM100"):
        magi_fa4_backend._prepare_default_magi_kernel(torch.device("cuda"))


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

    magi_fa4_backend._validate_magi_cutlass_inputs(query, query, 0.0, build_flags)


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


def test_magi_unsupported_mode_fails_before_backend_call(monkeypatch):
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_fa4_backend._prepare_default_magi_kernel.cache_clear()
    with pytest.raises(RuntimeError, match="does not support cpu"):
        magi_fa4_backend._prepare_default_magi_kernel(torch.device("cpu"))


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
    ("field", "value", "error_type", "expected_message"),
    [
        ("q_ranges", [[0, 8]], TypeError, "must be a torch.Tensor"),
        ("q_ranges", torch.tensor([[0, 8]], dtype=torch.int64), TypeError, "dtype torch.int32"),
        ("q_ranges", torch.tensor([0, 8], dtype=torch.int32), ValueError, "shape \\[num_ranges, 2\\]"),
        ("q_ranges", torch.empty(0, 2, dtype=torch.int32), ValueError, "at least one range"),
        ("attn_type_map", torch.tensor([0], dtype=torch.int64), TypeError, "dtype torch.int32"),
        ("attn_type_map", torch.tensor([[0]], dtype=torch.int32), ValueError, "shape \\[num_ranges\\]"),
    ],
)
def test_magi_attention_mask_rejects_invalid_structure(field, value, error_type, expected_message):
    values = {
        "q_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "k_ranges": torch.tensor([[0, 8]], dtype=torch.int32),
        "attn_type_map": None,
    }
    values[field] = value

    with pytest.raises(error_type, match=expected_message):
        magi_backend.MagiAttentionMask(**values)


def test_magi_attention_mask_requires_contiguous_type_map():
    ranges = torch.tensor([[0, 4], [4, 8]], dtype=torch.int32)
    noncontiguous_type_map = torch.arange(4, dtype=torch.int32)[::2]
    assert not noncontiguous_type_map.is_contiguous()

    with pytest.raises(ValueError, match="attn_type_map must be contiguous"):
        magi_backend.MagiAttentionMask(ranges, ranges.clone(), noncontiguous_type_map)


@pytest.mark.parametrize(
    ("q_ranges", "k_ranges", "attn_type_map", "expected_message"),
    [
        (
            torch.tensor([[-1, 8]], dtype=torch.int32),
            torch.tensor([[0, 8]], dtype=torch.int32),
            None,
            "q_ranges must contain non-empty half-open ranges with non-negative starts",
        ),
        (
            torch.tensor([[0, 8]], dtype=torch.int32),
            torch.tensor([[4, 4]], dtype=torch.int32),
            None,
            "k_ranges must contain non-empty half-open ranges with non-negative starts",
        ),
        (
            torch.tensor([[0, 8]], dtype=torch.int32),
            torch.tensor([[0, 8]], dtype=torch.int32),
            torch.tensor([4], dtype=torch.int32),
            "attn_type_map values must be in \\[0, 3\\]",
        ),
    ],
)
def test_magi_attention_mask_rejects_invalid_static_values(q_ranges, k_ranges, attn_type_map, expected_message):
    with pytest.raises(ValueError, match=expected_message):
        magi_backend.MagiAttentionMask(q_ranges, k_ranges, attn_type_map)


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

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
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from veomni.ops import build_ALL_OPS
from veomni.ops.kernels import attention as veomni_attention
from veomni.ops.kernels.attention import magi as magi_backend


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


def test_default_magi_backend_lazily_calls_package_fa4_facade(monkeypatch):
    captured = {}

    def fake_ffa_fa4(query, key, value, **kwargs):
        captured.update(query=query, key=key, value=value, kwargs=kwargs)
        return "output", "lse"

    class FakeAttnForwardMeta:
        def __init__(self, *, lse, max_logits):
            self.lse = lse
            self.max_logits = max_logits

    fake_package = ModuleType("magi_attention")
    fake_package.__path__ = []
    fake_api = ModuleType("magi_attention.api")
    fake_api.AttnForwardMeta = FakeAttnForwardMeta
    fake_functional = ModuleType("magi_attention.functional")
    fake_functional.ffa_fa4_func = fake_ffa_fa4
    monkeypatch.setitem(sys.modules, "magi_attention", fake_package)
    monkeypatch.setitem(sys.modules, "magi_attention.api", fake_api)
    monkeypatch.setitem(sys.modules, "magi_attention.functional", fake_functional)
    monkeypatch.setattr(
        magi_backend,
        "_prepare_default_magi_kernel",
        lambda device: (magi_backend._MAGI_KERNEL_CUTE_JIT, None),
    )
    query = torch.randn(8, 4, 16)
    key = torch.randn(8, 2, 16)
    value = torch.randn(8, 2, 16)
    ranges = torch.tensor([[0, 8]], dtype=torch.int32)

    result = magi_backend._default_magi_attention_forward(
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
    assert captured["query"] is query
    assert captured["key"] is key
    assert captured["value"] is value
    assert captured["kwargs"] == {
        "q_ranges": ranges,
        "k_ranges": ranges,
        "attn_type_map": None,
        "softmax_scale": 0.25,
        "softcap": 30.0,
    }


@pytest.mark.parametrize(
    ("compute_capability", "expected_mode"),
    [
        (75, magi_backend._MAGI_KERNEL_UNSUPPORTED),
        (80, magi_backend._MAGI_KERNEL_UNSUPPORTED),
        (89, magi_backend._MAGI_KERNEL_UNSUPPORTED),
        (90, magi_backend._MAGI_KERNEL_CUTLASS),
        (100, magi_backend._MAGI_KERNEL_CUTE_JIT),
        (120, magi_backend._MAGI_KERNEL_CUTE_JIT),
    ],
)
def test_magi_kernel_mode_follows_query_device(monkeypatch, compute_capability, expected_mode):
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: compute_capability)

    assert magi_backend._get_magi_kernel_mode(torch.device("cuda")) == expected_mode


@pytest.mark.parametrize("compute_capability", [75, 80, 89])
def test_magi_rejects_pre_sm90_gpus(monkeypatch, compute_capability):
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: compute_capability)
    magi_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(RuntimeError, match=rf"does not support SM{compute_capability}.*SM90.*SM100"):
        magi_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_sm90_reports_cutlass_installer(monkeypatch):
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(ImportError, match=r"install_magi_sm90\.sh"):
        magi_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_sm100_plus_does_not_require_cutlass_backend(monkeypatch):
    calls = 0

    def fake_compute_capability(device):
        nonlocal calls
        calls += 1
        return 100

    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", fake_compute_capability)
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_backend._prepare_default_magi_kernel.cache_clear()

    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))
    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))

    assert calls == 1
    assert magi_backend._prepare_default_magi_kernel.cache_info().currsize == 1


def test_magi_sm90_prepares_cutlass_device_once(monkeypatch):
    prepared = []
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: 90)
    _set_fake_cutlass_backend(monkeypatch, available=True)
    magi_backend._prepare_default_magi_kernel.cache_clear()
    monkeypatch.setattr(
        magi_backend,
        "_install_magi_tile_size_compatibility",
        lambda: prepared.append("tile-size"),
    )
    monkeypatch.setattr(
        magi_backend,
        "_ensure_magi_cutlass_stack_size",
        lambda device: prepared.append(device),
    )

    for device in (torch.device("cuda:0"), torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:1")):
        kernel_mode, build_flags = magi_backend._prepare_default_magi_kernel(device)
        assert kernel_mode == magi_backend._MAGI_KERNEL_CUTLASS
        assert build_flags is not None

    assert prepared == ["tile-size", torch.device("cuda:0"), "tile-size", torch.device("cuda:1")]
    assert magi_backend._prepare_default_magi_kernel.cache_info().currsize == 2


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
        magi_backend._validate_magi_cutlass_inputs(query, query, softcap, build_flags)


def test_magi_sm90_accepts_bf16_inputs_in_compiled_head_dim_bucket(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 64, dtype=torch.bfloat16)

    magi_backend._validate_magi_cutlass_inputs(query, query, 0.0, build_flags)


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
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: 90)
    magi_backend._prepare_default_magi_kernel.cache_clear()

    with pytest.raises(RuntimeError, match=build_flag):
        magi_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_sm90_rejects_different_query_value_head_dims(monkeypatch):
    build_flags = _set_fake_cutlass_backend(monkeypatch, available=True)
    assert build_flags is not None
    query = torch.empty(1, 1, 128, dtype=torch.bfloat16)
    value = torch.empty(1, 1, 129, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="same head dimension"):
        magi_backend._validate_magi_cutlass_inputs(query, value, 0.0, build_flags)


@pytest.mark.parametrize(
    ("current_stack_size", "expected_set_calls"),
    [
        (1024, 1),
        (magi_backend._MAGI_CUTLASS_STACK_SIZE, 0),
    ],
)
def test_magi_sm90_configures_cutlass_stack_size(monkeypatch, current_stack_size, expected_set_calls):
    set_calls = _set_fake_cuda_runtime(monkeypatch, current_stack_size=current_stack_size)

    magi_backend._ensure_magi_cutlass_stack_size(torch.device("cuda:1"))

    assert len(set_calls) == expected_set_calls
    if expected_set_calls:
        assert set_calls[0][1] == magi_backend._MAGI_CUTLASS_STACK_SIZE


def test_magi_sm90_rejects_clamped_cutlass_stack_size(monkeypatch):
    _set_fake_cuda_runtime(
        monkeypatch,
        current_stack_size=1024,
        configured_stack_size=4096,
    )

    with pytest.raises(RuntimeError, match="Failed to verify.*configured=4096"):
        magi_backend._ensure_magi_cutlass_stack_size(torch.device("cuda"))


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
        magi_backend._ensure_magi_cutlass_stack_size(torch.device("cuda"))


def test_magi_unsupported_mode_fails_before_backend_call(monkeypatch):
    _set_fake_cutlass_backend(monkeypatch, available=False)
    magi_backend._prepare_default_magi_kernel.cache_clear()
    with pytest.raises(RuntimeError, match="does not support cpu"):
        magi_backend._prepare_default_magi_kernel(torch.device("cpu"))


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

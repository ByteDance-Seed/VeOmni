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
    monkeypatch.setattr(magi_backend, "_prepare_default_magi_kernel", lambda device: None)
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
        (80, magi_backend._MAGI_KERNEL_CUTLASS),
        (89, magi_backend._MAGI_KERNEL_CUTLASS),
        (90, magi_backend._MAGI_KERNEL_CUTLASS),
        (100, magi_backend._MAGI_KERNEL_CUTE_JIT),
        (120, magi_backend._MAGI_KERNEL_CUTE_JIT),
    ],
)
def test_magi_kernel_mode_follows_query_device(monkeypatch, compute_capability, expected_mode):
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: compute_capability)

    assert magi_backend._get_magi_kernel_mode(torch.device("cuda")) == expected_mode


@pytest.mark.parametrize(("compute_capability", "build_arch"), [(80, "sm80"), (86, "sm80"), (90, "sm90")])
def test_magi_cutlass_mode_reports_arch_specific_installer(monkeypatch, compute_capability, build_arch):
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: compute_capability)
    monkeypatch.setattr(magi_backend, "_magi_cutlass_backend", None)

    with pytest.raises(
        ImportError,
        match=rf"install_magi_sm80_sm90\.sh {build_arch}",
    ):
        magi_backend._prepare_default_magi_kernel(torch.device("cuda"))


def test_magi_cute_jit_mode_does_not_require_cutlass_backend(monkeypatch):
    calls = 0

    def fake_compute_capability(device):
        nonlocal calls
        calls += 1
        return 100

    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", fake_compute_capability)
    monkeypatch.setattr(magi_backend, "_magi_cutlass_backend", None)
    monkeypatch.setattr(magi_backend, "_prepared_default_magi_devices", set())

    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))
    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))

    assert calls == 1
    assert magi_backend._prepared_default_magi_devices == {torch.device("cuda")}


def test_magi_cutlass_mode_prepares_device_once(monkeypatch):
    prepared_devices = []
    monkeypatch.setattr(magi_backend, "get_gpu_compute_capability", lambda device: 80)
    monkeypatch.setattr(magi_backend, "_magi_cutlass_backend", object())
    monkeypatch.setattr(magi_backend, "_prepared_default_magi_devices", set())
    monkeypatch.setattr(
        magi_backend,
        "_prepare_magi_cutlass_device",
        lambda device: prepared_devices.append(device),
    )

    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))
    magi_backend._prepare_default_magi_kernel(torch.device("cuda"))

    assert prepared_devices == [torch.device("cuda")]


def test_magi_unsupported_mode_fails_before_backend_call():
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

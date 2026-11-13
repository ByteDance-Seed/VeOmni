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

"""CPU contract checks for the opt-in AscendC retained-state adapter."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def adapter(monkeypatch):
    """Load the adapters with an identity guard for CPU contract tests."""
    from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import fla_npu, fla_npu_reuse

    name = "veomni.ops.kernels.gated_delta_rule.vendor.triton.utils"
    triton_utils = ModuleType(name)
    triton_utils.input_guard = lambda fn: fn
    monkeypatch.setitem(sys.modules, name, triton_utils)
    return fla_npu, fla_npu_reuse


def test_reuse_preserves_forward_states_and_skips_recomputation(adapter, monkeypatch):
    """Forward intermediates reach backward without rerunning forward stages."""
    main, reuse = adapter
    calls = []
    cached = {}

    def prepare(q, k, v, g, beta, chunk_size, **kwargs):
        """Capture the packed inputs and synthesize reusable intermediates."""
        calls.append("prepare")
        assert g.dtype == beta.dtype == torch.float32
        assert kwargs["cu_seqlens"] == [0, 2, 4]
        assert kwargs["chunk_indices"] == [0, 0, 1, 0]
        cached.update(g=g * 2, w=k.clone(), u=v.clone(), a=torch.ones(1))
        return None, None, None, None, None, cached["g"], cached["w"], cached["u"], cached["a"]

    def fwd_h(k, w, u, **kwargs):
        """Capture state inputs and synthesize state outputs."""
        calls.append("h")
        assert w is cached["w"] and u is cached["u"] and kwargs["g"] is cached["g"]
        cached.update(h=torch.ones(1), vn=torch.ones_like(u))
        return cached["h"], cached["vn"], None

    def fwd_o(q, k, v, h, scale, **kwargs):
        """Require the retained state and return a head-major output."""
        calls.append("o")
        assert v is cached["vn"] and h is cached["h"] and kwargs["output_layout"] == "BNSD"
        return q.clone()

    def dv_local(q, k, do, g, scale, chunk_size, **kwargs):
        """Check the local value gradient uses the composite API contract."""
        calls.append("dv")
        assert kwargs["A"] is None
        return torch.ones_like(do)

    def dhu(q, k, w, do, dv, scale, chunk_size, **kwargs):
        """Require the retained W tensor during the state backward."""
        calls.append("dhu")
        assert w is cached["w"] and kwargs["h0"] is None
        return torch.ones(1), None, dv

    def finalize(q, k, v, vn, do, dv, g, beta, h, dh, a, **kwargs):
        """Require retained tensors and synthesize distinct gate gradients."""
        calls.append("finalize")
        assert vn is cached["vn"] and h is cached["h"] and a is cached["a"]
        assert torch.equal(g, cached["g"]) and beta.dtype == torch.float32
        assert kwargs["use_exp2"] is False
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v), beta * 0 + 2, g * 0 + 3

    ops = SimpleNamespace(
        chunk_gated_delta_rule_fwd_prepare=prepare,
        chunk_fwd_h=fwd_h,
        chunk_fwd_o=fwd_o,
        chunk_bwd_dv_local=dv_local,
        chunk_gated_delta_rule_bwd_dhu=dhu,
        chunk_gated_delta_rule_bwd_finalize=finalize,
    )
    monkeypatch.setattr(reuse, "_reuse_ops", lambda q, cu: ops)
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    g = torch.ones(1, 4, 2, dtype=torch.float32)
    beta = torch.ones(1, 4, 2, dtype=torch.bfloat16)
    outputs, saved = reuse.forward(*values, g, beta, cu_seqlens=torch.tensor([0, 2, 4]))
    assert calls == ["prepare", "h", "o"]
    assert all(actual is cached[name] for actual, name in zip(saved.tensors[-3:], ("w", "h", "vn"), strict=True))
    grads = reuse.backward((torch.ones_like(outputs[0]), outputs[1]), saved)
    assert calls == ["prepare", "h", "o", "dv", "dhu", "finalize"]
    assert all(torch.equal(grad, torch.ones_like(value)) for grad, value in zip(grads[:3], values, strict=True))
    assert torch.equal(grads[3], torch.full_like(g, 3))
    assert torch.equal(grads[4], torch.full_like(beta, 2))


def test_reuse_rejects_unsupported_devices_boundaries_and_api(adapter, monkeypatch):
    """Reject unsupported hardware, malformed packing and incomplete APIs."""
    main, reuse = adapter
    q = SimpleNamespace(device=SimpleNamespace(type="npu", index=0), shape=(1, 2, 4, 128))
    npu = SimpleNamespace(get_device_name=lambda index: "Ascend910_9382")
    monkeypatch.setattr(torch, "npu", npu, raising=False)
    ops = SimpleNamespace(
        **{
            name: lambda *args, **kwargs: None
            for name in (
                "chunk_gated_delta_rule_fwd_prepare",
                "chunk_fwd_h",
                "chunk_fwd_o",
                "chunk_bwd_dv_local",
                "chunk_gated_delta_rule_bwd_dhu",
                "chunk_gated_delta_rule_bwd_finalize",
            )
        }
    )
    monkeypatch.setattr(main, "_get_gdn_ops", lambda: ops)
    assert reuse._reuse_ops(q, [0, 2, 4]) is ops
    for boundaries in ([0, 0, 4], [0, 3], [1, 4], [0, 3, 2, 4]):
        with pytest.raises(ValueError, match="strictly increasing"):
            reuse._reuse_ops(q, boundaries)
    monkeypatch.setattr(npu, "get_device_name", lambda index: "Ascend950")
    with pytest.raises(RuntimeError, match="910C only"):
        reuse._reuse_ops(q, [0, 2, 4])
    monkeypatch.setattr(npu, "get_device_name", lambda index: "Ascend910_9382")
    del ops.chunk_gated_delta_rule_bwd_finalize
    with pytest.raises(RuntimeError, match="decomposed GDN"):
        reuse._reuse_ops(q, [0, 2, 4])


def test_standard_backend_keeps_existing_saved_state_contract(adapter, monkeypatch):
    """The default path continues to store its original tensor state."""
    main, _ = adapter
    values = [torch.ones(1, 4, 2, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    monkeypatch.setattr(main, "_chunk_fwd", lambda q, k, v, g, *args: (g, q.transpose(1, 2), q.new_empty(0), None))
    monkeypatch.setattr(
        main,
        "_chunk_bwd",
        lambda q, k, v, g, beta, *args: (
            torch.ones_like(q),
            torch.ones_like(k),
            torch.ones_like(v),
            torch.zeros_like(beta),
            torch.zeros_like(g),
        ),
    )
    output, saved = main.forward(*values, gate, gate)
    assert len(saved.tensors) == 10 and not saved.metadata.reuse_intermediates
    grads = main.backward((torch.ones_like(output[0]), output[1]), saved)
    assert all(torch.equal(grad, value) for grad, value in zip(grads[:3], values, strict=True))


@pytest.mark.parametrize("use_reuse", (False, True))
def test_nonempty_packed_boundaries_and_state_are_not_replaced(adapter, monkeypatch, use_reuse):
    """The usual packed path reuses its boundary tensor and initial state."""
    main, reuse = adapter
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    cu = torch.tensor([0, 2, 4])
    initial = torch.ones(2, 2, 128, 128, dtype=torch.bfloat16)

    def forward(q, k, v, g, beta, scale, state, output_state, boundaries, cu_list, chunks, host_chunks, chunk_size):
        """Check original objects remain on the fast path."""
        assert boundaries is cu and state is initial and cu_list == [0, 2, 4]
        assert host_chunks[str(chunk_size)] == [0, 0, 1, 0]
        result = (g, q.transpose(1, 2), q.new_empty(0), None)
        return result + (q, k, v) if use_reuse else result

    monkeypatch.setattr(reuse if use_reuse else main, "_chunk_fwd_reuse" if use_reuse else "_chunk_fwd", forward)
    _, saved = (reuse if use_reuse else main).forward(
        *values, gate, gate, initial_state=initial, cu_seqlens=cu, chunk_indices_list={"64": [0, 0, 1, 0]}
    )
    assert saved.tensors[6] is initial and saved.tensors[7] is cu


@pytest.mark.parametrize("use_reuse", (False, True))
def test_empty_packed_sequence_preserves_boundaries_and_initial_states(adapter, monkeypatch, use_reuse):
    """Both backends compact empty segments and preserve original state owners."""
    main, reuse = adapter
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    initial = torch.arange(5, dtype=torch.bfloat16).view(5, 1, 1, 1).expand(5, 2, 128, 128).contiguous()
    calls = []

    def forward(q, k, v, g, beta, scale, state, output_state, cu, cu_list, chunks, host_chunks, chunk_size):
        """Require compact metadata and original nonempty owners."""
        calls.append("forward")
        assert cu.tolist() == cu_list == [0, 2, 4]
        assert host_chunks[str(chunk_size)] == [0, 0, 1, 0]
        torch.testing.assert_close(state, initial[[1, 3]], atol=0, rtol=0)
        result = (g, q.transpose(1, 2), q.new_empty(0), None)
        return result + (q, k, v) if use_reuse else result

    def backward(q, k, v, g, beta, a, scale, state, do, cu, cu_list, chunks, host_chunks, chunk_size, *retained):
        """Backward must use the same compact metadata and initial states."""
        calls.append("backward")
        assert cu.tolist() == cu_list == [0, 2, 4]
        assert host_chunks[str(chunk_size)] == [0, 0, 1, 0]
        torch.testing.assert_close(state, initial[[1, 3]], atol=0, rtol=0)
        assert len(retained) == (3 if use_reuse else 0)
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v), torch.zeros_like(beta), torch.zeros_like(g)

    if use_reuse:
        monkeypatch.setattr(reuse, "_chunk_fwd_reuse", forward)
        monkeypatch.setattr(reuse, "_chunk_bwd_reuse", backward)
    else:
        monkeypatch.setattr(main, "_chunk_fwd", forward)
        monkeypatch.setattr(main, "_chunk_bwd", backward)
    chosen = reuse if use_reuse else main
    output, saved = chosen.forward(
        *values,
        gate,
        gate,
        initial_state=initial,
        cu_seqlens=torch.tensor([0, 0, 2, 2, 4, 4]),
        chunk_indices_list={"64": [1, 0, 3, 0]},
    )
    assert calls == ["forward"] and saved.metadata.reuse_intermediates == use_reuse
    grads = chosen.backward((torch.ones_like(output[0]), output[1]), saved)
    assert calls == ["forward", "backward"]
    assert all(torch.equal(grad, value) for grad, value in zip(grads[:3], values, strict=True))


@pytest.mark.parametrize("use_reuse", (False, True))
@pytest.mark.parametrize("with_initial", (False, True))
def test_empty_packed_inference_restores_final_state_slots(adapter, monkeypatch, use_reuse, with_initial):
    """Empty segments keep their own initial state or zero in the final state."""
    main, reuse = adapter
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    initial = torch.arange(5, dtype=torch.float32).view(5, 1, 1, 1).expand(5, 2, 128, 128).contiguous()
    compact_final = torch.full((2, 2, 128, 128), 11.0)

    def forward(q, k, v, g, beta, scale, state, output_state, cu, cu_list, chunks, host_chunks, chunk_size):
        """Return distinct final states for each nonempty segment."""
        assert cu.tolist() == cu_list == [0, 2, 4]
        if with_initial:
            torch.testing.assert_close(state, initial[[1, 3]], atol=0, rtol=0)
        else:
            assert state is None
        result = (g, q.transpose(1, 2), q.new_empty(0), compact_final)
        return result + (q, k, v) if use_reuse else result

    monkeypatch.setattr(reuse if use_reuse else main, "_chunk_fwd_reuse" if use_reuse else "_chunk_fwd", forward)
    output, _ = (reuse if use_reuse else main).forward(
        *values,
        gate,
        gate,
        initial_state=initial if with_initial else None,
        cu_seqlens=torch.tensor([0, 0, 2, 2, 4, 4]),
        output_final_state=True,
    )
    expected = initial.clone() if with_initial else torch.zeros_like(initial)
    expected[[1, 3]] = compact_final
    torch.testing.assert_close(output[1], expected, atol=0, rtol=0)


@pytest.mark.parametrize("use_reuse", (False, True))
def test_zero_token_input_is_rejected(adapter, use_reuse):
    """Zero-token batches are outside the backend contract, with or without packing."""
    main, reuse = adapter
    values = [torch.empty(1, 0, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.empty(1, 0, 2)
    for cu in (None, torch.tensor([], dtype=torch.int32), torch.tensor([0]), torch.tensor([0, 0])):
        with pytest.raises(ValueError, match="requires at least one token"):
            (reuse if use_reuse else main).forward(*values, gate, gate, cu_seqlens=cu)


@pytest.mark.parametrize("use_reuse", (False, True))
def test_empty_packed_rejects_invalid_boundaries(adapter, use_reuse):
    """A repeated boundary must not hide a decreasing or truncated range."""
    main, reuse = adapter
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    for boundaries in ([0, 2, 1, 1, 4], [0, 2, 2, 3], [0, 3, 2, 4], [1, 4]):
        with pytest.raises(ValueError, match="nondecreasing and span"):
            (reuse if use_reuse else main).forward(*values, gate, gate, cu_seqlens=torch.tensor(boundaries))
    invalid_state = torch.zeros(3, 2, 64, 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="initial_state must match"):
        (reuse if use_reuse else main).forward(
            *values, gate, gate, initial_state=invalid_state, cu_seqlens=torch.tensor([0, 2, 2, 4])
        )


def test_invalid_packed_offsets_fail_before_chunk_metadata_build(adapter, monkeypatch):
    """A huge invalid offset cannot trigger an unbounded host index allocation."""
    main, _ = adapter
    values = [torch.ones(1, 4, 2, 128, dtype=torch.bfloat16) for _ in range(3)]
    gate = torch.ones(1, 4, 2)

    def unexpected_metadata(*args, **kwargs):
        """Detect any attempt to expand an invalid sequence range."""
        raise AssertionError("chunk metadata was built before boundary validation")

    monkeypatch.setattr(main, "_prepare_chunk_indices_list", unexpected_metadata)
    monkeypatch.setattr(main, "_prepare_chunk_indices", unexpected_metadata)
    with pytest.raises(ValueError, match="nondecreasing and span"):
        main.forward(*values, gate, gate, cu_seqlens=torch.tensor([0, 10**12, 4]))
    with pytest.raises(ValueError, match="nondecreasing and span"):
        main.precompute_varlen_metadata(torch.tensor([0, 10**12, 4]), num_heads=2)


def test_checkpoint_forward_failure_does_not_poison_later_backward():
    """An exception retained by pytest must not retain a checkpoint hook."""
    import traceback

    from tests.ops.gated_delta_rule._checkpoint_guard import checkpoint_nonreentrant

    calls = []

    def fail_forward(value):
        """Raise on the first checkpoint forward to exercise cleanup."""
        calls.append(value)
        raise RuntimeError("checkpoint sentinel")

    with pytest.raises(RuntimeError, match="checkpoint sentinel") as captured:
        checkpoint_nonreentrant(fail_forward, torch.ones(1, requires_grad=True))
    unrelated = torch.tensor([3.0], requires_grad=True)
    unrelated.square().sum().backward()
    assert unrelated.grad.item() == 6.0 and len(calls) == 1
    assert "fail_forward" in {frame.name for frame in traceback.extract_tb(captured.value.__traceback__)}

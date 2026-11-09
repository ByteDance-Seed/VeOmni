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


@pytest.fixture
def adapter(monkeypatch):
    from veomni.ops.kernels.gated_delta_rule.chunk_gated_delta_rule.standard import fla_npu, fla_npu_reuse

    name = "veomni.ops.kernels.gated_delta_rule.vendor.triton.utils"
    triton_utils = ModuleType(name)
    triton_utils.input_guard = lambda fn: fn
    monkeypatch.setitem(sys.modules, name, triton_utils)
    return fla_npu, fla_npu_reuse


def test_reuse_preserves_forward_states_and_skips_recomputation(adapter, monkeypatch):
    main, reuse = adapter
    calls = []
    cached = {}

    def prepare(q, k, v, g, beta, chunk_size, **kwargs):
        calls.append("prepare")
        assert g.dtype == beta.dtype == torch.float32
        assert kwargs["cu_seqlens"] == [0, 2, 4]
        assert kwargs["chunk_indices"] == [0, 0, 1, 0]
        cached.update(g=g * 2, w=k.clone(), u=v.clone(), a=torch.ones(1))
        return None, None, None, None, None, cached["g"], cached["w"], cached["u"], cached["a"]

    def fwd_h(k, w, u, **kwargs):
        calls.append("h")
        assert w is cached["w"] and u is cached["u"] and kwargs["g"] is cached["g"]
        cached.update(h=torch.ones(1), vn=torch.ones_like(u))
        return cached["h"], cached["vn"], None

    def fwd_o(q, k, v, h, scale, **kwargs):
        calls.append("o")
        assert v is cached["vn"] and h is cached["h"] and kwargs["output_layout"] == "BNSD"
        return q.clone()

    def dv_local(q, k, do, g, scale, chunk_size, **kwargs):
        calls.append("dv")
        assert kwargs["A"] is None
        return torch.ones_like(do)

    def dhu(q, k, w, do, dv, scale, chunk_size, **kwargs):
        calls.append("dhu")
        assert w is cached["w"] and kwargs["h0"] is None
        return torch.ones(1), None, dv

    def finalize(q, k, v, vn, do, dv, g, beta, h, dh, a, **kwargs):
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


def test_empty_packed_sequence_falls_back_to_standard_backend(adapter, monkeypatch):
    main, reuse = adapter
    values = [torch.ones(1, 4, 2, 8, dtype=torch.bfloat16, requires_grad=True) for _ in range(3)]
    gate = torch.ones(1, 4, 2)
    calls = []

    def standard_forward(q, k, v, g, *args):
        calls.append("standard")
        return g, q.transpose(1, 2), q.new_empty(0), None

    def standard_backward(q, k, v, g, beta, *args):
        calls.append("standard_backward")
        return torch.ones_like(q), torch.ones_like(k), torch.ones_like(v), torch.zeros_like(beta), torch.zeros_like(g)

    monkeypatch.setattr(main, "_chunk_fwd", standard_forward)
    monkeypatch.setattr(main, "_chunk_bwd", standard_backward)
    output, saved = reuse.forward(*values, gate, gate, cu_seqlens=torch.tensor([0, 2, 2, 4]))
    assert calls == ["standard"] and not saved.metadata.reuse_intermediates
    grads = reuse.backward((torch.ones_like(output[0]), output[1]), saved)
    assert calls == ["standard", "standard_backward"]
    assert all(torch.equal(grad, value) for grad, value in zip(grads[:3], values, strict=True))

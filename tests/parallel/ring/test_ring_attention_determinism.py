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

"""Verify the training determinism switch reaches each low-level FA2/FA4 backward."""

import pytest
import torch

import veomni.distributed.sequence_parallel.ring_attention.gpu as ring_gpu
from scripts.ci.accelerator.launch_plan import LaunchSpec


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _LocalRingComm:
    world_size = 1
    rank = 0

    def __init__(self, group):
        pass

    def send_recv(self, tensor, recv_buffer=None):
        return tensor.clone()

    def commit(self):
        pass

    def wait(self):
        pass


@pytest.mark.parametrize("backend", ["fa2", "fa4"])
@pytest.mark.parametrize("path", ["dense", "zigzag", "varlen"])
@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_backward_honors_determinism(monkeypatch, path, setting, backend):
    if setting is None:
        monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    else:
        # Set after module import, just as trainer setup sets the environment.
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", setting)
    monkeypatch.setattr(ring_gpu, "FA_BACKEND", backend)
    monkeypatch.setattr(ring_gpu, "RingComm", _LocalRingComm)
    monkeypatch.setattr(ring_gpu, "_FA4_DETERMINISTIC_AVAILABLE", True)
    monkeypatch.setattr(ring_gpu, "get_gpu_compute_capability", lambda: 90)
    calls = []

    def backward(*args, **kwargs):
        calls.append(kwargs.get("deterministic", False))
        for name in ("dq", "dk", "dv"):
            kwargs[name].zero_()

    monkeypatch.setattr(ring_gpu, "_fa4_bwd", backward, raising=False)
    monkeypatch.setattr(ring_gpu, "_flash_attn_backward", backward, raising=False)
    monkeypatch.setattr(ring_gpu, "_flash_attn_varlen_backward", backward, raising=False)
    q = torch.zeros(1, 8, 2, 4)
    lse = torch.zeros(1, 2, 8)
    if path == "dense":
        ring_gpu._fa_backward(q, q, q, q, q, lse, 0.5, True)
    elif path == "zigzag":
        ring_gpu._zigzag_ring_backward(None, q, q, q, q, q, lse, 0.5)
    else:
        q = q.squeeze(0)
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        ring_gpu._fa_varlen_backward(
            q, q, q, q, q, lse.squeeze(0), q.clone(), q.clone(), q.clone(), cu, cu, 4, 4, 0.5, True
        )
    assert calls == [setting == "1"]


@pytest.mark.parametrize("packed", [False, True])
def test_legacy_fa4_keeps_normal_backward_and_rejects_determinism(monkeypatch, packed):
    monkeypatch.setattr(ring_gpu, "FA_BACKENDS", frozenset({"fa4"}))
    monkeypatch.setattr(ring_gpu, "_FA4_DETERMINISTIC_AVAILABLE", False)
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    calls = []

    def old_backward(*args, **kwargs):
        assert "deterministic" not in kwargs
        calls.append(True)
        for name in ("dq", "dk", "dv"):
            kwargs[name].zero_()

    monkeypatch.setattr(ring_gpu, "_fa4_bwd", old_backward, raising=False)
    q = torch.zeros(1, 8, 2, 4)
    lse = torch.zeros(1, 2, 8)
    assert ring_gpu.resolve_ring_backend("fa4") == "fa4"
    if packed:
        q = q.squeeze(0)
        cu = torch.tensor([0, 4, 8], dtype=torch.int32)
        ring_gpu._fa_varlen_backward(
            q, q, q, q, q, lse.squeeze(0), q.clone(), q.clone(), q.clone(), cu, cu, 4, 4, 0.5, True, backend="fa4"
        )
    else:
        ring_gpu._fa_backward(q, q, q, q, q, lse, 0.5, True, backend="fa4")
    assert calls == [True]
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    # No group exists: capability rejection must happen before group lookup/communication.
    with pytest.raises(NotImplementedError, match="does not support deterministic"):
        if packed:
            ring_gpu.zigzag_ring_flash_attn_varlen_func(q, q, q, cu, 4, backend="fa4")
        else:
            ring_gpu.zigzag_ring_flash_attn_func(q, q, q, backend="fa4")


def test_fa4_base_capability_does_not_require_determinism():
    import inspect

    def forward():
        pass

    def backward():
        pass

    shared = {
        "softmax_scale",
        "causal",
        "window_size_left",
        "window_size_right",
        "cu_seqlens_q",
        "cu_seqlens_k",
        "max_seqlen_q",
        "max_seqlen_k",
    }
    for function, extra in ((forward, {"return_lse"}), (backward, {"dq", "dk", "dv"})):
        function.__signature__ = inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.KEYWORD_ONLY) for name in sorted(shared | extra)]
        )
    assert ring_gpu._supports_fa4_interface(forward, backward)


@pytest.mark.parametrize("packed", [False, True])
def test_fa4_sm12x_rejects_determinism_before_communication(monkeypatch, packed):
    from types import SimpleNamespace

    monkeypatch.setattr(ring_gpu, "FA_BACKENDS", frozenset({"fa4"}))
    monkeypatch.setattr(ring_gpu, "_FA4_DETERMINISTIC_AVAILABLE", True)
    monkeypatch.setattr(ring_gpu, "get_gpu_compute_capability", lambda: 120, raising=False)
    monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    assert ring_gpu.resolve_ring_backend("fa4") == "fa4"
    monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", "1")
    q = torch.zeros(8, 2, 4)
    # Neither a process group nor saved tensors exist; both entry points must
    # reject before touching communication or the backward state.
    with pytest.raises(NotImplementedError, match="SM12x"):
        if packed:
            cu = torch.tensor([0, 8], dtype=torch.int32)
            ring_gpu.zigzag_ring_flash_attn_varlen_func(q, q, q, cu, 8, backend="fa4")
        else:
            ring_gpu.zigzag_ring_flash_attn_func(q.unsqueeze(0), q.unsqueeze(0), q.unsqueeze(0), backend="fa4")
    with pytest.raises(NotImplementedError, match="SM12x"):
        function = ring_gpu._ZigzagRingVarlenFlashAttn if packed else ring_gpu._RingFlashAttn
        function.backward(SimpleNamespace(backend="fa4"), q)

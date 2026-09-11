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

"""Verify the training determinism switch reaches each low-level FA2 backward."""

import pytest
import torch

import veomni.distributed.sequence_parallel.ring_attention.gpu as ring_gpu


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


@pytest.mark.parametrize("path", ["dense", "zigzag", "varlen"])
@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_fa2_backward_honors_determinism(monkeypatch, path, setting):
    if setting is None:
        monkeypatch.delenv("FLASH_ATTENTION_DETERMINISTIC", raising=False)
    else:
        # Set after module import, just as trainer setup sets the environment.
        monkeypatch.setenv("FLASH_ATTENTION_DETERMINISTIC", setting)
    monkeypatch.setattr(ring_gpu, "FA_BACKEND", "fa2")
    monkeypatch.setattr(ring_gpu, "RingComm", _LocalRingComm)
    calls = []

    def backward(**kwargs):
        calls.append(kwargs["deterministic"])
        for name in ("dq", "dk", "dv"):
            kwargs[name].zero_()

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

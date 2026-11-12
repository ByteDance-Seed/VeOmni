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
"""Tests for dllm_mean_global_loss (global sample-count normalization).

Distribution primitives (``all_reduce`` / ``get_parallel_state``) are
monkeypatched so the pure normalization arithmetic can be checked without a
process group.
"""

import pytest
import torch

import veomni.utils.loss_utils as L


class _FakePS:
    def __init__(self, sp_enabled=False, fsdp_size=1):
        self.sp_enabled = sp_enabled
        self.fsdp_size = fsdp_size


def _patch(monkeypatch, sp_enabled=False, fsdp_size=1):
    # all_reduce(sum) over a single rank is identity.
    monkeypatch.setattr(L, "all_reduce", lambda data, op="mean", group=None: int(data))
    monkeypatch.setattr(L, "get_parallel_state", lambda: _FakePS(sp_enabled, fsdp_size))


def test_per_sample_mean(monkeypatch):
    _patch(monkeypatch, fsdp_size=1)
    out = L.dllm_mean_global_loss(torch.tensor(6.0), micro_batches_num_samples=3)
    torch.testing.assert_close(out, torch.tensor(2.0))


def test_fsdp_size_multiplier(monkeypatch):
    _patch(monkeypatch, fsdp_size=4)
    out = L.dllm_mean_global_loss(torch.tensor(6.0), micro_batches_num_samples=3)
    # 6/3 * 4
    torch.testing.assert_close(out, torch.tensor(8.0))


def test_zero_samples_returns_zero_with_graph(monkeypatch):
    _patch(monkeypatch, fsdp_size=1)
    ws = torch.tensor(5.0, requires_grad=True)
    out = L.dllm_mean_global_loss(ws, micro_batches_num_samples=0)
    assert out.item() == 0.0
    out.backward()
    assert ws.grad is not None


def test_sp_raises(monkeypatch):
    _patch(monkeypatch, sp_enabled=True)
    with pytest.raises(NotImplementedError):
        L.dllm_mean_global_loss(torch.tensor(1.0), micro_batches_num_samples=1)

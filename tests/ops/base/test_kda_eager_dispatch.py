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

"""The eager chunk_kda reference must never dispatch to fla off cuda devices.

fla's ``chunk_kda`` autograd entry crashes on non-cuda tensors (its
``custom_device_ctx`` indexes ``torch.<device_type>.device``, which
``torch.cpu`` lacks), so environments that install flash-linear-attention
(e.g. the GPU CI image) would break every eager CPU forward through the KDA
layers. The eager op must pick the portable naive fallback for non-cuda
tensors regardless of whether fla is importable.
"""

import sys
import types

import torch

from veomni.models.transformers.kimi_k3.source.kda_fallback import chunk_kda_naive
from veomni.ops import resolve_op


def test_eager_chunk_kda_skips_fla_on_non_cuda_tensors(monkeypatch):
    fake_kda = types.ModuleType("fla.ops.kda")

    def _fla_crash(*args, **kwargs):
        raise AssertionError("fla chunk_kda must not run for non-cuda tensors")

    fake_kda.chunk_kda = _fla_crash
    monkeypatch.setitem(sys.modules, "fla", types.ModuleType("fla"))
    monkeypatch.setitem(sys.modules, "fla.ops", types.ModuleType("fla.ops"))
    monkeypatch.setitem(sys.modules, "fla.ops.kda", fake_kda)

    q = torch.randn(1, 8, 2, 16, dtype=torch.float32)
    k = torch.randn(1, 8, 2, 16, dtype=torch.float32)
    v = torch.randn(1, 8, 2, 32, dtype=torch.float32)
    g = -torch.rand(1, 8, 2, 16, dtype=torch.float32)
    beta = torch.rand(1, 8, 2, dtype=torch.float32)

    chunk_kda = resolve_op("chunk_kda", "kimi_k3", "eager").wrapper
    o, final_state = chunk_kda(q=q, k=k, v=v, g=g, beta=beta)
    o_ref, final_state_ref = chunk_kda_naive(q=q, k=k, v=v, g=g, beta=beta)
    assert o.shape == (1, 8, 2, 32)
    torch.testing.assert_close(o, o_ref)
    assert final_state is None and final_state_ref is None

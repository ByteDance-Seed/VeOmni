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

from types import SimpleNamespace

import pytest
import torch

import veomni.distributed.sequence_parallel.ring_attention as ring_attention_module


def test_ring_attention_routes_fixed_shape_input(monkeypatch):
    calls = []

    def forward(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return query

    backend = SimpleNamespace(forward=forward, packed_forward=None)
    monkeypatch.setattr(ring_attention_module, "_load_backend", lambda device_type: backend)

    query = torch.randn(1, 8, 2, 4)
    output = ring_attention_module.ring_attention(
        query,
        query,
        query,
        group=object(),
        cp_size=2,
        device_type="cuda",
    )

    assert output is query
    assert len(calls) == 1
    assert calls[0][3]["dropout_p"] == 0.0


def test_ring_attention_routes_packed_input_with_local_offsets(monkeypatch):
    calls = []

    def packed_forward(query, key, value, cu_seqlens, max_seqlen, **kwargs):
        calls.append((query, key, value, cu_seqlens, max_seqlen, kwargs))
        return query

    backend = SimpleNamespace(forward=None, packed_forward=packed_forward)
    monkeypatch.setattr(ring_attention_module, "_load_backend", lambda device_type: backend)

    query = torch.randn(1, 8, 2, 4)
    full_cu_seqlens = torch.tensor([0, 8, 16], dtype=torch.int64)
    output = ring_attention_module.ring_attention(
        query,
        query,
        query,
        group=object(),
        cp_size=2,
        cu_seqlens=full_cu_seqlens,
        device_type="cuda",
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    assert torch.equal(calls[0][3], torch.tensor([0, 4, 8], dtype=torch.int32))
    assert calls[0][4] == 4


@pytest.mark.parametrize(
    ("causal", "attention_mask", "match"),
    [
        (False, None, "requires causal attention"),
        (True, torch.ones(1), "does not support explicit attention masks"),
    ],
)
def test_ring_attention_validates_supported_inputs(causal, attention_mask, match):
    query = torch.randn(1, 8, 2, 4)
    with pytest.raises(NotImplementedError, match=match):
        ring_attention_module.ring_attention(
            query,
            query,
            query,
            group=object(),
            cp_size=2,
            attention_mask=attention_mask,
            causal=causal,
            device_type="cuda",
        )

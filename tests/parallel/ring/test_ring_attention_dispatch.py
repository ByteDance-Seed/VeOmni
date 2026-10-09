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
from veomni.utils.device import get_device_type


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
        device_type=get_device_type(),
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
        device_type=get_device_type(),
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    assert torch.equal(calls[0][3], torch.tensor([0, 4, 8], dtype=torch.int32))
    assert calls[0][4] == 4


def test_ring_attention_routes_bidirectional_to_per_rank_offsets(monkeypatch):
    calls = []

    def bidirectional_packed_forward(query, key, value, cu_seqlens_per_rank, **kwargs):
        calls.append((query, cu_seqlens_per_rank, kwargs))
        return query

    backend = SimpleNamespace(
        forward=None, packed_forward=None, bidirectional_packed_forward=bidirectional_packed_forward
    )
    monkeypatch.setattr(ring_attention_module, "_load_backend", lambda device_type: backend)

    # cp=2, ulysses=2: vit seqs of 1380 and 112 rows -> cp local length 752.
    query = torch.randn(1, 752, 2, 4)
    output = ring_attention_module.ring_attention(
        query,
        query,
        query,
        group=object(),
        cp_size=2,
        ulysses_size=2,
        cu_seqlens=torch.tensor([0, 1380, 1492], dtype=torch.int32),
        causal=False,
        device_type="npu",
    )

    assert output.shape == query.shape
    assert len(calls) == 1
    assert calls[0][0].shape == (752, 2, 4)
    assert calls[0][1] == [[0, 692, 748, 752], [0, 688, 744, 752]]
    assert calls[0][2]["dropout_p"] == 0.0


def test_ring_attention_bidirectional_single_vit_seq_uses_packed_path(monkeypatch):
    calls = []
    backend = SimpleNamespace(
        forward=None,
        packed_forward=None,
        bidirectional_packed_forward=lambda q, k, v, cu, **kw: calls.append(cu) or q,
    )
    monkeypatch.setattr(ring_attention_module, "_load_backend", lambda device_type: backend)

    query = torch.randn(1, 8, 2, 4)
    ring_attention_module.ring_attention(
        query, query, query, group=object(), cp_size=2, cu_seqlens=torch.tensor([0, 12]), causal=False
    )
    assert calls == [[[0, 8, 8], [0, 4, 8]]]


def test_ring_attention_bidirectional_rejects_mismatched_local_length(monkeypatch):
    backend = SimpleNamespace(bidirectional_packed_forward=None)
    monkeypatch.setattr(ring_attention_module, "_load_backend", lambda device_type: backend)
    query = torch.randn(1, 6, 2, 4)
    with pytest.raises(ValueError, match="expected cp local length 8"):
        ring_attention_module.ring_attention(
            query, query, query, group=object(), cp_size=2, cu_seqlens=torch.tensor([0, 12]), causal=False
        )


def test_ring_attention_bidirectional_is_not_implemented_on_cuda():
    query = torch.randn(1, 8, 2, 4)
    with pytest.raises(NotImplementedError, match="only implemented for NPU"):
        ring_attention_module.ring_attention(
            query,
            query,
            query,
            group=object(),
            cp_size=2,
            cu_seqlens=torch.tensor([0, 12]),
            causal=False,
            device_type="cuda",
        )


def test_ring_attention_bidirectional_requires_cu_seqlens():
    query = torch.randn(1, 8, 2, 4)
    with pytest.raises(ValueError, match="requires the global cu_seqlens"):
        ring_attention_module.ring_attention(
            query, query, query, group=object(), cp_size=2, causal=False, device_type="cuda"
        )


def test_ring_attention_rejects_explicit_attention_mask():
    query = torch.randn(1, 8, 2, 4)
    for causal in (True, False):
        with pytest.raises(NotImplementedError, match="does not support explicit attention masks"):
            ring_attention_module.ring_attention(
                query,
                query,
                query,
                group=object(),
                cp_size=2,
                attention_mask=torch.ones(1),
                causal=causal,
                device_type="cuda",
            )

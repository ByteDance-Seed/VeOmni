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

"""Single-shard equivalence for the vocab-parallel (``emb``) embedding ops.

One shard owns the whole vocabulary, so both ops must reduce exactly to the
dense op they shard — :func:`F.embedding` and :func:`F.linear`. That is worth
pinning because the sharded path reaches its result through three collectives
writing into freshly ``empty`` buffers: an unsharded path that skips a
collective without aliasing its buffer returns uninitialized memory forward and
an all-zero weight gradient back, neither of which raises.

The multi-rank paths need a process group and live with the distributed tests.
"""

import pytest
import torch
import torch.nn.functional as F

from veomni.ops.kernels.embed import AllToAllEmbedding, VocabParallelLinear
from veomni.utils.device import get_device_type


VOCAB, HIDDEN = 5, 4


@pytest.fixture
def table() -> torch.Tensor:
    # float64 so gradcheck's finite differences are meaningful. These stay on
    # CPU: float64 autograd is what the reference comparison needs and is not
    # dependably supported on every accelerator this repo targets. The device
    # parity of the dispatch itself is covered in float32 below.
    return torch.arange(VOCAB * HIDDEN, dtype=torch.float64).reshape(VOCAB, HIDDEN)


def _device() -> torch.device:
    # Route through ``veomni.utils.device.get_device_type`` so the test follows
    # the selected accelerator and passes the device-api-check sanity job
    # (which forbids hardcoded device strings in tests).
    return torch.device(get_device_type())


def test_embedding_forward_matches_dense(table):
    ids = torch.tensor([0, 3, 1, 3])
    assert torch.equal(AllToAllEmbedding.apply(None, ids, table), F.embedding(ids, table))


def test_embedding_preserves_input_shape(table):
    ids = torch.tensor([[0, 3], [1, 4]])
    assert AllToAllEmbedding.apply(None, ids, table).shape == (2, 2, HIDDEN)


def test_embedding_backward_matches_dense_and_accumulates_repeats(table):
    """A repeated id must sum into one row, not overwrite it."""
    ids = torch.tensor([0, 3, 1, 3])
    grad = torch.randn(4, HIDDEN, dtype=torch.float64)

    sharded_w = table.clone().requires_grad_(True)
    dense_w = table.clone().requires_grad_(True)
    (AllToAllEmbedding.apply(None, ids, sharded_w) * grad).sum().backward()
    (F.embedding(ids, dense_w) * grad).sum().backward()

    assert torch.allclose(sharded_w.grad, dense_w.grad)
    assert torch.allclose(sharded_w.grad[3], grad[1] + grad[3])
    # Rows no id touched stay at zero rather than at whatever the buffer held.
    assert torch.equal(sharded_w.grad[2], torch.zeros(HIDDEN, dtype=torch.float64))


def test_embedding_gradcheck(table):
    ids = torch.tensor([0, 3, 1, 3])
    weight = table.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda w: AllToAllEmbedding.apply(None, ids, w), (weight,))


def test_embedding_handles_no_ids(table):
    ids = torch.empty(0, dtype=torch.long)
    assert AllToAllEmbedding.apply(None, ids, table).shape == (0, HIDDEN)


def test_linear_forward_and_backward_match_dense(table):
    hidden = torch.randn(2, HIDDEN, dtype=torch.float64)
    grad = torch.randn(2, VOCAB, dtype=torch.float64)

    sharded_h = hidden.clone().requires_grad_(True)
    sharded_w = table.clone().requires_grad_(True)
    dense_h = hidden.clone().requires_grad_(True)
    dense_w = table.clone().requires_grad_(True)

    sharded = VocabParallelLinear.apply(None, sharded_h, sharded_w)
    dense = F.linear(dense_h, dense_w)
    assert torch.allclose(sharded, dense)

    (sharded * grad).sum().backward()
    (dense * grad).sum().backward()
    assert torch.allclose(sharded_h.grad, dense_h.grad)
    assert torch.allclose(sharded_w.grad, dense_w.grad)


def test_linear_gradcheck(table):
    hidden = torch.randn(2, HIDDEN, dtype=torch.float64, requires_grad=True)
    weight = table.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(lambda h, w: VocabParallelLinear.apply(None, h, w), (hidden, weight))


def test_ops_match_dense_on_the_selected_device():
    """Both ops on the accelerator the GPU / NPU jobs actually select.

    The rest of this file pins the numerics in CPU float64. Index dispatch and
    the buffer writes behind it are device code, so run them where they ship.
    """
    device = _device()
    ids = torch.tensor([0, 3, 1, 3], device=device)
    weight = torch.arange(VOCAB * HIDDEN, dtype=torch.float32, device=device).reshape(VOCAB, HIDDEN)
    hidden = torch.randn(2, HIDDEN, dtype=torch.float32, device=device)

    assert torch.equal(AllToAllEmbedding.apply(None, ids, weight), F.embedding(ids, weight))

    sharded_w = weight.clone().requires_grad_(True)
    dense_w = weight.clone().requires_grad_(True)
    grad = torch.randn(4, HIDDEN, dtype=torch.float32, device=device)
    (AllToAllEmbedding.apply(None, ids, sharded_w) * grad).sum().backward()
    (F.embedding(ids, dense_w) * grad).sum().backward()
    assert torch.allclose(sharded_w.grad, dense_w.grad)

    assert torch.allclose(VocabParallelLinear.apply(None, hidden, weight), F.linear(hidden, weight))


def test_embedding_rejects_out_of_range_ids_instead_of_returning_garbage(table):
    """A negative id must fail, not silently produce an uninitialized row.

    Ids are bucketed by rank with a floor division, so a negative id lands in
    bucket -1 — a bucket nothing collects. It would drop out of the dispatch
    before the local-range check below could see it, leaving its row of the
    ``empty`` output at whatever the buffer held.
    """
    with pytest.raises(RuntimeError, match="local index out of range"):
        AllToAllEmbedding.apply(None, torch.tensor([0, -1]), table)

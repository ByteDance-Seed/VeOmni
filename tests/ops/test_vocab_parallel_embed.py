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

"""Vocab-parallel (``emb``) embedding ops against the dense ops they shard.

One shard owns the whole vocabulary, so both ops must reduce exactly to
:func:`F.embedding` and :func:`F.linear`. That is worth pinning because the
sharded path reaches its result through collectives writing into freshly
``empty`` buffers: an unsharded path that skips a collective without aliasing
its buffer returns uninitialized memory forward and an all-zero weight gradient
back, neither of which raises.

The sharded paths run on two CPU gloo ranks with uneven token counts (one rank
holds none) and are compared with a dense reference over every rank's tokens.
"""

import json

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
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


def test_embedding_backward_sums_repeats_in_fp32_and_returns_the_table_dtype():
    """bf16 running sums of many small grads would stall; only the result is rounded."""
    table = torch.zeros(VOCAB, HIDDEN, dtype=torch.bfloat16, requires_grad=True)
    ids = torch.full((4096,), 3)
    grad = torch.full((4096, HIDDEN), 1e-3, dtype=torch.bfloat16)
    (AllToAllEmbedding.apply(None, ids, table) * grad).sum().backward()

    assert table.grad.dtype is torch.bfloat16
    expected = (grad.float().sum(0)).to(torch.bfloat16)
    assert torch.equal(table.grad[3], expected)
    assert torch.count_nonzero(table.grad[torch.arange(VOCAB) != 3]) == 0


def test_embedding_accepts_int32_ids(table):
    ids = torch.tensor([0, 3, 1, 3], dtype=torch.int32)
    weight = table.clone().requires_grad_(True)
    out = AllToAllEmbedding.apply(None, ids, weight)
    out.sum().backward()
    assert torch.equal(out, F.embedding(ids, table))
    assert torch.equal(weight.grad[3], torch.full((HIDDEN,), 2.0, dtype=table.dtype))


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
    with pytest.raises(RuntimeError, match="token ids must lie in"):
        AllToAllEmbedding.apply(None, torch.tensor([0, -1]), table)
    with pytest.raises(RuntimeError, match="token ids must lie in"):
        AllToAllEmbedding.apply(None, torch.tensor([0, VOCAB]), table)


_WORLD = 2
_MR_VOCAB, _MR_HIDDEN = 6, 3
_RANK_IDS = [[0, 5, 3, 3, 1], []]  # rank 0 hits both shards and repeats an id; rank 1 holds none
_RANK_HIDDEN_ROWS = [3, 0]


def _mr_table() -> torch.Tensor:
    return torch.randn(_MR_VOCAB, _MR_HIDDEN, generator=torch.Generator().manual_seed(0), dtype=torch.float64)


def _mr_randn(rank: int, salt: int, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(100 * salt + rank), dtype=torch.float64)


def _parity_rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=_WORLD, rank=rank)
    try:
        table = _mr_table()
        rows = _MR_VOCAB // _WORLD
        chunk = slice(rank * rows, (rank + 1) * rows)

        ids = torch.tensor(_RANK_IDS[rank], dtype=torch.long)
        hidden = _mr_randn(rank, 1, _RANK_HIDDEN_ROWS[rank], _MR_HIDDEN).requires_grad_(True)
        emb_shard = table[chunk].clone().requires_grad_(True)
        lin_shard = table[chunk].clone().requires_grad_(True)
        embs = AllToAllEmbedding.apply(dist.group.WORLD, ids, emb_shard)
        logits = VocabParallelLinear.apply(dist.group.WORLD, hidden, lin_shard)
        (embs * _mr_randn(rank, 2, len(ids), _MR_HIDDEN)).sum().backward()
        (logits * _mr_randn(rank, 3, _RANK_HIDDEN_ROWS[rank], _MR_VOCAB)).sum().backward()

        # Dense reference over every rank's tokens: the shard grad sums the whole emb group.
        emb_dense = table.clone().requires_grad_(True)
        lin_dense = table.clone().requires_grad_(True)
        hidden_dense = None
        for r in range(_WORLD):
            ids_r = torch.tensor(_RANK_IDS[r], dtype=torch.long)
            hidden_r = _mr_randn(r, 1, _RANK_HIDDEN_ROWS[r], _MR_HIDDEN).requires_grad_(True)
            (F.embedding(ids_r, emb_dense) * _mr_randn(r, 2, len(ids_r), _MR_HIDDEN)).sum().backward()
            (F.linear(hidden_r, lin_dense) * _mr_randn(r, 3, _RANK_HIDDEN_ROWS[r], _MR_VOCAB)).sum().backward()
            if r == rank:
                hidden_dense = hidden_r

        result = {
            "embedding_forward": torch.equal(embs, F.embedding(ids, table)),
            "embedding_grad": torch.allclose(emb_shard.grad, emb_dense.grad[chunk]),
            "linear_forward": torch.allclose(logits, F.linear(hidden.detach(), table)),
            "linear_hidden_grad": torch.allclose(hidden.grad, hidden_dense.grad),
            "linear_weight_grad": torch.allclose(lin_shard.grad, lin_dense.grad[chunk]),
        }
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_sharded_ops_match_dense_over_all_ranks(tmp_path):
    mp.spawn(_parity_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=_WORLD, join=True)

    for rank in range(_WORLD):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert all(result.values()), (rank, result)


def _out_of_range_rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=_WORLD, rank=rank)
    try:
        ids = torch.tensor([0, _MR_VOCAB] if rank == 0 else [1], dtype=torch.long)
        shard = torch.zeros(_MR_VOCAB // _WORLD, _MR_HIDDEN)
        try:
            AllToAllEmbedding.apply(dist.group.WORLD, ids, shard)
            raised = False
        except RuntimeError as e:
            raised = "token ids must lie in" in str(e)
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(raised, f)
    finally:
        dist.destroy_process_group()


def test_an_out_of_range_id_on_one_rank_fails_every_rank(tmp_path):
    """The rank without a bad id must raise too, not block in the next collective."""
    mp.spawn(_out_of_range_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=_WORLD, join=True)

    assert [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(_WORLD)] == [True, True]

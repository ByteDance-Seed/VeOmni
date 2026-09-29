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

"""``EmbParallelMixin`` / ``VocabParallelEmbedding`` against the dense ops.

With ``emb`` off every module runs the plain ``nn.Embedding`` / ``F.linear``
path. With it on, lookup and tied projection must go through the embedding's
own FSDP2 unshard hooks, so the weight gradient is reduce-scattered and scaled
like any other parameter's: the multi-rank tests wrap the table with
``fully_shard`` on CPU gloo ranks and compare it with a dense reference over
every rank's tokens. The kernels themselves are covered in
``tests/ops/test_vocab_parallel_embed.py``.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from veomni.models.seed_omni.mixins import emb_parallel_mixin
from veomni.models.seed_omni.mixins.emb_parallel_mixin import EmbParallelMixin, VocabParallelEmbedding


def _emb_state(group=None, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        extra_parallel_sizes={"emb": 2} if enabled else {},
        extra_parallel_enabled=lambda name: enabled,
        extra_parallel_group=lambda name: group,
    )


def _install_state(module, state) -> None:
    """Point the mixin at ``state``; ``None`` means no parallel state was ever initialized."""
    module.is_parallel_state_initialized = lambda: state is not None
    module.get_parallel_state = lambda: state


@pytest.fixture
def emb_off(monkeypatch):
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: _emb_state(enabled=False))


@pytest.fixture
def emb_on_single_shard(monkeypatch):
    """``emb`` reported on, with a ``None`` group: the kernels run their one-shard path."""
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: _emb_state(group=None))


def test_emb_parallel_is_inactive_before_any_parallel_state(monkeypatch):
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: False)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: pytest.fail("must not build a state"))
    assert EmbParallelMixin.emb_parallel_active() is False


def test_emb_parallel_is_inactive_without_the_group(emb_off):
    assert EmbParallelMixin.emb_parallel_active() is False


def test_emb_off_matches_nn_embedding_and_linear(emb_off):
    embedding = VocabParallelEmbedding(6, 4)
    plain = nn.Embedding(6, 4)
    plain.weight = embedding.weight
    ids = torch.tensor([[0, 3], [5, 1]])
    hidden = torch.randn(2, 4)

    assert torch.equal(EmbParallelMixin.emb_parallel_lookup(embedding, ids), plain(ids))
    assert torch.equal(EmbParallelMixin.emb_parallel_lookup(plain, ids), plain(ids))
    assert torch.equal(EmbParallelMixin.emb_parallel_project(hidden, embedding), F.linear(hidden, embedding.weight))
    assert torch.equal(EmbParallelMixin.emb_parallel_project(hidden, plain), F.linear(hidden, plain.weight))


def test_projection_casts_the_weight_to_the_activation_dtype(emb_off):
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    assert EmbParallelMixin.emb_parallel_project(hidden, VocabParallelEmbedding(6, 4)).dtype is torch.bfloat16
    assert EmbParallelMixin.emb_parallel_project(hidden, nn.Embedding(6, 4)).dtype is torch.bfloat16


def test_emb_on_rejects_a_plain_embedding(emb_on_single_shard):
    """A plain lookup would index this rank's vocab shard with global ids."""
    with pytest.raises(TypeError, match="not a VocabParallelEmbedding"):
        EmbParallelMixin.emb_parallel_lookup(nn.Embedding(6, 4), torch.tensor([0]))
    with pytest.raises(TypeError, match="use VocabParallelEmbedding"):
        EmbParallelMixin.emb_parallel_project(torch.randn(1, 4), nn.Embedding(6, 4))


def test_emb_on_padding_row_gets_no_gradient(emb_on_single_shard):
    embedding = VocabParallelEmbedding(6, 4, padding_idx=2)
    ids = torch.tensor([2, 0, 2, 5])
    dense = embedding.weight.detach().clone().requires_grad_(True)

    out = EmbParallelMixin.emb_parallel_lookup(embedding, ids)
    assert torch.equal(out, F.embedding(ids, dense))
    out.sum().backward()
    F.embedding(ids, dense, padding_idx=2).sum().backward()
    assert torch.equal(embedding.weight.grad, dense.grad)
    assert torch.equal(embedding.weight.grad[2], torch.zeros(4))


def test_emb_on_rejects_max_norm(emb_on_single_shard):
    with pytest.raises(NotImplementedError, match="max_norm"):
        VocabParallelEmbedding(6, 4, max_norm=1.0)(torch.tensor([0]))


_VOCAB, _HIDDEN = 8, 4


def _table() -> torch.Tensor:
    return torch.randn(_VOCAB, _HIDDEN, generator=torch.Generator().manual_seed(0))


def _randn(rank: int, salt: int, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(100 * salt + rank))


def _loss(embs: torch.Tensor, logits: torch.Tensor, rank: int) -> torch.Tensor:
    """Fixed upstream grads per rank, as one training step would feed back."""
    return (embs * _randn(rank, 2, *embs.shape)).sum() + (logits * _randn(rank, 3, *logits.shape)).sum()


def _step(embedding: nn.Module, ids: torch.Tensor, hidden: torch.Tensor, rank: int):
    embs = EmbParallelMixin.emb_parallel_lookup(embedding, ids)
    logits = EmbParallelMixin.emb_parallel_project(hidden, embedding)
    return embs, logits, _loss(embs, logits, rank)


def _dense_grad(rank_ids, rank_hidden_rows) -> torch.Tensor:
    """Sum of every rank's weight gradient on the unsharded table."""
    dense = _table().requires_grad_(True)
    for r, (ids, rows) in enumerate(zip(rank_ids, rank_hidden_rows)):
        embs = F.embedding(torch.tensor(ids, dtype=torch.long), dense)
        logits = F.linear(_randn(r, 1, rows, _HIDDEN), dense)
        _loss(embs, logits, r).backward()
    return dense.grad


_EMB_IDS = [[0, 7, 3, 3], [], [5, 1], [6, 6, 2, 4, 0]]
_EMB_HIDDEN_ROWS = [2, 0, 3, 1]


def _emb_fsdp_rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    """emb=2 x emb_fsdp=2: vocab rows over ``emb``, hidden over ``emb_fsdp``."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard

    world = len(_EMB_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("emb_fsdp", "emb"))
        emb_rank = mesh["emb"].get_local_rank()
        _install_state(emb_parallel_mixin, _emb_state(group=mesh["emb"].get_group()))

        rows = _VOCAB // 2
        chunk = slice(emb_rank * rows, (emb_rank + 1) * rows)
        embedding = VocabParallelEmbedding(_VOCAB, _HIDDEN)
        embedding.weight = nn.Parameter(_table()[chunk].clone())
        # Default divide factor (the emb_fsdp size): gloo has no PREMUL_SUM, which a custom factor needs.
        fully_shard(embedding, mesh=mesh["emb_fsdp"], shard_placement_fn=lambda param: Shard(1))

        ids = torch.tensor(_EMB_IDS[rank], dtype=torch.long)
        hidden = _randn(rank, 1, _EMB_HIDDEN_ROWS[rank], _HIDDEN)
        embs, logits, loss = _step(embedding, ids, hidden, rank)
        loss.backward()
        grad = embedding.weight.grad.full_tensor()

        expected_grad = _dense_grad(_EMB_IDS, _EMB_HIDDEN_ROWS)[chunk] / mesh["emb_fsdp"].size()
        result = {
            "lookup": torch.allclose(embs, F.embedding(ids, _table())),
            "project": torch.allclose(logits, F.linear(hidden, _table()), atol=1e-6),
            "grad": torch.allclose(grad, expected_grad, atol=1e-5),
        }
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


def test_emb_sharded_table_matches_dense_through_fsdp(tmp_path):
    world = len(_EMB_IDS)
    mp.spawn(_emb_fsdp_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    for rank in range(world):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert all(result.values()), (rank, result)


_DP_IDS = [[0, 7, 3], [5, 5]]
_DP_HIDDEN_ROWS = [2, 1]


def _tied_fsdp_rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    """emb off, table sharded by plain FSDP2: the tied head must still reach the gradient reduction."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    world = len(_DP_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        _install_state(emb_parallel_mixin, None)
        embedding = VocabParallelEmbedding(_VOCAB, _HIDDEN)
        embedding.weight = nn.Parameter(_table())
        fully_shard(embedding, mesh=init_device_mesh("cpu", (world,)))

        _, _, loss = _step(
            embedding,
            torch.tensor(_DP_IDS[rank], dtype=torch.long),
            _randn(rank, 1, _DP_HIDDEN_ROWS[rank], _HIDDEN),
            rank,
        )
        loss.backward()
        grad = embedding.weight.grad.full_tensor()

        expected_grad = _dense_grad(_DP_IDS, _DP_HIDDEN_ROWS) / world
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(torch.allclose(grad, expected_grad, atol=1e-5), f)
    finally:
        dist.destroy_process_group()


def test_tied_head_gradient_is_reduced_by_fsdp_when_emb_is_off(tmp_path):
    world = len(_DP_IDS)
    mp.spawn(_tied_fsdp_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    assert [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(world)] == [True] * world

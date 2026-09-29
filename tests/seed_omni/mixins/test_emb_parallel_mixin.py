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


def _emb_state(group=None, enabled: bool = True, size: int = 2, rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        extra_parallel_sizes={"emb": size} if enabled else {},
        extra_parallel_enabled=lambda name: enabled,
        extra_parallel_group=lambda name: group,
        extra_parallel_rank=lambda name: rank,
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
    """``emb`` reported on over one whole-vocab shard: the kernels run their one-shard path."""
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: _emb_state(group=None, size=1))


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


def _sliced(embedding: VocabParallelEmbedding, rows: int) -> VocabParallelEmbedding:
    embedding.weight = nn.Parameter(torch.randn(rows, embedding.embedding_dim))
    return embedding


def test_a_sliced_table_is_rejected_when_emb_is_off(emb_off):
    """Global ids would silently read the wrong rows of a shard."""
    embedding = _sliced(VocabParallelEmbedding(8, 4), rows=4)
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows"):
        embedding(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows"):
        EmbParallelMixin.emb_parallel_project(torch.randn(1, 4), embedding)

    embedding = _sliced(VocabParallelEmbedding(8, 4, padding_idx=5), rows=4)
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows"):
        _ = embedding.padding_idx


def test_an_unsliced_table_is_rejected_when_emb_is_on(monkeypatch):
    """The kernels would treat the whole table as one rank's shard of a ``vocab * emb`` vocabulary."""
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: _emb_state(size=2))
    embedding = VocabParallelEmbedding(8, 4)
    with pytest.raises(RuntimeError, match="holds 8 of 8 vocab rows, but the emb group here has 2"):
        embedding(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="holds 8 of 8 vocab rows, but the emb group here has 2"):
        EmbParallelMixin.emb_parallel_project(torch.randn(1, 4), embedding)


@pytest.mark.parametrize(("emb_rank", "local_padding_idx"), [(0, None), (1, 1)])
def test_padding_idx_indexes_the_rows_this_rank_holds(monkeypatch, emb_rank, local_padding_idx):
    """Initializers that zero ``weight[padding_idx]`` must hit global row 5 on its owner only."""
    monkeypatch.setattr(emb_parallel_mixin, "is_parallel_state_initialized", lambda: True)
    monkeypatch.setattr(emb_parallel_mixin, "get_parallel_state", lambda: _emb_state(size=2, rank=emb_rank))
    embedding = VocabParallelEmbedding(8, 4, padding_idx=5)
    assert embedding.padding_idx == 5
    assert repr(embedding) == "VocabParallelEmbedding(8, 4, padding_idx=5)"

    _sliced(embedding, rows=4).reset_parameters()
    assert embedding.padding_idx == local_padding_idx
    zero_rows = [row for row in range(4) if torch.equal(embedding.weight[row], torch.zeros(4))]
    assert zero_rows == ([] if local_padding_idx is None else [local_padding_idx])


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


_EMB_IDS = [[0, 7, 3, 3], [], [5, 1], [6, 6, 2, 5, 0]]
_EMB_PADDING_IDX = 5


def _mid_weight() -> torch.Tensor:
    return torch.randn(_HIDDEN, _HIDDEN, generator=torch.Generator().manual_seed(1))


class _TiedModel(nn.Module):
    """Lookup, a layer, then the tied head: the table is used twice in one backward."""

    def __init__(self, embedding: nn.Module):
        super().__init__()
        self.embedding = embedding
        self.mid = nn.Linear(_HIDDEN, _HIDDEN, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        embs = EmbParallelMixin.emb_parallel_lookup(self.embedding, ids)
        return EmbParallelMixin.emb_parallel_project(torch.tanh(self.mid(embs)), self.embedding)


def _dense_tied_grads() -> tuple[torch.Tensor, torch.Tensor]:
    table = _table().requires_grad_(True)
    mid = _mid_weight().requires_grad_(True)
    for r, ids in enumerate(_EMB_IDS):
        embs = F.embedding(torch.tensor(ids, dtype=torch.long), table, padding_idx=_EMB_PADDING_IDX)
        logits = F.linear(torch.tanh(F.linear(embs, mid)), table)
        (logits * _randn(r, 3, *logits.shape)).sum().backward()
    return table.grad, mid.grad


def _emb_fsdp_rank_main(rank: int, rendezvous: str, out_dir: str, reshard_after_forward: bool) -> None:
    """emb=2 x emb_fsdp=2: vocab rows over ``emb``, hidden over ``emb_fsdp``, the rest over all ranks."""
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard

    world = len(_EMB_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("emb_fsdp", "emb"))
        emb_rank = mesh["emb"].get_local_rank()
        _install_state(emb_parallel_mixin, _emb_state(group=mesh["emb"].get_group(), rank=emb_rank))

        rows = _VOCAB // 2
        chunk = slice(emb_rank * rows, (emb_rank + 1) * rows)
        embedding = VocabParallelEmbedding(_VOCAB, _HIDDEN, padding_idx=_EMB_PADDING_IDX)
        embedding.weight = nn.Parameter(_table()[chunk].clone())
        model = _TiedModel(embedding)
        model.mid.weight = nn.Parameter(_mid_weight())
        # Default divide factor (the emb_fsdp size): gloo has no PREMUL_SUM, which a custom factor needs.
        fully_shard(
            embedding,
            mesh=mesh["emb_fsdp"],
            shard_placement_fn=lambda param: Shard(1),
            reshard_after_forward=reshard_after_forward,
        )
        fully_shard(model, mesh=init_device_mesh("cpu", (world,)), reshard_after_forward=reshard_after_forward)

        ids = torch.tensor(_EMB_IDS[rank], dtype=torch.long)
        logits = model(ids)
        (logits * _randn(rank, 3, *logits.shape)).sum().backward()

        expected_table_grad, expected_mid_grad = _dense_tied_grads()
        expected_logits = F.linear(torch.tanh(F.linear(F.embedding(ids, _table()), _mid_weight())), _table())
        result = {
            "logits": torch.allclose(logits, expected_logits, atol=1e-6),
            "table_grad": torch.allclose(
                embedding.weight.grad.full_tensor(), expected_table_grad[chunk] / mesh["emb_fsdp"].size(), atol=1e-5
            ),
            "mid_grad": torch.allclose(model.mid.weight.grad.full_tensor(), expected_mid_grad / world, atol=1e-5),
        }
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("reshard_after_forward", [True, False])
def test_emb_sharded_table_matches_dense_through_fsdp(tmp_path, reshard_after_forward):
    world = len(_EMB_IDS)
    mp.spawn(
        _emb_fsdp_rank_main,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), reshard_after_forward),
        nprocs=world,
        join=True,
    )

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
        # Until EmbParallelMixin registers ``project`` with FSDP, a direct call sees the sharded weight.
        try:
            embedding.project(torch.randn(1, _HIDDEN))
            direct_project_rejected = False
        except RuntimeError as err:
            direct_project_rejected = "outside its FSDP2 unshard hooks" in str(err)

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
            json.dump([torch.allclose(grad, expected_grad, atol=1e-5), direct_project_rejected], f)
    finally:
        dist.destroy_process_group()


def test_tied_head_gradient_is_reduced_by_fsdp_when_emb_is_off(tmp_path):
    world = len(_DP_IDS)
    mp.spawn(_tied_fsdp_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    assert [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(world)] == [[True, True]] * world

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

"""``AllToAllEmbedding`` / ``VocabParallelLinear`` / ``ShardedEmbedding`` against the dense ops they shard.

With one shard owning the whole vocabulary the ops must reduce exactly to
:func:`F.embedding` and :func:`F.linear`. That is worth pinning because the sharded path reaches its
result through collectives writing into freshly ``empty`` buffers: an unsharded
path that skips a collective without aliasing its buffer returns uninitialized
memory forward and an all-zero weight gradient back, neither of which raises.

The sharded paths run on CPU gloo ranks with uneven token counts (one rank holds
none) and are compared with a dense reference over every rank's tokens. The
module is also run through FSDP2 on an ``emb`` x ``emb_fsdp`` mesh, lookup and tied head
alike, so its weight gradient is reduce-scattered and scaled like any other parameter's.
"""

import json
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from veomni.distributed.emb_parallel import AllToAllEmbedding, ShardedEmbedding, VocabParallelLinear, sharded_embedding
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


def _emb_state(group=None, enabled: bool = True, size: int = 2, rank: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        extra_parallel_sizes={"emb": size} if enabled else {},
        extra_parallel_enabled=lambda name: enabled,
        extra_parallel_group=lambda name: group,
        extra_parallel_rank=lambda name: rank,
    )


def _install_state(state) -> None:
    """Point ``ShardedEmbedding`` at ``state``; ``None`` means no parallel state was ever initialized."""
    sharded_embedding.is_parallel_state_initialized = lambda: state is not None
    sharded_embedding.get_parallel_state = lambda: state


@pytest.fixture
def emb_state(monkeypatch):
    def install(state) -> None:
        monkeypatch.setattr(sharded_embedding, "is_parallel_state_initialized", lambda: state is not None)
        monkeypatch.setattr(sharded_embedding, "get_parallel_state", lambda: state)

    return install


@pytest.mark.parametrize(
    "state",
    [None, _emb_state(enabled=False), _emb_state(size=2)],
    ids=["no_parallel_state", "emb_off", "emb_on_but_table_not_in_the_plan"],
)
def test_an_unsplit_table_matches_nn_embedding(emb_state, state):
    emb_state(state)
    embedding = ShardedEmbedding(6, 4, padding_idx=2)
    plain = nn.Embedding(6, 4, padding_idx=2)
    plain.weight = embedding.weight
    ids = torch.tensor([[0, 3], [5, 2]])
    hidden = torch.randn(2, 4)
    assert torch.equal(embedding(ids), plain(ids))
    assert torch.equal(embedding.project(hidden), F.linear(hidden, plain.weight))


def test_project_casts_the_weight_to_the_activation_dtype():
    hidden = torch.randn(2, 4, dtype=torch.bfloat16)
    assert ShardedEmbedding(6, 4).project(hidden).dtype is torch.bfloat16


def _sliced(embedding: ShardedEmbedding, rows: int) -> ShardedEmbedding:
    embedding.weight = nn.Parameter(torch.randn(rows, embedding.embedding_dim))
    return embedding


def test_a_sliced_table_is_rejected_when_emb_is_off(emb_state):
    """Global ids would silently read the wrong rows of a shard."""
    emb_state(_emb_state(enabled=False))
    embedding = _sliced(ShardedEmbedding(8, 4), rows=4)
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows"):
        embedding(torch.tensor([1]))

    embedding = _sliced(ShardedEmbedding(8, 4, padding_idx=5), rows=4)
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows"):
        _ = embedding.padding_idx


def test_a_table_split_for_another_emb_size_is_rejected(emb_state):
    emb_state(_emb_state(size=4))
    with pytest.raises(RuntimeError, match="holds 4 of 8 vocab rows, but the emb group here has 4"):
        _sliced(ShardedEmbedding(8, 4), rows=4)(torch.tensor([1]))


@pytest.mark.parametrize("option", [{"max_norm": 1.0}, {"scale_grad_by_freq": True}, {"sparse": True}])
def test_a_split_table_rejects_unsupported_options(emb_state, option):
    emb_state(_emb_state(size=2))
    with pytest.raises(NotImplementedError, match="max_norm, scale_grad_by_freq and sparse"):
        _sliced(ShardedEmbedding(8, 4, **option), rows=4)(torch.tensor([1]))


def test_a_split_table_outside_its_own_fsdp_unit_is_rejected(emb_state):
    """A parent unit would shard and gather the rows over the whole FSDP mesh, mixing vocab slices."""
    emb_state(_emb_state(size=2))
    embedding = _sliced(ShardedEmbedding(8, 4), rows=4)
    with pytest.raises(RuntimeError, match="not the emb module's own FSDP2 unit"):
        embedding(torch.tensor([1]))
    with pytest.raises(RuntimeError, match="not the emb module's own FSDP2 unit"):
        embedding.project(torch.randn(1, 4))


@pytest.mark.parametrize(("emb_rank", "local_padding_idx"), [(0, None), (1, 1)])
def test_padding_idx_indexes_the_rows_this_rank_holds(emb_state, emb_rank, local_padding_idx):
    """Initializers that zero ``weight[padding_idx]`` must hit global row 5 on its owner only."""
    emb_state(_emb_state(size=2, rank=emb_rank))
    embedding = ShardedEmbedding(8, 4, padding_idx=5)
    assert embedding.padding_idx == 5
    assert repr(embedding) == "ShardedEmbedding(8, 4, padding_idx=5)"

    _sliced(embedding, rows=4).reset_parameters()
    assert embedding.padding_idx == local_padding_idx
    zero_rows = [row for row in range(4) if torch.equal(embedding.weight[row], torch.zeros(4))]
    assert zero_rows == ([] if local_padding_idx is None else [local_padding_idx])


_FSDP_VOCAB, _FSDP_HIDDEN = 8, 4
_FSDP_IDS = [[0, 7, 3, 3], [], [5, 1], [6, 6, 2, 5, 0]]
_FSDP_PADDING_IDX = 5


def _fsdp_weights() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    table = torch.randn(_FSDP_VOCAB, _FSDP_HIDDEN, generator=g)
    mid = torch.randn(_FSDP_HIDDEN, _FSDP_HIDDEN, generator=g)
    head = torch.randn(_FSDP_VOCAB, _FSDP_HIDDEN, generator=g)
    return table, mid, head


def _fsdp_upstream_grad(rank: int, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(300 + rank))


class _Model(nn.Module):
    """Lookup, a layer, then the head: tied (``embedding.project``) or a separate linear."""

    def __init__(self, embedding: ShardedEmbedding, tied: bool):
        super().__init__()
        self.embedding = embedding
        self.mid = nn.Linear(_FSDP_HIDDEN, _FSDP_HIDDEN, bias=False)
        self.head = None if tied else nn.Linear(_FSDP_HIDDEN, _FSDP_VOCAB, bias=False)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        hidden = torch.tanh(self.mid(self.embedding(ids)))
        return self.embedding.project(hidden) if self.head is None else self.head(hidden)


def _dense_logits(ids: torch.Tensor, table, mid, head) -> torch.Tensor:
    embs = F.embedding(ids, table, padding_idx=_FSDP_PADDING_IDX)
    return F.linear(torch.tanh(F.linear(embs, mid)), head)


def _dense_model_grads(tied: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sum of every rank's gradients on the unsharded weights."""
    table, mid, head = (w.requires_grad_(True) for w in _fsdp_weights())
    for r, ids in enumerate(_FSDP_IDS):
        logits = _dense_logits(torch.tensor(ids, dtype=torch.long), table, mid, table if tied else head)
        (logits * _fsdp_upstream_grad(r, *logits.shape)).sum().backward()
    return table.grad, mid.grad, head.grad


def _fsdp_rank_main(
    rank: int, rendezvous: str, out_dir: str, layout: str, tied: bool, reshard: bool, separate: bool
) -> None:
    """``layout="emb"``: emb=2 x emb_fsdp=2, vocab rows over ``emb``, hidden over ``emb_fsdp``.

    ``layout="dp"``: emb off, the whole table its own plain FSDP2 unit over all ranks.

    ``separate``: the lookup, the layer and the head run as separate calls outside the root
    forward, like an encode node and a decode node, with the layer its own unit.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard
    from torch.distributed.tensor import Shard

    world = len(_FSDP_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        world_mesh = init_device_mesh("cpu", (world,))
        table, mid, head = _fsdp_weights()
        embedding = ShardedEmbedding(_FSDP_VOCAB, _FSDP_HIDDEN, padding_idx=_FSDP_PADDING_IDX)
        if layout == "emb":
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("emb_fsdp", "emb"))
            emb_rank = mesh["emb"].get_local_rank()
            _install_state(_emb_state(group=mesh["emb"].get_group(), rank=emb_rank))
            rows = _FSDP_VOCAB // 2
            chunk = slice(emb_rank * rows, (emb_rank + 1) * rows)
            # Default divide factor (the emb_fsdp size): gloo has no PREMUL_SUM, which a custom factor needs.
            table_grad_divisor = mesh["emb_fsdp"].size()
            shard_kwargs = {"mesh": mesh["emb_fsdp"], "shard_placement_fn": lambda param: Shard(1)}
        else:
            _install_state(None)
            chunk = slice(None)
            table_grad_divisor = world
            shard_kwargs = {"mesh": world_mesh}
        embedding.weight = nn.Parameter(table[chunk].clone())
        model = _Model(embedding, tied)
        model.mid.weight = nn.Parameter(mid.clone())
        if not tied:
            model.head.weight = nn.Parameter(head.clone())
        fully_shard(embedding, reshard_after_forward=reshard, **shard_kwargs)
        if layout == "emb":
            embedding._extra_parallel_name = "emb"  # what the parallelizer sets on the emb_fsdp unit
        if separate:
            fully_shard(model.mid, mesh=world_mesh, reshard_after_forward=reshard)
        fully_shard(model, mesh=world_mesh, reshard_after_forward=reshard)

        ids = torch.tensor(_FSDP_IDS[rank], dtype=torch.long)
        if separate:
            logits = embedding.project(torch.tanh(model.mid(embedding(ids))))
        else:
            logits = model(ids)
        (logits * _fsdp_upstream_grad(rank, *logits.shape)).sum().backward()

        table_grad, mid_grad, head_grad = _dense_model_grads(tied)
        result = {
            "logits": torch.allclose(logits, _dense_logits(ids, table, mid, table if tied else head), atol=1e-6),
            "table_grad": torch.allclose(
                embedding.weight.grad.full_tensor(), table_grad[chunk] / table_grad_divisor, atol=1e-5
            ),
            "mid_grad": torch.allclose(model.mid.weight.grad.full_tensor(), mid_grad / world, atol=1e-5),
        }
        if not tied:
            result["head_grad"] = torch.allclose(model.head.weight.grad.full_tensor(), head_grad / world, atol=1e-5)
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(result, f)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("layout", "tied", "reshard", "separate"),
    [
        ("emb", False, True, False),
        ("emb", False, False, False),
        ("emb", True, True, False),
        ("emb", True, False, False),
        ("emb", True, True, True),
        ("dp", True, True, False),
    ],
    ids=[
        "emb-untied-reshard",
        "emb-untied-no_reshard",
        "emb-tied-reshard",
        "emb-tied-no_reshard",
        "emb-tied-separate_calls",
        "dp-tied",
    ],
)
def test_sharded_embedding_matches_dense_through_fsdp(tmp_path, layout, tied, reshard, separate):
    """The tied head uses the table a second time in the same backward, from the same FSDP2 unit."""
    world = len(_FSDP_IDS)
    mp.spawn(
        _fsdp_rank_main,
        args=(str(tmp_path / "rendezvous"), str(tmp_path), layout, tied, reshard, separate),
        nprocs=world,
        join=True,
    )

    for rank in range(world):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert all(result.values()), (rank, result)


def _wrong_mesh_rank_main(rank: int, rendezvous: str, out_dir: str) -> None:
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard

    world = len(_FSDP_IDS)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", world_size=world, rank=rank)
    try:
        mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("emb_fsdp", "emb"))
        emb_rank = mesh["emb"].get_local_rank()
        _install_state(_emb_state(group=mesh["emb"].get_group(), rank=emb_rank))
        rows = _FSDP_VOCAB // 2
        embedding = ShardedEmbedding(_FSDP_VOCAB, _FSDP_HIDDEN)
        embedding.weight = nn.Parameter(_fsdp_weights()[0][emb_rank * rows : (emb_rank + 1) * rows].clone())
        # Its own unit, but on the regular FSDP mesh: what a second planned table listed in
        # _no_split_modules gets, since only the plan's first entry is wrapped on emb_fsdp.
        fully_shard(embedding, mesh=init_device_mesh("cpu", (world,)))
        raised = {}
        for name, call in (
            ("forward", lambda: embedding(torch.tensor([1]))),
            ("project", lambda: embedding.project(torch.randn(1, _FSDP_HIDDEN))),
        ):
            try:
                call()
                raised[name] = False
            except RuntimeError as e:
                raised[name] = "emb_fsdp mesh" in str(e)
        with open(f"{out_dir}/rank{rank}.json", "w") as f:
            json.dump(raised, f)
    finally:
        dist.destroy_process_group()


def test_a_split_table_in_its_own_unit_on_the_wrong_mesh_is_rejected(tmp_path):
    """The gathered rows are correctly shaped but mix different ranks' vocab slices."""
    world = len(_FSDP_IDS)
    mp.spawn(_wrong_mesh_rank_main, args=(str(tmp_path / "rendezvous"), str(tmp_path)), nprocs=world, join=True)

    for rank in range(world):
        result = json.loads((tmp_path / f"rank{rank}.json").read_text())
        assert result == {"forward": True, "project": True}, (rank, result)

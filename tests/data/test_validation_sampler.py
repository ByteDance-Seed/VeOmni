import math
import random
import types

import pytest
import torch
from torch.utils.data import Dataset

from veomni.data import build_dataloader
from veomni.data.data_loader import ExactDistributedBatchSampler


@pytest.mark.parametrize(
    ("dataset_size", "num_replicas", "batch_size"),
    [(1, 1, 1), (3, 2, 2), (5, 2, 2), (8, 3, 2), (16, 4, 1), (23, 4, 3), (97, 8, 4)],
)
def test_exact_distributed_batch_sampler_is_an_exact_equal_step_partition(
    dataset_size: int, num_replicas: int, batch_size: int
):
    batches_by_rank = [
        list(ExactDistributedBatchSampler(dataset_size, batch_size, num_replicas, rank))
        for rank in range(num_replicas)
    ]

    assert len({len(batches) for batches in batches_by_rank}) == 1
    for batches in batches_by_rank:
        assert all(1 <= len(batch) <= batch_size for batch in batches)

    step_major_indices = [
        index
        for step in range(len(batches_by_rank[0]))
        for rank in range(num_replicas)
        for index in batches_by_rank[rank][step]
    ]
    assert step_major_indices == list(range(dataset_size))


def test_exact_distributed_batch_sampler_is_repeatable_without_touching_rng_state():
    sampler = ExactDistributedBatchSampler(dataset_size=23, batch_size=3, num_replicas=4, rank=2)
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()

    first = [tuple(batch) for batch in sampler]
    sampler.set_epoch(9)
    second = [tuple(batch) for batch in sampler]

    assert first == second
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_exact_distributed_batch_sampler_small_grid_property():
    for dataset_size in range(1, 41):
        for num_replicas in range(1, 9):
            for batch_size in range(1, 9):
                schedulable = math.ceil(dataset_size / (num_replicas * batch_size)) * num_replicas <= dataset_size
                if not schedulable:
                    with pytest.raises(ValueError, match="cannot be partitioned"):
                        ExactDistributedBatchSampler(dataset_size, batch_size, num_replicas, rank=0)
                    continue

                batches_by_rank = [
                    list(ExactDistributedBatchSampler(dataset_size, batch_size, num_replicas, rank))
                    for rank in range(num_replicas)
                ]
                assert len({len(batches) for batches in batches_by_rank}) == 1
                indices = [index for batches in batches_by_rank for batch in batches for index in batch]
                assert sorted(indices) == list(range(dataset_size))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"dataset_size": 0, "batch_size": 1, "num_replicas": 1, "rank": 0}, "non-empty"),
        ({"dataset_size": 1, "batch_size": 0, "num_replicas": 1, "rank": 0}, "positive"),
        ({"dataset_size": 1, "batch_size": 1, "num_replicas": 0, "rank": 0}, "positive"),
        ({"dataset_size": 2, "batch_size": 1, "num_replicas": 2, "rank": 2}, "Rank must be"),
        (
            {"dataset_size": 5, "batch_size": 1, "num_replicas": 2, "rank": 0},
            "cannot be partitioned",
        ),
    ],
)
def test_exact_distributed_batch_sampler_rejects_invalid_or_unschedulable_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ExactDistributedBatchSampler(**kwargs)


class _IndexDataset(Dataset):
    def __len__(self):
        return 5

    def __getitem__(self, index):
        return [{"id": torch.tensor(index)}]


def _collate_ids(features):
    return {"id": torch.stack([feature["id"] for feature in features])}


def test_native_dataloader_accepts_exact_batch_sampler(monkeypatch):
    import veomni.data.data_loader as data_loader_module

    parallel_state = types.SimpleNamespace(dp_size=2, dp_rank=0, sp_enabled=False, sp_size=1)
    monkeypatch.setattr(data_loader_module, "get_parallel_state", lambda: parallel_state)
    batch_sampler = ExactDistributedBatchSampler(dataset_size=5, batch_size=2, num_replicas=2, rank=0)
    generator = torch.Generator().manual_seed(17)
    global_rng_state = torch.random.get_rng_state().clone()

    dataloader = build_dataloader(
        "native",
        dataset=_IndexDataset(),
        micro_batch_size=2,
        global_batch_size=4,
        dataloader_batch_size=2,
        max_seq_len=8,
        train_steps=len(batch_sampler),
        dyn_bsz=False,
        num_workers=0,
        prefetch_factor=None,
        pin_memory=False,
        collate_fn=_collate_ids,
        batch_sampler=batch_sampler,
        generator=generator,
    )

    assert torch.equal(torch.random.get_rng_state(), global_rng_state)
    observed = [micro_batches[0]["id"].tolist() for micro_batches in dataloader]
    assert observed == [[0, 1], [3]]
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)
    dataloader.set_epoch(1)
    generator.manual_seed(17)
    assert [micro_batches[0]["id"].tolist() for micro_batches in dataloader] == observed
    assert torch.equal(torch.random.get_rng_state(), global_rng_state)


def test_native_dataloader_rejects_custom_batch_sampler_with_dynamic_batching(monkeypatch):
    import veomni.data.data_loader as data_loader_module

    parallel_state = types.SimpleNamespace(dp_size=1, dp_rank=0, sp_enabled=False, sp_size=1)
    monkeypatch.setattr(data_loader_module, "get_parallel_state", lambda: parallel_state)
    batch_sampler = ExactDistributedBatchSampler(dataset_size=5, batch_size=2, num_replicas=1, rank=0)

    with pytest.raises(ValueError, match="only when dyn_bsz=False"):
        build_dataloader(
            "native",
            dataset=_IndexDataset(),
            micro_batch_size=2,
            global_batch_size=2,
            dataloader_batch_size=1,
            max_seq_len=8,
            train_steps=1,
            dyn_bsz=True,
            num_workers=0,
            prefetch_factor=None,
            pin_memory=False,
            collate_fn=_collate_ids,
            batch_sampler=batch_sampler,
        )

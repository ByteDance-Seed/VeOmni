import copy
import gc
import json
from itertools import islice
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import IterableDataset, get_worker_info

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader, build_native_dataloader
from veomni.data.dataset import DynamicBatchingSizeDataset, IterativeDataset, ShardedIterableDataset


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class CursorRows(IterableDataset):
    def __init__(self, count=4096):
        self.count = count
        self.cursor = 0
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.cursor = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.cursor >= self.count:
            raise StopIteration
        row = {
            "id": self.cursor,
            "seed": str(torch.initial_seed()),
            "random": torch.rand(()).item() if get_worker_info() is not None else 0.0,
            "input_ids": torch.arange(400 + self.cursor % 100),
        }
        self.cursor += 1
        return row

    def state_dict(self):
        return {
            "cursor": self.cursor,
            "epoch": self.epoch,
            "worker_rng": torch.get_rng_state() if get_worker_info() is not None else None,
        }

    def load_state_dict(self, state):
        self.cursor = state["cursor"]
        self.epoch = state["epoch"]
        if get_worker_info() is not None:
            torch.set_rng_state(state["worker_rng"])


def row_transform(row):
    return [row]


def row_collate(rows):
    return [row["id"] for row in rows]


def row_signature(row):
    return row["id"], row["seed"], row["random"]


def close_loaders(*loaders):
    for loader in loaders:
        iterator = getattr(loader, "_iterator", None)
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        loader._iterator = None
    gc.collect()


def make_loader(workers=0, *, dataset=None, seed=42, snapshot_every_n_steps=1):
    return DistributedDataloader(
        CursorRows() if dataset is None else dataset,
        batch_size=None,
        num_workers=workers,
        generator=torch.Generator().manual_seed(seed),
        multiprocessing_context="spawn" if workers else None,
        snapshot_every_n_steps=snapshot_every_n_steps,
    )


@pytest.mark.parametrize("workers", [0, 1])
def test_native_builder_does_not_consume_training_cpu_rng(monkeypatch, workers):
    import veomni.data.data_loader as module

    monkeypatch.setattr(
        module,
        "get_parallel_state",
        lambda: SimpleNamespace(dp_rank=0, dp_size=1, sp_enabled=False, sp_size=1, sp_rank=0),
    )
    loader = build_native_dataloader(
        dataset=IterativeDataset(CursorRows(), row_transform),
        micro_batch_size=1,
        global_batch_size=1,
        dataloader_batch_size=1,
        max_seq_len=500,
        train_steps=1,
        dyn_bsz=False,
        num_workers=workers,
        pin_memory=False,
        prefetch_factor=2 if workers else None,
        multiprocessing_context="spawn" if workers else None,
        build_collate_fn=False,
        seed=42,
    )
    try:
        training_rng = torch.get_rng_state().clone()
        next(iter(loader))
        assert torch.equal(torch.get_rng_state(), training_rng)
    finally:
        close_loaders(loader)


@pytest.mark.parametrize("workers", [0, 1])
def test_native_restore_preserves_generator_and_worker_rng(workers):
    reference = make_loader(workers)
    resumed = make_loader(workers, seed=999)
    try:
        iterator = iter(reference)
        for _ in range(5):
            next(iterator)
        state = reference.state_dict()
        expected = [row_signature(next(iterator)) for _ in range(12)]
        training_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(state)
        actual = iter(resumed)
        assert [row_signature(next(actual)) for _ in range(12)] == expected
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
        assert torch.equal(torch.get_rng_state(), training_rng)
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize("workers", [0, 1])
def test_unstarted_native_snapshot_and_next_epoch(workers):
    reference = make_loader(workers)
    resumed = make_loader(workers, seed=999)
    try:
        resumed.load_state_dict(reference.state_dict())
        for epoch in (0, 1):
            reference.set_epoch(epoch)
            resumed.set_epoch(epoch)
            expected = iter(reference)
            actual = iter(resumed)
            assert [row_signature(next(expected)) for _ in range(3)] == [row_signature(next(actual)) for _ in range(3)]
            assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize("workers", [0, 2])
def test_native_shuffle_sampler_restore(workers):
    def loader(seed):
        return DistributedDataloader(
            list(range(128)),
            batch_size=None,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=workers,
            multiprocessing_context="spawn" if workers else None,
        )

    reference = loader(42)
    resumed = loader(999)
    try:
        iterator = iter(reference)
        for _ in range(9):
            next(iterator)
        state = reference.state_dict()
        expected = [next(iterator) for _ in range(20)]
        training_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(state)
        actual = iter(resumed)
        assert [next(actual) for _ in range(20)] == expected
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
        assert torch.equal(torch.get_rng_state(), training_rng)
    finally:
        close_loaders(reference, resumed)


def test_native_legacy_state_without_generator_metadata_loads():
    reference = make_loader()
    resumed = make_loader()
    try:
        iterator = iter(reference)
        for _ in range(5):
            next(iterator)
        state = reference.state_dict()
        state.pop("_veomni_generator")
        resumed.load_state_dict(state)
        assert next(iter(resumed))["id"] == next(iterator)["id"]
    finally:
        close_loaders(reference, resumed)


def test_native_generator_metadata_is_isolated():
    reference = make_loader()
    resumed = make_loader(seed=999)
    try:
        iterator = iter(reference)
        next(iterator)
        state = reference.state_dict()
        expected = copy.deepcopy(state["_veomni_generator"])
        resumed.load_state_dict(state)
        for tensor in state["_veomni_generator"].values():
            tensor.zero_()
        assert torch.equal(reference.state_dict()["_veomni_generator"]["before_iterator"], expected["before_iterator"])
        next(iter(resumed))
        assert torch.equal(resumed.generator.get_state(), expected["at_checkpoint"])
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize("rank", range(8))
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("stateful_source", [False, True])
def test_row_shard_cursor_and_repeat_restore(rank, repeat, stateful_source):
    def dataset():
        source = CursorRows(101) if stateful_source else [{"id": index} for index in range(101)]
        return ShardedIterableDataset(source, dp_rank=rank, dp_size=8, repeat=repeat, seed=42)

    reference = dataset()
    iterator = iter(reference)
    for _ in range(5):
        next(iterator)
    state = copy.deepcopy(reference.state_dict())
    expected = [row["id"] for row in islice(iterator, 40)]
    resumed = dataset()
    resumed.load_state_dict(state)
    assert [row["id"] for row in islice(resumed, 40)] == expected


@pytest.mark.parametrize("rank", [0, 3, 7])
@pytest.mark.parametrize("worker", [0, 1])
def test_row_shard_preserves_plain_worker_assignment(monkeypatch, rank, worker):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_worker_info", lambda: SimpleNamespace(id=worker, num_workers=2))
    source = [{"id": index} for index in range(101)]
    dataset = ShardedIterableDataset(source, dp_rank=rank, dp_size=8)
    expected = [index for index in range(96) if index % 8 == rank and (index // 8) % 2 == worker]
    iterator = iter(dataset)
    prefix = [next(iterator)["id"] for _ in range(2)]
    state = dataset.state_dict()
    resumed = ShardedIterableDataset(source, dp_rank=rank, dp_size=8)
    resumed.load_state_dict(state)
    assert prefix + [row["id"] for row in resumed] == expected


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("stateful_source", [False, True])
def test_wrapped_row_source_resume(workers, stateful_source):
    def loader():
        source = CursorRows(64) if stateful_source else [{"id": index} for index in range(64)]
        rows = ShardedIterableDataset(IterativeDataset(source), dp_rank=0, dp_size=2, repeat=True)
        return make_loader(workers, dataset=rows, snapshot_every_n_steps=3)

    reference = loader()
    resumed = loader()
    try:
        iterator = iter(reference)
        for _ in range(4):
            next(iterator)
        state = reference.state_dict()
        expected = [next(iterator)["id"] for _ in range(50)]
        resumed.load_state_dict(state)
        actual = iter(resumed)
        assert [next(actual)["id"] for _ in range(50)] == expected
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize("rank", [0, 4, 7])
def test_worker_packing_resume_preserves_raw_cursor_across_repeat(rank):
    def loader():
        rows = ShardedIterableDataset(CursorRows(101), dp_rank=rank, dp_size=8, repeat=True, seed=42)
        packed = DynamicBatchingSizeDataset(
            dataset=IterativeDataset(rows, row_transform),
            micro_batch_seq_length=1200,
            ready_for_micro_batch_threshold=16,
            dynamic_batching_collate_fn=row_collate,
            save_by_idx=False,
        )
        return make_loader(1, dataset=packed, snapshot_every_n_steps=10)

    reference = loader()
    resumed = loader()
    try:
        iterator = iter(reference)
        for _ in range(10):
            next(iterator)
        state = reference.state_dict()
        expected = [next(iterator) for _ in range(30)]
        resumed.load_state_dict(state)
        actual = iter(resumed)
        assert [next(actual) for _ in range(30)] == expected
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
    finally:
        close_loaders(reference, resumed)


def hf_row_transform(row):
    return [{"id": row["id"], "input_ids": torch.arange(500)}]


def make_interleave_config(tmp_path):
    sources = []
    for source_index in range(2):
        directory = tmp_path / f"source_{source_index}"
        directory.mkdir()
        for shard in range(2):
            path = directory / f"part_{shard}.jsonl"
            path.write_text(
                "\n".join(json.dumps({"id": source_index * 100 + shard * 50 + index}) for index in range(50)) + "\n"
            )
        sources.append(str(directory))
    path = tmp_path / "sources.json"
    path.write_text(
        json.dumps(
            {
                "sources": sources,
                "names": ["first", "second"],
                "schedule": [{"schedule_type": "const", "weights": [0.5, 0.5]}],
            }
        )
    )
    return path


@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize("shuffle", [False, True])
def test_default_hf_interleave_native_resume(tmp_path, monkeypatch, workers, repeat, shuffle):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=3, dp_size=8))
    path = make_interleave_config(tmp_path)

    def loader():
        options = {} if shuffle else {"shuffle": False}
        dataset = module.build_interleave_dataset(
            str(path), datasets_type="iterable", dataset_repeat=repeat, **options
        )
        assert dataset._data._dataset._shuffling is None
        return DistributedDataloader(
            dataset,
            batch_size=None,
            num_workers=workers,
            generator=torch.Generator().manual_seed(42),
        )

    reference = loader()
    resumed = loader()
    try:
        iterator = iter(reference)
        prefix = [next(iterator) for _ in range(5)]
        state = reference.state_dict()
        expected = list(islice(iterator, 45))
        assert expected
        training_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(state)
        actual = list(islice(resumed, 45))
        assert actual == expected
        assert torch.equal(torch.get_rng_state(), training_rng)
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
        if not repeat:
            assert len({row["id"] for row in prefix + actual}) == len(prefix) + len(actual)
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize("repeat", [False, True])
def test_hf_interleave_ignores_incomplete_saved_source_cursor(tmp_path, monkeypatch, repeat):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=3, dp_size=8))
    path = make_interleave_config(tmp_path)

    def dataset():
        return module.build_interleave_dataset(str(path), datasets_type="iterable", dataset_repeat=repeat)

    reference = dataset()
    iterator = iter(reference)
    for _ in range(5):
        next(iterator)
    state = reference.state_dict()
    state["dataset"]["source_state"] = reference._data._source.state_dict()
    expected = list(islice(iterator, 45))
    assert expected
    resumed = dataset()
    resumed.load_state_dict(state)
    assert list(islice(resumed, 45)) == expected


@pytest.mark.parametrize("composition", ["map", "filter", "concat", "mixed_interleave"])
def test_hf_nested_shuffle_composition_resume(composition):
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets, interleave_datasets

    def dataset():
        shuffled = HFDataset.from_dict({"id": list(range(100))}).to_iterable_dataset().shuffle(seed=42, buffer_size=8)
        plain = HFDataset.from_dict({"id": list(range(100, 200))}).to_iterable_dataset()
        if composition == "map":
            source = shuffled.map(lambda row: {"id": row["id"] + 1})
        elif composition == "filter":
            source = shuffled.filter(lambda row: row["id"] % 2 == 0)
        elif composition == "concat":
            source = concatenate_datasets([plain, shuffled])
        else:
            source = interleave_datasets([plain, shuffled], probabilities=[0.5, 0.5], seed=42)
        return ShardedIterableDataset(source, repeat=True)

    reference = dataset()
    reference.set_epoch(3)
    iterator = iter(reference)
    for _ in range(105 if composition == "concat" else 5):
        next(iterator)
    state = reference.state_dict()
    assert state["source_state"] is None
    expected = list(islice(iterator, 205))
    resumed = dataset()
    resumed.load_state_dict(state)
    assert list(islice(resumed, 205)) == expected


@pytest.mark.parametrize("repeat", [False, True])
@pytest.mark.parametrize(("rank", "world_size"), [(0, 1), (0, 8), (3, 8), (7, 8)])
@pytest.mark.parametrize("saved_source_cursor", [False, True])
def test_default_hf_shuffle_row_resume(tmp_path, monkeypatch, repeat, rank, world_size, saved_source_cursor):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=rank, dp_size=world_size))
    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"id": index}) for index in range(100)) + "\n")

    def dataset():
        return module.build_iterable_dataset(str(path), dataset_repeat=repeat)

    reference = dataset()
    iterator = iter(reference)
    prefix = [next(iterator)["id"] for _ in range(5)]
    state = reference.state_dict()
    if saved_source_cursor:
        state["dataset"]["source_state"] = reference._data._source.state_dict()
    expected = [row["id"] for row in islice(iterator, 205)]
    resumed = dataset()
    resumed.load_state_dict(state)
    actual = [row["id"] for row in islice(resumed, 205)]
    assert actual == expected
    if not repeat:
        assert len(prefix) + len(actual) == 100 // world_size
        assert len(set(prefix + actual)) == len(prefix) + len(actual)


@pytest.mark.parametrize("workers", [0, 1, 2])
@pytest.mark.parametrize("repeat", [False, True])
def test_default_hf_shuffle_native_worker_resume(tmp_path, monkeypatch, workers, repeat):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))
    for shard in range(2):
        path = tmp_path / f"part_{shard}.jsonl"
        path.write_text("\n".join(json.dumps({"id": shard * 100 + index}) for index in range(100)) + "\n")

    def loader():
        return make_loader(workers, dataset=module.build_iterable_dataset(str(tmp_path), dataset_repeat=repeat))

    reference = loader()
    resumed = loader()
    try:
        iterator = iter(reference)
        prefix = [next(iterator)["id"] for _ in range(5)]
        state = reference.state_dict()
        expected = [row["id"] for row in islice(iterator, 405)]
        training_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(state)
        actual = [row["id"] for row in islice(resumed, 405)]
        assert actual == expected
        assert torch.equal(torch.get_rng_state(), training_rng)
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
        if not repeat:
            assert sorted(prefix + actual) == list(range(200))
    finally:
        close_loaders(reference, resumed)


@pytest.mark.parametrize(
    ("count", "buffer_size", "split_at"), [(100, 10000, 5), (100, 8, 5), (100, 8, 95), (10101, 10000, 5)]
)
@pytest.mark.parametrize("wrapped", [False, True])
def test_hf_shuffle_buffer_replay_across_repeat(tmp_path, count, buffer_size, split_at, wrapped):
    from datasets import load_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"id": index}) for index in range(count)) + "\n")

    def dataset():
        source = load_dataset("json", data_files=str(path), split="train", streaming=True).shuffle(
            seed=42, buffer_size=buffer_size
        )
        if wrapped:
            source = IterativeDataset(source)
        return ShardedIterableDataset(source, repeat=True)

    reference = dataset()
    reference.set_epoch(3)
    iterator = iter(reference)
    for _ in range(split_at):
        next(iterator)
    state = reference.state_dict()
    assert state["source_state"] is None
    expected = [row["id"] for row in islice(iterator, count + 10)]
    resumed = dataset()
    resumed.load_state_dict(state)
    assert [row["id"] for row in islice(resumed, count + 10)] == expected


def test_unshuffled_hf_keeps_fast_source_cursor(tmp_path, monkeypatch):
    from datasets import IterableDataset as HFIterableDataset
    from datasets import load_dataset

    path = tmp_path / "rows.jsonl"
    path.write_text("\n".join(json.dumps({"id": index}) for index in range(100)) + "\n")

    def dataset():
        return ShardedIterableDataset(load_dataset("json", data_files=str(path), split="train", streaming=True))

    reference = dataset()
    iterator = iter(reference)
    for _ in range(5):
        next(iterator)
    state = reference.state_dict()
    assert state["source_state"] is not None
    expected = list(iterator)
    restores = []
    original_load = HFIterableDataset.load_state_dict

    def restore(source, cursor):
        restores.append(copy.deepcopy(cursor))
        return original_load(source, cursor)

    monkeypatch.setattr(HFIterableDataset, "load_state_dict", restore)
    resumed = dataset()
    resumed.load_state_dict(state)
    assert list(resumed) == expected
    assert len(expected) == 95
    assert restores == [state["source_state"]]


@pytest.mark.parametrize("extension", ["json", "parquet"])
@pytest.mark.parametrize("workers", [1, 2])
def test_hf_multifile_worker_packing_resume(tmp_path, extension, workers):
    from datasets import load_dataset

    paths = []
    for shard in range(3):
        path = tmp_path / f"part_{shard}.{extension}"
        records = [{"id": shard * 64 + index} for index in range(64)]
        if extension == "json":
            path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
        else:
            import pyarrow as arrow
            import pyarrow.parquet as parquet

            parquet.write_table(arrow.Table.from_pylist(records), path)
        paths.append(str(path))

    def loader():
        source = load_dataset(extension, data_files=paths, split="train", streaming=True)
        rows = ShardedIterableDataset(source, dp_rank=0, dp_size=8, repeat=True, seed=42)
        packed = DynamicBatchingSizeDataset(
            dataset=IterativeDataset(rows, hf_row_transform),
            micro_batch_seq_length=1200,
            ready_for_micro_batch_threshold=16,
            dynamic_batching_collate_fn=row_collate,
            save_by_idx=False,
        )
        return make_loader(workers, dataset=packed, snapshot_every_n_steps=10)

    reference = loader()
    resumed = loader()
    try:
        iterator = iter(reference)
        for _ in range(10):
            next(iterator)
        state = reference.state_dict()
        expected = [next(iterator) for _ in range(30)]
        training_rng = torch.get_rng_state().clone()
        resumed.load_state_dict(state)
        actual = iter(resumed)
        assert [next(actual) for _ in range(30)] == expected
        assert torch.equal(torch.get_rng_state(), training_rng)
        assert torch.equal(reference.generator.get_state(), resumed.generator.get_state())
    finally:
        close_loaders(reference, resumed)

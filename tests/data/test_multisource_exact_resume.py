"""Exact multi-source replay for real HF streaming and worker boundaries."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import IterableDataset as HFIterableDataset
from datasets.iterable_dataset import ExamplesIterable

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import (
    InterleavedIterableDataset,
    IterativeDataset,
    ShardedIterableDataset,
    WeightedMultiSourceDataset,
    _ReplayableHFStream,
    build_iterable_dataset,
)
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
INPUT_CONTRACT = "sha256:" + "a" * 64


def generate_rows(shards):
    for shard in shards:
        for index in range(80):
            yield f"{shard}-{index}", {"id": 1000 * shard + index}


def sample_length(_sample):
    return 1


def weighted_sample_length(sample):
    return float(sample["id"] % 7 + 1)


def make_loader(*, buffer_size, sharded, workers, persistent=False, context=None, packing_buffer=1):
    hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(
        seed=5, buffer_size=buffer_size
    )
    if sharded:
        hf = ShardedIterableDataset(hf, dp_rank=0, dp_size=1, seed=13)
    source = IterativeDataset(
        hf, exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
    )
    dataset = WeightedMultiSourceDataset(
        [source], [1.0], seed=5, upstream_sharded=sharded, sample_token_len_fn=sample_length
    )
    raw = DistributedDataloader(
        dataset, batch_size=None, num_workers=workers, persistent_workers=persistent, multiprocessing_context=context
    )
    return DynamicBatchSizeDataLoader(
        raw,
        TextBatchingStrategy(token_micro_bsz=1, buffer_size=packing_buffer, get_length_fn=sample_length),
        length=40,
    ), raw


def rows(iterator, n):
    return [next(iterator)[0][0]["id"] for _ in range(n)]


def close(raw):
    iterator = raw._iterator
    if iterator is not None and hasattr(iterator, "_shutdown_workers"):
        iterator._shutdown_workers()


@pytest.fixture(autouse=True)
def parallel_state(monkeypatch):
    import veomni.data.dataset as module

    monkeypatch.setattr(module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))


@pytest.mark.parametrize("buffer_size", [1, 7])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_hf_shuffle_replays_exactly_after_snapshot(buffer_size, sharded, workers):
    original, orig_raw = make_loader(buffer_size=buffer_size, sharded=sharded, workers=workers)
    resumed, resume_raw = make_loader(buffer_size=buffer_size, sharded=sharded, workers=workers)
    try:
        original.set_epoch(2)
        iterator = iter(original)
        rows(iterator, 6)
        state = copy.deepcopy(original.state_dict())
        expected = rows(iterator, 8)
        resumed.prepare_resume_epoch(2)
        resumed.load_state_dict(state)
        resumed.set_epoch(2)
        assert rows(iter(resumed), 8) == expected
    finally:
        close(orig_raw)
        close(resume_raw)


@pytest.mark.parametrize("level", ["sample", "token"])
@pytest.mark.parametrize("persistent", [False, True])
def test_two_source_sampling_and_independent_cursors_resume(level, persistent):
    def build():
        sources = []
        for index in range(2):
            hf = HFIterableDataset(
                ExamplesIterable(generate_rows, {"shards": list(range(index * 4, index * 4 + 4))})
            ).shuffle(seed=5 + index, buffer_size=7)
            sharded = ShardedIterableDataset(hf, dp_rank=0, dp_size=1, seed=13)
            sources.append(
                IterativeDataset(
                    sharded,
                    exact_stream_replay=True,
                    resume_input_contract=INPUT_CONTRACT,
                    resume_input_deterministic=True,
                )
            )
        dataset = WeightedMultiSourceDataset(
            sources, [0.3, 0.7], seed=5, upstream_sharded=True, level=level, sample_token_len_fn=weighted_sample_length
        )
        raw = DistributedDataloader(dataset, batch_size=None, num_workers=2, persistent_workers=persistent)
        return DynamicBatchSizeDataLoader(
            raw, TextBatchingStrategy(token_micro_bsz=1, buffer_size=5, get_length_fn=sample_length), length=100
        ), raw

    def samples(iterator, count):
        result = []
        for _ in range(count):
            sample = next(iterator)[0][0]
            result.append((sample["id"], sample["ds_idx"]))
        return result

    original, orig_raw = build()
    resumed, resume_raw = build()
    try:
        original.set_epoch(2)
        active = iter(original)
        samples(active, 12)
        state = copy.deepcopy(original.state_dict())
        expected = samples(active, 25)
        resumed.prepare_resume_epoch(2)
        resumed.load_state_dict(state)
        resumed.set_epoch(2)
        assert samples(iter(resumed), 25) == expected
    finally:
        close(orig_raw)
        close(resume_raw)


@pytest.mark.parametrize("sharded", [False, True])
def test_persistent_workers_restart_exactly_at_next_epoch(sharded):
    reused, reuse_raw = make_loader(buffer_size=7, sharded=sharded, workers=2, persistent=True)
    fresh, fresh_raw = make_loader(buffer_size=7, sharded=sharded, workers=2, persistent=False)
    try:
        reused.set_epoch(2)
        rows(iter(reused), 8)  # truncate the previous epoch with outstanding prefetch
        reused.set_epoch(3)
        fresh.set_epoch(3)
        assert rows(iter(reused), 12) == rows(iter(fresh), 12)
    finally:
        close(reuse_raw)
        close(fresh_raw)


def test_changed_source_weights_rejected_before_mutation():
    loader, raw = make_loader(buffer_size=7, sharded=False, workers=0)
    other, other_raw = make_loader(buffer_size=7, sharded=False, workers=0)
    try:
        dataset, target = raw.dataset, other_raw.dataset
        state = copy.deepcopy(dataset.state_dict())
        state["topology"]["weights"] = [0.25]
        before = copy.deepcopy(target.state_dict())
        with pytest.raises(ValueError, match="weights"):
            target.load_state_dict(state, reconcile_policy="strict")
        assert target.state_dict()["topology"] == before["topology"]
    finally:
        close(raw)
        close(other_raw)


@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("workers", [0, 2])
def test_fresh_hf_sample_order_is_unchanged(sharded, workers):
    def source():
        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(
            seed=5, buffer_size=7
        )
        return ShardedIterableDataset(hf, dp_rank=0, dp_size=1, seed=13) if sharded else hf

    baseline = DistributedDataloader(source(), batch_size=None, num_workers=workers)
    replay = DistributedDataloader(
        IterativeDataset(
            source(), exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
        ),
        batch_size=None,
        num_workers=workers,
    )
    try:
        baseline.set_epoch(2)
        replay.set_epoch(2)
        left, right = iter(baseline), iter(replay)
        assert [next(left)["id"] for _ in range(15)] == [next(right)["id"] for _ in range(15)]
    finally:
        close(baseline)
        close(replay)


def resume_in_fresh_process(_rank, checkpoint_path, result_path):
    import veomni.data.dataset as module

    module.get_parallel_state = lambda: SimpleNamespace(dp_rank=0, dp_size=1)
    loader, raw = make_loader(buffer_size=7, sharded=True, workers=2, context="spawn")
    try:
        saved = torch.load(checkpoint_path, weights_only=False)
        loader.prepare_resume_epoch(2)
        loader.load_state_dict(saved)
        loader.set_epoch(2)
        torch.save(rows(iter(loader), 8), result_path)
    finally:
        close(raw)


def test_hf_shuffle_cold_resume_in_a_new_process(tmp_path):
    import torch.multiprocessing as mp

    original, raw = make_loader(buffer_size=7, sharded=True, workers=2, context="spawn")
    try:
        original.set_epoch(2)
        iterator = iter(original)
        rows(iterator, 6)
        checkpoint = tmp_path / "loader.pt"
        result = tmp_path / "result.pt"
        torch.save(copy.deepcopy(original.state_dict()), checkpoint)
        expected = rows(iterator, 8)
        mp.spawn(resume_in_fresh_process, args=(str(checkpoint), str(result)), nprocs=1, join=True)
        assert torch.load(result, weights_only=True) == expected
    finally:
        close(raw)


@pytest.mark.parametrize("field", ["seed", "dp_size", "upstream_sharded", "worker_geometry"])
def test_changed_sampling_provenance_rejected(field):
    loader, raw = make_loader(buffer_size=7, sharded=False, workers=0)
    try:
        state = copy.deepcopy(raw.dataset.state_dict())
        state["topology"][field] = {"seed": 99, "dp_size": 4, "upstream_sharded": True, "worker_geometry": (0, 2)}[
            field
        ]
        with pytest.raises(ValueError, match=field):
            raw.dataset.load_state_dict(state)
    finally:
        close(raw)


def test_uninitialized_legacy_sampling_is_rejected():
    loader, raw = make_loader(buffer_size=7, sharded=False, workers=0)
    try:
        state = copy.deepcopy(raw.dataset.state_dict())
        del state["runtime"]["sampling_initialized"]
        with pytest.raises(ValueError, match="cannot prove sampling was initialized"):
            raw.dataset.load_state_dict(state)
    finally:
        close(raw)


def test_old_checkpoint_without_sampling_topology_is_rejected_by_default():
    loader, raw = make_loader(buffer_size=7, sharded=False, workers=0)
    try:
        state = copy.deepcopy(raw.dataset.state_dict())
        state["version"] = 0
        with pytest.raises(ValueError, match="lacks exact topology provenance"):
            raw.dataset.load_state_dict(state)
    finally:
        close(raw)


def test_legacy_hf_without_replay_state_is_rejected():
    loader, raw = make_loader(buffer_size=7, sharded=True, workers=0)
    try:
        state = copy.deepcopy(raw.dataset.state_dict())
        state["runtime"]["dataset_states"][raw.dataset._source_ids[0]] = {"dataset": None}
        with pytest.raises(ValueError, match="lacks exact replay state"):
            raw.dataset.load_state_dict(state)
    finally:
        close(raw)


def test_changed_hf_shuffle_buffer_is_rejected():
    loader, raw = make_loader(buffer_size=7, sharded=False, workers=0)
    other, other_raw = make_loader(buffer_size=3, sharded=False, workers=0)
    try:
        with pytest.raises(ValueError, match="fingerprint"):
            other_raw.dataset.load_state_dict(copy.deepcopy(raw.dataset.state_dict()))
    finally:
        close(raw)
        close(other_raw)


def test_streaming_parquet_builder_restores_exactly(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for shard in range(8):
        pq.write_table(pa.table({"id": [1000 * shard + i for i in range(50)]}), tmp_path / f"part-{shard}.parquet")

    def loader():
        source = build_iterable_dataset(
            str(tmp_path),
            seed=5,
            split_by_node=True,
            shuffle=True,
            exact_stream_replay=True,
            resume_input_contract=INPUT_CONTRACT,
            resume_input_deterministic=True,
        )
        dataset = WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)
        raw = DistributedDataloader(dataset, batch_size=None, num_workers=2)
        return raw

    original, resumed = loader(), loader()
    try:
        original.set_epoch(2)
        iterator = iter(original)
        [next(iterator) for _ in range(6)]
        state = copy.deepcopy(original.state_dict())
        expected = [next(iterator)["id"] for _ in range(8)]
        resumed.prepare_resume_epoch(2)
        resumed.load_state_dict(state)
        actual = [next(iterable)["id"] for iterable in [iter(resumed)] for _ in range(8)]
        assert actual == expected
    finally:
        close(original)
        close(resumed)


class GatedReplayStream(_ReplayableHFStream):
    def __init__(self, source, started, release, epoch=2, worker_id=0):
        super().__init__(source, INPUT_CONTRACT, True)
        self.started = started
        self.release = release
        self.gated_epoch = epoch
        self.gated_worker = worker_id

    def _new_iterator(self, epoch):
        worker = torch.utils.data.get_worker_info()
        if (
            epoch == self.gated_epoch
            and worker is not None
            and worker.id == self.gated_worker
            and not self.started.is_set()
        ):
            self.started.set()
            if not self.release.wait(30):
                raise RuntimeError("test did not release the old epoch worker")
        return super()._new_iterator(epoch)


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_persistent_worker_delayed_old_epoch_does_not_skip_new_epoch(context):
    import multiprocessing as mp

    ctx = mp.get_context(context)
    started, release = ctx.Event(), ctx.Event()
    reused, reuse_raw = make_loader(buffer_size=7, sharded=True, workers=2, persistent=True, context=context)
    fresh, fresh_raw = make_loader(buffer_size=7, sharded=True, workers=2, context=context)
    source = reuse_raw.dataset._datasets[0]
    source._data = GatedReplayStream(source._data._dataset, started, release)
    try:
        reused.set_epoch(2)
        iter(reused)  # Starts prefetch; worker zero pauses before reading epoch 2.
        assert started.wait(30)
        reused.set_epoch(3)
        release.set()
        fresh.set_epoch(3)
        assert rows(iter(reused), 12) == rows(iter(fresh), 12)
    finally:
        release.set()
        close(reuse_raw)
        close(fresh_raw)


def test_persistent_checkpoint_before_one_worker_starts_new_epoch():
    import multiprocessing as mp

    ctx = mp.get_context("fork")
    started, release = ctx.Event(), ctx.Event()
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2, persistent=True)
    resumed, resume_raw = make_loader(buffer_size=7, sharded=True, workers=2, persistent=True)
    source = orig_raw.dataset._datasets[0]
    source._data = GatedReplayStream(source._data._dataset, started, release, epoch=3, worker_id=1)
    try:
        original.set_epoch(2)
        rows(iter(original), 4)
        original.set_epoch(3)
        iterator = iter(original)
        assert started.wait(30)
        rows(iterator, 1)
        state = copy.deepcopy(original.state_dict())
        workers = state["dataloader_state"]["_snapshot"]["_worker_snapshots"]
        assert [workers[f"worker_{i}"]["dataset_state"]["runtime"]["epoch"] for i in range(2)] == [3, 3]
        assert workers["worker_1"]["dataset_state"]["runtime"]["dataset_states"]["source_0"]["dataset"]["yielded"] == 0
        release.set()
        expected = rows(iterator, 8)
        resumed.prepare_resume_epoch(3)
        resumed.load_state_dict(state)
        resumed.set_epoch(3)
        assert rows(iter(resumed), 8) == expected
    finally:
        release.set()
        close(orig_raw)
        close(resume_raw)


def test_loader_checks_hf_source_before_installing_worker_state():
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    changed, changed_raw = make_loader(buffer_size=3, sharded=True, workers=2)
    try:
        original.set_epoch(2)
        rows(iter(original), 4)
        state = copy.deepcopy(orig_raw.state_dict())
        with pytest.raises(ValueError, match="fingerprint"):
            changed_raw.load_state_dict(state)
        assert changed_raw.next_iter_state is None
        assert changed_raw.dataset._pending_epoch is None
    finally:
        close(orig_raw)
        close(changed_raw)


def test_replay_timeout_does_not_change_training_iterator_timeout():
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    resumed, resume_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    try:
        original.set_epoch(2)
        rows(iter(original), 4)
        state = copy.deepcopy(orig_raw.state_dict())
        resume_raw.load_state_dict(state)
        iterator = iter(resume_raw)
        assert resume_raw.timeout == 0
        assert iterator._timeout == 0
    finally:
        close(orig_raw)
        close(resume_raw)


def test_out_of_order_workers_are_rejected_before_resume():
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    target, target_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    try:
        original.set_epoch(2)
        rows(iter(original), 4)
        state = copy.deepcopy(orig_raw.state_dict())
        target_raw.in_order = False
        with pytest.raises(ValueError, match="in-order data workers"):
            target_raw.load_state_dict(state)
        assert target_raw.next_iter_state is None
    finally:
        close(orig_raw)
        close(target_raw)


@pytest.mark.parametrize("changed_field", ["batch_size", "drop_last", "missing_geometry"])
def test_loader_batch_geometry_is_proven_before_resume(changed_field):
    def build(batch_size=1, drop_last=False):
        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(
            seed=5, buffer_size=7
        )
        source = IterativeDataset(
            hf, exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
        )
        dataset = WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)
        return DistributedDataloader(dataset, batch_size=batch_size, drop_last=drop_last, num_workers=0)

    original = build()
    target = build(batch_size=2 if changed_field == "batch_size" else 1, drop_last=changed_field == "drop_last")
    active = iter(original)
    for _ in range(4):
        next(active)
    state = copy.deepcopy(original.state_dict())
    if changed_field == "missing_geometry":
        state.pop("_veomni_weighted_geometry")
    with pytest.raises(ValueError, match="batch or worker geometry"):
        target.load_state_dict(state)
    assert target.next_iter_state is None


def test_loader_records_effective_default_worker_start_method():
    import multiprocessing

    method = multiprocessing.get_context().get_start_method()
    alternatives = [value for value in multiprocessing.get_all_start_methods() if value != method]
    if not alternatives:
        pytest.skip("only one multiprocessing start method is available")
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    same, same_raw = make_loader(buffer_size=7, sharded=True, workers=2, context=method)
    changed, changed_raw = make_loader(buffer_size=7, sharded=True, workers=2, context=alternatives[0])
    try:
        original.set_epoch(2)
        active = iter(original)
        rows(active, 4)
        state = copy.deepcopy(original.state_dict())
        expected = rows(active, 4)
        same.prepare_resume_epoch(2)
        same.load_state_dict(copy.deepcopy(state))
        same.set_epoch(2)
        assert rows(iter(same), 4) == expected
        with pytest.raises(ValueError, match="batch or worker geometry"):
            changed.load_state_dict(copy.deepcopy(state))
    finally:
        close(orig_raw)
        close(same_raw)
        close(changed_raw)


@pytest.mark.parametrize("field", ["random_state", "epoch", "avg_len_count", "sampling_initialized"])
def test_loader_rejects_corrupt_worker_state_before_install(field):
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    target, target_raw = make_loader(buffer_size=7, sharded=True, workers=2)
    try:
        original.set_epoch(2)
        rows(iter(original), 4)
        state = copy.deepcopy(orig_raw.state_dict())
        runtime = state["_snapshot"]["_worker_snapshots"]["worker_0"]["dataset_state"]["runtime"]
        runtime[field] = {
            "random_state": ("broken",),
            "epoch": 99,
            "avg_len_count": {},
            "sampling_initialized": False,
        }[field]
        with pytest.raises(ValueError):
            target_raw.load_state_dict(state)
        assert target_raw.next_iter_state is None
        assert target_raw.dataset._pending_epoch is None
    finally:
        close(orig_raw)
        close(target_raw)


def test_changed_transform_code_is_rejected_before_restore():
    def make_dataset(label):
        def transform(row):
            return {**row, "label": label}

        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=5)
        source = IterativeDataset(
            hf,
            transform=transform,
            exact_stream_replay=True,
            resume_input_contract=INPUT_CONTRACT,
            resume_input_deterministic=True,
        )
        return WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)

    saved, changed = make_dataset(1), make_dataset(2)
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(copy.deepcopy(saved.state_dict()))


class UnrestorableSource(torch.utils.data.IterableDataset):
    def __init__(self):
        self.cursor = 0

    def __iter__(self):
        while True:
            self.cursor += 1
            yield {"id": self.cursor}


def test_unrestorable_custom_source_is_rejected():
    source = WeightedMultiSourceDataset([UnrestorableSource()], [1.0], seed=5, upstream_sharded=True)
    state = copy.deepcopy(source.state_dict())
    with pytest.raises(ValueError, match="cannot restore source"):
        source.load_state_dict(state)


def test_custom_iterable_without_source_identity_is_rejected():
    class StatefulSource:
        def __iter__(self):
            yield {"id": 1}

        def state_dict(self):
            return {"offset": 1}

        def load_state_dict(self, state):
            pass

    dataset = WeightedMultiSourceDataset([IterativeDataset(StatefulSource())], [1.0], seed=5)
    with pytest.raises(ValueError, match="source fingerprint"):
        dataset.load_state_dict(copy.deepcopy(dataset.state_dict()))


def test_changed_token_length_function_is_rejected():
    def make_dataset(length):
        def measure(_sample):
            return length

        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=5)
        source = IterativeDataset(
            hf, exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
        )
        return WeightedMultiSourceDataset([source], [1.0], seed=5, level="token", sample_token_len_fn=measure)

    before, changed = make_dataset(1), make_dataset(2)
    with pytest.raises(ValueError, match="fingerprint"):
        changed.load_state_dict(copy.deepcopy(before.state_dict()))


def test_numpy_token_lengths_survive_exact_resume():
    def measure(_sample):
        return np.float64(1.0)

    def dataset():
        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=5)
        source = IterativeDataset(
            hf, exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
        )
        return WeightedMultiSourceDataset([source], [1.0], seed=5, level="token", sample_token_len_fn=measure)

    original, resumed = dataset(), dataset()
    active = iter(original)
    for _ in range(4):
        next(active)
    state = copy.deepcopy(original.state_dict())
    expected = [next(active) for _ in range(8)]
    resumed.load_state_dict(state)
    resumed_iterator = iter(resumed)
    assert [next(resumed_iterator) for _ in range(8)] == expected


def test_persistent_custom_source_without_restart_contract_is_rejected():
    source = WeightedMultiSourceDataset([UnrestorableSource()], [1.0], seed=5, upstream_sharded=True)
    with pytest.raises(ValueError, match="persistent workers require exact epoch restart"):
        DistributedDataloader(source, batch_size=None, num_workers=2, persistent_workers=True)


def test_unattested_fresh_hf_persistent_epochs_work_but_exact_resume_is_rejected(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for shard in range(8):
        pq.write_table(pa.table({"id": [1000 * shard + i for i in range(50)]}), tmp_path / f"part-{shard}.parquet")

    def build(persistent):
        source = build_iterable_dataset(str(tmp_path), seed=5, split_by_node=False, shuffle=True)
        dataset = WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)
        return DistributedDataloader(dataset, batch_size=None, num_workers=2, persistent_workers=persistent)

    reused, fresh = build(True), build(False)
    try:
        reused.set_epoch(2)
        active = iter(reused)
        [next(active) for _ in range(8)]
        checkpoint = copy.deepcopy(reused.state_dict())
        reused.set_epoch(3)
        fresh.set_epoch(3)
        left, right = iter(reused), iter(fresh)
        assert [next(left)["id"] for _ in range(12)] == [next(right)["id"] for _ in range(12)]
        with pytest.raises(ValueError):
            reused.load_state_dict(checkpoint)
    finally:
        close(reused)
        close(fresh)


def test_unattested_custom_hf_generator_requires_explicit_epoch_contract():
    hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))}))
    dataset = WeightedMultiSourceDataset([IterativeDataset(hf)], [1.0], seed=5, upstream_sharded=True)
    with pytest.raises(ValueError, match="persistent workers require exact epoch restart"):
        DistributedDataloader(dataset, batch_size=None, num_workers=2, persistent_workers=True)


def test_reordered_source_ids_are_rejected():
    def source(seed):
        hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=seed)
        return IterativeDataset(
            hf, exact_stream_replay=True, resume_input_contract=INPUT_CONTRACT, resume_input_deterministic=True
        )

    original = WeightedMultiSourceDataset([source(1), source(2)], [0.5, 0.5], source_ids=["a", "b"], seed=5)
    reordered = WeightedMultiSourceDataset([source(2), source(1)], [0.5, 0.5], source_ids=["b", "a"], seed=5)
    state = copy.deepcopy(original.state_dict())
    with pytest.raises(ValueError, match="source_ids"):
        reordered.load_state_dict(state)


def test_hf_subclass_without_exact_replay_is_rejected():
    hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=1, buffer_size=4)
    source = InterleavedIterableDataset(hf)
    dataset = WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)
    state = copy.deepcopy(dataset.state_dict())
    with pytest.raises(ValueError, match="requires exact replay support"):
        dataset.load_state_dict(state)


def test_hf_replay_requires_deterministic_input_attestation():
    hf = HFIterableDataset(ExamplesIterable(generate_rows, {"shards": list(range(8))})).shuffle(seed=1, buffer_size=4)
    source = IterativeDataset(hf, exact_stream_replay=True)
    dataset = WeightedMultiSourceDataset([source], [1.0], seed=5, upstream_sharded=True)
    state = copy.deepcopy(dataset.state_dict())
    with pytest.raises(ValueError, match="attested deterministic input contract"):
        dataset.load_state_dict(state)


def test_cross_epoch_buffer_restores_without_replaying_old_epoch():
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=2, packing_buffer=5)
    resumed, resume_raw = make_loader(buffer_size=7, sharded=True, workers=2, packing_buffer=5)
    try:
        original.set_epoch(2)
        iterator = iter(original)
        rows(iterator, 40)
        state = copy.deepcopy(original.state_dict())
        assert len(state["batching_strategy"].buffer) > 0
        original.set_epoch(3)
        expected = rows(iter(original), 8)
        resume_raw.validate_weighted_resume_state(state["dataloader_state"], expected_epoch=2)
        resumed.load_sft_next_epoch_buffer(state)
        resumed.set_epoch(3)
        assert rows(iter(resumed), 8) == expected
    finally:
        close(orig_raw)
        close(resume_raw)


@pytest.mark.parametrize("steps", [6, 40])
@pytest.mark.parametrize("workers,persistent", [(0, False), (2, False), (2, True)])
def test_sft_callback_restores_hf_stream_and_rng(monkeypatch, steps, workers, persistent):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    torch.manual_seed(12345)
    original, orig_raw = make_loader(
        buffer_size=7, sharded=True, workers=workers, persistent=persistent, packing_buffer=5
    )
    resumed, resume_raw = make_loader(
        buffer_size=7, sharded=True, workers=workers, persistent=persistent, packing_buffer=5
    )
    try:
        original._strict_resume_contract = True
        original.set_epoch(2)
        active = iter(original)
        rows(active, steps)
        cursor, rng = copy.deepcopy(original.state_dict()), torch.get_rng_state().clone()
        if steps == 40:
            original.set_epoch(3)
            active = iter(original)
        expected = [(rows(active, 1)[0], torch.rand(()).item()) for _ in range(8)]
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")),
                data=SimpleNamespace(
                    train_path="test://immutable",
                    resume_input_contract=INPUT_CONTRACT,
                    resume_input_deterministic=True,
                    dataloader=SimpleNamespace(use_background_prefetcher=False),
                ),
                train_steps=40,
            ),
            state=SimpleNamespace(global_step=0, epoch=0),
            train_dataloader=resumed,
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
            _sft_dynamic_exact_resume=True,
        )
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 80 + steps,
            "train_dataloader": cursor,
            "torch_rng_state": rng,
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 2),
        }
        monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
        callback.load_global_state()
        resumed.set_epoch(trainer.start_epoch)
        actual = [(rows(iter(resumed) if i == 0 else resumed, 1)[0], torch.rand(()).item()) for i in range(8)]
        assert actual == expected
    finally:
        close(orig_raw)
        close(resume_raw)


def _toy_model_and_optimizer():
    model = torch.nn.Linear(1, 1)
    return model, torch.optim.AdamW(model.parameters(), lr=0.01)


def _toy_step(model, optimizer, iterator):
    sample = rows(iterator, 1)[0]
    features = torch.tensor([[sample / 1000.0]]) + torch.rand(())
    optimizer.zero_grad(set_to_none=True)
    loss = model(features).square().mean()
    loss.backward()
    gradient = [parameter.grad.clone() for parameter in model.parameters()]
    optimizer.step()
    states = [optimizer.state[parameter] for parameter in model.parameters()]
    return {
        "sample": sample,
        "loss": loss.detach().clone(),
        "gradient": gradient,
        "parameters": [parameter.detach().clone() for parameter in model.parameters()],
        "exp_avg": [state["exp_avg"].clone() for state in states],
        "exp_avg_sq": [state["exp_avg_sq"].clone() for state in states],
        "rng": torch.get_rng_state().clone(),
    }


def _toy_sft_callback(loader, cursor, rng, step, epoch):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")),
            data=SimpleNamespace(
                train_path="test://immutable",
                resume_input_contract=INPUT_CONTRACT,
                resume_input_deterministic=True,
                dataloader=SimpleNamespace(use_background_prefetcher=False),
            ),
            train_steps=40,
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
        train_dataloader=loader,
        channel_loss_callback=None,
        offline_cache_writer=None,
        environ_meter=None,
        _sft_dynamic_exact_resume=True,
    )
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": step,
        "train_dataloader": cursor,
        "torch_rng_state": rng,
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, epoch),
    }
    callback.load_global_state()
    loader.set_epoch(trainer.start_epoch)
    return callback


def _persistent_boundary_process(_rank, checkpoint_path, result_path, context):
    import veomni.data.dataset as module

    module.get_parallel_state = lambda: SimpleNamespace(dp_rank=0, dp_size=1)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    loader, raw = make_loader(
        buffer_size=7, sharded=True, workers=2, persistent=True, context=context, packing_buffer=5
    )
    second, second_raw = make_loader(
        buffer_size=7, sharded=True, workers=2, persistent=True, context=context, packing_buffer=5
    )
    try:
        model, optimizer = _toy_model_and_optimizer()
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        _toy_sft_callback(loader, checkpoint["cursor"], checkpoint["rng"], 120, 2)
        first = _toy_step(model, optimizer, iter(loader))
        next_cursor, next_rng = copy.deepcopy(loader.state_dict()), torch.get_rng_state().clone()
        model_state, optimizer_state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        model2, optimizer2 = _toy_model_and_optimizer()
        model2.load_state_dict(model_state)
        optimizer2.load_state_dict(optimizer_state)
        _toy_sft_callback(second, next_cursor, next_rng, 121, 3)
        next_step = _toy_step(model2, optimizer2, iter(second))
        torch.save([first, next_step], result_path)
    finally:
        close(raw)
        close(second_raw)


@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_persistent_epoch_boundary_fresh_process_and_next_update(tmp_path, context):
    import torch.multiprocessing as mp

    torch.manual_seed(12345)
    loader, raw = make_loader(
        buffer_size=7, sharded=True, workers=2, persistent=True, context=context, packing_buffer=5
    )
    try:
        model, optimizer = _toy_model_and_optimizer()
        loader._strict_resume_contract = True
        loader.set_epoch(2)
        active = iter(loader)
        rows(active, 40)
        # Establish a non-empty Adam state before the saved epoch boundary.
        optimizer.zero_grad(set_to_none=True)
        model(torch.tensor([[1.0]])).square().mean().backward()
        optimizer.step()
        cursor = copy.deepcopy(loader.state_dict())
        assert len(cursor["batching_strategy"].buffer) > 0
        checkpoint = {
            "cursor": cursor,
            "rng": torch.get_rng_state().clone(),
            "model": copy.deepcopy(model.state_dict()),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }
        path, result = tmp_path / "checkpoint.pt", tmp_path / "result.pt"
        torch.save(checkpoint, path)
        loader.set_epoch(3)
        active = iter(loader)
        expected = [_toy_step(model, optimizer, active), _toy_step(model, optimizer, active)]
        mp.spawn(_persistent_boundary_process, args=(str(path), str(result), context), nprocs=1, join=True)
        actual = torch.load(result, weights_only=True)
        for left, right in zip(actual, expected):
            assert left["sample"] == right["sample"]
            assert torch.equal(left["loss"], right["loss"])
            assert torch.equal(left["rng"], right["rng"])
            for name in ("gradient", "parameters", "exp_avg", "exp_avg_sq"):
                assert all(torch.equal(a, b) for a, b in zip(left[name], right[name]))
    finally:
        close(raw)


class SlowReplayStream(_ReplayableHFStream):
    def __init__(self, source, delay):
        super().__init__(source, INPUT_CONTRACT, True)
        self.delay = delay

    def load_state_dict(self, state):
        import time

        time.sleep(self.delay)
        return super().load_state_dict(state)


def distributed_resume_worker(rank, failure, endpoint):
    from datetime import timedelta

    import torch.distributed as dist

    import veomni.data.dataset as module
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    module.get_parallel_state = lambda: SimpleNamespace(dp_rank=0, dp_size=1)
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=3))
    original, orig_raw = make_loader(buffer_size=7, sharded=True, workers=1, packing_buffer=5)
    resumed, resume_raw = make_loader(buffer_size=7, sharded=True, workers=1, packing_buffer=5)
    source = resume_raw.dataset._datasets[0]
    source._data = SlowReplayStream(source._data._dataset, delay=4 if rank == 1 and failure == "slow" else 0)
    try:
        original._strict_resume_contract = True
        original.set_epoch(2)
        active = iter(original)
        rows(active, 6)
        cursor, rng = copy.deepcopy(original.state_dict()), torch.get_rng_state().clone()
        expected = rows(active, 4)
        if rank == 0 and failure == "corrupt":
            cursor["dataloader_state"]["_snapshot"]["_worker_snapshots"]["worker_0"]["dataset_state"]["runtime"][
                "random_state"
            ] = ("invalid",)
        if rank == 0 and failure == "replay_eof":
            runtime = cursor["dataloader_state"]["_snapshot"]["_worker_snapshots"]["worker_0"]["dataset_state"][
                "runtime"
            ]
            next(iter(runtime["dataset_states"].values()))["dataset"]["yielded"] = 1_000_000
        resume_raw.resume_replay_timeout_seconds = 15
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=rank, checkpoint=SimpleNamespace(load_path="/unused")),
                data=SimpleNamespace(
                    train_path="test://immutable",
                    resume_input_contract=INPUT_CONTRACT,
                    resume_input_deterministic=True,
                    dataloader=SimpleNamespace(use_background_prefetcher=False),
                ),
                train_steps=40,
            ),
            state=SimpleNamespace(global_step=0, epoch=0),
            train_dataloader=resumed,
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
            _sft_dynamic_exact_resume=True,
        )
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 86,
            "train_dataloader": cursor,
            "torch_rng_state": rng,
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 2),
        }
        if failure in ("corrupt", "replay_eof"):
            with pytest.raises((RuntimeError, ValueError)):
                callback.load_global_state()
            if failure == "corrupt":
                assert trainer.state.global_step == 0
                assert resume_raw.next_iter_state is None
        else:
            callback.load_global_state()
            resumed.set_epoch(trainer.start_epoch)
            assert rows(iter(resumed), 4) == expected
        value = torch.tensor([1])
        dist.all_reduce(value)
        assert value.item() == 2
    finally:
        close(orig_raw)
        close(resume_raw)
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("failure", ["slow", "corrupt", "replay_eof"])
def test_distributed_restore_coordinates_before_default_group_timeout(failure):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(distributed_resume_worker, args=(failure,), nprocs=2)

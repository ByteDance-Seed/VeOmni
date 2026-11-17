"""Regressions for the lifecycle of a multi-source sampling iterator."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import DynamicBatchingSizeDataset, WeightedMultiSourceDataset
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.base import VeOmniIter


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Source(torch.utils.data.IterableDataset):
    supports_exact_epoch_restart = True
    exact_resume_fingerprint = "synthetic-source-v1"

    def __init__(self):
        self.epoch = 0
        self.cursor = 0

    def set_epoch(self, epoch):
        if epoch != self.epoch:
            self.epoch, self.cursor = epoch, 0

    def reset_for_epoch(self, epoch):
        self.epoch, self.cursor = epoch, 0

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        while self.cursor < 20:
            self.cursor += 1
            yield {"id": self.epoch * 1000 + worker_id * 100 + self.cursor}

    def state_dict(self):
        return {"epoch": self.epoch, "cursor": self.cursor}

    def load_state_dict(self, state):
        self.epoch, self.cursor = state["epoch"], state["cursor"]

    def validate_exact_resume_state(self, state, worker_geometry=None, epoch=None):
        if not isinstance(state, dict) or type(state.get("cursor")) is not int:
            raise ValueError("invalid synthetic source state")
        if epoch is not None and state.get("epoch") != epoch:
            raise ValueError("synthetic source epoch mismatch")


def sample_length(_sample):
    return 1


@pytest.fixture(autouse=True)
def cpu_parallel_state(monkeypatch):
    import veomni.data.dataset as dataset_module

    monkeypatch.setattr(dataset_module, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))


def make_dataset(seed):
    return WeightedMultiSourceDataset(
        [Source(), Source(), Source()],
        weights=[0.3, 0.4, 0.3],
        source_ids=["a", "b", "c"],
        seed=seed,
        upstream_sharded=True,
        sample_token_len_fn=sample_length,
    )


def rows(iterator, count):
    samples = (next(iterator)[0][0] for _ in range(count))
    return [(item["id"], item["ds_idx"]) for item in samples]


@pytest.mark.parametrize(("workers", "persistent"), [(0, False), (2, False), (2, True)])
@pytest.mark.parametrize("epoch", [0, 2])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 29])
def test_worker_sampling_resume_before_each_worker_starts(workers, persistent, epoch, prefetch, seed):
    sources = []

    def make_loader():
        source = DistributedDataloader(
            make_dataset(seed), batch_size=None, num_workers=workers, persistent_workers=persistent
        )
        sources.append(source)
        return DynamicBatchSizeDataLoader(
            source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1, get_length_fn=sample_length), length=3
        )

    original = make_loader()
    original.set_epoch(epoch)
    iterator = VeOmniIter(original, use_background_prefetcher=prefetch)
    resumed_iterator = None
    try:
        rows(iterator, 1)
        state = copy.deepcopy(iterator.state_dict())
        rng = torch.get_rng_state().clone()
        expected = rows(iterator, 2)
        resumed = make_loader()
        resumed.prepare_resume_epoch(epoch)
        resumed.load_state_dict(state)
        torch.set_rng_state(rng)
        resumed.set_epoch(epoch)
        resumed_iterator = VeOmniIter(resumed, use_background_prefetcher=prefetch)
        assert rows(resumed_iterator, 2) == expected
    finally:
        iterator.stop()
        if resumed_iterator is not None:
            resumed_iterator.stop()
        for source in sources:
            source_iterator = getattr(source, "_iterator", None)
            if source_iterator is not None and hasattr(source_iterator, "_shutdown_workers"):
                source_iterator._shutdown_workers()


def test_snapshot_before_first_sample_initializes_worker_rng():
    source = make_dataset(seed=5)
    source.set_epoch(2)
    iterator = iter(source)
    state = copy.deepcopy(source.state_dict())
    expected = next(iterator)
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    restored.set_epoch(2)
    actual = next(iter(restored))
    assert (actual["id"], actual["ds_idx"]) == (expected["id"], expected["ds_idx"])


def test_discarded_new_iterator_does_not_change_active_resume_state():
    source = make_dataset(seed=5)
    active = iter(source)
    next(active)
    discarded = iter(source)
    del discarded
    state = copy.deepcopy(source.state_dict())
    expected = next(active)
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    actual = next(iter(restored))
    assert (actual["id"], actual["ds_idx"]) == (expected["id"], expected["ds_idx"])


def test_legacy_checkpoint_keeps_existing_started_iterator_behavior():
    source = make_dataset(seed=5)
    source.set_epoch(2)
    iterator = iter(source)
    next(iterator)
    state = copy.deepcopy(source.state_dict())
    state["runtime"].pop("epoch")
    state["runtime"].pop("sampling_initialized")
    expected = next(iterator)
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    restored.set_epoch(2)
    actual = next(iter(restored))
    assert (actual["id"], actual["ds_idx"]) == (expected["id"], expected["ds_idx"])


@pytest.mark.parametrize("worker_batching", [False, True])
@pytest.mark.parametrize("epoch", [0, 2])
def test_legacy_loader_resume_after_epoch_set_survives_next_epoch(worker_batching, epoch):
    def make_loader():
        dataset = make_dataset(seed=5)
        if worker_batching:
            dataset = DynamicBatchingSizeDataset(
                dataset,
                micro_batch_seq_length=1,
                ready_for_micro_batch_threshold=1,
                dynamic_batching_collate_fn=list,
                save_by_idx=False,
                get_length_fn=sample_length,
                get_physical_length_fn=sample_length,
            )
        return DistributedDataloader(dataset, batch_size=None, num_workers=0)

    original = make_loader()
    original.set_epoch(epoch)
    active = iter(original)
    for _ in range(4):
        next(active)
    state = copy.deepcopy(original.state_dict())
    dataset_state = state["dataset_state"]
    if worker_batching:
        dataset_state = dataset_state["dynamic_batch_upstream_dataset_state"]
    dataset_state["runtime"].pop("epoch")
    dataset_state["runtime"].pop("sampling_initialized")
    expected = [next(active) for _ in range(4)]

    resumed = make_loader()
    resumed.prepare_resume_epoch(epoch)
    resumed.load_state_dict(state)
    resumed.set_epoch(epoch)
    restored = iter(resumed)
    assert [next(restored) for _ in range(4)] == expected
    saved_again = copy.deepcopy(resumed.state_dict())["dataset_state"]
    if worker_batching:
        saved_again = saved_again["dynamic_batch_upstream_dataset_state"]
    assert saved_again["runtime"]["epoch"] == epoch

    resumed.set_epoch(epoch + 1)
    fresh = make_loader()
    fresh.set_epoch(epoch + 1)
    next_epoch, expected_epoch = iter(resumed), iter(fresh)
    assert [next(next_epoch) for _ in range(8)] == [next(expected_epoch) for _ in range(8)]


@pytest.mark.parametrize("workers", [0, 2])
def test_legacy_main_process_batching_rejects_unproven_epoch(workers, monkeypatch, tmp_path):
    from veomni.trainer.callbacks import base as callback_base
    from veomni.trainer.callbacks import global_state_callback as callback_module

    def make_loader():
        source = DistributedDataloader(make_dataset(seed=5), batch_size=None, num_workers=workers)
        strategy = TextBatchingStrategy(token_micro_bsz=1, buffer_size=1, get_length_fn=sample_length)
        return DynamicBatchSizeDataLoader(source, strategy, length=40)

    def sampling_states(value):
        if isinstance(value, dict):
            if "topology" in value and "runtime" in value and "dataset_states" in value["runtime"]:
                yield value["runtime"]
            for item in value.values():
                yield from sampling_states(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                yield from sampling_states(item)

    original, resumed, fresh = make_loader(), make_loader(), make_loader()
    try:
        original.set_epoch(2)
        active = iter(original)
        rows(active, 6)
        state = copy.deepcopy(original.state_dict())
        legacy_states = list(sampling_states(state["dataloader_state"]))
        assert len(legacy_states) == max(workers, 1)
        for runtime in legacy_states:
            runtime.pop("epoch")
            runtime.pop("sampling_initialized")
        saved_rng = torch.get_rng_state().clone()

        checkpoint = SimpleNamespace(
            save_steps=0,
            save_epochs=0,
            save_hf_weights=False,
            hf_save_steps=0,
            hf_save_epochs=0,
            load_path=str(tmp_path),
        )
        trainer = SimpleNamespace(
            args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint, global_rank=0), train_steps=40),
            train_dataloader=resumed,
            state=SimpleNamespace(global_step=0),
        )
        monkeypatch.setattr(callback_base, "get_parallel_state", lambda: SimpleNamespace())
        monkeypatch.setattr(callback_module, "set_device_rng_state", lambda _state: None)
        callback = callback_module.GlobalStateCallback(trainer)
        callback._read_current = lambda _path: {
            "global_step": 86,
            "train_dataloader": state,
            "torch_rng_state": saved_rng,
        }
        with pytest.raises(ValueError, match="worker epochs differ or are missing"):
            callback.load_global_state()
        assert trainer.state.global_step == 0
        assert torch.equal(torch.get_rng_state(), saved_rng)
    finally:
        for loader in (original, resumed, fresh):
            iterator = getattr(loader._dataloader, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()


def direct_rows(iterator, count):
    return [(sample["id"], sample["ds_idx"]) for sample in (next(iterator) for _ in range(count))]


def test_epoch_change_snapshot_before_first_sample_uses_new_epoch_rng():
    source = make_dataset(seed=5)
    previous = iter(source)
    direct_rows(previous, 4)
    previous.close()
    source.set_epoch(1)
    state = copy.deepcopy(source.state_dict())
    expected = direct_rows(iter(source), 8)
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    assert direct_rows(iter(restored), 8) == expected


def test_modern_restore_then_epoch_change_discards_old_sampling_state():
    source = make_dataset(seed=5)
    direct_rows(iter(source), 4)
    state = copy.deepcopy(source.state_dict())
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    restored.set_epoch(1)
    expected = make_dataset(seed=5)
    expected.set_epoch(1)
    assert direct_rows(iter(restored), 8) == direct_rows(iter(expected), 8)


def test_legacy_resume_then_second_snapshot_retains_started_rng():
    source = make_dataset(seed=5)
    source.set_epoch(2)
    iterator = iter(source)
    direct_rows(iterator, 4)
    legacy = copy.deepcopy(source.state_dict())
    legacy["runtime"].pop("epoch")
    legacy["runtime"].pop("sampling_initialized")
    restored = make_dataset(seed=5)
    restored.load_state_dict(legacy)
    restored.set_epoch(2)
    restored_iterator = iter(restored)
    assert direct_rows(restored_iterator, 3) == direct_rows(iterator, 3)
    second = copy.deepcopy(restored.state_dict())
    assert second["runtime"]["sampling_initialized"] is True
    expected = direct_rows(iterator, 8)
    restored_again = make_dataset(seed=5)
    restored_again.load_state_dict(second)
    assert direct_rows(iter(restored_again), 8) == expected


@pytest.mark.parametrize("epoch", [0, 2])
def test_legacy_resaved_before_epoch_is_supplied_retains_unknown_epoch(epoch):
    source = make_dataset(seed=5)
    source.set_epoch(epoch)
    iterator = iter(source)
    direct_rows(iterator, 4)
    state = copy.deepcopy(source.state_dict())
    state["runtime"].pop("epoch")
    state["runtime"].pop("sampling_initialized")
    for _ in range(2):
        restored = make_dataset(seed=5)
        restored.load_state_dict(state)
        state = copy.deepcopy(restored.state_dict())
        assert "epoch" not in state["runtime"]
    restored = make_dataset(seed=5)
    restored.load_state_dict(state)
    restored.set_epoch(epoch)
    restored.set_epoch(epoch)
    assert direct_rows(iter(restored), 8) == direct_rows(iterator, 8)


def test_legacy_consumption_without_epoch_remains_resumable_but_rejects_late_epoch():
    source = make_dataset(seed=5)
    source.set_epoch(2)
    iterator = iter(source)
    direct_rows(iterator, 4)
    legacy = copy.deepcopy(source.state_dict())
    legacy["runtime"].pop("epoch")
    legacy["runtime"].pop("sampling_initialized")
    restored = make_dataset(seed=5)
    restored.load_state_dict(legacy)
    assert direct_rows(iter(restored), 3) == direct_rows(iterator, 3)
    before_epoch = restored._epoch
    before_sources = [copy.deepcopy(dataset.state_dict()) for dataset in restored._datasets]
    with pytest.raises(ValueError, match="needs its epoch before sampling"):
        restored.set_epoch(2)
    assert restored._epoch == before_epoch
    assert [dataset.state_dict() for dataset in restored._datasets] == before_sources
    saved_again = copy.deepcopy(restored.state_dict())
    assert "epoch" not in saved_again["runtime"]
    resumed = make_dataset(seed=5)
    resumed.load_state_dict(saved_again)
    resumed.set_epoch(2)
    assert direct_rows(iter(resumed), 8) == direct_rows(iterator, 8)


def test_legacy_restore_does_not_reuse_an_epoch_from_before_sampling():
    source = make_dataset(seed=5)
    source.set_epoch(2)
    direct_rows(iter(source), 4)
    state = copy.deepcopy(source.state_dict())
    state["runtime"].pop("epoch")
    state["runtime"].pop("sampling_initialized")
    source.load_state_dict(state)
    direct_rows(iter(source), 1)
    with pytest.raises(ValueError, match="needs its epoch before sampling"):
        source.set_epoch(3)


@pytest.mark.parametrize("seed", [0, 1, 5, 29])
def test_uninterrupted_epoch_sampling_matches_independent_seed_reference(seed):
    source = make_dataset(seed=seed)
    for epoch in range(3):
        source.set_epoch(epoch)
        current_seed = np.random.SeedSequence([seed, epoch, 0, 0]).generate_state(1, dtype=np.uint32)[0]
        rng = np.random.RandomState(int(current_seed))
        counts = [0, 0, 0]
        expected = []
        for _ in range(8):
            index = int(rng.choice(3, p=[0.3, 0.4, 0.3]))
            counts[index] += 1
            expected.append((epoch * 1000 + counts[index], index))
        assert direct_rows(iter(source), 8) == expected


def test_invalid_initialization_marker_fails_before_mutating_sources():
    source = make_dataset(seed=5)
    state = copy.deepcopy(source.state_dict())
    state["runtime"]["sampling_initialized"] = 1
    state["runtime"]["dataset_states"]["a"]["cursor"] = 7
    with pytest.raises(ValueError, match="sampling_initialized"):
        source.load_state_dict(state)
    assert source._datasets[0].cursor == 0


def make_prefetch_loader():
    # The parent and spawned cold-resume process must use the same worker start method.
    source = DistributedDataloader(
        make_dataset(seed=0), batch_size=None, num_workers=2, multiprocessing_context="spawn"
    )
    return DynamicBatchSizeDataLoader(
        source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1, get_length_fn=sample_length), length=3
    )


def update(weight, optimizer, iterator):
    sample = rows(iterator, 1)[0]
    loss = (weight * (sample[0] + sample[1] + 1)).square()
    loss.backward()
    grad = weight.grad.detach().clone()
    optimizer.step()
    optimizer.zero_grad()
    return {"sample": sample, "loss": loss.detach(), "grad": grad, "weight": weight.detach().clone()}


def fresh_resume(_rank, checkpoint_path, result_path):
    import veomni.data.dataset as dataset_module

    dataset_module.get_parallel_state = lambda: SimpleNamespace(dp_rank=0, dp_size=1)
    saved = torch.load(checkpoint_path, weights_only=False)
    loader = make_prefetch_loader()
    loader.load_state_dict(saved["loader"])
    torch.set_rng_state(saved["rng"])
    loader.set_epoch(2)
    iterator = VeOmniIter(loader, use_background_prefetcher=True)
    weight = torch.nn.Parameter(saved["weight"].clone())
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    optimizer.load_state_dict(saved["optimizer"])
    try:
        result = update(weight, optimizer, iterator)
        result["optimizer"] = optimizer.state_dict()
        torch.save(result, result_path)
    finally:
        iterator.stop()
        source_iterator = getattr(loader._dataloader, "_iterator", None)
        if source_iterator is not None and hasattr(source_iterator, "_shutdown_workers"):
            source_iterator._shutdown_workers()


def test_multisource_worker_cold_resume_next_optimizer_update(tmp_path):
    import torch.multiprocessing as mp

    loader = make_prefetch_loader()
    loader.set_epoch(2)
    iterator = VeOmniIter(loader, use_background_prefetcher=True)
    weight = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    try:
        update(weight, optimizer, iterator)
        saved = {
            "loader": copy.deepcopy(iterator.state_dict()),
            "rng": torch.get_rng_state().clone(),
            "weight": weight.detach().clone(),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }
        expected = update(weight, optimizer, iterator)
        expected_state = optimizer.state_dict()["state"][0]
        checkpoint_path, result_path = tmp_path / "checkpoint.pt", tmp_path / "result.pt"
        torch.save(saved, checkpoint_path)
        mp.spawn(fresh_resume, args=(str(checkpoint_path), str(result_path)), nprocs=1, join=True)
        actual = torch.load(result_path, weights_only=True)
        assert actual["sample"] == expected["sample"]
        for key in ("loss", "grad", "weight"):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
        actual_state = actual["optimizer"]["state"][0]
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(actual_state[key], expected_state[key], rtol=0, atol=0)
    finally:
        iterator.stop()
        source_iterator = getattr(loader._dataloader, "_iterator", None)
        if source_iterator is not None and hasattr(source_iterator, "_shutdown_workers"):
            source_iterator._shutdown_workers()

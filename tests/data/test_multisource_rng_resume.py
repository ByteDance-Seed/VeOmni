"""Regressions for the lifecycle of a multi-source sampling iterator."""

import copy
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import WeightedMultiSourceDataset
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.base import VeOmniIter


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class Source(torch.utils.data.IterableDataset):
    def __init__(self):
        self.epoch = 0
        self.cursor = 0

    def set_epoch(self, epoch):
        if epoch != self.epoch:
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


def test_invalid_initialization_marker_fails_before_mutating_sources():
    source = make_dataset(seed=5)
    state = copy.deepcopy(source.state_dict())
    state["runtime"]["sampling_initialized"] = 1
    state["runtime"]["dataset_states"]["a"]["cursor"] = 7
    with pytest.raises(ValueError, match="sampling_initialized"):
        source.load_state_dict(state)
    assert source._datasets[0].cursor == 0


def make_prefetch_loader():
    source = DistributedDataloader(make_dataset(seed=0), batch_size=None, num_workers=2)
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

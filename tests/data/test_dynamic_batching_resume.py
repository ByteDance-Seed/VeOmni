# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""A saved dynamic loader cursor must reproduce the next optimizer input."""

import copy
from pathlib import Path

import pytest
import torch
import torch.multiprocessing as mp

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _StatefulSamples:
    def __init__(self):
        self.cursor = 0

    def __len__(self):
        return 100

    def __iter__(self):
        return self

    def __next__(self):
        self.cursor += 1
        return {"id": self.cursor, "attention_mask": torch.ones(1, dtype=torch.int64)}

    def state_dict(self):
        return {"cursor": self.cursor}

    def load_state_dict(self, state):
        self.cursor = state["cursor"]


def test_empty_source_fails_with_actionable_restart_error():
    loader = DynamicBatchSizeDataLoader(iter(()), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=1)
    with pytest.raises(RuntimeError, match="exhausted.*could not restart"):
        next(iter(loader))


def test_callable_code_identity_is_stable_across_python_hash_seeds():
    import os
    import subprocess
    import sys

    script = """
from veomni.data.dynamic_batching import _resume_callable_identity
def length(sample):
    return 1 if sample in {'alpha', 'beta', 'gamma', 'delta', 'epsilon'} else 0
length.__code__ = length.__code__.replace(
    co_consts=length.__code__.co_consts + ((frozenset({'one', 'two', 'three'}),),)
)
print(_resume_callable_identity(length)['code_sha256'])
"""
    digests = []
    for seed in ("0", "1", "2", "3"):
        output = subprocess.check_output(
            [sys.executable, "-c", script], env={**os.environ, "PYTHONHASHSEED": seed}, text=True
        )
        digests.append(output.strip().splitlines()[-1])
    assert len(set(digests)) == 1


def test_exhausted_iterator_fails_before_replaying_a_sample():
    sample = {"id": 7, "attention_mask": torch.ones(1, dtype=torch.int64)}
    loader = DynamicBatchSizeDataLoader(
        iter([sample]), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=2
    )
    iterator = iter(loader)
    assert [item["id"] for item in next(iterator)[0]] == [7]
    with pytest.raises(RuntimeError, match="exhausted.*could not restart"):
        next(iterator)


def test_non_strict_streaming_checkpoint_does_not_inspect_dataset_length():
    class UnsizedDataset:
        def __len__(self):
            raise RuntimeError("length unavailable")

    source = _loader()
    source._strict_resume_contract = False
    source._dataloader.dataset = UnsizedDataset()
    iterator = iter(source)
    next(iterator)
    saved = source.state_dict()
    assert "_veomni_resume_geometry" not in saved
    target = _loader()
    target._strict_resume_contract = False
    target._dataloader.dataset = UnsizedDataset()
    target.load_state_dict(saved)
    assert next(iter(target)) == next(iterator)


def test_restartable_finite_source_repeats_until_requested_length():
    samples = [{"id": 7, "attention_mask": torch.ones(1, dtype=torch.int64)}]
    loader = DynamicBatchSizeDataLoader(samples, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=2)
    iterator = iter(loader)
    assert [[item["id"] for item in next(iterator)[0]] for _ in range(2)] == [[7], [7]]


def test_oneshot_both_sources_exhausted_does_not_drain_buffer():
    """In this specific EOF layout, neither rank drains its remaining buffer."""
    from torchdata.stateful_dataloader import StatefulDataLoader

    class OneShot(torch.utils.data.IterableDataset):
        def __init__(self, count):
            self.source = iter([[{"id": index} for index in range(count)]])

        def __iter__(self):
            return self.source

    def identity(item):
        return item

    def loader(count):
        source = StatefulDataLoader(OneShot(count), batch_size=None, num_workers=0, collate_fn=identity)
        strategy = TextBatchingStrategy(token_micro_bsz=1, buffer_size=1, get_length_fn=lambda _: 1)
        return DynamicBatchSizeDataLoader(source, strategy, num_micro_batch=1, length=3, drop_last=True)

    iterators = [iter(loader(count)) for count in (3, 1)]
    for iterator in iterators:
        assert next(iterator)[0][0]["id"] == 0
    for iterator in iterators:
        with pytest.raises(RuntimeError, match="source exhausted and could not restart"):
            next(iterator)


@pytest.mark.parametrize("value", [True, "invalid"])
def test_exhausted_cursor_cannot_restart_source_silently(value):
    source = _loader()
    next(iter(source))
    state = source.state_dict()
    state["_veomni_source_exhausted"] = value
    resumed = _loader()
    with pytest.raises((RuntimeError, ValueError), match="exhausted"):
        resumed.load_state_dict(state)
    assert resumed._dataloader.cursor == 0


def test_unexhausted_cursor_from_previous_head_remains_loadable():
    source = _loader()
    iterator = iter(source)
    next(iterator)
    state = source.state_dict()
    state["_veomni_source_exhausted"] = False
    resumed = _loader()
    resumed.load_state_dict(state)
    assert next(iter(resumed)) == next(iterator)


@pytest.mark.parametrize("changed", ["length", "microbatches", "drop_last", "workers", "strategy", "length_fn"])
def test_dynamic_resume_rejects_changed_geometry_before_loading_source(changed):
    source = _loader()
    next(iter(source))
    saved = source.state_dict()
    target = _loader()
    if changed == "length":
        target._length += 1
    elif changed == "microbatches":
        target.num_micro_batch += 1
    elif changed == "drop_last":
        target._drop_last = not target._drop_last
    elif changed == "workers":
        target._dataloader.num_workers = 1
    elif changed == "length_fn":
        target.batching_strategy.buffer._get_length_fn = lambda sample: 1
    else:
        target.batching_strategy.token_micro_bsz += 1
    with pytest.raises(ValueError, match="geometry"):
        target.load_state_dict(saved)
    assert target._dataloader.cursor == 0
    assert target.step == 0


def test_dynamic_resume_rejects_legacy_cursor_without_geometry():
    source = _loader()
    next(iter(source))
    saved = source.state_dict()
    saved.pop("_veomni_resume_geometry", None)
    target = _loader()
    with pytest.raises(ValueError, match="geometry"):
        target.load_state_dict(saved)
    assert target._dataloader.cursor == 0


def test_dynamic_resume_rejects_malformed_geometry_and_phase():
    source = _loader()
    next(iter(source))
    saved = source.state_dict()
    saved["_veomni_resume_geometry"]["num_workers"] = False
    with pytest.raises(ValueError, match="geometry"):
        _loader().load_state_dict(saved)
    saved = source.state_dict()
    del saved["_veomni_resume_phase"]
    target = _loader()
    with pytest.raises(ValueError, match="phase"):
        target.load_state_dict(saved)
    assert target._checkpoint_phase == "running"
    saved = source.state_dict()
    saved["num_micro_batch"] += 1
    with pytest.raises(ValueError, match="geometry"):
        _loader().load_state_dict(saved)
    saved = source.state_dict()
    saved["batching_strategy"].buffer_size += 1
    with pytest.raises(ValueError, match="strategy"):
        _loader().load_state_dict(saved)


@pytest.mark.parametrize(
    "changed",
    [
        "epoch_steps",
        "epoch",
        "source",
        "prefetch",
        "old_context",
        "missing_contract",
        "determinism",
        "padding",
        "parallel",
        "out_of_order",
    ],
)
def test_trainer_rejects_unproven_dynamic_cursor_before_restoring(changed):
    from types import SimpleNamespace

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _loader()
    if changed == "out_of_order":
        source._dataloader.num_workers = 1
        source._dataloader.in_order = False
        source._dataloader.dataset = SimpleNamespace(_fingerprint="fixed")
    next(iter(source))
    cursor = source.state_dict()
    target = _loader()
    if changed == "out_of_order":
        target._dataloader.num_workers = 1
        target._dataloader.in_order = False
        target._dataloader.dataset = SimpleNamespace(_fingerprint="fixed")
    args = SimpleNamespace(
        train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")),
        data=SimpleNamespace(
            train_path="test://source-v1",
            resume_input_contract="sha256:" + "1" * 64,
            resume_input_deterministic=True,
            dataloader=SimpleNamespace(use_background_prefetcher=False),
        ),
        train_steps=100,
    )
    context = _dynamic_resume_context(args, 0)
    if changed == "epoch_steps":
        context["epoch_steps"] += 1
    elif changed == "epoch":
        context["saved_epoch"] = 1
    elif changed == "source":
        context["data"]["train_path"] = "test://different"
    elif changed == "prefetch":
        context["background_prefetch"] = True
    elif changed == "missing_contract":
        args.data.resume_input_contract = None
        context["data"]["resume_input_contract"] = None
    elif changed == "determinism":
        args.data.resume_input_deterministic = False
        context["data"]["resume_input_deterministic"] = False
    elif changed == "padding":
        context["train"]["pad_to_length"] = True
    elif changed == "parallel":
        context["parallel"]["cp_size"] = 4
    trainer = SimpleNamespace(
        args=args,
        state=SimpleNamespace(global_step=0, epoch=0),
        train_dataloader=target,
        _sft_dynamic_exact_resume=True,
    )
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    saved = {"global_step": 1, "train_dataloader": cursor}
    if changed != "old_context":
        saved["_veomni_dynamic_context"] = context
    callback._read_current = lambda _: saved
    with pytest.raises((RuntimeError, ValueError), match="dynamic"):
        callback.load_global_state()
    assert trainer.state.global_step == 0
    assert target._dataloader.cursor == 0


def _loader():
    loader = DynamicBatchSizeDataLoader(
        _StatefulSamples(),
        TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
        num_micro_batch=1,
        length=10,
    )
    loader._strict_resume_contract = True
    return loader


def _optimizer_step(weight, optimizer, batch):
    sample_ids = [sample["id"] for sample in batch[0]]
    loss = (weight * sum(sample_ids)).square()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return sample_ids


def _fresh_resume(_rank, checkpoint_path, result_path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loader = _loader()
    loader.load_state_dict(saved["loader"])
    weight = torch.nn.Parameter(saved["weight"].clone())
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    optimizer.load_state_dict(saved["optimizer"])
    ids = _optimizer_step(weight, optimizer, next(iter(loader)))
    torch.save({"ids": ids, "weight": weight.detach(), "optimizer": optimizer.state_dict()}, result_path)


def test_dynamic_loader_warmup_cursor_and_fresh_process_update_match(tmp_path: Path):
    loader = _loader()
    iterator = iter(loader)
    weight = torch.nn.Parameter(torch.tensor(0.5))
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    assert _optimizer_step(weight, optimizer, next(iterator)) == [1]
    checkpoint = {
        "loader": copy.deepcopy(loader.state_dict()),
        "weight": weight.detach().clone(),
        "optimizer": copy.deepcopy(optimizer.state_dict()),
    }
    assert checkpoint["loader"]["step"] == 1
    checkpoint_path = tmp_path / "step1.pt"
    torch.save(checkpoint, checkpoint_path)
    expected_ids = _optimizer_step(weight, optimizer, next(iterator))
    result_path = tmp_path / "fresh.pt"
    mp.spawn(_fresh_resume, args=(str(checkpoint_path), str(result_path)), nprocs=1, join=True)
    result = torch.load(result_path, map_location="cpu", weights_only=False)
    assert result["ids"] == expected_ids
    torch.testing.assert_close(result["weight"], weight.detach(), rtol=0, atol=0)
    expected_state = next(iter(optimizer.state_dict()["state"].values()))
    restored_state = next(iter(result["optimizer"]["state"].values()))
    for key in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(restored_state[key], expected_state[key], rtol=0, atol=0)


def test_uninterrupted_warmup_batch_sequence_is_unchanged():
    loader = _loader()
    iterator = iter(loader)
    batches = [[sample["id"] for sample in next(iterator)[0]] for _ in range(3)]
    assert batches == [[1], [2, 3], [4, 5, 6]]
    assert loader.state_dict()["step"] == 3


def test_multiple_microbatches_resume_with_the_same_next_group():
    def make_loader():
        return DynamicBatchSizeDataLoader(
            _StatefulSamples(),
            TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
            num_micro_batch=2,
            length=10,
        )

    original = make_loader()
    iterator = iter(original)
    next(iterator)
    state = copy.deepcopy(original.state_dict())
    expected = [[item["id"] for item in microbatch] for microbatch in next(iterator)]
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = [[item["id"] for item in microbatch] for microbatch in next(iter(resumed))]
    assert state["step"] == 1
    assert actual == expected


def test_resume_preserves_final_flush_when_drop_last_is_false():
    class FiniteSamples(_StatefulSamples):
        def __next__(self):
            if self.cursor >= 4:
                raise StopIteration
            return super().__next__()

    def make_loader():
        return DynamicBatchSizeDataLoader(
            FiniteSamples(),
            TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
            num_micro_batch=1,
            length=1,
            drop_last=False,
        )

    original = make_loader()
    iterator = iter(original)
    next(iterator)
    state = copy.deepcopy(original.state_dict())
    expected = [[item["id"] for item in microbatch] for microbatch in next(iterator)]
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = [[item["id"] for item in microbatch] for microbatch in next(iter(resumed))]
    assert actual == expected


def test_resume_performs_the_same_post_yield_fetch_with_ready_residual_buffer():
    class VariableSamples(_StatefulSamples):
        def __next__(self):
            self.cursor += 1
            lengths = [1, 1, 9, 3, 2, 4, 1, 5]
            return {"id": self.cursor, "attention_mask": torch.ones(lengths[(self.cursor - 1) % len(lengths)])}

    def make_loader():
        return DynamicBatchSizeDataLoader(
            VariableSamples(),
            TextBatchingStrategy(token_micro_bsz=10, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
            num_micro_batch=1,
            length=20,
        )

    original = make_loader()
    iterator = iter(original)
    assert [item["id"] for item in next(iterator)[0]] == [1]
    state = copy.deepcopy(original.state_dict())
    expected = [item["id"] for item in next(iterator)[0]]
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = [item["id"] for item in next(iter(resumed))[0]]
    assert actual == expected


def test_resume_in_flush_matches_all_remaining_batches():
    class FiniteSamples(_StatefulSamples):
        def __next__(self):
            if self.cursor >= 4:
                raise StopIteration
            return super().__next__()

    def make_loader():
        return DynamicBatchSizeDataLoader(
            FiniteSamples(),
            TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
            collate_fn=lambda items: {"ids": [item["id"] for item in items]},
            num_micro_batch=1,
            length=1,
            drop_last=False,
        )

    original = make_loader()
    iterator = iter(original)
    next(iterator)
    next(iterator)
    flushed = next(iterator)
    assert flushed[0]["ids"] == [4]
    flushed[0]["ids"][-1] = 99
    state = copy.deepcopy(original.state_dict())
    assert state["_veomni_resume_phase"] == "flushing"

    def remaining(loader):
        result = []
        while True:
            try:
                result.append(next(loader))
            except StopIteration:
                return result

    expected = remaining(iterator)
    resumed = make_loader()
    resumed.load_state_dict(state)
    assert remaining(iter(resumed)) == expected


def test_invalid_resume_phase_is_rejected():
    loader = _loader()
    next(iter(loader))
    state = loader.state_dict()
    state["_veomni_resume_phase"] = "invalid"
    with pytest.raises(ValueError, match="invalid dynamic loader resume phase"):
        _loader().load_state_dict(state)


def test_real_stateful_loader_resume_at_source_eof_preserves_flush():
    from torchdata.stateful_dataloader import StatefulDataLoader

    samples = [{"id": index, "attention_mask": torch.ones(1, dtype=torch.int64)} for index in range(1, 5)]

    def make_loader():
        source = StatefulDataLoader([samples], batch_size=None, num_workers=0)
        return DynamicBatchSizeDataLoader(
            source,
            TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
            collate_fn=lambda items: {"ids": [item["id"] for item in items]},
            num_micro_batch=1,
            length=1,
            drop_last=False,
        )

    original = make_loader()
    iterator = iter(original)
    assert next(iterator)[0]["ids"] == [1]
    state = copy.deepcopy(original.state_dict())
    expected = next(iterator)
    resumed = make_loader()
    resumed.load_state_dict(state)
    assert next(iter(resumed)) == expected


def test_background_prefetch_resume_preserves_next_unmutated_group():
    from veomni.trainer.base import BackgroundPrefetcher

    loader = _loader()
    prefetcher = BackgroundPrefetcher(loader, maxsize=1)
    try:
        first = next(prefetcher)
        assert [sample["id"] for sample in first[0]] == [1]
        state = copy.deepcopy(prefetcher.state_dict())
        expected = [[sample["id"] for sample in batch] for batch in next(prefetcher)]
    finally:
        prefetcher.stop()
    resumed = _loader()
    resumed.load_state_dict(state)
    actual = [[sample["id"] for sample in batch] for batch in next(iter(resumed))]
    assert actual == expected


def _epoch_test_collate(items):
    return items[0]


def test_trainer_rejects_dynamic_epoch_boundary_before_restoring_cursor(monkeypatch):
    from types import SimpleNamespace

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    loader = _loader()
    restored = []
    monkeypatch.setattr(loader, "load_state_dict", lambda state: restored.append(state))
    trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")), train_steps=6
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
        train_dataloader=loader,
    )
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {"global_step": 18, "train_dataloader": {}}

    with pytest.raises(RuntimeError, match="Cannot safely restore.*epoch boundary"):
        callback.load_global_state()
    assert trainer.state.global_step == 0
    assert restored == []


@pytest.mark.parametrize("cursor_field", [{}, {"train_dataloader": None}])
def test_trainer_allows_new_data_without_cursor_at_epoch_boundary(monkeypatch, cursor_field):
    from types import SimpleNamespace

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    trainer = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")), train_steps=6
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
        train_dataloader=_loader(),
        channel_loss_callback=None,
        offline_cache_writer=None,
        environ_meter=None,
    )
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {"global_step": 18, **cursor_field}
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)

    callback.load_global_state()
    assert trainer.state.global_step == 18
    assert trainer.start_epoch == 3
    assert trainer.start_step == 0


def _epoch_boundary_guard_worker(rank, endpoint):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        loader = _loader()
        restored = []
        loader.load_state_dict = lambda state: restored.append(state)
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=rank, checkpoint=SimpleNamespace(load_path="/unused")), train_steps=6
            ),
            state=SimpleNamespace(global_step=0, epoch=0),
            train_dataloader=loader,
        )
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {"global_step": 18 if rank == 0 else 17, "train_dataloader": {}}
        with pytest.raises(RuntimeError, match="epoch boundary|failed on another rank"):
            callback.load_global_state()
        assert trainer.state.global_step == 0
        assert restored == []
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_epoch_boundary_guard_propagates_before_any_rank_restores():
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_epoch_boundary_guard_worker, args=(), nprocs=2)


@pytest.mark.parametrize("workers", [0, 2])
def test_native_sft_mid_epoch_resume_restores_sampler_before_worker_prefetch(monkeypatch, workers):
    from types import SimpleNamespace

    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from veomni.data.data_loader import DistributedDataloader
    from veomni.data.dataset import MappingDataset
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    def make_loader():
        from datasets import Dataset

        samples = MappingDataset(
            Dataset.from_list([{"id": i, "attention_mask": [1], "labels": [1]} for i in range(60)]).with_format(
                "torch"
            ),
            seed=7,
        )
        sampler = StatefulDistributedSampler(samples, num_replicas=1, rank=0, shuffle=True, seed=7)
        options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
        source = DistributedDataloader(
            samples,
            batch_size=1,
            sampler=sampler,
            num_workers=workers,
            collate_fn=_epoch_test_collate,
            drop_last=True,
            snapshot_every_n_steps=1,
            **options,
        )
        loader = DynamicBatchSizeDataLoader(
            source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=6, drop_last=True
        )
        loader._strict_resume_contract = True
        return loader

    def trace(iterator):
        return [(next(iterator)[0][0]["id"], torch.rand(()).item()) for _ in range(3)]

    def close(loader):
        iterator = getattr(loader, "_data_iter", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()

    torch.manual_seed(12345)
    original = make_loader()
    try:
        original.set_epoch(2)
        iterator = iter(original)
        for _ in range(2):
            next(iterator)
        cursor, rng = original.state_dict(), torch.get_rng_state()
        expected = trace(iterator)
    finally:
        close(original)

    restored = make_loader()
    try:
        checkpoint = SimpleNamespace(load_path="/unused")
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=0, checkpoint=checkpoint),
                data=SimpleNamespace(
                    train_path="test://immutable",
                    resume_input_contract="sha256:" + "1" * 64,
                    resume_input_deterministic=True,
                    dataloader=SimpleNamespace(use_background_prefetcher=False),
                ),
                train_steps=6,
            ),
            state=SimpleNamespace(global_step=0, epoch=0),
            train_dataloader=restored,
            channel_loss_callback=None,
            offline_cache_writer=None,
            environ_meter=None,
            _sft_dynamic_exact_resume=True,
        )
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 14,
            "train_dataloader": cursor,
            "torch_rng_state": rng,
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 2),
        }
        monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
        callback.load_global_state()
        restored.set_epoch(trainer.start_epoch)
        assert trace(iter(restored)) == expected
    finally:
        close(restored)


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("drop_last", [False, True])
@pytest.mark.parametrize("num_micro_batch", [1, 2])
def test_native_sft_epoch_boundary_restores_buffer_and_rng(monkeypatch, workers, drop_last, num_micro_batch):
    from types import SimpleNamespace

    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from veomni.data.data_loader import DistributedDataloader
    from veomni.data.dataset import MappingDataset
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    def make_loader():
        from datasets import Dataset

        samples = MappingDataset(
            Dataset.from_list([{"id": i, "attention_mask": [1]} for i in range(60)]).with_format("torch"), seed=7
        )
        sampler = StatefulDistributedSampler(samples, num_replicas=1, rank=0, shuffle=True, seed=7)
        options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
        source = DistributedDataloader(
            samples,
            batch_size=1,
            sampler=sampler,
            num_workers=workers,
            collate_fn=_epoch_test_collate,
            drop_last=drop_last,
            snapshot_every_n_steps=1,
            **options,
        )
        loader = DynamicBatchSizeDataLoader(
            source,
            TextBatchingStrategy(token_micro_bsz=1, buffer_size=3),
            num_micro_batch=num_micro_batch,
            length=6,
            drop_last=drop_last,
        )
        loader._strict_resume_contract = True
        return loader

    def close(loader):
        iterator = getattr(loader, "_data_iter", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()

    def trace(iterator):
        return [(tuple(micro[0]["id"] for micro in next(iterator)), torch.rand(()).item()) for _ in range(3)]

    torch.manual_seed(12345)
    original = make_loader()
    try:
        original.set_epoch(2)
        iterator = iter(original)
        for _ in range(6):
            next(iterator)
        cursor, rng = original.state_dict(), torch.get_rng_state()
        original.set_epoch(3)
        expected = trace(iter(original))
    finally:
        close(original)

    resumed = make_loader()
    try:
        trainer = SimpleNamespace(
            args=SimpleNamespace(
                train=SimpleNamespace(global_rank=0, checkpoint=SimpleNamespace(load_path="/unused")),
                data=SimpleNamespace(
                    train_path="test://immutable",
                    resume_input_contract="sha256:" + "1" * 64,
                    resume_input_deterministic=True,
                    dataloader=SimpleNamespace(use_background_prefetcher=False),
                ),
                train_steps=6,
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
            "global_step": 18,
            "train_dataloader": cursor,
            "torch_rng_state": rng,
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 2),
        }
        monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
        callback.load_global_state()
        resumed.set_epoch(trainer.start_epoch)
        assert trace(iter(resumed)) == expected
    finally:
        close(resumed)


def _native_mapping_loader(rank=0, world_size=1, *, offset=0, workers=0, persistent=False, source_drop_last=False):
    from datasets import Dataset
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    from veomni.data.data_loader import DistributedDataloader
    from veomni.data.dataset import MappingDataset

    data = Dataset.from_list([{"id": i + offset, "attention_mask": [1]} for i in range(60)]).with_format("torch")
    dataset = MappingDataset(data, seed=7)
    sampler = StatefulDistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=7)
    options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
    source = DistributedDataloader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        persistent_workers=persistent,
        drop_last=source_drop_last,
        collate_fn=_epoch_test_collate,
        snapshot_every_n_steps=1,
        **options,
    )
    loader = DynamicBatchSizeDataLoader(source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=3), length=6)
    loader._strict_resume_contract = True
    return loader


def _native_resume_trainer(loader, rank=0):
    from types import SimpleNamespace

    return SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(global_rank=rank, checkpoint=SimpleNamespace(load_path="/unused")),
            data=SimpleNamespace(
                train_path="test://immutable",
                resume_input_contract="sha256:" + "1" * 64,
                resume_input_deterministic=True,
                dataloader=SimpleNamespace(use_background_prefetcher=False),
            ),
            train_steps=6,
        ),
        state=SimpleNamespace(global_step=0, epoch=0),
        train_dataloader=loader,
        channel_loss_callback=None,
        offline_cache_writer=None,
        environ_meter=None,
        _sft_dynamic_exact_resume=True,
    )


def test_dynamic_resume_rejects_same_path_changed_dataset():
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader()
    next(iter(source))
    target = _native_mapping_loader(offset=100)
    trainer = _native_resume_trainer(target)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "train_dataloader": source.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
    }
    with pytest.raises(ValueError, match="geometry"):
        callback.load_global_state()
    assert trainer.state.global_step == 0


def test_dynamic_resume_rejects_changed_source_drop_last():
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader(source_drop_last=False)
    next(iter(source))
    target = _native_mapping_loader(source_drop_last=True)
    trainer = _native_resume_trainer(target)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "train_dataloader": source.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
    }
    with pytest.raises(ValueError, match="geometry"):
        callback.load_global_state()
    assert trainer.state.global_step == 0


def test_dynamic_boundary_rejects_persistent_workers():
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader(workers=2, persistent=True)
    target = _native_mapping_loader(workers=2, persistent=True)
    try:
        source.set_epoch(0)
        iterator = iter(source)
        for _ in range(6):
            next(iterator)
        trainer = _native_resume_trainer(target)
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 6,
            "train_dataloader": source.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
        }
        with pytest.raises(RuntimeError, match="epoch boundary"):
            callback.load_global_state()
        assert trainer.state.global_step == 0
    finally:
        for loader in (source, target):
            iterator = getattr(loader, "_data_iter", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()


def test_epoch_boundary_resave_without_next_step_keeps_original_cursor(monkeypatch):
    from types import SimpleNamespace

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader()
    source.set_epoch(0)
    iterator = iter(source)
    for _ in range(6):
        next(iterator)
    cursor = source.state_dict()
    rng = torch.get_rng_state()
    trainer = _native_resume_trainer(_native_mapping_loader())
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    saved = {
        "global_step": 6,
        "train_dataloader": cursor,
        "torch_rng_state": rng,
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
    }
    callback._read_current = lambda _: saved
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
    callback.load_global_state()
    assert trainer.start_epoch == 1
    resaved = callback.state_dict(SimpleNamespace(global_step=6, epoch=0))
    assert resaved["train_dataloader"]["step"] == 6
    assert resaved["train_dataloader"]["_veomni_resume_phase"] == "after_yield"

    again = _native_resume_trainer(_native_mapping_loader())
    callback_again = GlobalStateCallback.__new__(GlobalStateCallback)
    callback_again.trainer = again
    callback_again._read_current = lambda _: resaved
    callback_again.load_global_state()
    trainer.train_dataloader.set_epoch(1)
    again.train_dataloader.set_epoch(1)
    assert next(iter(trainer.train_dataloader)) == next(iter(again.train_dataloader))


def test_epoch_start_without_completed_step_is_not_a_resume_point():
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader(workers=2)
    source.set_epoch(1)
    iter(source)
    trainer = _native_resume_trainer(_native_mapping_loader(workers=2))
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 6,
        "train_dataloader": source.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 1),
    }
    try:
        with pytest.raises(ValueError, match="completed training step"):
            callback.load_global_state()
        assert trainer.state.global_step == 0
    finally:
        iterator = getattr(source, "_data_iter", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


@pytest.mark.parametrize("closure", [False, True])
def test_dynamic_resume_rejects_changed_transform_with_same_dataset(closure):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    source = _native_mapping_loader()
    target = _native_mapping_loader()

    def transform(offset):
        return lambda sample: {**sample, "id": sample["id"] + offset}

    if closure:
        source._dataloader.dataset._transform = transform(0)
        target._dataloader.dataset._transform = transform(100)
    else:
        source._dataloader.dataset._transform = lambda sample: sample
        target._dataloader.dataset._transform = lambda sample: {**sample, "id": sample["id"] + 100}
    next(iter(source))
    trainer = _native_resume_trainer(target)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "train_dataloader": source.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
    }
    with pytest.raises(ValueError, match="geometry"):
        callback.load_global_state()
    assert trainer.state.global_step == 0


def test_dynamic_trainer_state_records_epoch_and_input_contract():
    from types import SimpleNamespace

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    loader = _native_mapping_loader()
    loader.set_epoch(2)
    next(iter(loader))
    trainer = _native_resume_trainer(loader)
    trainer.environ_meter = SimpleNamespace(state_dict=lambda: {})
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    state = callback.state_dict(SimpleNamespace(global_step=13, epoch=2))
    context = state["_veomni_dynamic_context"]
    assert context["epoch_steps"] == 6
    assert context["saved_epoch"] == 2
    assert context["data"]["resume_input_contract"] == "sha256:" + "1" * 64
    assert context["data"]["resume_input_deterministic"] is True
    assert state["train_dataloader"]["_veomni_resume_geometry"]["dataset_fingerprint"]


def test_non_sft_loader_retains_legacy_resume_path(monkeypatch):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    source = _loader()
    next(iter(source))
    saved = source.state_dict()
    saved.pop("_veomni_resume_geometry")
    saved.pop("_veomni_resume_phase")
    loader = _loader()
    loader._strict_resume_contract = False
    trainer = _native_resume_trainer(loader)
    trainer._sft_dynamic_exact_resume = False
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "train_dataloader": saved,
        "torch_rng_state": torch.get_rng_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
    callback.load_global_state()
    assert loader._dataloader.cursor == source._dataloader.cursor


def test_sft_iterable_source_uses_legacy_cursor_path(monkeypatch):
    from types import SimpleNamespace

    from torchdata.stateful_dataloader import StatefulDataLoader

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            for sample_id in range(10):
                yield {"id": sample_id, "attention_mask": torch.ones(1, dtype=torch.int64)}

    def make_loader():
        source = StatefulDataLoader(Stream(), batch_size=None, num_workers=0)
        return DynamicBatchSizeDataLoader(source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=6)

    source = make_loader()
    iterator = iter(source)
    next(iterator)
    trainer = _native_resume_trainer(source)
    trainer.environ_meter = SimpleNamespace(state_dict=lambda: {})
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    saved = callback.state_dict(SimpleNamespace(global_step=1, epoch=0))
    assert "_veomni_dynamic_context" not in saved
    assert "_veomni_resume_geometry" not in saved["train_dataloader"]

    target = make_loader()
    resumed_trainer = _native_resume_trainer(target)
    resumed_trainer.environ_meter = SimpleNamespace(load_state_dict=lambda _: None)
    resumed = GlobalStateCallback.__new__(GlobalStateCallback)
    resumed.trainer = resumed_trainer
    resumed._read_current = lambda _: saved
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
    resumed.load_global_state()
    assert next(iter(target))[0][0]["id"] == next(iterator)[0][0]["id"]


@pytest.mark.parametrize("target_kind", ["iterable", "classification"])
def test_strict_sft_cursor_cannot_silently_downgrade_to_legacy(monkeypatch, target_kind):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    class Stream(torch.utils.data.IterableDataset):
        def __iter__(self):
            return iter(())

    source = _native_mapping_loader()
    next(iter(source))
    target = _loader()
    trainer = _native_resume_trainer(target)
    if target_kind == "iterable":
        target._dataloader.dataset = Stream()
    else:
        trainer._sft_dynamic_exact_resume = False
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "train_dataloader": source.state_dict(),
        "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
    }
    monkeypatch.setattr(target, "load_state_dict", lambda _: pytest.fail("cursor must not be restored"))
    with pytest.raises(ValueError, match="strict dynamic SFT cursor"):
        callback.load_global_state()
    assert trainer.state.global_step == 0


def _native_test_update(weight, optimizer, batch):
    sample_id = int(batch[0][0]["id"])
    random_value = torch.rand(())
    loss = (weight * (sample_id + 1) - random_value).square().sum()
    loss.backward()
    gradient = float(weight.grad.item())
    optimizer.step()
    optimizer.zero_grad()
    state = optimizer.state[weight]
    return (
        sample_id,
        float(random_value.item()),
        float(loss.item()),
        gradient,
        float(weight.item()),
        float(state["exp_avg"].item()),
        float(state["exp_avg_sq"].item()),
    )


def _fresh_native_epoch_resume(_rank, checkpoint_path, result_path, workers):
    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback

    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    loader = _native_mapping_loader(workers=workers)
    trainer = _native_resume_trainer(loader)
    weight = torch.nn.Parameter(torch.tensor([0.125]))
    optimizer = torch.optim.AdamW([weight], lr=0.001, foreach=False)
    with torch.no_grad():
        weight.copy_(saved["weight"])
    optimizer.load_state_dict(saved["optimizer"])
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: saved
    try:
        callback.load_global_state()
        loader.set_epoch(trainer.start_epoch)
        iterator = iter(loader)
        result = [_native_test_update(weight, optimizer, next(iterator)) for _ in range(3)]
        torch.save(result, result_path)
    finally:
        iterator = getattr(loader, "_data_iter", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


@pytest.mark.parametrize("workers", [0, 2])
def test_dynamic_epoch_boundary_fresh_process_matches_uninterrupted(tmp_path, workers):
    from veomni.trainer.callbacks.global_state_callback import _dynamic_resume_context

    torch.manual_seed(12345)
    loader = _native_mapping_loader(workers=workers)
    trainer = _native_resume_trainer(loader)
    weight = torch.nn.Parameter(torch.tensor([0.125]))
    optimizer = torch.optim.AdamW([weight], lr=0.001, foreach=False)
    try:
        loader.set_epoch(0)
        iterator = iter(loader)
        for _ in range(6):
            _native_test_update(weight, optimizer, next(iterator))
        saved = {
            "global_step": 6,
            "train_dataloader": loader.state_dict(),
            "torch_rng_state": torch.get_rng_state(),
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
            "weight": weight.detach().clone(),
            "optimizer": copy.deepcopy(optimizer.state_dict()),
        }
        checkpoint_path = tmp_path / "cursor.pt"
        result_path = tmp_path / "result.pt"
        torch.save(saved, checkpoint_path)
        loader.set_epoch(1)
        iterator = iter(loader)
        expected = [_native_test_update(weight, optimizer, next(iterator)) for _ in range(3)]
    finally:
        iterator = getattr(loader, "_data_iter", None)
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
    mp.spawn(_fresh_native_epoch_resume, args=(str(checkpoint_path), str(result_path), workers), nprocs=1, join=True)
    assert torch.load(result_path, weights_only=False) == expected


def _dynamic_restore_failure_worker(rank, failure, endpoint):
    from datetime import timedelta
    from types import SimpleNamespace

    import torch.distributed as dist

    from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback, _dynamic_resume_context

    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2, timeout=timedelta(seconds=30))
    try:
        source = _native_mapping_loader(rank, 2)
        source.set_epoch(0)
        iterator = iter(source)
        steps = 0 if failure == "step_zero" else 6 if failure == "boundary_phase" else 1
        for _ in range(steps):
            next(iterator)
        cursor = source.state_dict()
        target = _native_mapping_loader(rank, 2)
        trainer = _native_resume_trainer(target, rank)
        if failure in ("non_strict_loaded_step", "corrupt_loaded_step"):
            trainer._sft_dynamic_exact_resume = False
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._last_saved_step = -1
        if failure == "save_collection":
            trainer.args.train.checkpoint.save_path = "/unused"

            def collect_state(_):
                if rank == 0:
                    raise RuntimeError("cannot collect rank-local cursor")
                return {}

            callback.state_dict = collect_state
            with pytest.raises(RuntimeError, match="cannot collect|another rank"):
                callback.save_global_state(SimpleNamespace(global_step=1))
            dist.barrier()
            return
        state = {
            "global_step": steps,
            "train_dataloader": cursor,
            "torch_rng_state": torch.get_rng_state(),
            "_veomni_dynamic_context": _dynamic_resume_context(trainer.args, 0),
        }
        if failure == "callback":

            class LocalCallback:
                def load_state_dict(self, value):
                    if value == "bad":
                        raise ValueError("corrupt callback state")

            trainer.channel_loss_callback = LocalCallback()
            state["channel_loss_callback"] = "bad" if rank == 0 else "valid"
        if rank == 0:
            if failure == "missing_cursor":
                state.pop("train_dataloader")
            elif failure == "source_state":
                cursor["dataloader_state"] = {"unexpected": 1}
            elif failure == "missing_rng":
                state.pop("torch_rng_state")
            elif failure == "cpu_rng_conflict":
                state["device_rng_state"] = torch.Generator().manual_seed(123).get_state()
            elif failure in ("boundary_phase", "mid_phase"):
                cursor["_veomni_resume_phase"] = "running"
            elif failure == "rank_context":
                state["_veomni_dynamic_context"]["data"]["train_path"] = "test://corrupt-saved-source"
            elif failure == "non_strict_loaded_step":
                state["global_step"] = 2
            elif failure == "corrupt_loaded_step":
                state["global_step"] = {"invalid": 1}

        def read_current(_):
            if rank == 0 and failure == "load_io":
                raise OSError("cannot read rank-local cursor")
            return state

        callback._read_current = read_current
        with pytest.raises((RuntimeError, ValueError, AssertionError, OSError)):
            callback.load_global_state()
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "failure",
    [
        "missing_cursor",
        "source_state",
        "missing_rng",
        "cpu_rng_conflict",
        "step_zero",
        "boundary_phase",
        "mid_phase",
        "callback",
        "rank_context",
        "non_strict_loaded_step",
        "corrupt_loaded_step",
        "load_io",
        "save_collection",
    ],
)
@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu", nproc=2))
def test_dynamic_resume_rank_local_failures_are_coordinated(failure):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_dynamic_restore_failure_worker, args=(failure,), nprocs=2)

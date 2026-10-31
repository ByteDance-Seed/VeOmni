# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""A restored flush must not leak the old source cursor into later epochs."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data.test_dpo_provenance_checkpoint import _trainer
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import MappingDataset, get_length_fn
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.text_dpo_trainer import _dpo_recovery_geometry


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _Samples(list):
    pass


def _unpack(items):
    return items[0]


def _loader(workers):
    samples = _Samples({"id": index} for index in range(12))
    samples._fingerprint = hashlib.sha256(repr(samples).encode()).hexdigest()
    dataset = MappingDataset(samples, seed=42)
    sampler = StatefulDistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True, seed=42)
    options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
    source = DistributedDataloader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=workers,
        collate_fn=_unpack,
        drop_last=True,
        snapshot_every_n_steps=1,
        **options,
    )
    loader = DynamicBatchSizeDataLoader(
        source,
        TextBatchingStrategy(token_micro_bsz=1, buffer_size=10, get_length_fn=get_length_fn),
        num_micro_batch=1,
        length=6,
    )
    loader._strict_resume_contract = True
    loader._dpo_disallow_implicit_restart = True
    return loader


def _trace(iterator, count):
    return [(next(iterator)[0][0]["id"], float(torch.rand(()))) for _ in range(count)]


def _close(loader):
    for iterator in (getattr(loader, "_data_iter", None), getattr(loader._dataloader, "_iterator", None)):
        if iterator is not None and hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()


@pytest.mark.parametrize("workers", [0, 2])
def test_flushing_resume_preserves_two_following_epochs_and_rng(workers):
    trainer = _trainer(True)
    trainer.args.train_steps = 6
    trainer.args.train.num_train_epochs = 4
    trainer.args.data.dataloader.shuffle = True
    trainer.args.data.dataloader.num_workers = workers
    trainer.args.data.dataloader.prefetch_factor = 1 if workers else None
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.args)
    torch.manual_seed(123)
    original, restored, restored_again = _loader(workers), _loader(workers), _loader(workers)
    try:
        original.set_epoch(0)
        iterator = iter(original)
        _trace(iterator, 4)
        cursor = original.state_dict()
        rng = torch.get_rng_state()
        assert cursor["_veomni_resume_phase"] == "flushing"
        trainer.base.train_dataloader = original
        protocol = trainer.dpo_provenance_checkpoint_state()
        expected_first = _trace(iterator, 1)
        before_save = torch.get_rng_state().clone()
        assert original.state_dict()["dataloader_state"]["_iterator_finished"] is True
        assert torch.equal(before_save, torch.get_rng_state())
        expected = [_trace(iterator, 1)]
        for epoch in (1, 2):
            original.set_epoch(epoch)
            expected.append(_trace(iter(original), 6))

        trainer.base.train_dataloader = restored
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": 4,
            "torch_rng_state": rng,
            "dpo_position": {"epoch": 0, "step_in_epoch": 4, "epoch_complete": False},
            "train_dataloader": cursor,
            "dpo_provenance_protocol": protocol,
        }
        callback.load_global_state()
        restored.set_epoch(trainer.start_epoch)
        iterator = iter(restored)
        assert _trace(iterator, 1) == expected_first
        before_save = torch.get_rng_state().clone()
        second_cursor = restored.state_dict()
        second_rng = torch.get_rng_state().clone()
        assert second_cursor["dataloader_state"]["_iterator_finished"] is True
        assert torch.equal(before_save, second_rng)
        actual = [_trace(iterator, 1)]
        for epoch in (1, 2):
            restored.set_epoch(epoch)
            actual.append(_trace(iter(restored), 6))
        assert actual == expected

        trainer.base.train_dataloader = restored_again
        callback._read_current = lambda _: {
            "global_step": 5,
            "torch_rng_state": second_rng,
            "dpo_position": {"epoch": 0, "step_in_epoch": 5, "epoch_complete": False},
            "train_dataloader": second_cursor,
            "dpo_provenance_protocol": protocol,
        }
        callback.load_global_state()
        restored_again.set_epoch(trainer.start_epoch)
        actual = [_trace(iter(restored_again), 1)]
        for epoch in (1, 2):
            restored_again.set_epoch(epoch)
            actual.append(_trace(iter(restored_again), 6))
        assert actual == expected
    finally:
        _close(original)
        _close(restored)
        _close(restored_again)


@pytest.mark.parametrize("workers", [0, 2])
def test_flushing_cursor_requires_proof_that_source_was_consumed(workers):
    loader = _loader(workers)
    try:
        next(iter(loader))
        cursor = loader.state_dict()
        assert cursor["_veomni_resume_phase"] == "after_yield"
        cursor["_veomni_resume_phase"] = "flushing"
        cursor["_veomni_flush_micro_batch"] = [{"id": 1}]
        with pytest.raises(ValueError, match="cannot prove that its source was exhausted"):
            loader.validate_resume_state(cursor)
    finally:
        _close(loader)

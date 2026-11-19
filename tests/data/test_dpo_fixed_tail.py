# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Fixed-batch DPO must not silently discard a short final optimizer batch."""

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.data.test_dpo_provenance_checkpoint import _trainer
from veomni.data.data_loader import DistributedDataloader
from veomni.data.dataset import MappingDataset
from veomni.data.dpo_collator import DPOMicroBatchCollator
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.text_dpo_trainer import _dpo_fixed_batch_steps, _dpo_recovery_geometry, _dpo_training_iterator


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _ids(features):
    return [feature["id"] for feature in features]


@pytest.mark.parametrize("microbatches", [1, 2, 4])
def test_incomplete_fixed_dpo_batch_keeps_every_pair(microbatches):
    dataset = MappingDataset([[{"id": index}] for index in range(5)])
    results = []
    for rank in (0, 1):
        sampler = StatefulDistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=False)
        loader = DistributedDataloader(
            dataset,
            sampler=sampler,
            batch_size=2,
            drop_last=False,
            num_workers=0,
            collate_fn=DPOMicroBatchCollator(microbatches, _ids),
        )
        batches = list(loader)
        assert len(batches) == _dpo_fixed_batch_steps(5, 2, 2) == 2
        assert all(group for batch in batches for group in batch)
        assert len(batches[-1]) == 1
        results.append([value for batch in batches for group in batch for value in group])
    # The native distributed sampler pads 5 samples to 6; no extra DPO
    # microbatch padding or replay is introduced by the collator.
    assert results == [[0, 2, 4], [1, 3, 0]]


@pytest.mark.parametrize("samples,dp,batch", [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
def test_fixed_dpo_invalid_geometry_fails_closed(samples, dp, batch):
    with pytest.raises(ValueError, match="nonempty data and positive"):
        _dpo_fixed_batch_steps(samples, dp, batch)


class _Samples(list):
    pass


def _transform(item):
    return [item]


def _fixed_loader(workers, rank):
    samples = _Samples({"id": index} for index in range(5))
    samples._fingerprint = hashlib.sha256(repr(samples).encode()).hexdigest()
    dataset = MappingDataset(samples, transform=_transform, seed=42)
    sampler = StatefulDistributedSampler(dataset, num_replicas=2, rank=rank, shuffle=True, seed=42)
    options = {"multiprocessing_context": "fork", "prefetch_factor": 1} if workers else {}
    return DistributedDataloader(
        dataset,
        sampler=sampler,
        batch_size=2,
        drop_last=False,
        num_workers=workers,
        collate_fn=DPOMicroBatchCollator(4, _ids),
        snapshot_every_n_steps=1,
        **options,
    )


def _step(iterator):
    return next(iterator), float(torch.rand(()))


@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("at_boundary", [False, True])
def test_fixed_tail_checkpoint_preserves_next_epochs(workers, rank, at_boundary):
    trainer = _trainer(True)
    trainer.args.train.dyn_bsz = False
    trainer.args.train.num_train_epochs = 3
    trainer.args.train.dataloader_batch_size = 2
    trainer.args.data.dataloader.drop_last = False
    trainer.args.data.dataloader.shuffle = True
    trainer.args.data.dataloader.num_workers = workers
    trainer.args.data.dataloader.prefetch_factor = 1 if workers else None
    trainer.args.train_steps = _dpo_fixed_batch_steps(5, 2, 2)
    trainer.policy_model.parallel_state = SimpleNamespace(dp_size=2, dp_rank=rank)
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    trainer.base.channel_loss_callback = None
    trainer.base.offline_cache_writer = None
    trainer.base.environ_meter = None
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.args)
    torch.manual_seed(123)
    original, restored = _fixed_loader(workers, rank), _fixed_loader(workers, rank)
    try:
        original.set_epoch(0)
        iterator = iter(original)
        assert len(_step(iterator)[0]) == 2
        step = 1
        if at_boundary:
            assert len(_step(iterator)[0]) == 1
            step = 2
        cursor, rng = original.state_dict(), torch.get_rng_state()
        trainer.base.train_dataloader = original
        protocol = trainer.dpo_provenance_checkpoint_state()
        expected = [] if at_boundary else [_step(iterator)]
        for epoch in (1, 2):
            original.set_epoch(epoch)
            iterator = iter(original)
            expected.extend(_step(iterator) for _ in range(2))

        trainer.base.train_dataloader = restored
        callback = GlobalStateCallback.__new__(GlobalStateCallback)
        callback.trainer = trainer
        callback._read_current = lambda _: {
            "global_step": step,
            "torch_rng_state": rng,
            "dpo_position": {"epoch": 0, "step_in_epoch": step, "epoch_complete": False},
            "train_dataloader": cursor,
            "dpo_provenance_protocol": protocol,
        }
        callback.load_global_state()
        actual = []
        if not at_boundary:
            restored.set_epoch(trainer.start_epoch)
            iterator = _dpo_training_iterator(restored, False, trainer._dpo_restore_fixed_iterator_rng)
            actual.append(_step(iterator))
        for epoch in (1, 2):
            restored.set_epoch(epoch)
            iterator = iter(restored)
            actual.extend(_step(iterator) for _ in range(2))
        assert actual == expected
    finally:
        for loader in (original, restored):
            iterator = getattr(loader, "_iterator", None)
            if iterator is not None and hasattr(iterator, "_shutdown_workers"):
                iterator._shutdown_workers()

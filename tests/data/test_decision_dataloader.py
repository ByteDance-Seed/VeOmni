"""Native map-style decision loader ordering, padding, workers, and resume."""

import copy
import math
import os
import random
import threading
from functools import partial
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.decision_arguments import VeOmniDecisionArguments
from veomni.arguments.parser import _instantiate_recursive
from veomni.data.data_loader import DistributedDataloader
from veomni.data.decision.collator import DecisionCollator
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.base import VeOmniIter
from veomni.trainer.decision_trainer import DecisionTrainer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _expected(size, epoch, rank, world=2, global_batch_size=4):
    order = list(range(size))
    random.Random(f"13:{epoch}").shuffle(order)
    order += [None] * (-len(order) % global_batch_size)
    return [
        order[start + rank : start + global_batch_size : world] for start in range(0, len(order), global_batch_size)
    ]


def _encode(case, *, training):
    assert training
    length = 2 + case % 3
    return {
        "input_ids": torch.full((length,), case, dtype=torch.long),
        "attention_mask": torch.ones(length, dtype=torch.long),
        "decision_meta": {},
        "decision_supervision": {
            "counts": torch.tensor([2]),
            "targets": torch.tensor([[1.0, 0.0]]),
            "weights": torch.tensor([1.0]),
            "ordinal_mask": torch.tensor([False]),
            "keys": (("a", "b"),),
            "question_keys": ("q",),
        },
    }


def _trainer(rank=0, size=11, epoch=2, workers=0, dynamic=False, runtime="main", micro_batch_size=1):
    trainer = object.__new__(DecisionTrainer)
    with patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": str(rank)}):
        trainer.args = _instantiate_recursive(
            VeOmniDecisionArguments,
            {
                "model": {"model_path": "/fixture"},
                "train": {
                    "global_batch_size": 4,
                    "micro_batch_size": micro_batch_size,
                    "seed": 13,
                    "dyn_bsz": dynamic,
                    "dyn_bsz_runtime": runtime,
                },
                "data": {
                    "max_seq_len": 8,
                    "train_size": 128,
                    "dyn_bsz_buffer_size": 1,
                    "dataloader": {
                        "num_workers": workers,
                        "pin_memory": False,
                        "prefetch_factor": 2 if workers else None,
                    },
                },
            },
        )
    trainer.model = SimpleNamespace(
        parallel_state=SimpleNamespace(dp_size=2, dp_rank=rank, sp_size=1, sp_enabled=False),
        unwrapped_module=SimpleNamespace(),
    )
    trainer.decision_data = SimpleNamespace(
        train_cases=[SimpleNamespace(case_id=str(i), request=i) for i in range(size)],
        padding_case=SimpleNamespace(case_id="0", request=0),
    )
    trainer.steps_per_epoch = math.ceil(size / 4)
    trainer._prepare_data = lambda: None
    trainer.data_transform = partial(_encode, training=True)
    trainer._build_dataset()
    trainer._build_collate_fn()
    with patch("veomni.data.data_loader.get_parallel_state", return_value=trainer.model.parallel_state):
        trainer._build_dataloader()
    trainer.train_dataset.set_epoch(epoch)
    trainer.train_dataloader.set_epoch(epoch)
    return trainer


def _ids(batches):
    result = []
    for batch in batches:
        supervision = batch["decision_supervision"]
        offset = 0
        for index, (length, padding) in enumerate(
            zip(batch["decision_request_lengths"], supervision["padding"], strict=True)
        ):
            assert supervision["weights"][index].item() == (0.0 if padding else 1.0)
            result.append(None if padding else batch["input_ids"][0, offset].item())
            offset += length
    return result


@pytest.mark.parametrize("size", [1, 7, 8, 11])
def test_native_decision_loader_partitions_cases_and_pads_tail(size):
    seen = []
    for rank in range(2):
        trainer = _trainer(rank, size)
        assert type(trainer.train_dataloader) is DistributedDataloader
        batches = [_ids(batch) for batch in trainer.train_dataloader]
        assert batches == _expected(size, 2, rank)
        seen.extend(case for batch in batches for case in batch if case is not None)
    assert sorted(seen) == list(range(size))


@pytest.mark.parametrize("consumed", [0, 1, 3])
def test_native_decision_loader_restores_remaining_batches_and_next_epoch(consumed):
    trainer = _trainer()
    iterator = iter(trainer.train_dataloader)
    prefix = [_ids(next(iterator)) for _ in range(consumed)]
    snapshot = copy.deepcopy(trainer.train_dataloader.state_dict())
    resumed = _trainer(epoch=0)
    resumed.train_dataloader.load_state_dict(snapshot)
    assert prefix + [_ids(batch) for batch in resumed.train_dataloader] == _expected(11, 2, 0)
    resumed.train_dataloader.set_epoch(3)
    assert [_ids(batch) for batch in resumed.train_dataloader] == _expected(11, 3, 0)


def test_decision_loader_resume_uses_consumed_prefetch_snapshot():
    trainer = _trainer()
    collate = trainer.train_dataloader.collate_fn
    prefetched = threading.Event()
    calls = 0

    def collate_with_signal(cases):
        nonlocal calls
        result = collate(cases)
        calls += 1
        if calls >= 2:
            prefetched.set()
        return result

    trainer.train_dataloader.collate_fn = collate_with_signal
    iterator = VeOmniIter(trainer.train_dataloader, use_background_prefetcher=True)
    try:
        first = _ids(next(iterator))
        assert prefetched.wait(timeout=5)
        snapshot = copy.deepcopy(iterator.state_dict())
    finally:
        iterator.stop()
    resumed = _trainer()
    resumed.train_dataloader.load_state_dict(snapshot)
    assert [first] + [_ids(batch) for batch in resumed.train_dataloader] == _expected(11, 2, 0)


def test_decision_collator_does_not_mutate_padding_case():
    trainer = _trainer()
    cached = _encode(0, training=True)
    trainer.collate_fn.padding_case = cached
    batch = trainer.collate_fn([cached, None])
    assert cached["decision_supervision"]["weights"].item() == 1.0
    assert batch["decision_supervision"]["weights"].tolist() == [1.0, 0.0]
    assert batch["decision_supervision"]["padding"] == (False, True)


def test_worker_preprocessing_keeps_order_and_resumes():
    trainer = _trainer(workers=2)
    iterator = iter(trainer.train_dataloader)
    first = _ids(next(iterator))
    snapshot = copy.deepcopy(trainer.train_dataloader.state_dict())
    remaining = [_ids(batch) for batch in iterator]
    resumed = _trainer(workers=2, epoch=0)
    resumed.train_dataloader.load_state_dict(snapshot)
    assert [_ids(batch) for batch in resumed.train_dataloader] == remaining
    assert [first, *remaining] == _expected(11, 2, 0)


@pytest.mark.parametrize("runtime,workers", [("main", 0), ("main", 2), ("worker", 0), ("worker", 2)])
def test_dynamic_decision_loader_packs_and_restores_batches(runtime, workers):
    trainer = _trainer(size=101, dynamic=True, runtime=runtime, workers=workers)
    if runtime == "main":
        assert isinstance(trainer.train_dataloader, DynamicBatchSizeDataLoader)
        assert isinstance(trainer.train_dataloader.batching_strategy, TextBatchingStrategy)
    iterator = iter(trainer.train_dataloader)
    first = next(iterator)
    snapshot = copy.deepcopy(trainer.train_dataloader.state_dict())
    remaining = [next(iterator) for _ in range(trainer.train_steps - 1)]
    for step in [first, *remaining]:
        assert len(step) == 2
        for batch in step:
            assert batch["input_ids"].numel() <= 8
            assert batch["cu_seq_lens_q"].diff().tolist() == list(batch["decision_request_lengths"])
    assert any(len(batch["decision_request_lengths"]) > 1 for batch in first)
    resumed = _trainer(size=101, dynamic=True, runtime=runtime, workers=workers, epoch=0)
    resumed.train_dataloader.load_state_dict(snapshot)
    resumed_iterator = iter(resumed.train_dataloader)
    actual = [next(resumed_iterator) for _ in remaining]
    assert [_ids(step) for step in actual] == [_ids(step) for step in remaining]


def test_fixed_decision_loader_packs_multiple_requests_and_keeps_zero_weight_tail():
    for rank in range(2):
        trainer = _trainer(rank=rank, size=7, micro_batch_size=2)
        steps = list(trainer.train_dataloader)
        assert all(len(step) == 1 for step in steps)
        assert [_ids(step) for step in steps] == _expected(7, 2, rank)


def test_decision_collator_aligns_unequal_option_counts():
    first, second = _encode(1, training=True), _encode(2, training=True)
    second["decision_supervision"].update(
        counts=torch.tensor([3]),
        targets=torch.tensor([[0.2, 0.3, 0.5]]),
        keys=(("x", "y", "z"),),
        question_keys=("other",),
        ordinal_mask=torch.tensor([True]),
    )
    batch = DecisionCollator()([first, second])
    supervision = batch["decision_supervision"]
    torch.testing.assert_close(supervision["targets"], torch.tensor([[1.0, 0.0, 0.0], [0.2, 0.3, 0.5]]))
    assert supervision["keys"] == (("a", "b"), ("x", "y", "z"))
    assert supervision["ordinal_mask"].tolist() == [False, True]
    assert batch["position_ids"].tolist() == [[0, 1, 2, 0, 1, 2, 3]]
    assert first["decision_supervision"]["targets"].shape == (1, 2)

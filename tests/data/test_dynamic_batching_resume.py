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


def _loader():
    return DynamicBatchSizeDataLoader(
        _StatefulSamples(),
        TextBatchingStrategy(token_micro_bsz=4, buffer_size=1, bsz_warmup_steps=3, bsz_warmup_init_mbtoken=1),
        num_micro_batch=1,
        length=10,
    )


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


def test_empty_source_fails_with_actionable_restart_error():
    loader = DynamicBatchSizeDataLoader(iter(()), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=1)
    with pytest.raises(RuntimeError, match="exhausted.*could not restart"):
        next(iter(loader))


def test_exhausted_iterator_fails_before_replaying_a_sample():
    sample = {"id": 7, "attention_mask": torch.ones(1, dtype=torch.int64)}
    loader = DynamicBatchSizeDataLoader(
        iter([sample]), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=2
    )
    iterator = iter(loader)
    assert [item["id"] for item in next(iterator)[0]] == [7]
    with pytest.raises(RuntimeError, match="exhausted.*could not restart"):
        next(iterator)


def test_restartable_finite_source_still_repeats_until_requested_length():
    samples = [{"id": 7, "attention_mask": torch.ones(1, dtype=torch.int64)}]
    loader = DynamicBatchSizeDataLoader(samples, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=2)
    iterator = iter(loader)
    assert [[item["id"] for item in next(iterator)[0]] for _ in range(2)] == [[7], [7]]


def test_exhausted_source_drains_ready_buffer_before_error():
    samples = [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3)]
    loader = DynamicBatchSizeDataLoader(
        iter([samples]), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3
    )
    iterator = iter(loader)
    assert [[item["id"] for item in next(iterator)[0]] for _ in range(3)] == [[1], [2], [3]]


def test_resume_exhausted_source_drains_the_same_buffered_samples():
    class OneBatch(_StatefulSamples):
        def __next__(self):
            if self.cursor:
                raise StopIteration
            self.cursor = 1
            return [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3)]

    def make_loader():
        return DynamicBatchSizeDataLoader(OneBatch(), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3)

    original = make_loader()
    iterator = iter(original)
    assert [item["id"] for item in next(iterator)[0]] == [1]
    state = original.state_dict()
    expected = [[item["id"] for item in next(iterator)[0]] for _ in range(2)]
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = [[item["id"] for item in next(iter(resumed))[0]]]
    actual.extend([[item["id"] for item in next(resumed)[0]]])
    assert actual == expected == [[2], [3]]


def test_exhausted_source_resume_preserves_gradient_accumulation_groups():
    class OneBatch(_StatefulSamples):
        def __next__(self):
            if self.cursor:
                raise StopIteration
            self.cursor = 1
            return [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3, 4)]

    def make_loader():
        return DynamicBatchSizeDataLoader(
            OneBatch(), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), num_micro_batch=2, length=2
        )

    original = make_loader()
    iterator = iter(original)
    first = [[item["id"] for item in microbatch] for microbatch in next(iterator)]
    state = original.state_dict()
    expected = [[item["id"] for item in microbatch] for microbatch in next(iterator)]
    resumed = make_loader()
    resumed.load_state_dict(state)
    actual = [[item["id"] for item in microbatch] for microbatch in next(iter(resumed))]
    assert first == [[1], [2]]
    assert actual == expected == [[3], [4]]


def test_exhausted_source_does_not_repeat_a_stateful_readiness_check():
    class ReservingStrategy:
        def __init__(self):
            self.pending = []
            self.reserved = None

        def put_item(self, item):
            self.pending.append(item)

        def is_ready_for_micro_batch(self):
            if self.pending:
                self.reserved = self.pending.pop(0)
                return True
            return False

        def get_micro_batch(self, step):
            return [self.reserved]

        def empty(self):
            return not self.pending

    samples = [{"id": i} for i in (1, 2, 3)]
    loader = DynamicBatchSizeDataLoader(iter([samples]), ReservingStrategy(), length=3)
    iterator = iter(loader)
    assert [[item["id"] for item in next(iterator)[0]] for _ in range(3)] == [[1], [2], [3]]


def test_exhausted_source_with_unsupported_tail_fails_closed():
    samples = [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3)]
    loader = DynamicBatchSizeDataLoader(
        iter([samples]), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3, drop_last=False
    )
    iterator = iter(loader)
    assert [item["id"] for item in next(iterator)[0]] == [1]
    with pytest.raises(RuntimeError, match="exhausted.*could not restart"):
        next(iterator)


def test_exhausted_source_is_not_reopened_for_each_buffered_microbatch():
    class OneBatch(_StatefulSamples):
        def __init__(self):
            super().__init__()
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            return self

        def __next__(self):
            if self.cursor:
                raise StopIteration
            self.cursor = 1
            return [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in range(10)]

    source = OneBatch()
    loader = DynamicBatchSizeDataLoader(source, TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=10)
    iterator = iter(loader)
    assert [[item["id"] for item in next(iterator)[0]] for _ in range(10)] == [[i] for i in range(10)]
    assert source.iter_calls == 2


def test_resume_does_not_reopen_exhausted_one_shot_source():
    class ClosedAfterRestart(_StatefulSamples):
        def __init__(self):
            super().__init__()
            self.iter_calls = 0

        def __iter__(self):
            self.iter_calls += 1
            if self.iter_calls > 2:
                raise RuntimeError("source already closed")
            return self

        def __next__(self):
            if self.cursor:
                raise StopIteration
            self.cursor = 1
            return [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3)]

        def state_dict(self):
            return {"cursor": self.cursor, "iter_calls": self.iter_calls}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]
            self.iter_calls = state["iter_calls"]

    def make_loader():
        return DynamicBatchSizeDataLoader(
            ClosedAfterRestart(), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3
        )

    original = make_loader()
    iterator = iter(original)
    assert [item["id"] for item in next(iterator)[0]] == [1]
    assert [item["id"] for item in next(iterator)[0]] == [2]
    state = original.state_dict()
    assert state["_veomni_source_exhausted"] is True
    resumed = make_loader()
    resumed.load_state_dict(state)
    assert [item["id"] for item in next(iter(resumed))[0]] == [3]


def test_checkpoint_rejects_exhausted_source_without_state_protocol():
    samples = [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in (1, 2, 3)]
    loader = DynamicBatchSizeDataLoader(
        iter([samples]), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3
    )
    iterator = iter(loader)
    assert [item["id"] for item in next(iterator)[0]] == [1]
    assert [item["id"] for item in next(iterator)[0]] == [2]
    with pytest.raises(RuntimeError, match="exhausted source.*state_dict.*load_state_dict"):
        loader.state_dict()


def test_exhausted_source_resume_preserves_next_epoch_iterator_state():
    class EpochSource:
        def __init__(self):
            self.calls = 0

        def __iter__(self):
            self.calls += 1
            ids = {1: [1, 2, 3], 2: [], 3: [100], 4: [200]}.get(self.calls, [])
            return iter([[{"id": i, "attention_mask": torch.ones(1)} for i in ids]] if ids else [])

        def state_dict(self):
            return {"calls": self.calls}

        def load_state_dict(self, state):
            self.calls = state["calls"]

    def make_loader():
        return DynamicBatchSizeDataLoader(
            EpochSource(), TextBatchingStrategy(token_micro_bsz=1, buffer_size=1), length=3
        )

    def value(iterator):
        return next(iterator)[0][0]["id"]

    original = make_loader()
    iterator = iter(original)
    assert [value(iterator), value(iterator)] == [1, 2]
    state = original.state_dict()
    assert state["_veomni_source_exhausted"] is True
    resumed = make_loader()
    resumed.load_state_dict(state)
    assert value(iter(resumed)) == value(iterator) == 3
    assert value(iter(original)) == value(iter(resumed)) == 100

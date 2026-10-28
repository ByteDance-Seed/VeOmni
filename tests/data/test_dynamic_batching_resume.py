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


@pytest.mark.parametrize("dpo_recovery", [False, True])
def test_restore_copies_tensor_buffer_only_for_dpo(dpo_recovery):
    original = _loader()
    next(iter(original))
    state = original.state_dict()
    marker = torch.arange(8)
    state["payload_marker"] = marker
    resumed = _loader()
    resumed._dpo_disallow_implicit_restart = dpo_recovery

    resumed.load_state_dict(state)

    assert torch.equal(resumed.payload_marker, marker)
    assert (resumed.payload_marker.data_ptr() != marker.data_ptr()) is dpo_recovery
    assert state["_veomni_resume_phase"] == "after_yield"


class _FiniteSamples(_StatefulSamples):
    def __init__(self, count):
        super().__init__()
        self.count = count

    def __next__(self):
        if self.cursor >= self.count:
            raise StopIteration
        return super().__next__()


def _finite_dpo_loader(count, *, buffer_size=1, num_micro_batch=1, drop_last=True):
    loader = DynamicBatchSizeDataLoader(
        _FiniteSamples(count),
        TextBatchingStrategy(token_micro_bsz=1, buffer_size=buffer_size),
        num_micro_batch=num_micro_batch,
        length=5,
        drop_last=drop_last,
    )
    loader._dpo_disallow_implicit_restart = True
    return loader


def test_dpo_drains_buffer_before_eof_and_resumes_during_flush():
    loader = _finite_dpo_loader(4, buffer_size=3)
    iterator = iter(loader)
    assert [[sample["id"] for sample in next(iterator)[0]] for _ in range(3)] == [[1], [2], [3]]
    checkpoint = loader.state_dict()
    assert checkpoint["_veomni_resume_phase"] == "flushing"
    assert [sample["id"] for sample in next(iterator)[0]] == [4]
    with pytest.raises(RuntimeError, match="implicit data replay is disabled"):
        next(iterator)

    resumed = _finite_dpo_loader(4, buffer_size=3)
    resumed.load_state_dict(checkpoint)
    resumed_iterator = iter(resumed)
    assert [sample["id"] for sample in next(resumed_iterator)[0]] == [4]
    with pytest.raises(RuntimeError, match="implicit data replay is disabled"):
        next(resumed_iterator)


@pytest.mark.parametrize("drop_last", [False, True])
def test_dpo_partial_final_step_never_replays_or_pads(drop_last):
    loader = _finite_dpo_loader(3, num_micro_batch=2, drop_last=drop_last)
    iterator = iter(loader)
    assert [[sample["id"] for sample in microbatch] for microbatch in next(iterator)] == [[1], [2]]
    with pytest.raises(RuntimeError, match="implicit data replay is disabled|neither is supported"):
        next(iterator)


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


def test_dpo_dynamic_loader_rejects_implicit_dataset_replay():
    source = [{"id": 1, "attention_mask": torch.ones(1, dtype=torch.int64)}]

    def make_loader(disallow_restart):
        loader = DynamicBatchSizeDataLoader(
            source,
            TextBatchingStrategy(token_micro_bsz=1, buffer_size=1),
            collate_fn=lambda samples: {"ids": [sample["id"] for sample in samples]},
            length=2,
        )
        loader._dpo_disallow_implicit_restart = disallow_restart
        return iter(loader)

    regular = make_loader(False)
    assert next(regular) == [{"ids": [1]}]
    assert next(regular) == [{"ids": [1]}]

    dpo_loader = make_loader(True)
    assert next(dpo_loader) == [{"ids": [1]}]
    with pytest.raises(RuntimeError, match="implicit data replay is disabled"):
        next(dpo_loader)


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

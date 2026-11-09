# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Deterministic cyclic packing, commit boundaries and cold replay."""

import copy
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.cyclic_packing import CyclicPackedLoader


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CONTRACT = "sha256:" + "a" * 64


class Samples:
    def __init__(self, lengths=(3, 2, 4, 1, 3, 2, 1, 4, 1, 2, 3)):
        self.rows = [
            {
                "input_ids": torch.full((length,), index + 1, dtype=torch.long),
                "labels": torch.full((length,), index + 1, dtype=torch.long),
                "loss_weights": torch.full((length,), 0.25 + (index % 4) * 0.25),
                "sample_id": index,
                "pixel_values": torch.full((index % 3 + 1, 4), index, dtype=torch.float32),
            }
            for index, length in enumerate(lengths)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        # Deliberately return cached tensors; the collator must not mutate them.
        return [self.rows[index]]


def collate(features):
    # Mimic the legacy packing collator's in-place boundary masking.
    for feature in features[1:]:
        feature["labels"][0] = -100
    return {
        "ids": [feature["sample_id"] for feature in features],
        "tokens": torch.cat([feature["input_ids"] for feature in features]),
        "labels": torch.cat([feature["labels"] for feature in features]),
        "weights": torch.cat([feature["loss_weights"] for feature in features]),
        "pixels": torch.cat([feature["pixel_values"] for feature in features]),
    }


collate.supports_stateless_cyclic_replay = True
collate.checkpoint_collator_identity = "fixture-collator-v1"


def loader(dataset, step, **overrides):
    collator = overrides.pop("collate_fn", collate)
    config = dict(
        max_seq_len=5,
        micro_batches_per_step=3,
        buffer_size=3,
        max_buffer_bytes=320,
        input_contract=_CONTRACT,
        source_fingerprint="immutable-fixture-v1",
        deterministic=True,
        train_steps=8,
        train_epochs=1,
        sp_size=1,
        step_getter=lambda: step.value,
    )
    config.update(overrides)
    return CyclicPackedLoader(dataset, collator, **config)


def advance(data, step):
    batch = next(data)
    step.value += 1
    data.commit_step()
    return batch


def compact(batch):
    return [
        {key: value.tolist() if isinstance(value, torch.Tensor) else value for key, value in micro.items()}
        for micro in batch
    ]


def test_multiple_cycles_emit_exact_coverage_and_short_tail():
    dataset, step = Samples(), SimpleNamespace(value=0)
    data = loader(dataset, step)
    cycles = []
    previous = data._cycle
    seen = []
    for _ in range(8):
        batch = advance(data, step)
        if data._cycle != previous:
            cycles.append(seen)
            seen = []
            previous = data._cycle
        seen.extend(index for group in batch for index in group["ids"])
        assert all(len(group["tokens"]) <= 5 for group in batch)
    assert len(cycles) >= 2
    for seen_cycle in cycles:
        assert sorted(seen_cycle) == list(range(len(dataset)))
        assert len(seen_cycle) == len(set(seen_cycle))
    assert all(torch.equal(row["labels"], row["input_ids"]) for row in dataset.rows)


def test_packing_budget_is_independent_of_backing_storage_capacity():
    compact_rows, padded_rows = Samples(), Samples()
    for row in padded_rows.rows:
        for key, value in row.items():
            if isinstance(value, torch.Tensor):
                backing = torch.empty(value.numel() + 256, dtype=value.dtype)
                backing[: value.numel()].copy_(value.reshape(-1))
                row[key] = backing[: value.numel()].reshape(value.shape)
    compact_step, padded_step = SimpleNamespace(value=0), SimpleNamespace(value=0)
    compact_loader = loader(compact_rows, compact_step)
    padded_loader = loader(padded_rows, padded_step)
    for _ in range(3):
        assert compact(advance(compact_loader, compact_step)) == compact(advance(padded_loader, padded_step))
        assert compact_loader.state_dict() == padded_loader.state_dict()


def test_cyclic_collator_ragged_fallback_does_not_alias_source(monkeypatch):
    from veomni.data import data_collator
    from veomni.data.cyclic_packing import _StatelessMainCollator

    monkeypatch.setattr(data_collator, "get_parallel_state", lambda: SimpleNamespace(sp_enabled=False))
    samples = Samples(lengths=(3, 2, 3, 2))
    backing = torch.arange(10, dtype=torch.float32)
    for index, row in enumerate(samples.rows):
        row.pop("loss_weights")
        row["extra_metadata"] = {"nested": [backing[: index + 2]]}
    original = backing.clone()
    data = loader(samples, SimpleNamespace(value=0), collate_fn=_StatelessMainCollator(data_collator.MainCollator()))
    batches = next(data)
    first = batches[0]["extra_metadata"][0]["nested"][0]
    other = batches[1]["extra_metadata"][0]["nested"][0]
    other_before = other.clone()
    first.add_(100)
    assert torch.equal(backing, original)
    assert torch.equal(other, other_before)


def test_uncommitted_or_inconsistent_step_cannot_be_saved_or_advanced():
    step = SimpleNamespace(value=0)
    data = loader(Samples(), step)
    next(data)
    with pytest.raises(RuntimeError, match="uncommitted"):
        data.state_dict()
    with pytest.raises(RuntimeError, match="optimizer step"):
        data.commit_step()
    with pytest.raises(RuntimeError, match="uncommitted"):
        data.state_dict()


@pytest.mark.parametrize("checkpoint_step", (1, 2, 3, 4, 5, 6))
def test_replay_matches_continuation_across_cycle_boundaries(checkpoint_step):
    dataset, step = Samples(), SimpleNamespace(value=0)
    data = loader(dataset, step)
    for _ in range(checkpoint_step):
        advance(data, step)
    checkpoint = copy.deepcopy(data.state_dict())
    reference = [compact(advance(data, step)) for _ in range(8 - checkpoint_step)]

    restored_step = SimpleNamespace(value=checkpoint_step)
    restored = loader(Samples(), restored_step, require_resume=True)
    restored.load_state_dict(checkpoint)
    actual = [compact(advance(restored, restored_step)) for _ in range(8 - checkpoint_step)]
    assert actual == reference
    with pytest.raises(StopIteration):
        next(restored)


def test_changed_contract_or_buffered_feature_rejected():
    step = SimpleNamespace(value=0)
    data = loader(Samples(), step, micro_batches_per_step=1)
    advance(data, step)
    checkpoint = data.state_dict()

    wrong_contract = loader(Samples(), SimpleNamespace(value=1), input_contract="sha256:" + "b" * 64)
    with pytest.raises(ValueError, match="contract or geometry"):
        wrong_contract.load_state_dict(checkpoint)
    with pytest.raises(RuntimeError, match="uncommitted"):
        wrong_contract.state_dict()

    changed = Samples()
    # Last item loaded into the lookahead/buffer is not part of the first group.
    changed.rows[2]["pixel_values"].add_(1)
    restored = loader(changed, SimpleNamespace(value=1), micro_batches_per_step=1, require_resume=True)
    with pytest.raises(ValueError, match="buffered frontier"):
        restored.load_state_dict(checkpoint)

    collate.checkpoint_collator_identity = "fixture-collator-v2"
    try:
        changed_collator = loader(Samples(), SimpleNamespace(value=1), micro_batches_per_step=1, require_resume=True)
    finally:
        collate.checkpoint_collator_identity = "fixture-collator-v1"
    with pytest.raises(ValueError, match="contract or geometry"):
        changed_collator.load_state_dict(checkpoint)


def test_unattested_random_or_legacy_inputs_fail_closed():
    with pytest.raises(ValueError, match="deterministic-input"):
        loader(Samples(), SimpleNamespace(value=0), deterministic=False)
    with pytest.raises(ValueError, match="sha256 input contract"):
        loader(Samples(), SimpleNamespace(value=0), input_contract="moving-path")
    missing = loader(Samples(), SimpleNamespace(value=0), require_resume=True)
    with pytest.raises(RuntimeError, match="matching loader checkpoint"):
        next(missing)
    legacy = loader(Samples(), SimpleNamespace(value=0), require_resume=True)
    with pytest.raises(ValueError, match="unsupported cyclic"):
        legacy.load_state_dict({"legacy": True})
    old_kind = loader(Samples(), SimpleNamespace(value=0)).state_dict()
    old_kind["kind"] = "veomni_cyclic_packing_v1"
    with pytest.raises(ValueError, match="unsupported cyclic"):
        loader(Samples(), SimpleNamespace(value=0), require_resume=True).load_state_dict(old_kind)

    class RandomSamples(Samples):
        def __getitem__(self, index):
            random.random()
            return super().__getitem__(index)

    with pytest.raises(ValueError, match="consume global RNG"):
        next(loader(RandomSamples(), SimpleNamespace(value=0)))


def test_source_row_with_multiple_features_is_rejected():
    from veomni.data.data_transform import process_plaintext_example

    class SplitRow(Samples):
        def __getitem__(self, index):
            tokenizer = SimpleNamespace(encode=lambda *_args, **_kwargs: list(range(10)), eos_token_id=0)
            return process_plaintext_example({"content_split": "long row"}, tokenizer, max_seq_len=5)

    with pytest.raises(ValueError, match="exactly one feature"):
        next(loader(SplitRow(), SimpleNamespace(value=0)))


def test_pin_memory_is_disabled_without_an_accelerator(monkeypatch):
    monkeypatch.setattr(torch.accelerator, "is_available", lambda: False)
    data = loader(Samples(), SimpleNamespace(value=0), pin=True)
    assert data._pin is False
    assert next(data)


def test_empty_supervision_cannot_advance_optimizer():
    samples = Samples()
    for row in samples.rows:
        row["labels"].fill_(-100)
    data = loader(samples, SimpleNamespace(value=0))
    with pytest.raises(ValueError, match="no supervised tokens"):
        next(data)
    with pytest.raises(RuntimeError, match="uncommitted"):
        data.state_dict()


@pytest.mark.parametrize("key", ("input_ids", "labels", "loss_weights", "position_ids", "image_mask"))
def test_rejects_misaligned_sequence_fields(key):
    samples = Samples()
    samples.rows[0][key] = torch.ones(2, 2)
    with pytest.raises(ValueError, match="one-dimensional|aligned"):
        next(loader(samples, SimpleNamespace(value=0)))


def test_rejects_collator_without_replay_contract():
    def unmarked(features):
        return features

    data = loader(Samples(), SimpleNamespace(value=0))
    with pytest.raises(ValueError, match="certified stateless collator"):
        CyclicPackedLoader(
            Samples(),
            unmarked,
            max_seq_len=5,
            micro_batches_per_step=1,
            buffer_size=2,
            max_buffer_bytes=1024,
            input_contract=_CONTRACT,
            source_fingerprint="fixture",
            deterministic=True,
            train_steps=1,
            train_epochs=1,
            sp_size=1,
            step_getter=lambda: 0,
        )
    assert data._committed_step == 0


def _one_update(data, step, weight, optimizer):
    batch = next(data)
    losses, weights = [], []
    for micro in batch:
        target = micro["labels"]
        valid = target.ne(-100)
        target = torch.where(valid, target - 1, target)
        logits = micro["tokens"].float().unsqueeze(-1) * weight.unsqueeze(0)
        losses.append((F.cross_entropy(logits, target, ignore_index=-100, reduction="none") * micro["weights"]).sum())
        weights.append(micro["weights"][valid].sum())
    loss = sum(losses) / sum(weights)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    step.value += 1
    data.commit_step()
    return float(loss.detach()), weight.detach().clone(), compact(batch)


def _fresh_process_resume(checkpoint_path, result_path):
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    step = SimpleNamespace(value=saved["step"])
    weight = torch.nn.Parameter(saved["weight"].clone())
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    optimizer.load_state_dict(saved["optimizer"])
    data = loader(Samples(), step, require_resume=True)
    data.load_state_dict(saved["loader"])
    loss, updated, batch = _one_update(data, step, weight, optimizer)
    torch.save({"loss": loss, "weight": updated, "batch": batch, "optimizer": optimizer.state_dict()}, result_path)


def test_new_process_resume_matches_next_weighted_update(tmp_path):
    step = SimpleNamespace(value=0)
    weight = torch.nn.Parameter(torch.linspace(-0.2, 0.3, 12))
    optimizer = torch.optim.AdamW([weight], lr=0.01)
    data = loader(Samples(), step)
    for _ in range(3):
        _one_update(data, step, weight, optimizer)
    checkpoint = tmp_path / "checkpoint.pt"
    result_path = tmp_path / "result.pt"
    torch.save(
        {
            "step": step.value,
            "weight": weight.detach(),
            "optimizer": optimizer.state_dict(),
            "loader": data.state_dict(),
        },
        checkpoint,
    )
    loss, updated, batch = _one_update(data, step, weight, optimizer)
    command = (
        "from tests.data.test_cyclic_packing import _fresh_process_resume; "
        f"_fresh_process_resume({str(checkpoint)!r}, {str(result_path)!r})"
    )
    completed = subprocess.run([sys.executable, "-c", command], env=os.environ.copy(), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    actual = torch.load(result_path, map_location="cpu", weights_only=False)
    assert actual["loss"] == loss and torch.equal(actual["weight"], updated) and actual["batch"] == batch
    expected_state = optimizer.state_dict()["state"][0]
    for key, value in expected_state.items():
        torch.testing.assert_close(actual["optimizer"]["state"][0][key], value, rtol=0, atol=0)

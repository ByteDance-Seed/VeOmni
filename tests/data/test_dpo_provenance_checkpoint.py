# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DPO loader buffers must not silently cross data-semantics versions."""

import copy
import io
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.data.data_transform import dpo_producer_identity, process_dpo_example
from veomni.data.dataset import DynamicBatchingSizeDataset, get_length_fn
from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback
from veomni.trainer.text_dpo_trainer import TextDPOTrainer, _dpo_objective_parameters, _dpo_recovery_geometry
from veomni.utils.dpo_provenance import DPO_FEATURE_KEY, dpo_tokenizer_fingerprint


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _DeclaredTokenizer:
    dpo_unpadded_output_contract = True
    dpo_unpadded_output_contract_id = "checkpoint-test-tokenizer-v1"


def _trainer(enabled, allow_legacy=False, allow_legacy_objective=False):
    trainer = TextDPOTrainer.__new__(TextDPOTrainer)
    config = SimpleNamespace(
        explicit_segment_provenance=enabled,
        allow_legacy_data_resume=allow_legacy,
        allow_legacy_objective_resume=allow_legacy_objective,
        beta=0.1,
        label_smoothing=0.0,
        reference_free=False,
        loss_type="sigmoid",
        average_log_prob=False,
        refer_model_precision="bfloat16",
    )
    checkpoint = SimpleNamespace(load_path="/unused")
    train = SimpleNamespace(
        checkpoint=checkpoint,
        global_rank=0,
        world_size=1,
        micro_batch_size=1,
        global_batch_size=1,
        gradient_accumulation_steps=1,
        dataloader_batch_size=1,
        dyn_bsz=True,
        dyn_bsz_runtime="main",
        dyn_bsz_count_mode="total",
        dyn_bsz_physical_overflow_ratio=1.5,
        pad_to_length=False,
        bsz_warmup_ratio=0,
        bsz_warmup_init_mbtoken=200,
        seed=42,
        num_train_epochs=1,
        max_steps=None,
    )
    loader = SimpleNamespace(
        type="native",
        num_workers=0,
        worker_num_threads=None,
        prefetch_factor=None,
        persistent_workers=False,
        in_order=True,
        drop_last=True,
        shuffle=False,
        pin_memory=False,
        use_background_prefetcher=False,
    )
    data = SimpleNamespace(
        train_path="fixture",
        train_size=64,
        train_sample=1,
        dyn_bsz_buffer_size=1,
        max_seq_len=64,
        data_type="dpo",
        datasets_type="mapping",
        dataset_repeat=False,
        multisource_datasets_type="mapping",
        dataset_name="mapping",
        enable_multisource=False,
        text_keys="chosen",
        silent_exception=False,
        dataloader=loader,
    )
    accelerator = SimpleNamespace(
        dp_size=1,
        dp_replicate_size=1,
        dp_shard_size=1,
        ep_size=1,
        ep_outside=False,
        extra_parallel_sizes=[],
        extra_parallel_names=[],
        extra_parallel_placement_innermost=[],
        tp_size=1,
        pp_size=1,
        ulysses_size=1,
        cp_size=1,
    )
    args = SimpleNamespace(
        dpo_config=config, train=train, data=data, model=SimpleNamespace(accelerator=accelerator), train_steps=10
    )
    trainer.base = SimpleNamespace(
        args=args,
        train_dataloader=MagicMock(),
    )
    trainer.policy_model = SimpleNamespace(tokenizer=_DeclaredTokenizer(), chat_template=None)
    trainer._dpo_producer_identity = dpo_producer_identity(trainer.policy_model.tokenizer, None, None)
    trainer._dpo_transform_max_seq_len = 64
    trainer._dpo_objective_parameters = _dpo_objective_parameters(config)
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(args)
    trainer._dpo_epoch_step = 0
    trainer._dpo_epoch_complete = False
    return trainer


@pytest.mark.parametrize(
    "current,saved,allow_legacy,allow_legacy_objective,raises",
    [
        (False, None, False, False, True),
        (False, None, True, False, True),
        (False, None, True, True, True),
        (True, None, True, True, True),
        (True, {"version": 2, "enabled": True, "plaintext_transform_version": 1}, False, False, True),
        (True, {"version": 2, "enabled": True, "plaintext_transform_version": 1}, False, True, True),
        (False, {"version": 2, "enabled": False, "plaintext_transform_version": 1}, False, True, True),
        (False, {"version": 2, "enabled": True, "plaintext_transform_version": 1}, True, True, True),
        (True, {"version": 1, "enabled": True}, True, True, True),
        (True, {"version": 4, "enabled": True}, True, True, True),
        (True, {"version": 2, "enabled": 1, "plaintext_transform_version": 1}, True, True, True),
        (True, {"version": 2, "enabled": True, "plaintext_transform_version": 2}, True, True, True),
        (
            True,
            {
                "version": 3,
                "enabled": True,
                "plaintext_transform_version": 1,
                "objective_normalization": "global_pair_mean_v1",
            },
            False,
            False,
            True,
        ),
        (
            False,
            {
                "version": 3,
                "enabled": False,
                "plaintext_transform_version": 1,
                "objective_normalization": "global_pair_mean_v1",
            },
            False,
            False,
            True,
        ),
        (
            True,
            {
                "version": 5,
                "enabled": True,
                "plaintext_transform_version": 1,
                "conversation_transform_version": 1,
                "objective_normalization": "global_pair_mean_v1",
                "producer_contract_version": 1,
                "audited_unpadded_tokenizer_sha256": None,
                "producer_identity": dpo_producer_identity(_DeclaredTokenizer(), None, None),
                "transform_max_seq_len": 64,
                "objective_parameters": {
                    "beta": 0.1,
                    "label_smoothing": 0.0,
                    "reference_free": False,
                    "loss_type": "sigmoid",
                    "average_log_prob": False,
                    "refer_model_precision": "bfloat16",
                    "audited_unpadded_tokenizer_sha256": None,
                },
            },
            False,
            False,
            False,
        ),
        (
            True,
            {"version": 3, "enabled": True, "plaintext_transform_version": 1, "objective_normalization": "unknown"},
            True,
            True,
            True,
        ),
    ],
)
def test_checkpoint_mode_contract(current, saved, allow_legacy, allow_legacy_objective, raises):
    trainer = _trainer(current, allow_legacy, allow_legacy_objective)
    if not raises:
        saved = trainer.dpo_provenance_checkpoint_state()
    if raises:
        with pytest.raises(ValueError):
            trainer.validate_dpo_provenance_checkpoint(saved)
    else:
        trainer.validate_dpo_provenance_checkpoint(saved)


def test_checkpoint_rejects_a_different_audited_tokenizer():
    trainer = _trainer(True)
    trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256 = "a" * 64
    trainer._dpo_objective_parameters = _dpo_objective_parameters(trainer.base.args.dpo_config)
    saved = trainer.dpo_provenance_checkpoint_state()
    trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256 = "b" * 64
    trainer._dpo_objective_parameters = _dpo_objective_parameters(trainer.base.args.dpo_config)
    with pytest.raises(ValueError, match="provenance mode"):
        trainer.validate_dpo_provenance_checkpoint(saved)


@pytest.mark.parametrize(
    "dynamic,ending",
    [
        (False, "step"),
        (True, "step"),
        (True, "step_buffer"),
        (False, "early_eof"),
        (False, "within"),
        (True, "within"),
    ],
)
def test_dpo_epoch_boundary_resume_starts_fresh_sampler(monkeypatch, dynamic, ending):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    class Loader(StatefulDataLoader):
        def set_epoch(self, epoch):
            self.sampler.set_epoch(epoch)

    def build():
        data = [{"id": i, "attention_mask": torch.ones(1, dtype=torch.int64)} for i in range(6)]
        sampler = StatefulDistributedSampler(data, num_replicas=1, rank=0, shuffle=True, seed=11)
        loader = Loader(data, sampler=sampler, batch_size=1, num_workers=0, collate_fn=lambda x: x[0])
        if dynamic:
            return DynamicBatchSizeDataLoader(
                loader,
                TextBatchingStrategy(token_micro_bsz=1, buffer_size=2 if ending == "step_buffer" else 1),
                num_micro_batch=1,
                length=3,
            )
        return loader

    def sample_id(item):
        return item[0][0]["id"] if dynamic else item["id"]

    original = build()
    original.set_epoch(0)
    it = iter(original)
    n = {"step": 3, "step_buffer": 3, "early_eof": 6, "within": 2}[ending]
    for _ in range(n):
        next(it)
    if ending == "early_eof":
        with pytest.raises(StopIteration):
            next(it)
    cursor = copy.deepcopy(original.state_dict())
    if ending == "within":
        expected = [sample_id(next(it))]
    else:
        original.set_epoch(1)
        expected_iter = iter(original)
        expected = [sample_id(next(expected_iter)) for _ in range(3)]

    trainer = _trainer(True)
    trainer.base.args.train.num_train_epochs = 2
    trainer.base.args.train.dyn_bsz = dynamic
    trainer.base.args.train_steps = 10 if ending == "early_eof" else 3
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.base.args)
    trainer.base.train_dataloader = build()
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    checkpoint = {
        "global_step": n,
        "dpo_position": {"epoch": 0, "step_in_epoch": n, "epoch_complete": ending == "early_eof"},
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
        "train_dataloader": cursor,
    }
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: checkpoint
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    callback.load_global_state()
    expected_position = (0, 2) if ending == "within" else (1, 0)
    assert (trainer.start_epoch, trainer.start_step) == expected_position
    trainer.train_dataloader.set_epoch(expected_position[0])
    restored_iter = iter(trainer.train_dataloader)
    actual = [sample_id(next(restored_iter)) for _ in range(len(expected))]
    assert actual == expected


def test_dpo_epoch_end_refreshes_checkpoint_position_without_changing_sft_behavior():
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = _trainer(True)
    callback.dcp_every_n_epochs = 0
    callback.save_hf_weights = False
    callback._last_saved_step = 3
    callback.save_global_state = MagicMock()
    state = SimpleNamespace(epoch=0, global_step=3)
    callback.on_epoch_end(state)
    callback.save_global_state.assert_called_once_with(state)

    callback.trainer = SimpleNamespace()
    callback.save_global_state.reset_mock()
    callback.on_epoch_end(state)
    callback.save_global_state.assert_not_called()


@pytest.mark.parametrize("same_output", [False, True])
def test_finished_dpo_resume_preserves_position_at_train_end(monkeypatch, same_output):
    from torchdata.stateful_dataloader import StatefulDataLoader

    trainer = _trainer(True)
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    trainer.base.train_dataloader = StatefulDataLoader(list(range(6)), batch_size=1, num_workers=0)
    trainer.base.environ_meter = MagicMock()
    trainer.base.args.train.checkpoint.save_path = "/output/checkpoints"
    trainer.base.args.train.checkpoint.load_path = (
        "/output/checkpoints/global_step_2" if same_output else "/old/checkpoints/global_step_2"
    )
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.base.args)
    cursor = trainer.train_dataloader.state_dict()
    position = {"epoch": 0, "step_in_epoch": 2, "epoch_complete": True}
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 2,
        "dpo_position": position,
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    callback.load_global_state()
    assert (trainer.start_epoch, trainer.start_step) == (1, 0)
    assert (trainer.state.epoch, trainer._dpo_epoch_step, trainer._dpo_epoch_complete) == (0, 2, True)
    callback.save_hf_weights = True
    saved = []
    callback.save_global_state = lambda state: saved.append(callback.state_dict(state))
    callback.on_train_end(trainer.state)
    if same_output:
        assert not saved
    else:
        assert saved[0]["dpo_position"] == position
        assert saved[0]["train_dataloader"] is cursor


@pytest.mark.parametrize("dynamic", [False, True])
def test_pending_epoch_end_does_not_prefetch_the_previous_epoch(monkeypatch, dynamic):
    from torchdata.stateful_dataloader import StatefulDataLoader
    from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

    class Loader(StatefulDataLoader):
        def set_epoch(self, epoch):
            self.sampler.set_epoch(epoch)

    def build():
        data = [{"id": i} for i in range(12)]
        sampler = StatefulDistributedSampler(data, num_replicas=1, rank=0, shuffle=True, seed=11)
        loader = Loader(data, sampler=sampler, batch_size=1, num_workers=2, collate_fn=lambda x: x[0])
        if dynamic:
            return DynamicBatchSizeDataLoader(
                loader,
                TextBatchingStrategy(token_micro_bsz=1, buffer_size=2, get_length_fn=lambda _: 1),
                num_micro_batch=1,
                length=3,
            )
        return loader

    def sample_id(item):
        return item[0][0]["id"] if dynamic else item["id"]

    original = build()
    original.set_epoch(0)
    iterator = iter(original)
    for _ in range(3):
        next(iterator)
    cursor = copy.deepcopy(original.state_dict())
    original.set_epoch(1)
    iterator = iter(original)
    expected = [sample_id(next(iterator)) for _ in range(3)]

    trainer = _trainer(True)
    trainer.base.args.train.num_train_epochs = 2
    trainer.base.args.train_steps = 3
    trainer.base.args.train.dyn_bsz = dynamic
    trainer.base.args.data.dataloader.num_workers = 2
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.base.args)
    trainer.base.train_dataloader = build()
    trainer.base.state = SimpleNamespace(global_step=0, epoch=0)
    trainer.base.environ_meter = MagicMock()
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 3,
        "dpo_position": {"epoch": 0, "step_in_epoch": 3, "epoch_complete": False},
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
        "train_dataloader": cursor,
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    callback.load_global_state()
    assert trainer._dpo_pending_epoch_end == (0, 3)
    trainer._dpo_epoch_complete = True
    callback.dcp_every_n_epochs = 1
    callback.save_hf_weights = False
    saved_states = []
    callback.save_global_state = lambda state: saved_states.append(callback.state_dict(state))
    callback.on_epoch_end(trainer.state)
    assert saved_states[0]["dpo_position"]["epoch_complete"] is True
    trainer._dpo_pending_epoch_end = None
    trainer._dpo_pending_loader_state = None
    trainer.train_dataloader.set_epoch(1)
    resumed = iter(trainer.train_dataloader)
    actual = [sample_id(next(resumed)) for _ in range(3)]
    assert actual == expected


def test_checkpoint_mode_is_checked_before_loader_restore(monkeypatch):
    trainer = _trainer(True)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {"global_step": 3, "train_dataloader": {"cursor": 7}}
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="trainer state|provenance"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_checkpoint_requires_rank_local_loader_cursor(monkeypatch):
    trainer = _trainer(True)
    protocol = trainer.dpo_provenance_checkpoint_state()
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {"global_step": 3, "dpo_provenance_protocol": protocol}
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="rank-local dataloader cursor"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("field", ["gradient_accumulation_steps", "global_batch_size", "dyn_bsz_count_mode"])
def test_checkpoint_rejects_changed_batch_geometry(field):
    trainer = _trainer(True)
    saved = copy.deepcopy(trainer.dpo_provenance_checkpoint_state())
    setattr(trainer.base.args.train, field, "changed")
    with pytest.raises(ValueError, match="geometry changed"):
        trainer.validate_dpo_provenance_checkpoint(saved)


@pytest.mark.parametrize("field", ["dataset_repeat", "datasets_type", "multisource_datasets_type"])
def test_checkpoint_rejects_changed_dataset_policy(field):
    trainer = _trainer(True)
    saved = copy.deepcopy(trainer.dpo_provenance_checkpoint_state())
    setattr(trainer.base.args.data, field, "changed")
    with pytest.raises(ValueError, match="geometry changed"):
        trainer.validate_dpo_provenance_checkpoint(saved)


def test_checkpoint_rejects_changed_loader_type():
    trainer = _trainer(True)
    saved = copy.deepcopy(trainer.dpo_provenance_checkpoint_state())
    trainer.base.args.data.dataloader.type = "other"
    with pytest.raises(ValueError, match="geometry changed"):
        trainer.validate_dpo_provenance_checkpoint(saved)


@pytest.mark.parametrize("dataset_name", ["iterable", "interleave", "veomni_weighted_multisource"])
def test_checkpoint_rejects_unverified_iterable_and_multisource_cursor(dataset_name):
    trainer = _trainer(True)
    trainer.base.args.data.dataset_name = dataset_name
    trainer._dpo_recovery_geometry = _dpo_recovery_geometry(trainer.base.args)
    saved = trainer.dpo_provenance_checkpoint_state()
    with pytest.raises(ValueError, match="mapping dataset with a verified cursor"):
        trainer.validate_dpo_provenance_checkpoint(saved)


@pytest.mark.parametrize(
    ("epoch", "step", "complete", "expected"),
    [(0, 5, False, (0, 5)), (0, 5, True, (1, 0)), (0, 10, False, (1, 0)), (1, 2, False, (1, 2))],
)
def test_dpo_restores_explicit_position_after_short_epoch(epoch, step, complete, expected):
    trainer = _trainer(True)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._restore_position(
        {
            "global_step": 12,
            "dpo_position": {"epoch": epoch, "step_in_epoch": step, "epoch_complete": complete},
        }
    )
    assert (trainer.start_epoch, trainer.start_step) == expected


@pytest.mark.parametrize("case", ["out_of_order", "worker_dynamic", "prefetch_flush"])
def test_dpo_rejects_unsupported_exact_resume_data_loader(case):
    trainer = _trainer(True)
    if case == "out_of_order":
        trainer.base.args.data.dataloader.num_workers = 2
        trainer.base.args.data.dataloader.in_order = False
    elif case == "worker_dynamic":
        trainer.base.args.train.dyn_bsz_runtime = "worker"
    else:
        trainer.base.args.data.dataloader.use_background_prefetcher = True
        trainer.base.args.data.dataloader.drop_last = False
    with pytest.raises(ValueError, match="DPO exact recovery"):
        _dpo_recovery_geometry(trainer.base.args)


def test_dpo_recovery_rejects_changed_dynamic_microbatch_count(monkeypatch):
    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy

    trainer = _trainer(True)
    trainer.base.train_dataloader = DynamicBatchSizeDataLoader(
        list(range(8)), TextBatchingStrategy(token_micro_bsz=2, buffer_size=1), num_micro_batch=2, length=8
    )
    cursor = trainer.base.train_dataloader.state_dict()
    cursor["num_micro_batch"] = 1
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "dpo_position": {"epoch": 0, "step_in_epoch": 1, "epoch_complete": False},
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="microbatch geometry"):
        callback.load_global_state()


@pytest.mark.parametrize("cursor", [{}, {"bogus": 7}])
def test_real_stateful_loader_rejects_empty_or_lazy_invalid_cursor(monkeypatch, cursor):
    from torchdata.stateful_dataloader import StatefulDataLoader

    trainer = _trainer(True)
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.start_step = 1
    trainer.base.train_dataloader = StatefulDataLoader(list(range(8)), batch_size=1, num_workers=0)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 3,
        "dpo_position": {"epoch": 0, "step_in_epoch": 3, "epoch_complete": False},
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError, AssertionError)):
        callback.load_global_state()


def test_real_stateful_loader_eager_validation_preserves_next_item(monkeypatch):
    from torchdata.stateful_dataloader import StatefulDataLoader

    original = StatefulDataLoader(list(range(8)), batch_size=1, num_workers=0)
    iterator = iter(original)
    next(iterator)
    next(iterator)
    cursor = original.state_dict()
    expected = next(iterator)
    trainer = _trainer(True)
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.start_step = 1
    trainer.base.start_epoch = 0
    trainer.base.train_dataloader = StatefulDataLoader(list(range(8)), batch_size=1, num_workers=0)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 2,
        "dpo_position": {"epoch": 0, "step_in_epoch": 2, "epoch_complete": False},
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    callback.load_global_state()
    torch.testing.assert_close(next(iter(trainer.train_dataloader)), expected, rtol=0, atol=0)


def test_dpo_recovery_rejects_legacy_dynamic_cursor_without_phase(monkeypatch):
    from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader, TextBatchingStrategy

    trainer = _trainer(True)
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.train_dataloader = DynamicBatchSizeDataLoader(
        list(range(8)), TextBatchingStrategy(token_micro_bsz=2, buffer_size=1), length=8
    )
    cursor = {"step": 1, "num_micro_batch": 1, "dataloader_state": {"cursor": 1}}
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "dpo_position": {"epoch": 0, "step_in_epoch": 1, "epoch_complete": False},
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="exact-resume phase"):
        callback.load_global_state()


def test_dpo_recovery_eager_validation_preserves_dynamic_loader_cursor(monkeypatch):
    from tests.data.test_dynamic_batching_resume import _loader

    original = _loader()
    iterator = iter(original)
    next(iterator)
    cursor = original.state_dict()
    expected = [[sample["id"] for sample in batch] for batch in next(iterator)]
    trainer = _trainer(True)
    trainer.base.state = SimpleNamespace(global_step=0)
    trainer.base.start_step = 1
    trainer.base.start_epoch = 0
    trainer.base.train_dataloader = _loader()
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 1,
        "dpo_position": {"epoch": 0, "step_in_epoch": 1, "epoch_complete": False},
        "train_dataloader": cursor,
        "dpo_provenance_protocol": trainer.dpo_provenance_checkpoint_state(),
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    callback.load_global_state()
    actual = [[sample["id"] for sample in batch] for batch in next(iter(trainer.train_dataloader))]
    assert actual == expected


@pytest.mark.parametrize("mutation", ["template", "tokenizer", "split_special_tokens", "max_seq_len", "beta"])
def test_live_producer_is_checked_before_loader_restore(monkeypatch, mutation):
    from tokenizers import Tokenizer as BackendTokenizer
    from tokenizers.models import WordLevel
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast

    from veomni.data.chat_template import ChatmlTemplate, DefaultTemplate

    trainer = _trainer(True)
    backend = BackendTokenizer(WordLevel({"[UNK]": 0, "[PAD]": 1, "a": 2}, unk_token="[UNK]"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
    trainer.policy_model.tokenizer = tokenizer
    trainer.policy_model.chat_template = ChatmlTemplate(tokenizer)
    trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256 = dpo_tokenizer_fingerprint(tokenizer)
    trainer._dpo_objective_parameters = _dpo_objective_parameters(trainer.base.args.dpo_config)
    trainer._dpo_producer_identity = dpo_producer_identity(
        tokenizer, trainer.policy_model.chat_template, trainer.base.args.dpo_config.audited_unpadded_tokenizer_sha256
    )
    saved = trainer.dpo_provenance_checkpoint_state()
    if mutation == "template":
        trainer.policy_model.chat_template = DefaultTemplate(tokenizer)
    elif mutation == "tokenizer":
        tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
            single="[PAD] [PAD] $A", special_tokens=[("[PAD]", 1)]
        )
    elif mutation == "split_special_tokens":
        tokenizer.split_special_tokens = not tokenizer.split_special_tokens
    elif mutation == "max_seq_len":
        trainer._dpo_transform_max_seq_len = 32
    else:
        trainer.base.args.dpo_config.beta = 0.2
        trainer._dpo_objective_parameters = _dpo_objective_parameters(trainer.base.args.dpo_config)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: {
        "global_step": 3,
        "dpo_position": {"epoch": 0, "step_in_epoch": 3, "epoch_complete": False},
        "train_dataloader": {"cursor": 7, "buffer": [{"input_ids": [1, 2, 3]}]},
        "dpo_provenance_protocol": saved,
    }
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError)):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
def test_missing_checkpoint_trainer_state_does_not_assume_safe_mode(monkeypatch, enabled):
    trainer = _trainer(enabled)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: None
    monkeypatch.setattr("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", lambda *_: None)
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="trainer state|provenance"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_explicit_legacy_resume_cannot_bypass_required_provenance(monkeypatch):
    trainer = _trainer(False, allow_legacy=True, allow_legacy_objective=True)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    callback._read_current = lambda _: None
    monkeypatch.setattr("veomni.checkpoint.legacy_v0_1_12.apply_legacy_global_state", lambda *_: None)
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.dist.is_initialized", lambda: False)
    with pytest.raises((ValueError, RuntimeError), match="provenance"):
        callback.load_global_state()
    trainer.train_dataloader.load_state_dict.assert_not_called()


def test_full_sample_buffer_serialization_keeps_typed_provenance():
    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return list(range(len(text)))

    feature = process_dpo_example(
        {"chosen": "abc", "rejected": "d"}, tokenizer=Tokenizer(), record_segment_provenance=True
    )[0]
    buffer = io.BytesIO()
    torch.save(copy.deepcopy({"buffer": [feature]}), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)["buffer"][0]
    marker = restored[DPO_FEATURE_KEY]
    assert (marker.chosen, marker.rejected, marker.producer_contract_version) == (3, 1, 1)
    assert len(marker.feature_digest) == 64


@pytest.mark.parametrize("save_by_idx", [False, True])
def test_dynamic_batch_buffer_restore_preserves_provenance(save_by_idx):
    class Tokenizer:
        dpo_unpadded_output_contract = True
        dpo_unpadded_output_contract_id = "test-tokenizer-v1"

        def encode(self, text, add_special_tokens=True):
            return list(range(len(text)))

    class Dataset(torch.utils.data.IterableDataset):
        output_index_for_resume = False

        def get_item(self, index):
            return process_dpo_example(
                {"chosen": "abc", "rejected": "d"}, tokenizer=Tokenizer(), record_segment_provenance=True
            )

        def __iter__(self):
            return iter([])

    def build():
        return DynamicBatchingSizeDataset(
            Dataset(),
            micro_batch_seq_length=8,
            ready_for_micro_batch_threshold=2,
            dynamic_batching_collate_fn=lambda x: x,
            get_length_fn=get_length_fn,
            save_by_idx=save_by_idx,
        )

    original = build()
    feature = original.dataset.get_item(0)[0]
    original._buffer = [(feature, 4, 4)]
    original._buffer_of_output_index = [(0, 0)]
    original._buffer_token_count = original._buffer_physical_token_count = 4
    serialized = io.BytesIO()
    torch.save(original.state_dict(), serialized)
    serialized.seek(0)
    resumed = build()
    resumed.load_state_dict(torch.load(serialized, weights_only=False))
    marker = resumed._buffer[0][0][DPO_FEATURE_KEY]
    assert (marker.chosen, marker.rejected, marker.producer_contract_version) == (3, 1, 1)
    assert len(marker.feature_digest) == 64

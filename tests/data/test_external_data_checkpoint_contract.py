"""An opt-in external data provider must be checked before checkpoint I/O."""

import copy
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import (
    GlobalStateCallback,
    _dynamic_resume_context,
    _validate_external_data_state,
    prepare_external_data_contract,
)


class Provider:
    def __init__(self):
        self.events = []
        self.identity = "provider-and-source-v1"
        self.global_identity = "provider-and-global-source-v1"

    def install(self):
        self.events.append("install")

    def checkpoint_identity(self):
        return self.identity

    def checkpoint_global_identity(self):
        return self.global_identity

    def validate_before_checkpoint(self, trainer, state=None):
        self.events.append(("preflight", state))

    def validate_cursor(self, trainer, cursor, *, require_cursor):
        self.events.append(("cursor", require_cursor))
        if not isinstance(cursor, dict) or not cursor:
            raise ValueError("missing cursor")


def _trainer():
    provider = Provider()
    args = SimpleNamespace(
        train_steps=10,
        train=SimpleNamespace(
            global_rank=0,
            num_train_epochs=4,
            checkpoint=SimpleNamespace(save_async=False, save_path="/not-used"),
        ),
        data=SimpleNamespace(),
        model=SimpleNamespace(accelerator=SimpleNamespace()),
    )
    trainer = SimpleNamespace(
        args=args,
        _veomni_require_exact_data_resume=True,
        _veomni_exact_epoch_position={"epoch": 0, "step_in_epoch": 1},
        train_dataloader=SimpleNamespace(
            _veomni_exact_checkpoint_contract=provider,
            _veomni_consumed_in_epoch=1,
        ),
        model=SimpleNamespace(
            optimizer=SimpleNamespace(_is_multi_optimizer=True),
            checkpoint=SimpleNamespace(_loaded_optimizer_generation=None),
        ),
    )
    return trainer, provider


def test_provider_is_installed_and_checked_before_model_save():
    trainer, provider = _trainer()
    events = provider.events
    trainer.save_dcp = lambda state: events.append(("save", state.global_step))
    callback = CheckpointCallback.__new__(CheckpointCallback)
    callback.trainer = trainer
    state = SimpleNamespace(global_step=1, epoch=0)
    prepare_external_data_contract(trainer, install=True)
    callback._save_dcp(state)
    assert events == ["install", ("preflight", None), ("preflight", state), ("save", 1)]


def test_unproven_provider_rejects_before_model_save():
    trainer, provider = _trainer()
    provider.validate_before_checkpoint = lambda *_: (_ for _ in ()).throw(ValueError("bad cursor"))
    writes = []
    trainer.save_dcp = lambda state: writes.append(state)
    callback = CheckpointCallback.__new__(CheckpointCallback)
    callback.trainer = trainer
    with pytest.raises((RuntimeError, ValueError), match="bad cursor|validating"):
        callback._save_dcp(SimpleNamespace(global_step=1, epoch=0))
    assert writes == []


def test_unsupported_optimizer_and_async_are_rejected_before_model_save():
    trainer, _ = _trainer()
    trainer.model.optimizer._is_multi_optimizer = False
    with pytest.raises(RuntimeError, match="MultiOptimizer"):
        prepare_external_data_contract(trainer, install=True)
    trainer.model.optimizer._is_multi_optimizer = True
    trainer.args.train.checkpoint.save_async = True
    with pytest.raises(RuntimeError, match="synchronous"):
        prepare_external_data_contract(trainer, install=True)


def test_external_checkpoint_state_contains_provider_identity_and_position(monkeypatch):
    trainer, provider = _trainer()
    trainer._veomni_exact_epoch_position = {"epoch": 1, "step_in_epoch": 1}
    trainer.train_dataloader.state_dict = lambda: {"_veomni_consumed_in_epoch": 1}
    trainer.environ_meter = SimpleNamespace(state_dict=lambda: {})
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.get_device_rng_state", lambda: None)
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    trainer.global_state_callback = callback
    result = callback.state_dict(SimpleNamespace(global_step=11, epoch=1))
    assert result["_veomni_external_data"]["identity"] == provider.identity
    assert result["_veomni_external_data"]["position"] == {"epoch": 1, "step_in_epoch": 1}
    assert result["train_dataloader"]["_veomni_consumed_in_epoch"] == 1


def test_external_boundary_state_restores_one_iterator_before_next_epoch(monkeypatch):
    trainer, provider = _trainer()
    trainer.args.train.world_size = 1
    trainer.args.train.checkpoint.load_path = "/unused"
    trainer.args.train.checkpoint.save_path = None
    trainer._veomni_exact_epoch_position = {"epoch": 1, "step_in_epoch": 10}
    trainer.state = SimpleNamespace(global_step=20, epoch=1)
    loader = trainer.train_dataloader
    loader._veomni_resume_iterator_prepared = True
    loader.state_dict = lambda: {"_veomni_consumed_in_epoch": 10, "pending": [77]}
    loader.load_state_dict = lambda cursor: setattr(loader, "restored", cursor)
    trainer.environ_meter = SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda _: None)
    trainer.channel_loss_callback = None
    trainer.offline_cache_writer = None
    callback = GlobalStateCallback.__new__(GlobalStateCallback)
    callback.trainer = trainer
    trainer.global_state_callback = callback
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.get_device_rng_state", lambda: None)
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.set_device_rng_state", lambda _: None)
    state = callback.state_dict(trainer.state)
    state["_veomni_optimizer_generations"] = {"model": "generation-20"}
    checkpoint = SimpleNamespace(
        _loaded_optimizer_global_step=20,
        _loaded_optimizer_generation="generation-20",
    )
    monkeypatch.setattr(
        "veomni.trainer.callbacks.global_state_callback._multi_optimizer_checkpoints",
        lambda _: [("model", checkpoint)],
    )
    monkeypatch.setattr("veomni.trainer.callbacks.global_state_callback.layout.checkpoint_is_complete", lambda _: True)
    monkeypatch.setattr(
        "veomni.trainer.callbacks.global_state_callback.layout.read_manifest",
        lambda _: {"global_step": 20, "world_size": 1, "model_state": True},
    )
    callback._read_current = lambda _: state
    callback.load_global_state()
    assert (trainer.start_epoch, trainer.start_step) == (2, 0)
    assert loader.restored == {"_veomni_consumed_in_epoch": 10, "pending": [77]}
    assert provider.events[-1] == ("cursor", True)
    # The restored loader may already point to the next epoch. A zero-step
    # export validates and reuses the old boundary rather than snapshotting it.
    loader._veomni_consumed_in_epoch = 0
    prepare_external_data_contract(trainer, trainer.state)
    assert callback.state_dict(trainer.state)["train_dataloader"] == loader.restored


@pytest.mark.parametrize("step", [1, 10])
def test_exact_external_cursor_binds_source_context_and_completed_position(step):
    trainer, provider = _trainer()
    position = {"epoch": 1, "step_in_epoch": step}
    metadata = {
        "version": 1,
        "identity": provider.identity,
        "global_identity": provider.global_identity,
        "position": position,
        "context": _dynamic_resume_context(trainer.args, 1),
    }
    cursor = {"_veomni_consumed_in_epoch": step}
    assert _validate_external_data_state(trainer, provider, metadata, cursor, 10 + step) == 1
    for damage in ("source", "global_source", "context", "position", "cursor"):
        changed, changed_cursor = copy.deepcopy(metadata), dict(cursor)
        if damage == "source":
            changed["identity"] = "other"
        elif damage == "global_source":
            changed["global_identity"] = "other"
        elif damage == "context":
            changed["context"]["epoch_steps"] = 11
        elif damage == "position":
            changed["position"]["step_in_epoch"] += 1
        else:
            changed_cursor["_veomni_consumed_in_epoch"] += 1
        with pytest.raises(ValueError):
            _validate_external_data_state(trainer, provider, changed, changed_cursor, 10 + step)


def _provider_failure_worker(rank, store):
    dist.init_process_group(
        "gloo", init_method="file://" + store, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        for scenario in (
            "mode",
            "provider",
            "baseexception",
            "missing",
            "identity",
            "context",
            "optimizer",
            "owners",
            "order",
            "unbound",
            "wrapped_unbound",
        ):
            trainer, provider = _trainer()
            if scenario == "mode":
                trainer._veomni_require_exact_data_resume = rank == 0
            elif scenario == "provider" and rank == 0:
                provider.validate_before_checkpoint = lambda *_: (_ for _ in ()).throw(ValueError("injected"))
            elif scenario == "baseexception" and rank == 0:
                provider.validate_before_checkpoint = lambda *_: (_ for _ in ()).throw(SystemExit("injected"))
            elif scenario == "missing" and rank == 0:
                trainer.train_dataloader._veomni_exact_checkpoint_contract = None
            elif scenario == "identity" and rank == 0:
                provider.global_identity = "other-source"
            elif scenario == "context" and rank == 0:
                trainer.args.data.train_path = Path("/not-json-serializable")
            elif scenario == "optimizer" and rank == 0:
                trainer.model.optimizer._is_multi_optimizer = False
            elif scenario == "owners" and rank == 0:
                trainer.model.module_runtimes = {
                    "extra": SimpleNamespace(
                        optimizer=SimpleNamespace(_is_multi_optimizer=True),
                        checkpoint=SimpleNamespace(_loaded_optimizer_generation=None),
                    )
                }
            elif scenario == "order":
                names = ("first", "second") if rank == 0 else ("second", "first")
                trainer.model.module_runtimes = {
                    name: SimpleNamespace(
                        optimizer=SimpleNamespace(_is_multi_optimizer=True),
                        checkpoint=SimpleNamespace(_loaded_optimizer_generation=None),
                    )
                    for name in names
                }
            elif scenario == "unbound":
                trainer._veomni_require_exact_data_resume = False
                trainer.train_dataloader._veomni_exact_resume_requested = rank == 0
            elif scenario == "wrapped_unbound":
                from veomni.data.dynamic_batching import DynamicBatchSizeDataLoader

                trainer._veomni_require_exact_data_resume = False
                inner = SimpleNamespace(_veomni_exact_resume_requested=rank == 0)
                trainer.train_dataloader = object.__new__(DynamicBatchSizeDataLoader)
                trainer.train_dataloader._dataloader = inner
            with pytest.raises((RuntimeError, ValueError, TypeError, SystemExit)):
                prepare_external_data_contract(trainer, install=True)
            marker = torch.tensor([1], dtype=torch.int32)
            dist.all_reduce(marker)
            assert marker.item() == 2
    finally:
        dist.destroy_process_group()


def test_rank_local_provider_failures_are_coordinated(tmp_path):
    mp.spawn(_provider_failure_worker, args=(str(tmp_path / "provider-store"),), nprocs=2, join=True)

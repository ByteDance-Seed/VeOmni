"""Exact-data save scheduling and incomplete-checkpoint recovery safeguards."""

import time
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from veomni.checkpoint import layout
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback
from veomni.trainer.callbacks.global_state_callback import GlobalStateCallback


class Provider:
    ready = True
    error = None

    def checkpoint_ready(self, trainer, state):
        if self.error is not None:
            raise self.error
        return self.ready

    def install(self):
        pass

    def checkpoint_identity(self):
        return "source"

    def checkpoint_global_identity(self):
        return "global-source"

    def validate_before_checkpoint(self, trainer, state=None):
        pass

    def validate_cursor(self, trainer, cursor, *, require_cursor):
        pass


def callbacks(*, step_frequency=2, epoch_frequency=0, hf_frequency=0):
    provider = Provider()
    trainer = SimpleNamespace(
        _veomni_require_exact_data_resume=True,
        train_dataloader=SimpleNamespace(_veomni_exact_checkpoint_contract=provider),
        args=SimpleNamespace(train=SimpleNamespace(checkpoint=SimpleNamespace(save_async=False))),
        wait_for_pending_save=lambda: None,
    )
    checkpoint = CheckpointCallback.__new__(CheckpointCallback)
    checkpoint.trainer = trainer
    checkpoint.dcp_every_n_steps = step_frequency
    checkpoint.dcp_every_n_epochs = epoch_frequency
    checkpoint.save_hf_weights = bool(hf_frequency)
    checkpoint.hf_every_n_steps = hf_frequency
    checkpoint.hf_every_n_epochs = 0
    checkpoint._last_dcp_step = -1
    checkpoint._last_hf_step = -1
    checkpoint._async_save_failed = False
    checkpoint._external_pending_dcp = False
    checkpoint._external_pending_hf = False
    checkpoint._external_checkpoint_event = None
    global_state = GlobalStateCallback.__new__(GlobalStateCallback)
    global_state.trainer = trainer
    global_state._last_saved_step = -1
    global_state._unadvanced_boundary_state = None
    trainer.checkpoint_callback = checkpoint
    trainer.global_state_callback = global_state
    events = []

    def dcp(state):
        events.append(("dcp", state.global_step))
        checkpoint._last_dcp_step = state.global_step

    def hf(state, stage="step_end"):
        # MultiOptimizer exports also require a same-step DCP.
        if checkpoint._last_dcp_step != state.global_step:
            dcp(state)
        events.append(("hf", state.global_step))
        checkpoint._last_hf_step = state.global_step

    def cursor(state, *, require_optimizer_checkpoint=False):
        events.append(("cursor", state.global_step, require_optimizer_checkpoint))
        global_state._last_saved_step = state.global_step

    checkpoint._save_dcp = dcp
    checkpoint._save_hf = hf
    global_state.save_global_state = cursor
    return provider, checkpoint, global_state, events


def event(checkpoint, global_state, name, step, epoch=0):
    state = SimpleNamespace(global_step=step, epoch=epoch)
    getattr(checkpoint, "on_" + name)(state)
    getattr(global_state, "on_" + name)(state)


def test_scheduled_save_waits_and_uses_actual_step_for_all_components():
    provider, checkpoint, global_state, events = callbacks(step_frequency=2, hf_frequency=3)
    event(checkpoint, global_state, "step_end", 1)
    provider.ready = False
    event(checkpoint, global_state, "step_end", 2)
    event(checkpoint, global_state, "step_end", 3)
    assert events == []
    provider.ready = True
    event(checkpoint, global_state, "step_end", 4)
    assert events == [("dcp", 4), ("hf", 4), ("cursor", 4, True)]
    assert not checkpoint._external_pending_dcp and not checkpoint._external_pending_hf


def test_same_step_epoch_and_step_save_are_deduplicated():
    provider, checkpoint, global_state, events = callbacks(step_frequency=2, epoch_frequency=1)
    event(checkpoint, global_state, "step_end", 2)
    # Taking the cursor snapshot can itself populate a replay buffer.
    provider.ready = False
    event(checkpoint, global_state, "epoch_end", 2)
    event(checkpoint, global_state, "train_end", 2)
    assert events == [("dcp", 2), ("cursor", 2, True)]


def test_pending_save_at_train_end_fails_without_writing():
    provider, checkpoint, global_state, events = callbacks()
    provider.ready = False
    event(checkpoint, global_state, "step_end", 2)
    with pytest.raises(RuntimeError, match="ended before"):
        event(checkpoint, global_state, "train_end", 2)
    assert events == []


def test_train_end_closes_provider_after_its_final_cursor():
    provider, checkpoint, global_state, events = callbacks(step_frequency=0, hf_frequency=3)
    provider.close = lambda: events.append(("close", 3))
    event(checkpoint, global_state, "train_end", 3)
    assert events == [("dcp", 3), ("hf", 3), ("cursor", 3, True), ("close", 3)]


def test_train_end_propagates_provider_close_failure():
    provider, checkpoint, global_state, _ = callbacks()

    def fail():
        raise OSError("close failed")

    provider.close = fail
    with pytest.raises((OSError, RuntimeError), match="close|failed"):
        event(checkpoint, global_state, "train_end", 1)


def test_train_end_pending_checkpoint_closes_locally_and_preserves_both_errors():
    provider, checkpoint, global_state, _ = callbacks()
    provider.ready = False
    event(checkpoint, global_state, "step_end", 2)

    def fail_close():
        raise OSError("injected cleanup failure")

    provider.close = fail_close
    with pytest.raises(RuntimeError, match="ended before") as caught:
        event(checkpoint, global_state, "train_end", 2)
    assert any("injected cleanup failure" in note for note in caught.value.__notes__)


def test_train_end_failed_cursor_closes_provider_locally():
    provider, checkpoint, global_state, events = callbacks(step_frequency=0, hf_frequency=3)
    provider.close = lambda: events.append(("close", 3))

    def fail_cursor(*args, **kwargs):
        raise OSError("cursor failed")

    global_state.save_global_state = fail_cursor
    with pytest.raises(OSError, match="cursor failed"):
        event(checkpoint, global_state, "train_end", 3)
    assert events == [("dcp", 3), ("hf", 3), ("close", 3)]


def test_epoch_save_deferred_into_next_epoch_uses_new_completed_step():
    provider, checkpoint, global_state, events = callbacks(step_frequency=0, epoch_frequency=1)
    provider.ready = False
    event(checkpoint, global_state, "epoch_end", 4, epoch=0)
    assert events == []
    provider.ready = True
    event(checkpoint, global_state, "step_end", 5, epoch=1)
    assert events == [("dcp", 5), ("cursor", 5, True)]


def test_checkpoint_write_error_never_commits_a_cursor():
    _, checkpoint, global_state, events = callbacks()

    def fail_save(state):
        raise OSError("injected checkpoint failure")

    checkpoint._save_dcp = fail_save
    with pytest.raises(OSError, match="injected"):
        event(checkpoint, global_state, "step_end", 2)
    with pytest.raises(RuntimeError, match="did not run"):
        global_state.on_step_end(SimpleNamespace(global_step=2, epoch=0))
    assert events == []


@pytest.mark.parametrize("failure", ["cursor", "write", "manifest"])
def test_cursor_failure_leaves_existing_model_files_incomplete(tmp_path, monkeypatch, failure):
    _, checkpoint, global_state, _ = callbacks()
    trainer = checkpoint.trainer
    trainer.args.train.global_rank = 0
    trainer.args.train.world_size = 1
    trainer.args.train.checkpoint.save_path = str(tmp_path)
    root = layout.step_dir(str(tmp_path), 2)

    def model_save(state):
        # Represent a completed model DCP, including its individual marker.
        weights = tmp_path / "global_step_2" / "model" / "ckpt"
        weights.mkdir(parents=True)
        (weights / ".metadata").touch()
        checkpoint._last_dcp_step = state.global_step

    checkpoint._save_dcp = model_save
    global_state.save_global_state = GlobalStateCallback.save_global_state.__get__(global_state)
    global_state.state_dict = lambda state: {"global_step": state.global_step, "train_dataloader": {"next": 3}}
    monkeypatch.setattr(
        "veomni.trainer.callbacks.global_state_callback._saved_multi_optimizer_generations", lambda *_: {}
    )
    if failure == "cursor":

        def fail_cursor(state):
            raise OSError("cursor failed")

        global_state.state_dict = fail_cursor
    elif failure == "write":

        def fail_write(*args, **kwargs):
            raise OSError("write failed")

        monkeypatch.setattr(torch, "save", fail_write)
    else:

        def fail_manifest(*args, **kwargs):
            raise OSError("manifest failed")

        monkeypatch.setattr(layout, "write_manifest", fail_manifest)
    with pytest.raises((OSError, RuntimeError)):
        event(checkpoint, global_state, "step_end", 2)
    assert (tmp_path / "global_step_2" / "model" / "ckpt" / ".metadata").exists()
    assert not layout.checkpoint_is_complete(root)
    assert global_state._last_saved_step == -1


def test_provider_errors_and_unordered_callbacks_fail_closed():
    provider, checkpoint, global_state, events = callbacks()
    provider.error = ValueError("readiness failed")
    with pytest.raises((RuntimeError, ValueError), match="readiness|failed"):
        event(checkpoint, global_state, "step_end", 2)
    with pytest.raises(RuntimeError, match="did not run"):
        global_state.on_step_end(SimpleNamespace(global_step=2, epoch=0))
    assert events == []


def test_hf_only_save_requires_same_step_optimizer_receipt():
    _, checkpoint, global_state, events = callbacks(step_frequency=0, hf_frequency=3)
    event(checkpoint, global_state, "step_end", 3)
    assert events == [("dcp", 3), ("hf", 3), ("cursor", 3, True)]


def test_unadvanced_boundary_does_not_bypass_the_provider_replay_guard():
    provider, checkpoint, global_state, events = callbacks(step_frequency=0, hf_frequency=3)
    provider.ready = False
    global_state._unadvanced_boundary_state = {"global_step": 3, "_veomni_external_data": {"identity": "v1"}}
    with pytest.raises(RuntimeError, match="ended before"):
        event(checkpoint, global_state, "train_end", 3)
    assert events == []


def _distributed_deferral(rank, store):
    dist.init_process_group(
        "gloo", init_method="file://" + store, rank=rank, world_size=2, timeout=timedelta(seconds=30)
    )
    try:
        provider, checkpoint, global_state, events = callbacks()
        provider.ready = rank == 0
        event(checkpoint, global_state, "step_end", 2)
        assert events == []
        provider.ready = True
        event(checkpoint, global_state, "step_end", 3)
        assert events == [("dcp", 3), ("cursor", 3, True)]
        provider.error = ValueError("one-rank-error") if rank == 0 else None
        with pytest.raises((RuntimeError, ValueError)):
            event(checkpoint, global_state, "step_end", 4)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        provider, checkpoint, global_state, _ = callbacks()
        if rank == 0:

            def fail_close():
                raise OSError("one-rank close failure")

            provider.close = fail_close
        with pytest.raises((OSError, RuntimeError)):
            event(checkpoint, global_state, "train_end", 1)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        # A peer can fail before the data callback is reached. Closing must
        # not enter a new collective that the failing peer cannot join.
        provider, checkpoint, global_state, _ = callbacks()
        if rank == 0:

            def fail_pending():
                raise OSError("one-rank pending-save failure")

            checkpoint.trainer.wait_for_pending_save = fail_pending
        with pytest.raises((OSError, RuntimeError)):
            event(checkpoint, global_state, "train_end", 1)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        provider, checkpoint, global_state, _ = callbacks()
        state = SimpleNamespace(global_step=1, epoch=0)
        checkpoint.on_train_end(state)
        if rank == 0:
            checkpoint._external_checkpoint_event = None
        with pytest.raises(RuntimeError):
            global_state.on_train_end(state)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        # Both decisions can be individually valid while demanding different
        # collective sequences. Reject before saving either cursor.
        provider, checkpoint, global_state, _ = callbacks()
        state = SimpleNamespace(global_step=1, epoch=0)
        checkpoint.on_train_end(state)
        if rank == 0:
            checkpoint._external_checkpoint_event = ("train_end", 1, "saved", True)
        with pytest.raises(RuntimeError, match="decision differs"):
            global_state.on_train_end(state)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        provider, checkpoint, global_state, _ = callbacks()
        checkpoint._external_pending_hf = rank == 0
        with pytest.raises(RuntimeError, match="decision differs"):
            event(checkpoint, global_state, "train_end", 1)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2

        provider, checkpoint, global_state, _ = callbacks()
        if rank == 0:
            provider.close = lambda: time.sleep(0.2)
        event(checkpoint, global_state, "train_end", 1)
        alive = torch.ones(1)
        dist.all_reduce(alive)
        assert alive.item() == 2
    finally:
        dist.destroy_process_group()


def test_two_ranks_agree_on_deferral_and_error(tmp_path):
    mp.spawn(_distributed_deferral, args=(str(tmp_path / "store"),), nprocs=2, join=True)

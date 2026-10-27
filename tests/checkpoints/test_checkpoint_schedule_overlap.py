"""HF export must not trigger a second write of the same DCP step."""

from types import SimpleNamespace

import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.trainer.callbacks import base as callback_base
from veomni.trainer.callbacks import global_state_callback
from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.mark.parametrize("implicit_dcp", [True, False])
def test_hf_step_then_epoch_dcp_saves_at_most_once(monkeypatch, implicit_dcp):
    checkpoint = SimpleNamespace(save_steps=0, save_epochs=1, save_hf_weights=True, hf_save_steps=3, hf_save_epochs=0)
    manager = SimpleNamespace(_last_saved_step=-1)
    calls = []
    trainer = SimpleNamespace(
        args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint)), model=SimpleNamespace(checkpoint=manager)
    )

    def save_hf(state, stage):
        calls.append("hf")
        if implicit_dcp:
            calls.append("dcp")
            manager._last_saved_step = state.global_step

    def save_dcp(state):
        calls.append("dcp")
        manager._last_saved_step = state.global_step

    trainer.save_hf_or_lora = save_hf
    trainer.save_dcp = save_dcp
    monkeypatch.setattr(callback_base, "get_parallel_state", lambda: SimpleNamespace())
    monkeypatch.setattr(global_state_callback, "validate_resumed_prefetch_is_drained", lambda *args: None)
    callback = CheckpointCallback(trainer)
    state = SimpleNamespace(global_step=3, epoch=0)
    callback.on_step_end(state)
    callback.on_epoch_end(state)
    assert calls == ["hf", "dcp"]


def test_exact_runtime_identity_is_checked_before_any_checkpoint_io(monkeypatch):
    monkeypatch.delenv("VEOMNI_EXACT_RESUME_RUNTIME_ID", raising=False)
    monkeypatch.setattr(global_state_callback, "_has_file_profile", lambda trainer: True)
    monkeypatch.setattr(
        global_state_callback,
        "_exact_checkpoint_contract",
        lambda *a, **k: SimpleNamespace(install=lambda: None, checkpoint_identity=lambda: "provider"),
    )
    trainer = SimpleNamespace(save_model_assets=lambda: pytest.fail("checkpoint I/O began"))
    callback = CheckpointCallback.__new__(CheckpointCallback)
    callback.trainer = trainer
    with pytest.raises(RuntimeError, match="runtime/source identity"):
        callback.on_train_begin(SimpleNamespace(global_step=0))

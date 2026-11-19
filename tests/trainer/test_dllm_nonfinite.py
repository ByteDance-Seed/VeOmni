from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from veomni.trainer import dllm_trainer
from veomni.trainer.dllm_trainer import DllmTrainer


class _MustNotStep:
    def step(self):
        raise AssertionError("optimizer must not run after a non-finite gradient")

    def zero_grad(self):
        raise AssertionError("zero_grad must not run after a non-finite gradient")


def test_train_step_fails_before_optimizer_on_nonfinite_gradient(monkeypatch):
    trainer = DllmTrainer.__new__(DllmTrainer)
    runtime_model = object()
    optimizer = _MustNotStep()
    trainer.base = SimpleNamespace(
        args=SimpleNamespace(model=SimpleNamespace(optimizer=SimpleNamespace(max_grad_norm=1.0))),
        state=SimpleNamespace(global_step=0),
        model=SimpleNamespace(
            model=runtime_model,
            optimizer=optimizer,
            lr_scheduler=_MustNotStep(),
            parallel_state="base",
        ),
        model_reshard=lambda *_: None,
        _reset_async_activation_offload_if_enabled=lambda *_: None,
        sync_before_train_step=lambda: None,
        _configure_hsdp_allreduce=lambda *_: None,
    )
    trainer.on_step_begin = lambda **_: None
    trainer.on_micro_batches_ready = lambda _: None
    trainer.forward_backward_step = lambda _: (
        torch.tensor(1.0),
        {"foundation_loss": torch.tensor(1.0)},
    )
    monkeypatch.setattr(dllm_trainer, "synchronize", lambda: None)
    monkeypatch.setattr(dllm_trainer, "use_parallel_state", lambda *_a, **_k: nullcontext())

    def fail_closed(model, max_norm, *, error_if_nonfinite):
        assert model is runtime_model
        assert max_norm == 1.0
        assert error_if_nonfinite is True
        raise RuntimeError("non-finite gradient")

    monkeypatch.setattr(dllm_trainer, "veomni_clip_grad_norm", fail_closed)

    with pytest.raises(RuntimeError, match="non-finite gradient"):
        trainer.train_step(iter([[{"loss_weight": torch.ones(1, 1)}]]))

    assert trainer.base.state.global_step == 1

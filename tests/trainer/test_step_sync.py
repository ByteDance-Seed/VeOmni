import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import veomni.trainer.base as base_module
from veomni.trainer.base import BaseTrainer, _resolve_offload_config
from veomni.trainer.dit_trainer import DiTTrainer
from veomni.trainer.text_dpo_trainer import TextDPOTrainer
from veomni.trainer.text_trainer import TextTrainer
from veomni.trainer.vlm_trainer import VLMTrainer


def _trainer(sync_each_train_step: bool):
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(train=SimpleNamespace(sync_each_train_step=sync_each_train_step))
    return trainer


def test_sync_before_train_step_honors_training_flag(monkeypatch):
    calls = []
    monkeypatch.setattr(base_module, "synchronize", lambda: calls.append("sync"))

    _trainer(True).sync_before_train_step()
    _trainer(False).sync_before_train_step()

    assert calls == ["sync"]


def test_train_step_uses_sync_helper():
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)

        assert "sync_before_train_step()" in source
        assert "synchronize()" not in source


def test_reset_async_activation_offload_skips_missing_config(monkeypatch):
    calls = []
    monkeypatch.setattr(base_module, "reset_async_activation_offload", lambda model: calls.append(model))

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.model = object()
    trainer.args = SimpleNamespace(model=SimpleNamespace(accelerator=SimpleNamespace()))

    trainer._reset_async_activation_offload_if_enabled()
    assert calls == []

    trainer.args.model.accelerator.offload_config = SimpleNamespace(enable_async_activation=True)
    trainer._reset_async_activation_offload_if_enabled()
    assert calls == [trainer.model]


def test_resolve_offload_config_defaults_when_absent():
    cfg = _resolve_offload_config(SimpleNamespace(model=SimpleNamespace(accelerator=SimpleNamespace())))

    assert cfg.enable_async_activation is False
    assert cfg.enable_activation is False


def test_build_training_context_without_offload_config(monkeypatch):
    contexts = (nullcontext(), nullcontext())
    monkeypatch.setattr(base_module, "build_activation_offloading_context", lambda *args, **kwargs: contexts)

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            accelerator=SimpleNamespace(
                gradient_checkpointing=SimpleNamespace(enable=False),
            ),
        )
    )

    trainer._build_training_context()

    assert trainer.model_fwd_context is contexts[0]
    assert trainer.model_bwd_context is contexts[1]


def test_configure_hsdp_allreduce_toggles_outer_micro_steps():
    calls = []
    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        model=SimpleNamespace(
            accelerator=SimpleNamespace(
                fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"),
                dp_replicate_size=2,
            )
        )
    )
    trainer.model = SimpleNamespace(set_requires_all_reduce=calls.append)

    for micro_step in range(4):
        trainer._configure_hsdp_allreduce(micro_step, 4)

    assert calls == [False, True]


def test_train_step_uses_async_offload_reset_helper():
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)

        assert "_reset_async_activation_offload_if_enabled()" in source


def test_train_step_uses_hsdp_allreduce_helper():
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)

        assert "_configure_hsdp_allreduce(" in source

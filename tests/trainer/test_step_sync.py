import inspect
from types import SimpleNamespace

import veomni.trainer.base as base_module
from veomni.trainer.base import BaseTrainer
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


def test_train_step_uses_hsdp_allreduce_helper():
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)

        assert "_configure_hsdp_allreduce(" in source

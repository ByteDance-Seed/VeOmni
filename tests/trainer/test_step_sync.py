import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

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


def test_reset_async_activation_offload_skips_missing_config(monkeypatch):
    calls = []
    monkeypatch.setattr(base_module, "reset_async_activation_offload", lambda model: calls.append(model))

    trainer = BaseTrainer.__new__(BaseTrainer)
    model = SimpleNamespace(args=SimpleNamespace(accelerator=SimpleNamespace()))

    trainer._reset_async_activation_offload_if_enabled(model)
    assert calls == []

    model.args.accelerator.offload_config = SimpleNamespace(enable_async_activation=True)
    trainer._reset_async_activation_offload_if_enabled(model)
    assert calls == [model]


def test_unset_base_trainer_model_raises():
    trainer = BaseTrainer.__new__(BaseTrainer)
    with pytest.raises(AttributeError, match="model is unset"):
        _ = trainer.model


def test_unset_base_trainer_model_is_getattr_defaultable():
    trainer = BaseTrainer.__new__(BaseTrainer)
    assert getattr(trainer, "model", None) is None


def test_build_training_context_requires_model_args():
    trainer = BaseTrainer.__new__(BaseTrainer)
    with pytest.raises(AttributeError):
        trainer._build_training_context(object())


def test_build_training_context_without_offload_config(monkeypatch):
    contexts = (nullcontext(), nullcontext())
    monkeypatch.setattr(base_module, "build_activation_offloading_context", lambda *args, **kwargs: contexts)

    trainer = BaseTrainer.__new__(BaseTrainer)
    model = SimpleNamespace(
        args=SimpleNamespace(
            accelerator=SimpleNamespace(
                gradient_checkpointing=SimpleNamespace(enable=False),
            ),
        )
    )

    trainer._build_training_context(model)

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

        assert "_reset_async_activation_offload_if_enabled(" in source


def test_train_step_uses_hsdp_allreduce_helper():
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)

        assert "_configure_hsdp_allreduce(" in source


def test_train_step_only_counts_step_after_batch_fetch_succeeds():
    """global_step must not advance for a step whose batch fetch raised StopIteration.

    `BaseTrainer.train()` catches `StopIteration` from `train_step()` and ends the
    epoch, so a loader that runs dry before `train_steps` is reached must not leave
    `global_step` counting a step that never ran (no optimizer step, no resume
    cursor movement for it).
    """
    for wrapper_cls in (BaseTrainer, TextTrainer, VLMTrainer, TextDPOTrainer, DiTTrainer):
        source = inspect.getsource(wrapper_cls.train_step)
        global_step_line = next(line for line in source.splitlines() if "state.global_step += 1" in line)
        next_batch_line = next(
            line for line in source.splitlines() if "next(data_iterator)" in line and "else" not in line
        )
        global_step_idx = source.index(global_step_line)
        next_batch_idx = source.index(next_batch_line)

        assert global_step_idx > next_batch_idx, (
            f"{wrapper_cls.__name__}.train_step increments global_step before the batch "
            "fetch that can raise StopIteration"
        )


def test_base_trainer_global_step_not_overcounted_on_stop_iteration():
    """End-to-end: train() must not overcount global_step when the loader runs dry."""

    trainer = BaseTrainer.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(
        train_steps=10,
        train=SimpleNamespace(local_rank=0, num_train_epochs=1),
        data=SimpleNamespace(dataloader=SimpleNamespace(use_background_prefetcher=False, drop_last=False)),
    )
    trainer.state = base_module.TrainerState()
    trainer.start_epoch = 0
    trainer.start_step = 0

    n_batches = 3
    trainer.train_dataloader = [[{"labels": object()}]] * n_batches

    trainer.on_train_begin = lambda: None
    trainer.on_train_end = lambda: None
    trainer.on_epoch_begin = lambda: None
    trainer.on_epoch_end = lambda: None
    trainer.on_step_begin = lambda **kwargs: None

    optimizer_steps = []
    trainer.on_step_end = lambda **kwargs: optimizer_steps.append(1)

    def fake_train_step(data_iterator):
        next(data_iterator)
        trainer.state.global_step += 1
        trainer.on_step_end()

    trainer.train_step = fake_train_step

    trainer.train()

    assert trainer.state.global_step == n_batches == len(optimizer_steps)

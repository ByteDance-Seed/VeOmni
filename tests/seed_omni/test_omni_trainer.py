"""Unit tests for the :class:`OmniTrainer` orchestration pieces that need no process group."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import MultiLRScheduler, OmniModelRuntime
from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime
from veomni.trainer.callbacks.base import TrainerState
from veomni.trainer.omni.omni_trainer import OmniTrainer, cascade_module_reshard


def test_the_export_stage_reaches_every_module_checkpoint_manager():
    """``train_end`` is what lets a manager drop the optimizer before the final
    export, so the stage ``CheckpointCallback`` passes must survive the fan-out."""
    managers = {name: MagicMock() for name in ("a", "b")}
    module_runtimes = {}
    for name, manager in managers.items():
        runtime = ModuleRuntime.__new__(ModuleRuntime)
        runtime.checkpoint = manager
        module_runtimes[name] = runtime
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.model = OmniModelRuntime.__new__(OmniModelRuntime)
    trainer.model.module_runtimes = module_runtimes
    state = TrainerState(global_step=2)

    trainer.save_hf_or_lora(state, stage="train_end")

    for manager in managers.values():
        manager.save_hf_or_lora.assert_called_once_with(state, stage="train_end")


@pytest.mark.parametrize(
    ("micro_step", "num_micro_steps", "expected"),
    [(0, 1, None), (0, 3, False), (1, 3, None), (2, 3, True)],
)
def test_cascade_module_reshard_keeps_params_gathered_between_micro_steps(micro_step, num_micro_steps, expected):
    runtimes = {"a": MagicMock(), "b": MagicMock()}

    cascade_module_reshard(runtimes, micro_step, num_micro_steps)

    for runtime in runtimes.values():
        if expected is None:
            runtime._model_reshard.assert_not_called()
        else:
            runtime._model_reshard.assert_called_once_with(expected)


def _accumulate_grads(num_micro_steps: int) -> torch.Tensor:
    weight = torch.nn.Parameter(torch.tensor([1.0, 2.0]))

    def forward(micro_batch):
        loss = (weight * micro_batch["x"]).pow(2).mean()
        return {"loss": loss, "losses": {}}

    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.model = SimpleNamespace(forward=forward, module_runtimes={})
    trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
    trainer.model_fwd_context = nullcontext()
    trainer.model_bwd_context = nullcontext()
    trainer.preforward = lambda micro_batch: micro_batch

    micro_batch = {"x": torch.tensor([3.0, -1.0])}
    for _ in range(num_micro_steps):
        trainer.forward_backward_step(micro_batch, num_micro_steps=num_micro_steps)
    return weight.grad


def test_gradient_accumulation_averages_micro_batch_gradients():
    """Accumulating N copies of one micro-batch must match a single step on it."""
    torch.testing.assert_close(_accumulate_grads(2), _accumulate_grads(1))


def test_an_exhausted_iterator_does_not_count_a_step():
    """``train`` catches the ``StopIteration`` and runs ``epoch_end``, whose
    checkpoints (and a later resume) read ``global_step``."""
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.state = TrainerState(global_step=3)

    with pytest.raises(StopIteration):
        trainer.train_step(iter(()))

    assert trainer.state.global_step == 3


def test_a_frozen_module_has_no_async_save_to_drain():
    runtime = ModuleRuntime.__new__(ModuleRuntime)
    runtime.checkpoint = None

    runtime.wait_for_pending_save()


def test_async_activation_offload_is_reset_only_on_modules_that_enable_it(monkeypatch):
    def _module(enable_async: bool) -> SimpleNamespace:
        offload_config = SimpleNamespace(enable_async_activation=enable_async)
        return SimpleNamespace(
            args=SimpleNamespace(accelerator=SimpleNamespace(offload_config=offload_config)),
            model=MagicMock(),
        )

    runtimes = {"offloaded": _module(True), "plain": _module(False)}
    reset = MagicMock()
    monkeypatch.setattr("veomni.trainer.omni.omni_trainer.reset_async_activation_offload", reset)
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.model = SimpleNamespace(module_runtimes=runtimes)

    trainer._reset_async_activation_offload_if_enabled()

    reset.assert_called_once_with(runtimes["offloaded"].model)


def test_multi_lr_scheduler_without_schedulers_reports_zero_lr():
    assert MultiLRScheduler({}).get_last_lr() == [0.0]


def test_an_all_frozen_model_runtime_has_no_optimizer_or_lr_scheduler():
    frozen = SimpleNamespace(optimizer=None, lr_scheduler=None, _build_lr_scheduler=MagicMock())
    runtime = OmniModelRuntime(MagicMock(), module_runtimes={"frozen": frozen})

    runtime._build_optimizer()
    runtime._build_lr_scheduler(total_steps=10)

    assert runtime.optimizer is None and runtime.lr_scheduler is None


def test_the_trainer_refuses_a_model_with_nothing_to_train(monkeypatch):
    monkeypatch.setattr("veomni.trainer.omni.omni_trainer.build_omni_model_runtime_args", MagicMock())
    monkeypatch.setattr(
        "veomni.trainer.omni.omni_trainer.build_omni_model_runtime",
        MagicMock(return_value=SimpleNamespace(optimizer=None)),
    )
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(train=SimpleNamespace())

    with pytest.raises(ValueError, match="every module is frozen"):
        trainer._build_model_runtime()


@pytest.mark.parametrize(("global_rank", "writes"), [(0, True), (1, False)])
def test_model_runtime_writes_root_assets_without_weights_on_rank_zero(global_rank, writes):
    train_args = SimpleNamespace(global_rank=global_rank, checkpoint=SimpleNamespace(model_assets_dir="/out/assets"))
    runtime = OmniModelRuntime(MagicMock(), train_args=train_args)
    runtime.save_pretrained = MagicMock()

    runtime.save_model_assets()

    if writes:
        runtime.save_pretrained.assert_called_once_with("/out/assets", save_module_weights=False)
    else:
        runtime.save_pretrained.assert_not_called()


def test_model_runtime_steps_only_the_trainable_modules():
    trainable = SimpleNamespace(optimizer=MagicMock(), lr_scheduler=None)
    trainable._build_lr_scheduler = lambda total_steps: setattr(trainable, "lr_scheduler", MagicMock())
    frozen = SimpleNamespace(optimizer=None, lr_scheduler=None, _build_lr_scheduler=MagicMock())
    runtime = OmniModelRuntime(MagicMock(), module_runtimes={"trainable": trainable, "frozen": frozen})
    assert runtime.optimizer is None and runtime.lr_scheduler is None

    runtime._build_optimizer()
    runtime._build_lr_scheduler(total_steps=10)

    assert list(runtime.optimizer.optimizers) == ["trainable"]
    assert list(runtime.lr_scheduler.schedulers) == ["trainable"]
    frozen._build_lr_scheduler.assert_called_once_with(10)

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
from veomni.trainer.callbacks.omni_callbacks import OmniModuleDcpCallback, OmniModuleHfCallback
from veomni.trainer.omni.omni_trainer import OmniTrainer, cascade_module_reshard


def _hf_callback(*, hf_save_steps: int = 0, hf_save_epochs: int = 0) -> OmniModuleHfCallback:
    checkpoint = SimpleNamespace(save_hf_weights=True, hf_save_steps=hf_save_steps, hf_save_epochs=hf_save_epochs)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint)))
    trainer.save_hf_or_lora = MagicMock()
    return OmniModuleHfCallback(trainer)


def test_hf_export_at_train_end_is_not_skipped_by_a_dcp_save_at_the_same_step():
    """The module managers' ``last_saved_step`` counts DCP saves; HF export must not read it."""
    callback = _hf_callback()
    state = TrainerState(global_step=2, stage="train_end")

    callback.on_train_end(state)

    callback.trainer.save_hf_or_lora.assert_called_once_with(state)


def test_hf_export_is_written_once_per_step():
    callback = _hf_callback(hf_save_steps=2)
    state = TrainerState(global_step=2, stage="step_end")
    callback.on_step_end(state)
    state.stage = "train_end"
    callback.on_train_end(state)

    assert callback.trainer.save_hf_or_lora.call_count == 1


def test_failed_hf_export_is_retried_at_train_end():
    callback = _hf_callback(hf_save_steps=2)
    callback.trainer.save_hf_or_lora.side_effect = [RuntimeError("disk full"), None]
    state = TrainerState(global_step=2, stage="step_end")
    with pytest.raises(RuntimeError):
        callback.on_step_end(state)
    state.stage = "train_end"
    callback.on_train_end(state)

    assert callback.trainer.save_hf_or_lora.call_count == 2


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


def test_the_moe_router_monitor_is_refused_before_anything_is_built():
    args = SimpleNamespace(train=SimpleNamespace(moe_load_balance_monitor_interval=10))

    with pytest.raises(AssertionError, match="moe_load_balance_monitor_interval"):
        OmniTrainer(args)


def test_callback_hooks_publish_the_stage_before_dispatch():
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.state = TrainerState()
    callback = MagicMock()
    callback.on_step_end.side_effect = lambda state, **kwargs: seen.append(state.stage)
    trainer._callbacks = [callback]
    seen = []

    trainer.on_step_end(loss=1.0, loss_dict={"lm": 1.0}, grad_norm=0.5)

    assert seen == ["step_end"]
    callback.on_step_end.assert_called_once_with(trainer.state, loss=1.0, loss_dict={"lm": 1.0}, grad_norm=0.5)


def test_an_exhausted_iterator_does_not_count_a_step():
    """``train`` catches the ``StopIteration`` and runs ``epoch_end``, whose
    checkpoints (and a later resume) read ``global_step``."""
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.state = TrainerState(global_step=3)

    with pytest.raises(StopIteration):
        trainer.train_step(iter(()))

    assert trainer.state.global_step == 3


def test_train_end_drains_in_flight_async_dcp_saves():
    checkpoint = SimpleNamespace(save_steps=0, save_epochs=0)
    trainer = SimpleNamespace(args=SimpleNamespace(train=SimpleNamespace(checkpoint=checkpoint)))
    trainer.wait_for_pending_save = MagicMock()

    OmniModuleDcpCallback(trainer).on_train_end(TrainerState(global_step=2, stage="train_end"))

    trainer.wait_for_pending_save.assert_called_once_with()


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

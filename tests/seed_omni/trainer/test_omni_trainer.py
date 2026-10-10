"""Unit tests for the :class:`OmniTrainer` orchestration pieces that need no process group."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import OmniModelRuntime
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


def test_the_trainer_refuses_a_model_with_nothing_to_train(monkeypatch):
    monkeypatch.setattr("veomni.trainer.omni.omni_trainer.build_omni_model_runtime_args", MagicMock())
    monkeypatch.setattr(
        "veomni.trainer.omni.omni_trainer.build_omni_model_runtime",
        MagicMock(return_value=SimpleNamespace(optimizer=None)),
    )
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(train=SimpleNamespace(training_task="online_training"))

    with pytest.raises(ValueError, match="every module is frozen"):
        trainer._build_model_runtime()


def test_an_offline_embedding_run_may_build_no_optimizer(monkeypatch):
    """It runs only the frozen modules' ``offline_encode``, so it trains nothing by design."""
    monkeypatch.setattr("veomni.trainer.omni.omni_trainer.build_omni_model_runtime_args", MagicMock())
    built = SimpleNamespace(optimizer=None)
    monkeypatch.setattr("veomni.trainer.omni.omni_trainer.build_omni_model_runtime", MagicMock(return_value=built))
    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.args = SimpleNamespace(train=SimpleNamespace(training_task="offline_embedding"))

    assert trainer._build_model_runtime() is built


def test_offline_cache_step_writes_every_micro_batch_without_autograd():
    """The cache is written from ``forward``'s return value: FSDP2 may hand the graph a copy of the batch."""
    grad_enabled = []

    def forward(micro_batch):
        grad_enabled.append(torch.is_grad_enabled())
        return {"loss": None, "losses": {}, "conversation_list": [*micro_batch["conversation_list"], "encoded"]}

    trainer = OmniTrainer.__new__(OmniTrainer)
    trainer.model = SimpleNamespace(forward=forward)
    trainer.args = SimpleNamespace(train=SimpleNamespace(enable_batch_invariant_mode=False))
    trainer.model_fwd_context = nullcontext()
    trainer.state = TrainerState(global_step=0)
    trainer.offline_cache_writer = MagicMock()
    trainer.on_step_begin = MagicMock()
    trainer.on_step_end = MagicMock()
    trainer.sync_before_train_step = MagicMock()
    trainer.model_reshard = MagicMock()
    trainer.preforward = lambda micro_batch: micro_batch

    trainer.offline_cache_step(iter([[{"conversation_list": ["a"]}, {"conversation_list": ["b"]}]]))

    assert grad_enabled == [False, False]
    assert [c.args[0] for c in trainer.offline_cache_writer.save_conversation_list.call_args_list] == [
        ["a", "encoded"],
        ["b", "encoded"],
    ]
    assert trainer.state.global_step == 1
    trainer.on_step_end.assert_called_once_with(loss=0.0, loss_dict={}, grad_norm=0.0)

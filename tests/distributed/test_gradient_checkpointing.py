import types
import weakref

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import noop_context_fn

from veomni.arguments import GradientCheckpointingConfig, MixedPrecisionConfig
from veomni.distributed.checkpoint import CheckpointFunction
from veomni.distributed.torch_parallelize import build_parallelize_model


class _CheckpointingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


class _RetainingCheckpointModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.recomputed_output = None
        self.recomputed_input_ref = None

    def forward(self, value):
        output = value.sin()
        if torch.is_grad_enabled():
            self.recomputed_output = output
            self.recomputed_input_ref = weakref.ref(value)
        return output * 2


@pytest.mark.parametrize("early_stop", [True, False])
@pytest.mark.parametrize("use_reentrant", [True, False])
def test_build_parallelize_model_forwards_checkpoint_early_stop(monkeypatch, early_stop, use_reentrant):
    import veomni.distributed.torch_parallelize as torch_parallelize

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: types.SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2"),
    )
    monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", lambda model, **kwargs: model)
    model = _CheckpointingModel()

    result = build_parallelize_model(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False),
        early_stop=early_stop,
        enable_reentrant=use_reentrant,
    )

    assert result is model
    expected = {
        "use_reentrant": use_reentrant,
        "context_fn": noop_context_fn,
    }
    if not use_reentrant:
        expected["early_stop"] = early_stop
    assert model.gradient_checkpointing_kwargs == expected


def test_gradient_checkpointing_config_enables_early_stop_by_default():
    assert GradientCheckpointingConfig().early_stop is True


def test_reentrant_checkpoint_releases_recomputed_input_grad():
    module = _RetainingCheckpointModule()
    value = torch.randn(8, requires_grad=True)

    output = CheckpointFunction.apply(module, False, value)
    output.sum().backward()

    recomputed_input = module.recomputed_input_ref()
    assert recomputed_input is not None
    assert recomputed_input.grad is None
    torch.testing.assert_close(value.grad, 2 * value.detach().cos())

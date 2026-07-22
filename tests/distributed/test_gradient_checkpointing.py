import types

import pytest
import torch.nn as nn
from torch.utils.checkpoint import noop_context_fn

from veomni.arguments import GradientCheckpointingConfig, MixedPrecisionConfig
from veomni.distributed.torch_parallelize import build_parallelize_model


class _CheckpointingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing_kwargs = None

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.gradient_checkpointing_kwargs = gradient_checkpointing_kwargs


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

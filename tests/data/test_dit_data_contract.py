"""DiT examples must reach the native parser and shared data transform."""

import sys
from pathlib import Path

import pytest

from veomni.arguments import parse_args
from veomni.trainer.dit_trainer import VeOmniDiTArguments


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("name", ["wan2.2_I2V_5B", "wan2.2_I2V_14B"])
def test_wan_i2v_yaml_uses_declared_arguments(name, monkeypatch):
    # Kernel availability is independent of the example's dataclass contract.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_dit.py",
            str(ROOT / "configs" / "dit" / f"{name}.yaml"),
            "--model.ops_implementation.load_balancing_loss_implementation",
            "eager",
        ],
    )
    args = parse_args(VeOmniDiTArguments)
    assert args.model.accelerator.fsdp_config.fsdp_mode == "fsdp2"
    assert args.model.accelerator.fsdp_config.mixed_precision.param_dtype == "bfloat16"
    assert not args.model.accelerator.offload_config.enable_async_activation
    assert not args.model.accelerator.offload_config.enable_activation
    assert args.model.accelerator.gradient_checkpointing.enable
    assert args.train.training_task == "online_training"

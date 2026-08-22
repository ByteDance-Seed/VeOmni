"""Behavioral tests for Qwen3.5-MoE EP load-balance configuration."""

import sys

import pytest
import yaml

from veomni.arguments import (
    AcceleratorConfig,
    CheckpointConfig,
    DataArguments,
    FSDPConfig,
    GradientCheckpointingConfig,
    ModelArguments,
    OffloadConfig,
    OpsImplementationConfig,
    TrainingArguments,
    VeOmniArguments,
    parse_args,
)
from veomni.arguments.arguments_types import MoEEPBalanceConfig


def _arguments(
    *,
    enabled=False,
    max_replicas_per_rank=1,
    ep_size=2,
    dp_replicate_size=-1,
    dp_shard_size=-1,
    fsdp_mode="fsdp2",
    ep_outside=False,
    fsdp_offload=False,
    activation_offload=False,
    load_path=None,
    save_steps=0,
    save_hf_weights=True,
    gradient_checkpointing_enable=False,
    moe_implementation="fused_triton",
    lora_config=None,
):
    return VeOmniArguments(
        model=ModelArguments(
            config_path="test-config",
            lora_config={} if lora_config is None else lora_config,
            ops_implementation=OpsImplementationConfig(
                moe_implementation=moe_implementation,
                load_balancing_loss_implementation="eager",
            ),
        ),
        data=DataArguments(train_path="test-data"),
        train=TrainingArguments(
            accelerator=AcceleratorConfig(
                ep_size=ep_size,
                dp_replicate_size=dp_replicate_size,
                dp_shard_size=dp_shard_size,
                ep_outside=ep_outside,
                fsdp_config=FSDPConfig(fsdp_mode=fsdp_mode, offload=fsdp_offload),
                offload_config=OffloadConfig(enable_activation=activation_offload),
            ),
            checkpoint=CheckpointConfig(load_path=load_path, save_steps=save_steps, save_hf_weights=save_hf_weights),
            gradient_checkpointing=GradientCheckpointingConfig(enable=gradient_checkpointing_enable),
            moe_ep_load_balance=MoEEPBalanceConfig(
                enabled=enabled,
                max_replicas_per_rank=max_replicas_per_rank,
            ),
        ),
    )


def test_moe_ep_load_balance_defaults_are_disabled():
    args = _arguments()

    assert args.train.moe_ep_load_balance == MoEEPBalanceConfig(enabled=False, max_replicas_per_rank=1)


def test_moe_ep_load_balance_parses_nested_yaml_and_cli_override(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "model": {
                    "config_path": "test-config",
                    "ops_implementation": {
                        "moe_implementation": "fused_triton",
                        "load_balancing_loss_implementation": "eager",
                    },
                },
                "data": {"train_path": "test-data"},
                "train": {
                    "accelerator": {"ep_size": 2},
                    "moe_ep_load_balance": {"enabled": False, "max_replicas_per_rank": 3},
                },
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "veomni",
            str(config_path),
            "--train.moe_ep_load_balance.enabled=true",
        ],
    )

    args = parse_args(VeOmniArguments)

    assert args.train.moe_ep_load_balance == MoEEPBalanceConfig(enabled=True, max_replicas_per_rank=3)


@pytest.mark.parametrize("max_replicas_per_rank", [0, -1])
def test_moe_ep_load_balance_requires_positive_replica_count(max_replicas_per_rank):
    with pytest.raises(ValueError, match="max_replicas_per_rank must be positive"):
        _arguments(enabled=True, max_replicas_per_rank=max_replicas_per_rank)


def test_moe_ep_load_balance_requires_expert_parallelism():
    with pytest.raises(ValueError, match="ep_size > 1"):
        _arguments(enabled=True, ep_size=1)


@pytest.mark.parametrize("moe_implementation", ["eager", "fused_quack"])
def test_moe_ep_load_balance_requires_supported_backend(moe_implementation):
    with pytest.raises(ValueError, match="fused_npu.*fused_triton"):
        _arguments(enabled=True, moe_implementation=moe_implementation)


def test_moe_ep_load_balance_requires_full_training():
    with pytest.raises(ValueError, match="full.*non-LoRA"):
        _arguments(enabled=True, lora_config={"r": 8})


def test_moe_ep_load_balance_requires_fsdp2():
    with pytest.raises(ValueError, match="fsdp_mode='fsdp2'"):
        _arguments(enabled=True, fsdp_mode="ddp")


def test_moe_ep_load_balance_rejects_hsdp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    with pytest.raises(ValueError, match="dp_replicate_size == 1"):
        _arguments(enabled=True, dp_replicate_size=2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"ep_outside": True}, "ep_outside=False"),
        ({"fsdp_offload": True}, "fsdp_config.offload=False"),
        ({"activation_offload": True}, "offload_config.enable_activation=False"),
        ({"load_path": "resume/global_step_10"}, "checkpoint.load_path=None"),
    ],
)
def test_moe_ep_load_balance_rejects_unsupported_enabled_combinations(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _arguments(enabled=True, **kwargs)


def test_moe_ep_load_balance_allows_checkpoint_saving_and_gradient_checkpointing():
    args = _arguments(
        enabled=True,
        save_steps=50,
        save_hf_weights=False,
        gradient_checkpointing_enable=True,
    )

    assert args.train.checkpoint.save_steps == 50
    assert args.train.checkpoint.save_hf_weights is False
    assert args.train.gradient_checkpointing.enable is True


def test_moe_ep_load_balance_disabled_preserves_legacy_combinations(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")

    args = _arguments(
        enabled=False,
        max_replicas_per_rank=0,
        ep_size=1,
        dp_replicate_size=2,
        fsdp_mode="ddp",
        ep_outside=True,
        fsdp_offload=True,
        activation_offload=True,
        load_path="resume/global_step_10",
        moe_implementation="eager",
        lora_config={"r": 8},
    )

    assert args.train.moe_ep_load_balance.enabled is False
    assert args.train.moe_ep_load_balance.max_replicas_per_rank == 0
    assert args.train.accelerator.dp_replicate_size == 2
    assert args.train.accelerator.fsdp_config.fsdp_mode == "ddp"
    assert args.train.accelerator.ep_outside is True
    assert args.train.accelerator.fsdp_config.offload is True
    assert args.train.accelerator.offload_config.enable_activation is True
    assert args.train.checkpoint.load_path == "resume/global_step_10"

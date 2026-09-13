import pytest
import torch

from veomni.arguments import GradientCheckpointingConfig, InterLayerReplayConfig, TrainingArguments


def test_inter_layer_replay_is_disabled_by_default() -> None:
    args = TrainingArguments()

    assert not args.inter_layer_replay.enable
    assert args.inter_layer_replay.window_size == 0
    assert not hasattr(args.inter_layer_replay, "replay_layer")


def test_inter_layer_replay_requires_gradient_checkpointing() -> None:
    with pytest.raises(ValueError, match="requires gradient checkpointing"):
        TrainingArguments(
            gradient_checkpointing=GradientCheckpointingConfig(enable=False),
            inter_layer_replay=InterLayerReplayConfig(enable=True),
        )


@pytest.mark.parametrize(
    ("config", "message"),
    [
        (InterLayerReplayConfig(enable=True, current_layer=-2), "current_layer must be -1 or non-negative"),
        (InterLayerReplayConfig(enable=True, window_size=-1), "window_size must be non-negative"),
        (
            InterLayerReplayConfig(enable=True, current_layer=7, window_size=8),
            "reaches below decoder layer 0",
        ),
    ],
)
def test_inter_layer_replay_rejects_invalid_windows(config: InterLayerReplayConfig, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingArguments(inter_layer_replay=config)


class _ParallelState:
    fsdp_enabled = True
    dp_mode = "fsdp2"
    tp_size = 1
    pp_size = 1
    ulysses_size = 1
    cp_size = 1


def test_inter_layer_replay_rejects_ulysses_parallelism(monkeypatch) -> None:
    from veomni.distributed import torch_parallelize

    state = _ParallelState()
    state.ulysses_size = 2
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: state)

    with pytest.raises(ValueError, match="ulysses_size=2"):
        torch_parallelize.build_parallelize_model(
            torch.nn.Linear(2, 2),
            inter_layer_replay_config=InterLayerReplayConfig(enable=True),
            inter_layer_replay_moe_implementation="fused_npu",
        )


def test_inter_layer_replay_requires_fused_npu_moe(monkeypatch) -> None:
    from veomni.distributed import torch_parallelize

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: _ParallelState())

    with pytest.raises(ValueError, match="moe_implementation='fused_npu'"):
        torch_parallelize.build_parallelize_model(
            torch.nn.Linear(2, 2),
            inter_layer_replay_config=InterLayerReplayConfig(enable=True),
            inter_layer_replay_moe_implementation="fused_torch",
        )

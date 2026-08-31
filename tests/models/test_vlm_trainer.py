import pytest

from veomni.models import build_foundation_model
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMModelRuntime,
    _get_vlm_visual_module,
)

from ..tools.training_utils import make_eager_ops_config, unbuilt_runtime


_FREEZE_VIT_VLM_CASES = [
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe"),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl"),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl"),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe"),
]


@pytest.mark.parametrize("freeze_vit", [False, True])
@pytest.mark.parametrize("config_path", _FREEZE_VIT_VLM_CASES)
def test_freeze_vit_on_vlm_model(config_path, freeze_vit):
    # This test only constructs the model on `meta` and verifies freeze
    # behaviour — it never runs forward. Use an all-eager ops config so the
    # build works everywhere: it pins every per-op field (including the
    # Qwen3.5 GatedDeltaNet trio that has no FLA backend on NPU and the
    # GPU-only liger/triton defaults that fail NPU validation). Eager paths
    # that raise only at forward time are fine because this test never
    # forwards.
    ops_implementation = make_eager_ops_config()
    model = build_foundation_model(
        config_path=config_path,
        weights_path=None,
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=ops_implementation,
    )
    visual = _get_vlm_visual_module(model)
    assert visual is not None

    args = VeOmniVLMArguments(
        model=VLMMModelArguments(config_path=config_path, ops_implementation=ops_implementation),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit
    runtime = unbuilt_runtime(args.model, cls=VLMModelRuntime, train=args.train)
    runtime.model = model
    runtime.model_config = model.config

    runtime.freeze_model()

    assert all(param.requires_grad is not freeze_vit for param in visual.parameters())

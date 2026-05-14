from types import SimpleNamespace

import pytest

from veomni.models import build_foundation_model
from veomni.trainer.vlm_trainer import (
    VeOmniVLMArguments,
    VLMMDataArguments,
    VLMMModelArguments,
    VLMTrainer,
    _get_vlm_visual_module,
)
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

from ..tools.training_utils import make_eager_ops_config


# Per-case version gate so the v4 lane has at least one collected test
# (otherwise the whole module would skip with ``allow_module_level=True``,
# pytest reports ``0 collected / 1 skipped`` and exits with code 5, failing CI).
_v5_only = pytest.mark.skipif(
    not is_transformers_version_greater_or_equal_to("5.0.0"),
    reason="Requires transformers >= 5.0.0",
)


_FREEZE_VIT_VLM_CASES = [
    # qwen2_vl keeps a v4 monkey-patch fallback (see
    # ``veomni/models/transformers/qwen2_vl/__init__.py``) so it runs on both
    # transformers stacks and acts as the v4 lane's smoke test for this file.
    pytest.param("./tests/toy_config/qwen2vl_toy/config.json", id="qwen2_vl"),
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5", marks=_v5_only),
    pytest.param("./tests/toy_config/qwen3_5_moe_toy/config.json", id="qwen3_5_moe", marks=_v5_only),
    pytest.param("./tests/toy_config/qwen25vl_toy/config.json", id="qwen2_5_vl", marks=_v5_only),
    pytest.param("./tests/toy_config/qwen3vl_toy/config.json", id="qwen3_vl", marks=_v5_only),
    pytest.param("./tests/toy_config/qwen3vlmoe_toy/config.json", id="qwen3_vl_moe", marks=_v5_only),
]


@pytest.mark.parametrize(
    "freeze_vit",
    [
        pytest.param(False, id="freeze_vit_disabled"),
        pytest.param(True, id="freeze_vit_enabled"),
    ],
)
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
        model=VLMMModelArguments(
            config_path=config_path,
            ops_implementation=make_eager_ops_config(),
        ),
        data=VLMMDataArguments(train_path="dummy"),
    )
    args.train.freeze_vit = freeze_vit

    trainer = VLMTrainer.__new__(VLMTrainer)
    trainer.base = SimpleNamespace(
        args=args,
        model=model,
        model_config=model.config,
    )

    trainer._freeze_model_module()

    if freeze_vit:
        assert all(not param.requires_grad for param in visual.parameters())
    else:
        assert all(param.requires_grad for param in visual.parameters())

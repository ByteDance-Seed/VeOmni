"""Regression tests for VLM-style optimizer param groups + DCP resume."""

import pytest
import torch
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import OptimizerConfig
from veomni.optim.optimizer import (
    build_optimizer,
    filter_empty_param_groups,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class _TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.backbone(x)


def test_optimizer_config_positional_learning_rate_preserves_adamw_update():
    config = OptimizerConfig("adamw", 0.02, 0.003, 0.001)
    assert (config.lr, config.lr_min, config.lr_start) == (0.02, 0.003, 0.001)

    model = _TinyModel()
    reference = _TinyModel()
    reference.load_state_dict(model.state_dict())
    optimizer = build_optimizer(model, lr=config.lr, betas=config.betas, weight_decay=0.0, optimizer_type=config.type)
    expected = torch.optim.AdamW(reference.parameters(), lr=0.02, betas=config.betas, weight_decay=0.0)
    for actual_param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
        actual_param.grad = torch.ones_like(actual_param)
        reference_param.grad = torch.ones_like(reference_param)
    optimizer.step()
    expected.step()
    for actual_param, reference_param in zip(model.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(actual_param, reference_param, rtol=0, atol=0)


def test_optimizer_config_offload_options_remain_keyword_compatible():
    config = OptimizerConfig(
        "cpu_offload_adamw",
        0.02,
        cpu_offload_param_patterns=["backbone.weight"],
        cpu_offload_resident_moment_dtype="bfloat16",
        cpu_offload_cpu_moment_dtype="float32",
    )
    assert config.lr == 0.02
    assert config.cpu_offload_param_patterns == ["backbone.weight"]
    assert config.cpu_offload_resident_moment_dtype == "bfloat16"
    assert config.cpu_offload_cpu_moment_dtype == "float32"


@pytest.mark.parametrize("optimizer_type", ["cpu_offload_adamw", "adamw", "anyprecision_adamw"])
def test_optimizer_config_rejects_old_draft_offload_positional_options(optimizer_type):
    with pytest.raises(TypeError, match="CPU-offload options by keyword"):
        OptimizerConfig(optimizer_type, ["backbone.weight"], "bfloat16", "float32")


def test_filter_empty_param_groups_drops_frozen_vit_group():
    p = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [], "lr": 1e-6},
        {"params": [p], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [p]


def test_filter_empty_param_groups_drops_frozen_only_group():
    """A group whose params are all requires_grad=False has no optimizer state
    and must be dropped (otherwise the same KeyError surfaces on resume)."""
    frozen = torch.nn.Parameter(torch.randn(2, 2), requires_grad=False)
    trainable = torch.nn.Parameter(torch.randn(2, 2))
    groups = [
        {"params": [frozen], "lr": 1e-6},
        {"params": [frozen, trainable], "lr": 2e-5},
    ]
    filtered = filter_empty_param_groups(groups)
    assert len(filtered) == 1
    assert filtered[0]["params"] == [trainable]  # frozen dropped, trainable kept


def test_build_optimizer_skips_empty_vlm_param_groups():
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )
    assert len(opt.param_groups) == 1
    assert opt.param_groups[0]["lr"] == 2e-5


def test_vlm_style_optimizer_dcp_roundtrip_step_succeeds():
    """Reproduces production freeze_vit resume: save/load must keep betas."""
    model = _TinyModel()
    opt = build_optimizer(
        model,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model.parameters()), "lr": 2e-5},
        ],
    )

    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()

    optim_sd = get_optimizer_state_dict(
        model=model,
        optimizers=opt,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )

    model2 = _TinyModel()
    opt2 = build_optimizer(
        model2,
        lr=2e-5,
        param_groups=[
            {"params": [], "lr": 1e-6},
            {"params": list(model2.parameters()), "lr": 2e-5},
        ],
    )
    set_optimizer_state_dict(
        model=model2,
        optimizers=opt2,
        optim_state_dict=optim_sd,
        options=StateDictOptions(flatten_optimizer_state_dict=True),
    )

    for group in opt2.param_groups:
        assert "betas" in group

    loss2 = model2(torch.randn(2, 4)).sum()
    loss2.backward()
    opt2.step()

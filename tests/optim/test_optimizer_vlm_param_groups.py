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


@pytest.mark.parametrize("optimizer_type", ["cpu_offload_adamw", "full_fp32_adamw", "master_fp32_adamw"])
def test_precision_optimizer_exposes_live_groups_to_omni(optimizer_type):
    from veomni.models.seed_omni.accelerated.omni_model.omni_model_runtime import MultiOptimizer as OmniMultiOptimizer
    from veomni.optim.lr_scheduler import build_lr_scheduler

    model = _TinyModel()
    optimizer = build_optimizer(
        model,
        lr=0.02,
        weight_decay=0,
        optimizer_type=optimizer_type,
        cpu_offload_param_patterns=["backbone.weight"] if optimizer_type == "cpu_offload_adamw" else None,
    )
    other = torch.nn.Linear(4, 4, bias=False)
    other_optimizer = torch.optim.AdamW(other.parameters(), lr=0.03)
    combined = OmniMultiOptimizer({"precision": optimizer, "other": other_optimizer})
    leaf_group = optimizer.optimizers_dict[optimizer.key_names[0]].param_groups[0]
    assert combined.param_groups[0] is leaf_group
    combined.param_groups[0]["lr"] = 0.01
    assert leaf_group["lr"] == 0.01

    schedule = build_lr_scheduler(optimizer, train_steps=4, lr=0.01, lr_decay_style="linear")
    before = model.backbone.weight.detach().clone()
    for param in (*model.parameters(), *other.parameters()):
        param.grad = torch.ones_like(param)
    combined.step()
    schedule.step()
    assert not torch.equal(model.backbone.weight, before)
    assert combined.param_groups[0]["lr"] == schedule.get_last_lr()[0]


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

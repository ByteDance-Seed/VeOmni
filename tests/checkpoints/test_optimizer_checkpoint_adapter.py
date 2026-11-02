# Copyright 2025 Bytedance Ltd. and/or its affiliates
# Licensed under the Apache License, Version 2.0.

"""Exercise the optimizer checkpoint extension without optional optimizer packages."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer, OptimizerState
from veomni.optim.checkpoint_adapter import register_rank_local_adamw
from veomni.optim.optimizer import MultiOptimizer


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


class ToyAdamW(torch.optim.Optimizer):
    def __init__(self, params, scale=1.0):
        super().__init__(params, {"lr": 0.01})
        self.scale = scale

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if not state:
                    state.update(
                        step=torch.tensor(0.0),
                        master_weight=param.detach().float().clone(),
                        exp_avg=torch.zeros_like(param, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(param, dtype=torch.float32),
                    )
                state["step"] += 1
                state["exp_avg"].mul_(0.9).add_(param.grad, alpha=0.1)
                state["exp_avg_sq"].mul_(0.95).addcmul_(param.grad, param.grad, value=0.05)
                state["master_weight"].add_(state["exp_avg"], alpha=-group["lr"] * self.scale)
                param.copy_(state["master_weight"])


register_rank_local_adamw(
    ToyAdamW,
    lambda _optimizer: nullcontext(),
    lambda optimizer: ("toy-v1", optimizer.scale),
    lambda _optimizer, _param, _field: torch.float32,
    lambda _optimizer, _param: False,
)


def _build(scale=1.0):
    torch.manual_seed(57)
    model = torch.nn.Linear(2, 2)
    leaf = ToyAdamW(model.parameters(), scale=scale)
    return model, MultiOptimizer(model, {"toy": leaf}, ["toy"])


def test_yaml_beta_sequence_is_immutable():
    from veomni.arguments.arguments_types import OptimizerConfig
    from veomni.arguments.parser import _instantiate_recursive

    config = _instantiate_recursive(OptimizerConfig, {"betas": [0.9, 0.95]})
    assert config.betas == (0.9, 0.95) and isinstance(config.betas, tuple)


def test_registered_optimizer_preserves_lazy_state_and_next_update(tmp_path):
    source, optimizer = _build()
    source.weight.grad = torch.ones_like(source.weight)
    optimizer.step()
    optimizer.zero_grad()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _build()
    DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    assert target.bias not in restored.optimizers_dict["toy"].state
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        left.grad = torch.full_like(left, 0.125)
        right.grad = left.grad.clone()
    optimizer.step()
    restored.step()
    for left, right in zip(source.parameters(), target.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_registered_optimizer_raw_snapshot_is_detached_but_restore_requires_proof():
    model, optimizer = _build()
    model.weight.grad = torch.ones_like(model.weight)
    optimizer.step()
    optimizer.zero_grad()
    snapshot = optimizer.state_dict()
    live = optimizer.optimizers_dict["toy"].state[model.weight]["master_weight"]
    saved = snapshot["state.weight.master_weight"]
    assert saved.data_ptr() != live.data_ptr()
    before = live.detach().clone()
    saved.add_(7)
    torch.testing.assert_close(live, before, rtol=0, atol=0)
    with pytest.raises(RuntimeError, match="Unverified MultiOptimizer state"):
        optimizer.load_state_dict(snapshot)


def test_registered_optimizer_rejects_changed_algorithm_before_loading(tmp_path):
    source, optimizer = _build()
    parallel = SimpleNamespace(dp_mode="ddp")
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = _build(scale=2.0)
    with pytest.raises(RuntimeError, match="layout|identity|checkpoint"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)


@pytest.mark.parametrize("changed_field", ["master_weight", "step"])
def test_lazy_adapter_state_dtype_policy_is_part_of_checkpoint_identity(tmp_path, changed_field):
    class ConfigurableToy(ToyAdamW):
        def __init__(self, params, dtype):
            super().__init__(params)
            self.configured_dtype = dtype

    register_rank_local_adamw(
        ConfigurableToy,
        lambda _optimizer: nullcontext(),
        lambda _optimizer: ("same-algorithm",),
        lambda optimizer, _param, field: optimizer.configured_dtype if field == changed_field else torch.float32,
        lambda _optimizer, _param: False,
    )

    def build(dtype):
        model = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
        leaf = ConfigurableToy(model.parameters(), dtype)
        return model, MultiOptimizer(model, {"leaf": leaf}, ["leaf"])

    parallel = SimpleNamespace(dp_mode="ddp")
    source, optimizer = build(torch.float32)
    assert not optimizer.optimizers_dict["leaf"].state
    DistributedCheckpointer.save(str(tmp_path), {"model": source, "optimizer": optimizer}, parallel_state=parallel)
    target, restored = build(torch.bfloat16)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="layout|identity|checkpoint"):
        DistributedCheckpointer.load(str(tmp_path), {"model": target, "optimizer": restored}, parallel_state=parallel)
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_registered_single_optimizer_is_rejected_before_any_save_or_model_load(tmp_path):
    # Registration alone declares exact state precision. In particular, an
    # implementation need not also define requires_exact_state_dtypes.
    source = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    optimizer = ToyAdamW(source.parameters())
    parallel = SimpleNamespace(dp_mode="ddp")
    assert not hasattr(optimizer, "requires_exact_state_dtypes")
    source.weight.grad = torch.ones_like(source.weight)
    optimizer.step()
    assert optimizer.state[source.weight]["master_weight"].dtype == torch.float32

    invalid_dir = tmp_path / "single-optimizer"
    with pytest.raises(RuntimeError, match="require MultiOptimizer"):
        DistributedCheckpointer.save(
            str(invalid_dir), {"model": source, "optimizer": optimizer}, parallel_state=parallel
        )
    assert not invalid_dir.exists()

    raw = OptimizerState(source, optimizer, parallel_state=parallel)
    with pytest.raises(RuntimeError, match="require MultiOptimizer"):
        raw.state_dict()
    with pytest.raises(RuntimeError, match="require MultiOptimizer"):
        raw.load_state_dict({})

    model_dir = tmp_path / "model-only"
    DistributedCheckpointer.save(str(model_dir), {"model": source}, parallel_state=parallel)
    target = torch.nn.Linear(2, 2, dtype=torch.bfloat16)
    with torch.no_grad():
        target.weight.fill_(42)
    before = target.weight.detach().clone()
    with pytest.raises(RuntimeError, match="require MultiOptimizer"):
        DistributedCheckpointer.load(
            str(model_dir), {"model": target, "optimizer": ToyAdamW(target.parameters())}, parallel_state=parallel
        )
    torch.testing.assert_close(target.weight, before, rtol=0, atol=0)


def test_registered_adapter_cannot_fabricate_optimizer_group_options(tmp_path):
    class FabricatingToy(ToyAdamW):
        def state_dict(self):
            state = super().state_dict()
            state["param_groups"][0]["factor"] = 2
            return state

    register_rank_local_adamw(
        FabricatingToy,
        lambda _optimizer: nullcontext(),
        lambda _optimizer: ("toy-v1",),
        lambda _optimizer, _param, _field: torch.float32,
        lambda _optimizer, _param: False,
    )
    model = torch.nn.Linear(2, 2)
    leaf = FabricatingToy(model.parameters())
    optimizer = MultiOptimizer(model, {"leaf": leaf}, ["leaf"])
    with pytest.raises(RuntimeError, match="fabricated checkpoint group options"):
        DistributedCheckpointer.save(
            str(tmp_path), {"model": model, "optimizer": optimizer}, parallel_state=SimpleNamespace(dp_mode="ddp")
        )
    assert not list(tmp_path.iterdir())

# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Contracts of the ``extra_parallel_only`` path that no other test reaches single-process."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.parallel_plan import SpecInfo
from veomni.distributed.torch_compile import CompileConfig
from veomni.distributed.torch_parallelize import (
    _divide_extra_parallel_grads_like_fsdp,
    parallelize_model_fsdp2,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _param_with_spec(*, persistent: int | None) -> nn.Parameter:
    param = nn.Parameter(torch.zeros(4, 4))
    param.spec_info = SpecInfo(
        para_name="ple",
        placement=Shard(0),
        fqn="table",
        para_fsdp_mesh=None,
        persistent_fsdp_shard_dim=persistent,
    )
    return param


def _apply_backward_hooks(param: nn.Parameter, value: torch.Tensor) -> torch.Tensor:
    """Run ``param``'s tensor hooks the way autograd would."""
    for hook in (param._backward_hooks or {}).values():
        value = hook(value)
    return value


def test_the_gradient_hook_skips_persistent_parameters():
    """Their operator already averages, so a hook here divides twice.

    Qwen4-Exp PLE's ``_Qwen4ExpScaleGradient`` divides by
    ``extra_parallel_gradient_divide_factor("ple")`` in its own backward. A second
    division from this hook would leave the gradient ``world_size`` times too
    small, which is silent -- training still runs, just far too slowly.
    """
    model = nn.Module()
    model.persistent = _param_with_spec(persistent=0)
    model.plain = _param_with_spec(persistent=None)
    # A plain assignment of a Parameter to an `nn.Module` attribute registers it,
    # so `named_parameters()` sees both.
    assert {"persistent", "plain"} == {name for name, _ in model.named_parameters()}
    parallel_state = SimpleNamespace(
        extra_parallel_enabled=lambda name: True,
        extra_parallel_gradient_divide_factor=lambda name: 8,
    )
    spec_info = {fqn: p.spec_info for fqn, p in model.named_parameters()}

    _divide_extra_parallel_grads_like_fsdp(model, parallel_state, spec_info)

    grad = torch.full((4, 4), 8.0)
    assert torch.all(_apply_backward_hooks(model.persistent, grad) == 8.0), (
        "a persistent param must not be divided; its operator already did"
    )
    assert torch.all(_apply_backward_hooks(model.plain, grad) == 1.0), (
        "a non-persistent plan shard is divided by the factor"
    )


def test_extra_parallel_only_rejects_torch_compile_instead_of_running_eager(monkeypatch):
    """The unwrapped path returns before ``validate_compile_runtime``.

    Without this refusal a config that asks for compile would silently train
    eager -- the config claims one thing and the run does another, with no
    warning.
    """
    monkeypatch.setattr(
        "veomni.distributed.torch_parallelize.get_parallel_state",
        lambda: SimpleNamespace(fsdp_enabled=True, dp_mode="fsdp2"),
    )

    with pytest.raises(RuntimeError, match="extra_parallel_only"):
        parallelize_model_fsdp2(
            nn.Linear(2, 2),
            extra_parallel_only=True,
            compile_config=CompileConfig(enable=True),
            init_device="meta",
        )


def test_a_plan_less_module_under_an_enabled_extra_parallel_dim_says_how_to_fix_it(monkeypatch):
    """The assertion is right to fire; what it says decides whether it is actionable.

    A plan-less module is normally a per-module overlay that inherited the global
    mesh (a vision tower under a backbone's `ep`). The bare message only pointed at
    example plans, which does not tell that user the cheaper fix is to turn the
    dimension off for this module.
    """
    from types import SimpleNamespace

    import torch.nn as nn

    from veomni.distributed import torch_parallelize

    monkeypatch.setattr(
        torch_parallelize,
        "get_parallel_state",
        lambda: SimpleNamespace(
            fsdp_enabled=True,
            dp_mode="fsdp2",
            any_extra_parallel_enabled=True,
            extra_parallel_names=("ep",),
            extra_parallel_enabled=lambda name: True,
            extra_parallel_fsdp_device_mesh={"ep": None},
        ),
    )
    monkeypatch.setattr(torch_parallelize, "get_runtime_parallel_plan", lambda model: None)

    with pytest.raises(AssertionError) as excinfo:
        torch_parallelize.parallelize_model_fsdp2(nn.Linear(2, 2), init_device="meta")

    message = str(excinfo.value)
    assert "parallel plan" in message
    assert "ep_size: 1" in message, "must name the per-module opt-out, which is the usual fix"
    assert "qwen3_moe/parallel_plan.py" in message, "must still point at a worked example"

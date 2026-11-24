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
from veomni.distributed.torch_parallelize import (
    _divide_extra_parallel_grads_like_fsdp,
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


def test_shared_lora_params_are_collected_when_the_model_is_unwrapped():
    """Shared-expert LoRA params are collected as plain parameters on this path.

    The DTensor assertion below this guards call order; ``extra_parallel_only``
    never wraps, so the snapshot is already final and the plain parameters must
    be accepted -- otherwise the mode cannot be combined with shared-expert LoRA.
    """
    import torch
    import torch.nn as nn

    from veomni.optim.optimizer import _collect_ep_replicated_lora_param_ids

    class LoraSharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 2))

    model = nn.Sequential(LoraSharedExperts())
    model._extra_parallel_only = True

    assert _collect_ep_replicated_lora_param_ids(model) == {id(model[0].gate_up_proj)}


def test_shared_lora_grad_accumulation_survives_multiple_microbatches():
    """One division per backward, applied to the whole accumulated grad.

    ``LoraSharedExperts`` reduces through ``register_post_accumulate_grad_hook``,
    which fires on every backward over the *whole* accumulated ``p.grad``: the
    all-reduce SUMs every rank's contribution, so a replicated parameter ends up
    holding ``world_size`` times its own value, and dividing that buffer once
    restores it. Modelled here at the level the hooks operate on, since the real
    one needs a process group.

    What this pins is that the division rides on the same callback as the sum. A
    division installed as a tensor hook instead runs on each new contribution
    while the sum covers the accumulation, so micro-batch 2 re-scales
    micro-batch 1 and the grad grows by the EP width per step.
    """
    import torch
    import torch.nn as nn

    world_size = 4
    # Each EP rank holds the same replicated adapter and computes only its own
    # experts' contribution, for two micro-batches.
    my_contributions = [1.0, 3.0]
    expected = sum(my_contributions)  # replicated: the value is rank-invariant

    param = nn.Parameter(torch.zeros(1))

    def ep_sum_hook(p):
        # SUM over the EP group: a replicated param accumulates world_size copies
        # of its own value. Then undo it, which is what FSDP's divide factor did.
        p.grad.mul_(world_size).div_(world_size)

    param.register_post_accumulate_grad_hook(ep_sum_hook)

    for contribution in my_contributions:
        # `+=` is what an accumulation step does to `.grad`.
        if param.grad is None:
            param.grad = torch.tensor([contribution])
        else:
            param.grad += torch.tensor([contribution])
        for hook in list(param._post_accumulate_grad_hooks.values()):
            hook(param)

    assert torch.allclose(param.grad, torch.tensor([expected])), (
        f"grad accumulation drifted: {param.grad.item()} vs {expected}"
    )

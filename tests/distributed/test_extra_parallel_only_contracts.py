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

    Getting the fix right took two attempts. `ep_size: 1` *is* the answer for
    `ep`, because `_module_base` folds an inherited `ep` entry into `ep_size`
    before the overlay merges; clearing the three lists is what fails, since the
    folded width comes back through `AcceleratorConfig.__post_init__`. The
    message must therefore point at `ep_size` and must not send users to the
    lists.
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
            extra_parallel_sizes={"ep": 4},
            extra_parallel_enabled=lambda name: True,
            extra_parallel_fsdp_device_mesh={"ep": None},
        ),
    )
    monkeypatch.setattr(torch_parallelize, "get_runtime_parallel_plan", lambda model: None)

    with pytest.raises(AssertionError) as excinfo:
        torch_parallelize.parallelize_model_fsdp2(nn.Linear(2, 2), init_device="meta")

    message = str(excinfo.value)
    assert "parallel plan" in message
    assert "qwen3_moe/parallel_plan.py" in message, "must still point at a worked example"
    assert "ep_size: 1" in message, "must name the actual opt-out for `ep`"
    assert "does NOT work" in message, "must warn that clearing the lists is not the fix"
    assert "Enabled dimensions" in message


def test_cpu_offload_is_refused_even_when_every_parameter_is_plan_sharded():
    """The narrower guard missed the tied-`TextEncoder` shape.

    Its only parameter is `emb`-sharded, so nothing was replicated and a
    `replicated_bytes`-only guard passed -- but `ShardedEmbedding._project`
    all-gathers the vocabulary table with accelerator activations, and a
    host-resident table cannot take part. Driven at `_finish_without_fsdp_wrap`:
    that is the function that owns the check, and reaching it through
    `parallelize_model_fsdp2` would need a stub for every earlier config check.
    """
    from types import SimpleNamespace

    import torch.nn as nn
    from torch.distributed.tensor import Shard

    from veomni.distributed.torch_parallelize import _finish_without_fsdp_wrap

    model = nn.Linear(4, 4)
    spec_info = {
        fqn: SimpleNamespace(para_name="emb", placement=Shard(0), persistent_fsdp_shard_dim=None)
        for fqn, _ in model.named_parameters()
    }
    parallel_state = SimpleNamespace(
        extra_parallel_names=("emb",),
        extra_parallel_enabled=lambda name: True,
        extra_parallel_fsdp_size=lambda name: 1,
    )

    with pytest.raises(ValueError, match="offload cannot be combined"):
        _finish_without_fsdp_wrap(
            model,
            weights_path=None,
            parallel_state=parallel_state,
            fqn2spec_info=spec_info,
            should_skip_hf_weight_load=True,
            kwargs={"enable_fsdp_offload": True, "init_device": "meta"},
        )


def test_shared_expert_lora_grads_are_divided_like_the_skipped_wrap_would(monkeypatch):
    """Shared-expert LoRA stays usable without the wrap, with the same grad scale.

    The adapters are replicated across the EP group; ``_ensure_ep_grad_sync_hooks``
    SUMs their grads over that group and counts on the wrap's
    ``set_gradient_divide_factor(world_size)`` to turn that sum into
    ``total_grad / world_size``. Without a wrap the division has to be reinstated,
    or every step is ``world_size`` times too large.

    Driven through ``_finish_without_fsdp_wrap`` rather than through the helper
    directly: calling the helper would still pass if the call site forgot it.
    """
    from types import SimpleNamespace

    import torch
    import torch.nn as nn
    from torch.distributed.tensor import Shard

    from veomni.distributed import torch_parallelize

    class LoraSharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 2))

    model = nn.Sequential(LoraSharedExperts())
    spec_info = {
        fqn: SimpleNamespace(para_name="ep", placement=Shard(0), persistent_fsdp_shard_dim=None)
        for fqn, _ in model.named_parameters()
    }
    parallel_state = SimpleNamespace(
        extra_parallel_names=("ep",),
        extra_parallel_enabled=lambda name: True,
        extra_parallel_fsdp_size=lambda name: 1,
        world_size=8,
        extra_parallel_gradient_divide_factor=lambda name: 8,
    )

    seen = {}

    def _record(m, ps):
        seen["model"] = m
        seen["world_size"] = ps.world_size

    monkeypatch.setattr(torch_parallelize, "_divide_replicated_lora_grads_like_fsdp", _record)
    torch_parallelize._finish_without_fsdp_wrap(
        model,
        weights_path=None,
        parallel_state=parallel_state,
        fqn2spec_info=spec_info,
        should_skip_hf_weight_load=True,
        kwargs={"init_device": "meta"},
    )

    assert seen.get("model") is model, "the no-wrap path must reinstate the shared-LoRA divide"
    assert seen.get("world_size") == 8


def test_the_shared_lora_divide_is_by_world_size_and_skips_frozen_params():
    """The factor itself: grads shrink by ``world_size``, frozen params get no hook."""
    from types import SimpleNamespace

    import torch
    import torch.nn as nn

    from veomni.distributed.torch_parallelize import _divide_replicated_lora_grads_like_fsdp

    class LoraSharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 2))
            self.frozen = nn.Parameter(torch.zeros(2, 2), requires_grad=False)

    # factor 1 registers nothing: the wrapped path's divide factor is 1 there too.
    plain = nn.Sequential(LoraSharedExperts())
    _divide_replicated_lora_grads_like_fsdp(plain, SimpleNamespace(world_size=1))
    assert not plain[0].gate_up_proj._backward_hooks

    model = nn.Sequential(LoraSharedExperts())
    _divide_replicated_lora_grads_like_fsdp(model, SimpleNamespace(world_size=8))
    param = model[0].gate_up_proj
    assert not model[0].frozen._backward_hooks, "a frozen parameter has no grad to divide"
    # A real backward, so the hook's output lands in `.grad` as it will in training.
    param.sum().backward()
    assert torch.equal(param.grad, torch.full((2, 2), 1.0 / 8))


def test_shared_lora_params_are_collected_unwrapped_but_a_missing_wrap_still_raises():
    """The DTensor assertion guards call order, not the unwrapped path.

    It exists because a snapshot taken before ``fully_shard`` rewraps the module
    goes stale immediately. ``extra_parallel_only`` never wraps, so plain
    parameters there are final and must be accepted -- otherwise the mode cannot
    be combined with shared-expert LoRA at all. A model *without* that marker
    still has to fail, or the misuse the assertion was written for returns.
    """
    import torch
    import torch.nn as nn

    from veomni.optim.optimizer import _collect_ep_replicated_lora_param_ids

    class LoraSharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 2))

    unwrapped = nn.Sequential(LoraSharedExperts())
    unwrapped._extra_parallel_only = True
    assert _collect_ep_replicated_lora_param_ids(unwrapped) == {id(unwrapped[0].gate_up_proj)}

    # No marker: still the call-order mistake, so it must keep raising.
    missing = nn.Sequential(LoraSharedExperts())
    try:
        _collect_ep_replicated_lora_param_ids(missing)
    except AssertionError as exc:
        assert "not a DTensor" in str(exc)
    else:  # pragma: no cover - the assertion is the point
        raise AssertionError("a plain parameter on an unwrapped-by-accident model must still be refused")

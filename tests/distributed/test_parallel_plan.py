# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

"""Which modules ``ParallelPlan`` hands the parallelizer to wrap on each para mesh, plus ``extra_parallel_only`` unit contracts."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.distributed.parallel_plan import ParallelPlan, SpecInfo
from veomni.distributed.torch_parallelize import _divide_extra_parallel_grads_like_fsdp
from veomni.distributed.utils import sort_fqn_by_submodule_first
from veomni.models.transformers.qwen3_5_moe.parallel_plan import get_parallel_plan as qwen3_5_moe_plan


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_every_planned_parameter_owner_is_a_para_module():
    """An owner left out would be re-sharded over the regular FSDP mesh, mixing ranks' slices."""
    assert qwen3_5_moe_plan().extra_parallel_fsdp_no_shard_module == {
        "ep": {"model.language_model.layers.*.mlp.experts", "mtp.layers.*.mlp.experts"}
    }


def test_owners_of_one_module_collapse_to_one_entry():
    plan = ParallelPlan(
        extra_parallel_plan={"ep": {"layers.*.experts.gate_up_proj": Shard(0), "layers.*.experts.down_proj": Shard(0)}}
    )
    assert plan.extra_parallel_fsdp_no_shard_module == {"ep": {"layers.*.experts"}}


def test_every_owner_is_found_in_the_model():
    model = nn.Module()
    model.embed_tokens = nn.Embedding(8, 4)
    model.decoder = nn.Module()
    model.decoder.ngram = nn.Embedding(8, 4)
    model.decoder.ngram_extra = nn.Embedding(8, 4)
    plan = ParallelPlan(
        extra_parallel_plan={"emb": {"embed_tokens.weight": Shard(0), "decoder.ngram.weight": Shard(0)}}
    )

    found = plan.get_extra_parallel_fsdp_no_shard_info(model, "emb")
    assert found == {"embed_tokens": model.embed_tokens, "decoder.ngram": model.decoder.ngram}


def test_nested_owners_are_rejected():
    with pytest.raises(ValueError, match="'decoder' and 'decoder.ngram' are nested"):
        ParallelPlan(extra_parallel_plan={"emb": {"decoder.weight": Shard(0), "decoder.ngram.weight": Shard(0)}})


def test_owners_that_only_share_a_name_prefix_are_not_nested():
    plan = ParallelPlan(extra_parallel_plan={"emb": {"ngram.weight": Shard(0), "ngram_extra.weight": Shard(0)}})
    assert plan.extra_parallel_fsdp_no_shard_module == {"emb": {"ngram", "ngram_extra"}}


def test_sort_fqn_by_submodule_first_preserves_siblings_and_orders_descendants():
    layers = [f"model.layers.{i}" for i in range(15)]
    assert sort_fqn_by_submodule_first(layers) == layers
    # Non-adjacent ancestor/descendant pair must still place child before parent while preserving sibling order.
    assert sort_fqn_by_submodule_first(["decoder", "embed_tokens", "decoder.embed_tokens"]) == [
        "decoder.embed_tokens",
        "decoder",
        "embed_tokens",
    ]
    assert sort_fqn_by_submodule_first(["layers.0", "layers.1", "layers.0.experts"]) == [
        "layers.0.experts",
        "layers.0",
        "layers.1",
    ]
    # Numeric prefix "model.layers.1" must not be treated as an ancestor of "model.layers.10".
    assert sort_fqn_by_submodule_first(
        ["model.layers.0", "model.layers.1", "model.layers.1.sub", "model.layers.10"]
    ) == [
        "model.layers.0",
        "model.layers.1.sub",
        "model.layers.1",
        "model.layers.10",
    ]
    assert sort_fqn_by_submodule_first(
        ["", "model", "model.layers.0", "model.layers.1", "model.layers.0.mlp.experts", "model.layers.0.mlp"]
    ) == [
        "model.layers.0.mlp.experts",
        "model.layers.0.mlp",
        "model.layers.0",
        "model.layers.1",
        "model",
        "",
    ]


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
    for hook in (param._backward_hooks or {}).values():
        value = hook(value)
    return value


def test_the_gradient_hook_skips_persistent_parameters():
    """Persistent ExtraParallel params divide in their own backward and must not be divided twice."""
    model = nn.Module()
    model.persistent = _param_with_spec(persistent=0)
    model.plain = _param_with_spec(persistent=None)
    parallel_state = SimpleNamespace(
        extra_parallel_enabled=lambda name: True,
        extra_parallel_gradient_divide_factor=lambda name: 8,
    )
    spec_info = {fqn: p.spec_info for fqn, p in model.named_parameters()}

    _divide_extra_parallel_grads_like_fsdp(model, parallel_state, spec_info)

    grad = torch.full((4, 4), 8.0)
    assert torch.all(_apply_backward_hooks(model.persistent, grad) == 8.0)
    assert torch.all(_apply_backward_hooks(model.plain, grad) == 1.0)


def test_shared_lora_params_are_collected_when_the_model_is_unwrapped(monkeypatch):
    """``extra_parallel_only`` keeps shared-expert LoRA params as plain tensors, which the collector accepts."""
    from veomni.optim import optimizer as opt_mod
    from veomni.optim.optimizer import _collect_ep_replicated_lora_param_ids

    class LoraSharedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(2, 2))

    model = nn.Sequential(LoraSharedExperts())
    monkeypatch.setattr(opt_mod, "get_parallel_state", lambda: SimpleNamespace(extra_parallel_only=True))

    assert _collect_ep_replicated_lora_param_ids(model) == {id(model[0].gate_up_proj)}


def test_shared_lora_grad_accumulation_survives_multiple_microbatches(monkeypatch):
    """Unwrapped ``LoraSharedExperts`` divides its EP-summed grad once per backward, preserving accumulation."""
    import torch.distributed as dist

    from veomni.lora import moe_layers
    from veomni.lora.moe_layers import LoraSharedExperts

    world_size = 4
    layer = object.__new__(LoraSharedExperts)
    nn.Module.__init__(layer)
    layer.lora_A = nn.Parameter(torch.zeros(2, 2))
    layer._ep_grad_hooks_done = False

    fake_ps = SimpleNamespace(ep_enabled=True, ep_group="ep_group", world_size=world_size, extra_parallel_only=True)

    def fake_all_reduce(tensor, op=None, group=None):
        assert group == "ep_group"
        tensor.mul_(world_size)

    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", fake_all_reduce)
    monkeypatch.setattr(moe_layers, "get_parallel_state", lambda: fake_ps)

    layer._ensure_ep_grad_sync_hooks()
    hooks = list(layer.lora_A._post_accumulate_grad_hooks.values())
    assert len(hooks) == 1

    for value in (1.0, 3.0):
        step_grad = torch.full((2, 2), value)
        layer.lora_A.grad = step_grad if layer.lora_A.grad is None else layer.lora_A.grad + step_grad
        for hook in hooks:
            hook(layer.lora_A)

    torch.testing.assert_close(layer.lora_A.grad, torch.full((2, 2), 4.0))

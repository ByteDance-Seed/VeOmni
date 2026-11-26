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
"""CPU tests for native MoE-LoRA via ``VeOmniLoraModel`` (Phase 2).

Exercises the injection + save/load contract *without* running the experts
forward (which needs the fused kernel / distributed state), so the suite is
CPU-only and dependency-light:

* ``target_parameters`` injection installs the right wrapper flavour
  (independent Mode 1 / shared Mode 2) at the matched experts FQNs.
* MoE metadata is embedded in ``adapter_config.json`` (``veomni_lora.moe_mode``
  + ``target_parameters``) — no ``veomni_moe_lora.json`` sidecar.
* ``get_lora_state_dict`` emits PEFT-format spec keys with the right rank
  (3-D per-expert for independent, 2-D for shared).
* ``save_pretrained`` -> ``from_pretrained`` (wrappers rebuilt from config) +
  native weight load round-trips the adapter tensors bit-exact.
* MoE mode is inferred from tensor shapes when the config lacks a
  ``veomni_lora`` block (stock-PEFT-style adapter).
* A two-rank Gloo probe checks shared-LoRA FSDP2/EP gradient accumulation,
  activation recomputation, and parameter updates without the fused GPU kernel.
"""

from __future__ import annotations

import copy
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
import torch.nn as nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import spawn_with_rendezvous
from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.moe_layers import (
    LoraSharedExperts,
    is_lora_independent_experts,
    is_lora_moe_experts,
    is_lora_shared_experts,
)
from veomni.lora.state_dict import get_lora_state_dict, load_adapter_state_dict
from veomni.lora.weight_loading import init_lora_parameter, load_lora_weights


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
_CI_CPU_2 = LaunchSpec(nproc=2, device="cpu")


torch.manual_seed(0)

NUM_EXPERTS = 4
HIDDEN = 16
INTER = 32
RANK = 4


class ToyExperts(nn.Module):
    """Minimal fused-layout experts module (v5 Qwen3-MoE-style)."""

    def __init__(self) -> None:
        super().__init__()
        self.num_experts = NUM_EXPERTS
        self.hidden_dim = HIDDEN
        self.intermediate_dim = INTER
        self.act_fn = nn.SiLU()
        self.gate_up_proj = nn.Parameter(torch.randn(NUM_EXPERTS, 2 * INTER, HIDDEN) * 0.02)
        self.down_proj = nn.Parameter(torch.randn(NUM_EXPERTS, HIDDEN, INTER) * 0.02)


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = ToyExperts()


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _MLP()


class ToyMoE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Layer()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # unused; kept for nn.Module completeness
        return x


TARGET_PARAMS = [
    "layers.*.mlp.experts.gate_up_proj",
    "layers.*.mlp.experts.down_proj",
]


def _moe_config(mode: str) -> VeOmniLoraConfig:
    return VeOmniLoraConfig.from_yaml(
        {
            "rank": RANK,
            "alpha": 8,
            "target_parameters": TARGET_PARAMS,
            "share_expert_lora": mode == "shared",
        }
    )


def _experts(model: VeOmniLoraModel) -> nn.Module:
    return model.base_model.model.get_submodule("layers.0.mlp.experts")


def _randomize_lora_b(model: VeOmniLoraModel) -> None:
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_B." in name:
                p.normal_(0.0, 0.02)


_SHARED_LORA_PROBE_MODULES = tuple(
    f"{projection}.lora_{factor}.default"
    for projection in ("gate_proj", "up_proj", "down_proj")
    for factor in ("A", "B")
)


class _SharedLoraAccumulationProbe(LoraSharedExperts):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        self._forward_calls = getattr(self, "_forward_calls", 0) + 1
        self._ensure_ep_grad_sync_hooks()
        # Use each A/B independently: zero-initialized B must not mask A's
        # gradient, and every parameter gets a distinct, linear oracle.
        outputs = []
        for coefficient, name in enumerate(_SHARED_LORA_PROBE_MODULES, start=1):
            linear = self.get_submodule(name)
            outputs.append(coefficient * linear(x[:, : linear.in_features]))
        return tuple(outputs)


def _shared_lora_ep_accumulation_worker(rank: int, checkpointed: bool, defer_reshard: bool, endpoint) -> None:
    import torch.distributed as dist
    from torch.distributed._composable.fsdp import fully_shard
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard
    from torch.utils.checkpoint import DefaultDeviceType, checkpoint

    previous_device_type = DefaultDeviceType.get_device_type()
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=_CI_CPU_2.nproc)
    try:
        # Spawned workers do not inherit the pytest parent's CPU device policy.
        DefaultDeviceType.set_device_type("cpu")
        mesh = init_device_mesh("cpu", (2, 1), mesh_dim_names=("ep", "ep_fsdp"))
        experts = _SharedLoraAccumulationProbe(ToyExperts(), r=RANK, lora_alpha=8)
        fully_shard(experts, mesh=mesh["ep_fsdp"], shard_placement_fn=lambda param: Shard(1))
        experts.set_gradient_divide_factor(2)
        weights = {name: experts.get_submodule(name).weight for name in _SHARED_LORA_PROBE_MODULES}
        assert {f"{name}.weight" for name in weights} == {
            name for name, param in experts.named_parameters() if param.requires_grad
        }
        initial = {name: weight.to_local().detach().clone() for name, weight in weights.items()}
        optimizer = torch.optim.SGD(weights.values(), lr=0.1, foreach=False)
        state = SimpleNamespace(ep_enabled=True, ep_group=mesh["ep"].get_group())
        forward_checks = []
        with mock.patch("veomni.lora.moe_layers.get_parallel_state", return_value=state):
            # d(sum(c * linear(x)))/dW[i,j] = c * x[j]. The rank and
            # microbatch factors sum to (1 + 2) * (1 + 2) = 9; FSDP's
            # divide factor gives 4.5 * c * (j + 1) for every output row.
            for micro in (1, 2):
                if defer_reshard:
                    experts.set_reshard_after_backward(micro == 2)
                x = torch.arange(1, INTER + 1, dtype=torch.float32, device="cpu").unsqueeze(0)
                x = x * micro * (rank + 1)
                outputs = checkpoint(experts, x, use_reentrant=False) if checkpointed else experts(x)
                for coefficient, (name, output) in enumerate(zip(weights, outputs), start=1):
                    expected = coefficient * (x[:, : initial[name].shape[1]] @ initial[name].T)
                    forward_checks.append((name, output.detach().clone(), expected))
                sum(output.sum() for output in outputs).backward()

        actual_grads = {name: weight.grad.to_local().detach().clone() for name, weight in weights.items()}
        optimizer.step()
        replicas = {}
        for name, weight in weights.items():
            grad_replicas = [torch.empty_like(initial[name]) for _ in range(2)]
            weight_replicas = [torch.empty_like(initial[name]) for _ in range(2)]
            dist.all_gather(grad_replicas, actual_grads[name], group=state.ep_group)
            dist.all_gather(weight_replicas, weight.to_local(), group=state.ep_group)
            replicas[name] = (grad_replicas, weight_replicas)

        # Complete every collective before asserting, including fault-injected
        # runs where one parameter has missed its synchronization hook.
        for name, actual, expected in forward_checks:
            torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6, msg=name)
        assert experts._forward_calls == (4 if checkpointed else 2)
        for coefficient, (name, weight) in enumerate(weights.items(), start=1):
            columns = torch.arange(1, initial[name].shape[1] + 1, dtype=torch.float32, device="cpu")
            expected_grad = (4.5 * coefficient * columns).expand_as(initial[name])
            torch.testing.assert_close(actual_grads[name], expected_grad, rtol=0, atol=0, msg=f"{name} gradient")
            torch.testing.assert_close(
                weight.to_local(), initial[name] - 0.1 * expected_grad, rtol=1e-6, atol=1e-7, msg=f"{name} update"
            )
            grad_replicas, weight_replicas = replicas[name]
            torch.testing.assert_close(
                grad_replicas[0], grad_replicas[1], rtol=0, atol=0, msg=f"{name} gradient replicas"
            )
            torch.testing.assert_close(
                weight_replicas[0], weight_replicas[1], rtol=0, atol=0, msg=f"{name} weight replicas"
            )
    finally:
        DefaultDeviceType.set_device_type(previous_device_type)
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=_CI_CPU_2)
@pytest.mark.parametrize("checkpointed", [False, True])
@pytest.mark.parametrize("defer_reshard", [False, True])
def test_shared_lora_ep_gradient_accumulation(checkpointed, defer_reshard):
    spawn_with_rendezvous(
        _shared_lora_ep_accumulation_worker,
        args=(checkpointed, defer_reshard),
        nprocs=_CI_CPU_2.nproc,
    )


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_moe_injection_structure(mode):
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    experts = _experts(model)
    assert is_lora_moe_experts(experts)
    if mode == "independent":
        assert is_lora_independent_experts(experts)
    else:
        assert is_lora_shared_experts(experts)

    names = [n for n, _ in model.named_parameters()]
    assert "base_model.model.layers.0.mlp.experts.gate_up_proj.base_layer.weight" in names
    assert "base_model.model.layers.0.mlp.experts.down_proj.base_layer.weight" in names
    for spec in ("gate_proj", "up_proj", "down_proj"):
        assert f"base_model.model.layers.0.mlp.experts.{spec}.lora_A.default.weight" in names
        assert f"base_model.model.layers.0.mlp.experts.{spec}.lora_B.default.weight" in names

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable
    assert all(".lora_A." in n or ".lora_B." in n for n in trainable)


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_moe_config_embeds_mode(tmp_path, mode):
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    model.get_lora_config().save_pretrained(str(tmp_path))
    with open(tmp_path / "adapter_config.json") as f:
        raw = json.load(f)
    assert raw["veomni_lora"]["moe_mode"] == mode
    assert raw["target_parameters"] == TARGET_PARAMS
    # No sidecar in the native format.
    assert not (tmp_path / "veomni_moe_lora.json").exists()


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_moe_state_dict_peft_key_format(mode):
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    sd = get_lora_state_dict(model, config=model.get_lora_config())
    key = "base_model.model.layers.0.mlp.experts.gate_proj.lora_A.weight"
    assert key in sd, sorted(sd)
    assert all(".default." not in k for k in sd)
    expected_ndim = 3 if mode == "independent" else 2
    assert sd[key].ndim == expected_ndim
    if mode == "independent":
        assert sd[key].shape == (NUM_EXPERTS, RANK, HIDDEN)
    else:
        assert sd[key].shape == (RANK, HIDDEN)


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_moe_save_reload_param_roundtrip(tmp_path, mode):
    base = ToyMoE()
    model = VeOmniLoraModel(copy.deepcopy(base), _moe_config(mode))
    _randomize_lora_b(model)
    saved = {k: v.detach().clone() for k, v in get_lora_state_dict(model, config=model.get_lora_config()).items()}

    model.save_pretrained(str(tmp_path), safe_serialization=True)

    reloaded = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(tmp_path))
    load_lora_weights(reloaded, str(tmp_path), init_device="cpu")

    got = get_lora_state_dict(reloaded, config=reloaded.get_lora_config())
    assert set(got) == set(saved)
    for k in saved:
        torch.testing.assert_close(got[k], saved[k])


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_moe_mode_inferred_when_config_lacks_block(tmp_path, mode):
    """A stock-PEFT-style adapter (no veomni_lora block) infers mode from shapes."""
    base = ToyMoE()
    model = VeOmniLoraModel(copy.deepcopy(base), _moe_config(mode))
    _randomize_lora_b(model)
    model.save_pretrained(str(tmp_path), safe_serialization=True)

    # Strip the veomni_lora block to simulate a stock-PEFT config.
    cfg_path = tmp_path / "adapter_config.json"
    with open(cfg_path) as f:
        raw = json.load(f)
    raw.pop("veomni_lora", None)
    with open(cfg_path, "w") as f:
        json.dump(raw, f)

    reloaded = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(tmp_path))
    assert reloaded.get_lora_config().moe_mode == mode
    # And weights still load into the rebuilt wrappers.
    load_lora_weights(reloaded, str(tmp_path), init_device="cpu")
    on_disk = load_adapter_state_dict(str(tmp_path), device="cpu")
    live = get_lora_state_dict(reloaded, config=reloaded.get_lora_config())
    assert set(on_disk) == set(live)


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_init_lora_parameter_preserves_loaded_weights(mode):
    """GAP-1 / BUG-3 guard: post-load init must NOT reset a populated wrapper.

    ``post_process_after_weight_loading`` walks missing LoRA names through
    ``init_lora_parameter``. For a MoE wrapper the reset only fires when *all*
    of its LoRA tensors are in ``parameter_names_left``; once
    ``load_lora_weights`` has materialised them they are removed from that set,
    so the reset must be skipped and cannot re-randomise ``lora_A`` / re-zero
    ``lora_B``. Assert both the values survive and that the wrapper's
    ``reset_lora_parameters`` is never called.
    """
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    experts = _experts(model)

    # Fill every LoRA tensor with a recognizable non-zero sentinel, mimicking
    # weights just loaded from a checkpoint (lora_B is otherwise zero-init).
    sentinel = 0.1234
    with torch.no_grad():
        for name, p in model.named_parameters():
            if ".lora_A." in name or ".lora_B." in name:
                p.fill_(sentinel)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if ".lora_" in n}
    assert before, "expected LoRA params on the wrapped MoE model"

    # Fully loaded adapter -> empty missing set (matches post_process after a
    # successful load_lora_weights).
    with mock.patch.object(experts, "reset_lora_parameters", wraps=experts.reset_lora_parameters) as reset_spy:
        for name in before:
            init_lora_parameter(model, name, parameter_names_left=set())
        reset_spy.assert_not_called()

    after = {n: p for n, p in model.named_parameters() if ".lora_" in n}
    assert set(after) == set(before)
    for n, prev in before.items():
        torch.testing.assert_close(after[n], prev, msg=lambda s, n=n: f"{n} was clobbered by init_lora_parameter\n{s}")


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_init_lora_parameter_resets_when_all_missing(mode):
    """Fresh wrapper (every LoRA name still missing) must kaiming-A / zero-B once."""
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    experts = _experts(model)
    lora_names = {n for n, _ in model.named_parameters() if ".lora_A." in n or ".lora_B." in n}
    assert lora_names

    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in lora_names:
                p.fill_(float("nan"))

    with mock.patch.object(experts, "reset_lora_parameters", wraps=experts.reset_lora_parameters) as reset_spy:
        for name in sorted(lora_names):
            init_lora_parameter(model, name, parameter_names_left=set(lora_names))
        assert reset_spy.call_count == 1

    for name, p in model.named_parameters():
        if ".lora_B." in name:
            assert torch.count_nonzero(p).item() == 0, name
        if ".lora_A." in name:
            assert torch.isfinite(p).all() and not torch.isnan(p).any(), name


@pytest.mark.parametrize("mode", ["independent", "shared"])
def test_init_lora_parameter_raises_on_partial_missing(mode):
    """Partial adapter resume must fail closed instead of clobbering loaded A/B."""
    model = VeOmniLoraModel(ToyMoE(), _moe_config(mode))
    lora_names = sorted(n for n, _ in model.named_parameters() if ".lora_A." in n or ".lora_B." in n)
    assert len(lora_names) >= 2
    missing = {lora_names[0]}
    with pytest.raises(RuntimeError, match="only partially loaded"):
        init_lora_parameter(model, lora_names[0], parameter_names_left=missing)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

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

import copy
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from veomni.models.transformers.deepseek_v4.generated import patched_modeling_deepseek_v4_gpu as modeling
from veomni.ops.dispatch import OpSlot
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_REGISTRY_MODULE = "veomni.ops.kernel_registry"


class _RecordingSlot:
    use_non_eager_impl = True

    def __init__(self, output):
        self.output = output
        self.args = None

    def __call__(self, *args):
        self.args = args
        return self.output


def test_deepseek_v4_declares_liger_opslots():
    assert isinstance(modeling.veomni_rms_norm, OpSlot)
    assert modeling.veomni_rms_norm.op_name == "rms_norm"
    assert modeling.veomni_rms_norm.variant == "standard"
    assert isinstance(modeling.veomni_unweighted_rms_norm, OpSlot)
    assert modeling.veomni_unweighted_rms_norm.op_name == "rms_norm"
    assert modeling.veomni_unweighted_rms_norm.variant == "unweighted"
    assert isinstance(modeling.veomni_swiglu_mlp, OpSlot)
    assert modeling.veomni_swiglu_mlp.op_name == "swiglu_mlp"
    assert modeling.veomni_swiglu_mlp.variant == "standard"


@patch("veomni.utils.import_utils.is_torch_npu_available", return_value=True)
@patch(f"{_REGISTRY_MODULE}.IS_NPU_AVAILABLE", True)
def test_deepseek_v4_npu_binds_weighted_and_unweighted_rmsnorm(_mock_npu, monkeypatch):
    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.models.auto import _bind_veomni_ops

    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    fake_module = SimpleNamespace(
        veomni_rms_norm=OpSlot("rms_norm", "standard"),
        veomni_unweighted_rms_norm=OpSlot("rms_norm", "unweighted"),
    )
    ops_config = OpsImplementationConfig(
        attn_implementation="eager",
        moe_implementation="eager",
        cross_entropy_loss_implementation="eager",
        rms_norm_implementation="npu",
        swiglu_mlp_implementation="eager",
        rotary_pos_emb_implementation="eager",
        load_balancing_loss_implementation="eager",
        rms_norm_gated_implementation="eager",
        causal_conv1d_implementation="eager",
        chunk_gated_delta_rule_implementation="eager",
    )

    assert _bind_veomni_ops(fake_module, ops_config)
    assert fake_module.veomni_rms_norm.use_non_eager_impl
    assert fake_module.veomni_unweighted_rms_norm.use_non_eager_impl


def test_deepseek_v4_unweighted_rmsnorm_dispatches_without_weight(monkeypatch):
    output = torch.randn(2, 4, 8)
    slot = _RecordingSlot(output)
    monkeypatch.setattr(modeling, "veomni_unweighted_rms_norm", slot)

    norm = modeling.DeepseekV4UnweightedRMSNorm(eps=1e-6)
    hidden_states = torch.randn_like(output)

    assert norm(hidden_states) is output
    assert slot.args[0] is hidden_states
    assert slot.args[1:] == (None, norm.eps)


def test_deepseek_v4_mlp_applies_swiglu_limit_before_liger(monkeypatch):
    captured = {}

    class _FakeLigerSiLUMulFunction:
        @staticmethod
        def apply(gate, up):
            captured["gate"] = gate.detach().clone()
            captured["up"] = up.detach().clone()
            return torch.nn.functional.silu(gate) * up

    monkeypatch.setitem(sys.modules, "liger_kernel", ModuleType("liger_kernel"))
    monkeypatch.setitem(sys.modules, "liger_kernel.ops", ModuleType("liger_kernel.ops"))
    swiglu_mod = ModuleType("liger_kernel.ops.swiglu")
    swiglu_mod.LigerSiLUMulFunction = _FakeLigerSiLUMulFunction
    monkeypatch.setitem(sys.modules, "liger_kernel.ops.swiglu", swiglu_mod)

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", _RecordingSlot(None))

    config = SimpleNamespace(
        hidden_size=8,
        intermediate_size=16,
        mlp_bias=False,
        hidden_act="silu",
        swiglu_limit=0.5,
    )
    mlp = modeling.DeepseekV4MLP(config)
    hidden_states = torch.linspace(-2.0, 2.0, 64).reshape(2, 4, 8)

    output = mlp(hidden_states)
    assert output.shape == hidden_states.shape
    assert captured["gate"].max().item() <= 0.5 + 1e-6
    assert captured["up"].min().item() >= -0.5 - 1e-6
    assert captured["up"].max().item() <= 0.5 + 1e-6


def _require_liger_cuda():
    pytest.importorskip("liger_kernel")
    if not IS_CUDA_AVAILABLE:
        pytest.skip("Liger kernels require CUDA")


def _run_rmsnorm_forward_backward(module, hidden_states, grad_output):
    output = module(hidden_states)
    output.backward(grad_output)
    weight_grad = module.weight.grad.detach().clone() if hasattr(module, "weight") else None
    return output.detach(), hidden_states.grad.detach().clone(), weight_grad


@pytest.mark.parametrize("weighted", [True, False])
def test_deepseek_v4_liger_rmsnorm_matches_eager(monkeypatch, weighted):
    _require_liger_cuda()
    torch.manual_seed(0)
    device = get_device_type()
    shape = (2, 8, 128)

    if weighted:
        eager_module = modeling.DeepseekV4RMSNorm(shape[-1], eps=1e-6).to(device=device, dtype=torch.bfloat16)
    else:
        eager_module = modeling.DeepseekV4UnweightedRMSNorm(eps=1e-6).to(device=device)
    liger_module = copy.deepcopy(eager_module)

    eager_input = torch.randn(shape, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_input = eager_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(eager_input)

    variant = "standard" if weighted else "unweighted"
    slot_name = "veomni_rms_norm" if weighted else "veomni_unweighted_rms_norm"
    monkeypatch.setattr(modeling, slot_name, OpSlot("rms_norm", variant))
    eager_result = _run_rmsnorm_forward_backward(eager_module, eager_input, grad_output)

    liger_slot = OpSlot("rms_norm", variant)
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(modeling, slot_name, liger_slot)
    liger_result = _run_rmsnorm_forward_backward(liger_module, liger_input, grad_output)

    for eager_value, liger_value in zip(eager_result, liger_result, strict=True):
        if eager_value is not None:
            torch.testing.assert_close(liger_value, eager_value, atol=2e-2, rtol=2e-2)


def _run_mlp_forward_backward(module, hidden_states, grad_output):
    output = module(hidden_states)
    output.backward(grad_output)
    parameter_grads = {name: parameter.grad.detach().clone() for name, parameter in module.named_parameters()}
    return output.detach(), hidden_states.grad.detach().clone(), parameter_grads


def test_deepseek_v4_liger_shared_expert_swiglu_matches_eager(monkeypatch):
    _require_liger_cuda()
    torch.manual_seed(1)
    device = get_device_type()
    config = SimpleNamespace(
        hidden_size=128, intermediate_size=256, mlp_bias=False, hidden_act="silu", swiglu_limit=1e9
    )
    eager_module = modeling.DeepseekV4MLP(config).to(device=device, dtype=torch.bfloat16)
    liger_module = copy.deepcopy(eager_module)

    eager_input = torch.randn(2, 8, config.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    liger_input = eager_input.detach().clone().requires_grad_()
    grad_output = torch.randn_like(eager_input)

    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", OpSlot("swiglu_mlp", "standard"))
    eager_output, eager_input_grad, eager_parameter_grads = _run_mlp_forward_backward(
        eager_module, eager_input, grad_output
    )

    liger_slot = OpSlot("swiglu_mlp", "standard")
    liger_slot.bind("liger_kernel")
    monkeypatch.setattr(modeling, "veomni_swiglu_mlp", liger_slot)
    liger_output, liger_input_grad, liger_parameter_grads = _run_mlp_forward_backward(
        liger_module, liger_input, grad_output
    )

    torch.testing.assert_close(liger_output, eager_output, atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(liger_input_grad, eager_input_grad, atol=3e-2, rtol=3e-2)
    for name, eager_grad in eager_parameter_grads.items():
        torch.testing.assert_close(liger_parameter_grads[name], eager_grad, atol=3e-2, rtol=3e-2)

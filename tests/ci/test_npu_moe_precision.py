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

"""Local validation for optional NPU expert gradient precision."""

from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments.arguments_types import MixedPrecisionConfig
from veomni.distributed.npu_moe_precision import validate_expert_precision_config, validate_qwen4_experts


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _accelerator(enabled):
    mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=enabled)
    fsdp = SimpleNamespace(mixed_precision=mixed, fsdp_mode="fsdp2", low_precision_reduce_scatter_comm=False)
    return SimpleNamespace(fsdp_config=fsdp, tp_size=1, pp_size=1, torch_compile=SimpleNamespace(enable=False))


def _validate_config(mixed, *, mode="fsdp2", low_comm=False, tp=1, pp=1, compile_enabled=False):
    validate_expert_precision_config(mixed, mode, low_comm, tp, pp, compile_enabled)


def test_valid_or_disabled_configuration_needs_no_process_group():
    _validate_config(MixedPrecisionConfig(npu_expert_fp32_wgrad=True))
    _validate_config(MixedPrecisionConfig(npu_expert_fp32_wgrad=False), mode="ddp", tp=2)


@pytest.mark.parametrize(
    "mixed_changes,parallel_changes",
    [
        ({"enable": False}, {}),
        ({"param_dtype": "float32"}, {}),
        ({"reduce_dtype": "bfloat16"}, {}),
        ({}, {"mode": "ddp"}),
        ({}, {"low_comm": True}),
        ({}, {"tp": 2}),
        ({}, {"pp": 2}),
        ({}, {"compile_enabled": True}),
    ],
)
def test_unsupported_configuration_is_rejected(mixed_changes, parallel_changes):
    mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=True, **mixed_changes)
    with pytest.raises(ValueError, match="FP32 wgrad requires"):
        _validate_config(mixed, **parallel_changes)


@pytest.mark.parametrize("value", ["false", 0, 1, None])
def test_expert_precision_option_requires_real_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        MixedPrecisionConfig(npu_expert_fp32_wgrad=value)


def test_direct_fsdp2_rejects_unsupported_precision(monkeypatch):
    from veomni.distributed import torch_parallelize

    state = SimpleNamespace(dp_mode="fsdp2", tp_size=1, pp_size=1, fsdp_enabled=True)
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: state)
    mixed = MixedPrecisionConfig(npu_expert_fp32_wgrad=True, reduce_dtype="bfloat16")
    with pytest.raises(ValueError, match="FP32 wgrad requires"):
        torch_parallelize.parallelize_model_fsdp2(object(), mixed_precision=mixed)


class Qwen4ExpTextExperts(torch.nn.Module):
    def __init__(self, impl="fused_npu", swiglu=True, dtype=torch.float32):
        super().__init__()
        self.veomni_moe = SimpleNamespace(impl=impl)
        self.use_swiglu_mlp = swiglu
        self.weight = torch.nn.Parameter(torch.ones(2, 2, dtype=dtype))


def _model_with_experts(*, model_type="qwen4_exp", impl="fused_npu", swiglu=True, dtype=torch.float32):
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type=model_type)
    model.experts = Qwen4ExpTextExperts(impl, swiglu, dtype)
    return model


def test_local_qwen4_experts_are_accepted():
    model = _model_with_experts()
    assert validate_qwen4_experts(model) == {id(model.experts)}


def test_qwen4_model_without_experts_is_rejected():
    model = torch.nn.Module()
    model.config = SimpleNamespace(model_type="qwen4_exp")
    with pytest.raises(ValueError, match="requires Qwen4-Exp expert modules"):
        validate_qwen4_experts(model)


@pytest.mark.parametrize("wrapper_name", ["LoraSharedExperts", "LoraIndependentExperts"])
def test_qwen4_model_with_partial_moe_lora_is_rejected(wrapper_name):
    from veomni.lora import moe_layers

    wrapper_type = getattr(moe_layers, wrapper_name)
    wrapper = wrapper_type.__new__(wrapper_type)
    torch.nn.Module.__init__(wrapper)
    model = _model_with_experts()
    model.lora_experts = wrapper
    with pytest.raises(ValueError, match="does not support MoE LoRA expert wrappers"):
        validate_qwen4_experts(model)


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"model_type": "other"}, "requires Qwen4-Exp expert modules"),
        ({"impl": "eager"}, "requires Qwen4-Exp SiLU fused_npu"),
        ({"swiglu": False}, "requires Qwen4-Exp SiLU fused_npu"),
        ({"dtype": torch.bfloat16}, "requires FP32 expert master parameters"),
    ],
)
def test_invalid_local_qwen4_experts_are_rejected(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_qwen4_experts(_model_with_experts(**kwargs))


def test_qwen4_vlm_decoder_layers_are_fsdp_targets():
    import torch
    from transformers import Qwen4ExpConfig

    from tests.models.compare import eager_ops_config, ops_config_scope
    from veomni.distributed.torch_parallelize import _fsdp_target_classes, _is_fsdp_wrap_target
    from veomni.models.transformers.qwen4_exp.generated.patched_modeling_qwen4_exp_npu import (
        Qwen4ExpForConditionalGeneration,
    )

    config = Qwen4ExpConfig.from_pretrained("tests/toy_config/qwen4_exp_toy/config.json")
    config._attn_implementation = "eager"
    with ops_config_scope(eager_ops_config()), torch.device("meta"):
        model = Qwen4ExpForConditionalGeneration(config)
    targets = _fsdp_target_classes(model, [], True)
    blocks = {
        name for name, module in model.named_modules() if _is_fsdp_wrap_target(name, type(module).__name__, targets)
    }
    experts = [name for name, module in model.named_modules() if type(module).__name__ == "Qwen4ExpTextExperts"]
    assert experts and all(any(name.startswith(block + ".") for block in blocks) for name in experts)
    # Also cover a root that has not inherited the child model's no-split list.
    assert "Qwen4ExpTextDecoderLayer" in _fsdp_target_classes(SimpleNamespace(_no_split_modules=None), [], True)
    assert not _fsdp_target_classes(SimpleNamespace(_no_split_modules=None), [], False)


@pytest.mark.parametrize("for_inference", [False, True])
def test_seed_omni_module_does_not_silently_ignore_opt_in(for_inference):
    from veomni.models.seed_omni.accelerated.omni_module.omni_module_runtime import ModuleRuntime

    accelerator = _accelerator(True)
    args = SimpleNamespace(accelerator=accelerator)
    with pytest.raises(ValueError, match="unsupported by SeedOmni"):
        ModuleRuntime(args, "test", module_config=None, global_accelerator=accelerator, for_inference=for_inference)

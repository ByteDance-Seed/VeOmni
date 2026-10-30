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

import builtins
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.ci.accelerator.launch_plan import LaunchSpec
from veomni.arguments import AcceleratorConfig, Float8TrainingConfig, MixedPrecisionConfig
from veomni.arguments.parser import parse_args, save_args
from veomni.distributed.float8 import convert_float8_linears, validate_float8_training
from veomni.utils import device


cpu = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
TARGET = "model.layers.0.mlp.up_proj"


def _precision(*targets):
    return MixedPrecisionConfig(float8=Float8TrainingConfig(enable=True, target_modules=list(targets)))


def _state(**overrides):
    values = dict(
        dp_mode="fsdp2", sp_enabled=False, any_extra_parallel_enabled=False, tp_size=1, pp_size=1, dp_replicate_size=1
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class DenseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(model_type="qwen3")
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        self.model.layers[0].mlp = nn.Module()
        self.model.layers[0].mlp.up_proj = nn.Linear(32, 32, bias=False)
        self.lm_head = nn.Linear(32, 32, bias=False)

    @property
    def fc(self):
        return self.model.layers[0].mlp.up_proj


@cpu
def test_float8_disabled_never_imports_torchao(monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        assert not name.startswith("torchao")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    model = DenseModel()
    precision = MixedPrecisionConfig()
    original = model.fc
    validate_float8_training(precision, _state(dp_mode="ddp"), cpu_offload=True, compile_enabled=True)
    assert convert_float8_linears(model, precision.float8) == []
    assert model.fc is original
    assert not asdict(precision)["float8"]["enable"]


@cpu
@pytest.mark.parametrize("targets", [[], [""], "fc", [1]])
def test_float8_requires_explicit_module_patterns(targets):
    with pytest.raises(ValueError, match="target_modules"):
        Float8TrainingConfig(enable=True, target_modules=targets)


@cpu
def test_float8_yaml_cli_and_saved_config_roundtrip(monkeypatch, tmp_path):
    path = tmp_path / "accelerator.yaml"
    path.write_text(
        "fsdp_config:\n  mixed_precision:\n    float8:\n      enable: true\n"
        "      target_modules:\n        - model.layers.*.mlp.up_proj\n"
    )
    monkeypatch.setattr("sys.argv", ["test", str(path)])
    config = parse_args(AcceleratorConfig)
    assert isinstance(config.fsdp_config.mixed_precision.float8, Float8TrainingConfig)
    assert config.fsdp_config.mixed_precision.float8.enable
    assert config.fsdp_config.mixed_precision.float8.target_modules == ["model.layers.*.mlp.up_proj"]
    save_args(config, str(tmp_path / "saved"))
    monkeypatch.setattr("sys.argv", ["test", str(tmp_path / "saved" / "veomni_cli.yaml")])
    assert asdict(parse_args(AcceleratorConfig)) == asdict(config)
    monkeypatch.setattr("sys.argv", ["test", str(path), "--fsdp_config.mixed_precision.float8.enable", "false"])
    assert not parse_args(AcceleratorConfig).fsdp_config.mixed_precision.float8.enable
    path.write_text("fsdp_config:\n  mixed_precision:\n    float8:\n      enable: true\n      target_modules: []\n")
    monkeypatch.setattr("sys.argv", ["test", str(path)])
    with pytest.raises(ValueError, match="target_modules"):
        parse_args(AcceleratorConfig)


@cpu
@pytest.mark.parametrize(
    "changes, kwargs, message",
    [
        ({"dp_mode": "ddp"}, {}, "FSDP2"),
        ({"sp_enabled": True}, {}, "sequence"),
        ({"any_extra_parallel_enabled": True}, {}, "ExtraParallel"),
        ({"tp_size": 2}, {}, "tensor"),
        ({"pp_size": 2}, {}, "pipeline"),
        ({"dp_replicate_size": 2}, {}, "HSDP"),
        ({}, {"cpu_offload": True}, "offload"),
        ({}, {"compile_enabled": True}, "compile"),
        ({}, {"is_peft_model": True}, "LoRA"),
    ],
)
def test_float8_rejects_unvalidated_combinations(monkeypatch, changes, kwargs, message):
    monkeypatch.setattr(device, "get_device_type", lambda: "cuda")
    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda: 89)
    with pytest.raises(ValueError, match=message):
        validate_float8_training(_precision("fc"), _state(**changes), **kwargs)


@cpu
@pytest.mark.parametrize("param_dtype, reduce_dtype", [("float16", "float32"), ("bfloat16", "bfloat16")])
def test_float8_requires_bf16_compute_and_fp32_reduction(param_dtype, reduce_dtype):
    precision = _precision("fc")
    precision.param_dtype, precision.reduce_dtype = param_dtype, reduce_dtype
    with pytest.raises(ValueError, match="bfloat16.*float32"):
        validate_float8_training(precision, _state())


@cpu
@pytest.mark.parametrize("device_type, capability", [("cpu", 0), ("npu", 0), ("cuda", 80)])
def test_float8_rejects_unsupported_hardware(monkeypatch, device_type, capability):
    monkeypatch.setattr(device, "get_device_type", lambda: device_type)
    monkeypatch.setattr(device, "IS_CUDA_AVAILABLE", device_type == "cuda")
    monkeypatch.setattr(device, "get_gpu_compute_capability", lambda: capability)
    with pytest.raises(ValueError, match="SM89"):
        validate_float8_training(_precision("fc"), _state())


@cpu
@pytest.mark.parametrize(
    "case, message",
    [
        ("missing", "matched no"),
        ("shape", "divisible"),
        ("head", "LM head"),
        ("tied", "shared"),
        ("subclass", "plain"),
        ("model", "Qwen3"),
        ("dtype", "FP32"),
    ],
)
def test_float8_selection_validates_before_optional_import(case, message):
    model = DenseModel()
    targets = [TARGET]
    if case == "missing":
        targets.append("missing")
    elif case == "shape":
        model.model.layers[0].mlp.up_proj = nn.Linear(32, 8, bias=False)
    elif case == "head":
        targets = ["lm_head"]
    elif case == "tied":
        model.alias = model.fc
    elif case == "subclass":

        class CustomLinear(nn.Linear):
            pass

        model.model.layers[0].mlp.up_proj = CustomLinear(32, 32, bias=False)
    elif case == "model":
        model.config.model_type = "qwen3_moe"
    elif case == "dtype":
        model.fc.bfloat16()
    original = model.fc
    with pytest.raises(ValueError, match=message):
        convert_float8_linears(model, Float8TrainingConfig(enable=True, target_modules=targets))
    assert model.fc is original


def _require_native_float8():
    if not device.IS_CUDA_AVAILABLE or torch.version.hip or device.get_gpu_compute_capability() < 89:
        pytest.skip("native tensorwise float8 training requires NVIDIA SM89+")
    pytest.importorskip("torchao.float8")


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_veomni_mlp_preserves_float8_projection_forwards(family, monkeypatch):
    """Generated MLPs must call Float8Linear.forward instead of consuming .weight."""
    _require_native_float8()
    from torch.profiler import ProfilerActivity, profile
    from torchao.float8 import Float8LinearConfig, convert_to_float8_training
    from transformers import LlamaConfig, Qwen3Config

    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaMLP
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3MLP

    monkeypatch.setattr("veomni.ops.config._ops_config", None)
    config_cls, mlp_cls = (Qwen3Config, Qwen3MLP) if family == "qwen3" else (LlamaConfig, LlamaMLP)
    mlp = mlp_cls(config_cls(hidden_size=64, intermediate_size=128, hidden_act="silu")).cuda().bfloat16()
    convert_to_float8_training(mlp, config=Float8LinearConfig(pad_inner_dim=True))
    inputs = torch.randn(17, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        mlp(inputs).square().mean().backward()
        device.synchronize()
    assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == 9


@pytest.mark.parametrize("tokens", [16, 17, 31, 33])
def test_float8_native_kernels_and_ragged_bf16_reference(tokens):
    _require_native_float8()
    from torch.profiler import ProfilerActivity, profile

    torch.manual_seed(13)
    model = DenseModel().cuda()
    original_weight = model.fc.weight
    reference = deepcopy(model.fc).bfloat16()
    untouched = model.lm_head
    assert convert_float8_linears(model, Float8TrainingConfig(enable=True, target_modules=[TARGET])) == [TARGET]
    assert model.fc.weight is original_weight
    assert model.lm_head is untouched
    model.fc.bfloat16()
    inputs = torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    reference_inputs = inputs.detach().clone().requires_grad_()
    grad_output = torch.randn(tokens, 32, device="cuda", dtype=torch.bfloat16)
    expected = reference(reference_inputs)
    expected.backward(grad_output)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        actual = model.fc(inputs)
        actual.backward(grad_output)
        device.synchronize()
    assert actual.shape == expected.shape
    for candidate, baseline, tolerance in (
        (actual, expected, 0.08),
        (inputs.grad, reference_inputs.grad, 0.15),
        (model.fc.weight.grad, reference.weight.grad, 0.15),
    ):
        assert torch.isfinite(candidate).all()
        relative_error = (candidate.float() - baseline.float()).norm() / baseline.float().norm()
        assert relative_error < tolerance
    assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == 3


@pytest.mark.parametrize("family", ["qwen3", "llama"])
def test_float8_fsdp2_meta_load_update_and_dcp_resume(tmp_path, monkeypatch, family):
    """Exercise the real wrap/load path at one rank; multi-card CI is separate."""
    _require_native_float8()
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torchao.float8.float8_linear import Float8Linear
    from transformers import LlamaConfig, Qwen3Config
    from transformers import LlamaForCausalLM as HFLlamaForCausalLM
    from transformers import Qwen3ForCausalLM as HFQwen3ForCausalLM

    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.distributed.parallel_state import _init_parallel_state, clear_parallel_state
    from veomni.distributed.torch_parallelize import parallelize_model_fsdp2
    from veomni.models.checkpoint.weights import init_empty_weights, save_model_weights
    from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaForCausalLM
    from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3ForCausalLM

    monkeypatch.setattr("veomni.ops.config._ops_config", None)
    config_cls, model_cls, hf_model_cls = (
        (Qwen3Config, Qwen3ForCausalLM, HFQwen3ForCausalLM)
        if family == "qwen3"
        else (LlamaConfig, LlamaForCausalLM, HFLlamaForCausalLM)
    )
    config = config_cls(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    checkpoint = tmp_path / "hf"
    torch.manual_seed(0)
    original = model_cls(config).float()
    expected_weights = {name: param.detach().clone() for name, param in original.named_parameters()}
    original.save_pretrained(checkpoint)
    del original
    dist.init_process_group("nccl", init_method=f"file://{tmp_path / 'rdzv'}", rank=0, world_size=1)
    try:
        _init_parallel_state(dp_size=1, dp_mode="fsdp2")
        precision = _precision("model.layers.*.mlp.*_proj", "model.layers.*.self_attn.*_proj")

        def build(resume=False):
            with init_empty_weights():
                model = model_cls(config).float()
            return parallelize_model_fsdp2(
                model,
                weights_path=None if resume else str(checkpoint),
                init_device="meta",
                mixed_precision=precision,
                should_skip_hf_weight_load=resume,
            )

        model = build()
        assert sum(isinstance(module, Float8Linear) for module in model.modules()) == 7
        assert not isinstance(model.lm_head, Float8Linear)
        assert model.state_dict().keys() == expected_weights.keys()
        for name, param in model.named_parameters():
            assert not param.is_meta and param.dtype == torch.float32
            torch.testing.assert_close(param.full_tensor().cpu(), expected_weights[name], rtol=0, atol=0)
        inputs = torch.arange(17, device="cuda").reshape(1, 17)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        def step(module, optim):
            optim.zero_grad()
            loss = None
            for _ in range(2):
                loss = module(input_ids=inputs, labels=inputs, use_cache=False).loss / 2
                assert torch.isfinite(loss)
                loss.backward()
            assert all(
                param.grad is not None and torch.isfinite(param.grad.to_local()).all() for param in module.parameters()
            )
            optim.step()
            return loss.detach()

        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
            step(model, optimizer)
            device.synchronize()
        # Seven projections, three GEMMs each, two accumulated microbatches.
        assert sum(event.count for event in trace.key_averages() if event.key == "aten::_scaled_mm") == 42
        updated = dict(model.named_parameters())["model.layers.0.mlp.up_proj.weight"]
        assert updated.dtype == torch.float32
        assert not torch.equal(updated.full_tensor().cpu(), expected_weights["model.layers.0.mlp.up_proj.weight"])
        dcp.save(
            {"model": ModelState(model), "optimizer": OptimizerState(model, optimizer)}, checkpoint_id=tmp_path / "dcp"
        )
        expected_loss = step(model, optimizer)
        expected_parameters = {
            name: param.full_tensor().detach().cpu().clone() for name, param in model.named_parameters()
        }
        resumed = build(resume=True)
        resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=1e-3)
        dcp.load(
            {"model": ModelState(resumed), "optimizer": OptimizerState(resumed, resumed_optimizer, load=True)},
            checkpoint_id=tmp_path / "dcp",
        )
        torch.testing.assert_close(step(resumed, resumed_optimizer), expected_loss, rtol=0, atol=0)
        for name, param in resumed.named_parameters():
            torch.testing.assert_close(param.full_tensor().cpu(), expected_parameters[name], rtol=0, atol=0)
        export = tmp_path / "export"
        save_model_weights(
            str(export), resumed.state_dict(), global_rank=0, save_dtype="float32", model_assets=[config]
        )
        exported = hf_model_cls.from_pretrained(export, dtype=torch.float32)
        for name, param in exported.named_parameters():
            torch.testing.assert_close(param, expected_parameters[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()
        clear_parallel_state()

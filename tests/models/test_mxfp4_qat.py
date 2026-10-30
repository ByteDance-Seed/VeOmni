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

"""CPU execution contracts for the emulated Dense Qwen3 MXFP4 QAT recipe."""

import copy
import subprocess
import sys
import textwrap
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.tiny_configs import tiny_qwen3_config, tiny_qwen3_vl_config
from tests.tools.training_utils import make_eager_ops_config
from veomni.arguments import MixedPrecisionConfig
from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.arguments.parser import parse_args, save_args
from veomni.distributed.parallel_state import ParallelState, use_parallel_state
from veomni.models import build_foundation_model
from veomni.models.configuration_base import as_veomni_config
from veomni.models.qat import (
    export_mxfp4,
    load_mxfp4,
    prepare_mxfp4_qat,
    validate_mxfp4_qat_parallelism,
)


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))

pytest.importorskip("torchao", minversion="0.18.0")

MXTensor = pytest.importorskip("torchao.prototype.mx_formats.mx_tensor").MXTensor
MXFakeQuantizedLinear = pytest.importorskip("torchao.prototype.qat").MXFakeQuantizedLinear


_TARGETS = ["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.*_proj"]


@pytest.fixture(autouse=True)
def _veomni_backend(monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    with use_parallel_state(ParallelState(device_type="cpu")):
        yield


def _build(*, targets=None, dtype="float32", init_device="cpu", **config_overrides):
    config = tiny_qwen3_config(num_hidden_layers=1, **config_overrides)
    return build_foundation_model(
        config,
        weights_path=None,
        torch_dtype=dtype,
        init_device=init_device,
        ops_implementation=make_eager_ops_config(
            qat_implementation="torchao_mxfp4",
            qat_target_modules=_TARGETS if targets is None else targets,
        ),
    )


def _batch():
    return torch.tensor([[3, 7, 12, 5, 11, 8, 4], [6, 9, 15, 14, 10, 3, 13]])


def _step(model, optimizer):
    optimizer.zero_grad()
    input_ids = _batch()
    for microbatch in input_ids.split(1):
        output = model(input_ids=microbatch, labels=microbatch, use_cache=False)
        (output.loss / len(input_ids)).backward()
    optimizer.step()


def test_generated_qwen3_executes_every_selected_projection_and_updates_fp32_masters():
    torch.manual_seed(31)
    model = _build()
    prepared = {name: module for name, module in model.named_modules() if isinstance(module, MXFakeQuantizedLinear)}
    assert len(prepared) == 7
    assert type(model.lm_head) is torch.nn.Linear
    assert "generated.patched_modeling_qwen3_gpu" in type(model).__module__

    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    keys = set(model.state_dict())
    assert keys == set(before)
    assert all(param.dtype == torch.float32 for param in model.parameters())
    calls = dict.fromkeys(prepared, 0)
    handles = []
    for name, module in prepared.items():

        def count(_module, _inputs, _output, *, name=name):
            calls[name] += 1

        handles.append(module.register_forward_hook(count))

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = _batch()
    try:
        for microbatch in input_ids.split(1):
            # Exercise BF16 compute-facing parameters while retaining differentiable
            # FP32 masters. This checks casting semantics, not distributed FSDP.
            compute_params = {name: param.to(torch.bfloat16) for name, param in model.named_parameters()}
            output = torch.func.functional_call(
                model,
                compute_params,
                (),
                {"input_ids": microbatch, "labels": microbatch, "use_cache": False},
            )
            assert torch.isfinite(output.loss)
            (output.loss / len(input_ids)).backward()
    finally:
        for handle in handles:
            handle.remove()

    assert all(count == 2 for count in calls.values()), calls
    for name in prepared:
        grad = model.get_parameter(name + ".weight").grad
        assert grad is not None and grad.dtype == torch.float32
        assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    optimizer.step()
    assert set(model.state_dict()) == keys
    for name in prepared:
        assert not torch.equal(model.get_parameter(name + ".weight"), before[name + ".weight"])


def test_qat_model_and_adam_state_resume_preserves_next_update(tmp_path):
    torch.manual_seed(32)
    model = _build()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _step(model, optimizer)
    checkpoint = tmp_path / "training.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)

    restored = _build()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    state = torch.load(checkpoint, weights_only=True)
    restored.load_state_dict(state["model"])
    restored_optimizer.load_state_dict(state["optimizer"])
    _step(model, optimizer)
    _step(restored, restored_optimizer)
    for name, param in model.named_parameters():
        assert torch.equal(param, restored.get_parameter(name)), name
    for current, resumed in zip(optimizer.state.values(), restored_optimizer.state.values()):
        for name, value in current.items():
            assert torch.equal(value, resumed[name]), name


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
def test_hf_checkpoint_reload_reconstitutes_qat_and_preserves_outputs(tmp_path, dtype):
    torch.manual_seed(34)
    model = _build(dtype=dtype).eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _step(model, optimizer)
    input_ids = _batch()
    with torch.no_grad():
        expected = model(input_ids=input_ids, use_cache=False).logits
    checkpoint = tmp_path / "hf"
    model.save_pretrained(checkpoint)

    restored = build_foundation_model(
        str(checkpoint),
        weights_path=str(checkpoint),
        torch_dtype=dtype,
        init_device="cpu",
        ops_implementation=make_eager_ops_config(qat_implementation="torchao_mxfp4", qat_target_modules=_TARGETS),
    ).eval()
    assert len([module for module in restored.modules() if isinstance(module, MXFakeQuantizedLinear)]) == 7
    assert restored.config.ops_config.qat_implementation == "torchao_mxfp4"
    for name, param in model.named_parameters():
        assert torch.equal(param, restored.get_parameter(name)), name
    with torch.no_grad():
        actual = restored(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_mxfp4_export_reload_matches_prepared_qwen3_without_mutating_training_model(tmp_path):
    torch.manual_seed(33)
    model = _build().eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _step(model, optimizer)
    before = copy.deepcopy(model.state_dict())
    input_ids = _batch()
    with torch.no_grad():
        # FSDP casts compute parameters, retaining FP32 nonpersistent RoPE buffers.
        compute_params = {name: param.to(torch.bfloat16) for name, param in model.named_parameters()}
        expected = torch.func.functional_call(
            model, compute_params, (), {"input_ids": input_ids, "use_cache": False}
        ).logits

    export_mxfp4(model, tmp_path / "export")
    restored = load_mxfp4(tmp_path / "export", device="cpu")
    assert len([module for module in model.modules() if isinstance(module, MXFakeQuantizedLinear)]) == 7
    assert all(param.dtype == torch.float32 for param in model.parameters())
    assert len([param for param in restored.parameters() if isinstance(param, MXTensor)]) == 7
    assert all(not buffer.is_meta for buffer in restored.buffers())
    assert not restored.model.rotary_emb.inv_freq.is_meta
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name]), name
    with torch.no_grad():
        actual = restored(input_ids=input_ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("targets", "config_overrides", "message"),
    [
        (["lm_head"], {}, "projection"),
        (["model.layers.*.missing"], {}, "match"),
        (["model.layers.*.self_attn.q_proj"], {"attention_bias": True}, "bias"),
    ],
)
def test_qat_rejects_unsupported_targets(targets, config_overrides, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        _build(targets=targets, **config_overrides)


def test_qat_rejects_nested_qwen3_vl_text_recipe():
    config = as_veomni_config(tiny_qwen3_vl_config())
    config._set_ops_implementation(make_eager_ops_config())
    config.text_config._set_ops_implementation(
        make_eager_ops_config(qat_implementation="torchao_mxfp4", qat_target_modules=["layers.*.self_attn.q_proj"])
    )
    with pytest.raises(ValueError, match="standalone"):
        build_foundation_model(config, init_device="cpu", torch_dtype="float32")


def test_meta_preparation_preserves_parameter_keys_and_is_idempotent():
    model = _build(init_device="meta")
    parameters = dict(model.named_parameters())
    assert all(param.is_meta for param in parameters.values())
    prepare_mxfp4_qat(model)
    assert set(model.state_dict()) == set(parameters)
    assert all(model.get_parameter(name) is param for name, param in parameters.items())


def test_qat_yaml_cli_and_saved_configuration_roundtrip(tmp_path, monkeypatch):
    import yaml

    config_path = tmp_path / "recipe.yaml"
    original = make_eager_ops_config(qat_implementation="torchao_mxfp4", qat_target_modules=_TARGETS)
    config_path.write_text(yaml.safe_dump(asdict(original)))
    monkeypatch.setattr(sys, "argv", ["train", str(config_path)])
    parsed = parse_args(OpsImplementationConfig)
    assert parsed.qat_implementation == "torchao_mxfp4"
    assert parsed.qat_target_modules == _TARGETS

    attention_targets = ["model.layers.*.self_attn.q_proj", "model.layers.*.self_attn.v_proj"]
    monkeypatch.setattr(sys, "argv", ["train", str(config_path), "--qat_target_modules", *attention_targets])
    overridden = parse_args(OpsImplementationConfig)
    assert overridden.qat_target_modules == attention_targets
    output_dir = tmp_path / "saved"
    save_args(overridden, str(output_dir))
    monkeypatch.setattr(sys, "argv", ["train", str(output_dir / "veomni_cli.yaml")])
    assert asdict(parse_args(OpsImplementationConfig)) == asdict(overridden)


def test_disabled_qat_build_does_not_import_torchao():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
                import sys
                sys.modules["torchao"] = None

                import torch
                from tests.models.tiny_configs import tiny_qwen3_config
                from tests.tools.training_utils import make_eager_ops_config
                from veomni.models import build_foundation_model

                model = build_foundation_model(
                    tiny_qwen3_config(num_hidden_layers=1),
                    init_device="cpu",
                    torch_dtype="float32",
                    ops_implementation=make_eager_ops_config(),
                )
                assert type(model.model.layers[0].mlp.gate_proj) is torch.nn.Linear
                input_ids = torch.tensor([[1, 2, 3]])
                output = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                assert torch.isfinite(output.loss)
                assert sys.modules["torchao"] is None
                """
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("targets", [None, "model.layers.*.mlp.*_proj"])
def test_recipe_requires_explicit_target_list(targets):
    with pytest.raises(ValueError, match="nonempty qat_target_modules list"):
        OpsImplementationConfig(qat_implementation="torchao_mxfp4", qat_target_modules=targets)


@pytest.mark.parametrize(
    ("parallel_overrides", "precision_overrides", "kwargs", "message"),
    [
        ({"dp_mode": "ddp"}, {}, {}, "FSDP2"),
        ({"sp_enabled": True}, {}, {}, "SP/CP"),
        ({}, {"param_dtype": "float32"}, {}, "BF16"),
        ({}, {"reduce_dtype": "bfloat16"}, {}, "FP32"),
        ({}, {}, {"is_peft_model": True}, "LoRA"),
        ({}, {}, {"enable_fsdp_offload": True}, "offload"),
    ],
)
def test_distributed_recipe_rejects_unsupported_policy(
    monkeypatch, parallel_overrides, precision_overrides, kwargs, message
):
    model = _build()
    parallel = SimpleNamespace(
        dp_mode="fsdp2",
        sp_enabled=False,
        tp_enabled=False,
        pp_enabled=False,
        any_extra_parallel_enabled=False,
        dp_replicate_enabled=False,
    )
    for name, value in parallel_overrides.items():
        setattr(parallel, name, value)
    precision = {"enable": True, "param_dtype": "bfloat16", "reduce_dtype": "float32"}
    precision.update(precision_overrides)
    monkeypatch.setattr("veomni.distributed.parallel_state.get_parallel_state", lambda: parallel)
    with pytest.raises(ValueError, match=message):
        validate_mxfp4_qat_parallelism(model, MixedPrecisionConfig(**precision), **kwargs)


def test_distributed_recipe_rejects_bf16_master_parameters(monkeypatch):
    model = _build(dtype="bfloat16")
    # Reach the pure policy gate without invoking sharding or accelerator kernels.
    monkeypatch.setattr("veomni.utils.device.IS_CUDA_AVAILABLE", True)
    with pytest.raises(ValueError, match="FP32 master parameters"):
        validate_mxfp4_qat_parallelism(
            model, MixedPrecisionConfig(enable=True, param_dtype="bfloat16", reduce_dtype="float32")
        )

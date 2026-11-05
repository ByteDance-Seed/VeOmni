# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU-emulated MXFP8 through production Qwen3 loading and HF export.

Only the hardware gate and GEMM emulation seam are replaced. Real two-rank
Gloo FSDP2, meta materialization, checkpoint dispatch and fused AdamW run here;
this does not validate native MXFP8 kernels or physical multi-GPU collectives.
"""

import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.tools.launch_utils import torchrun


pytest.importorskip("torchao")
pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))


def _snapshot(model):
    from veomni.distributed.mxfp8 import unwrap_mxfp8_tensor

    result = {}
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32, name
        full = unwrap_mxfp8_tensor(parameter).full_tensor()
        assert type(full) is torch.Tensor and full.device.type == "cpu", name
        result[name] = full.detach().clone()
    return result


def _build_fsdp(path, recipe=None):
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_config, build_foundation_model

    model = build_foundation_model(build_config(str(path)), init_device="meta", torch_dtype="float32")
    assert ".generated." in type(model).__module__
    assert all(parameter.is_meta for parameter in model.parameters())
    precision = MixedPrecisionConfig(mxfp8=recipe)
    model = build_parallelize_model(
        model,
        weights_path=str(path),
        init_device="meta",
        mixed_precision=precision,
        enable_gradient_checkpointing=True,
        enable_reentrant=False,
        # Use load_model_weights -> _dispatch_parameter; do not substitute a
        # load_state_dict roundtrip for the production in-place loader.
        broadcast_model_weights_from_rank0=False,
    )
    model.train()
    assert model.is_gradient_checkpointing
    return model


def _assert_prepared(model, all_gather):
    from veomni.distributed.mxfp8.tensors import MXFP8Linear, MXFP8Weight

    prepared = {name: module for name, module in model.named_modules() if isinstance(module, MXFP8Linear)}
    assert set(prepared) == {f"model.layers.0.mlp.{name}_proj" for name in ("gate", "up", "down")}
    for module in prepared.values():
        assert module._emulate
        assert isinstance(module.weight.to_local(), MXFP8Weight) == all_gather
    assert type(model.lm_head) is torch.nn.Linear
    return prepared


def _optimizer_step(model, path, all_gather):
    from tests.tools.training_utils import unbuilt_runtime
    from veomni.arguments import ModelArguments

    prepared = _assert_prepared(model, all_gather)
    before = _snapshot(model)
    args = ModelArguments(model_path=str(path))
    args.optimizer.lr = 1e-3
    runtime = unbuilt_runtime(args)
    runtime.model = model
    runtime._build_optimizer()
    optimizer = runtime.optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["fused"] and group["foreach"] is False for group in optimizer.param_groups)
    called = set()
    handles = [
        module.register_forward_hook(lambda _module, _args, _output, name=name: called.add(name))
        for name, module in prepared.items()
    ]
    try:
        # A non-block-aligned token count exercises the padded MXFP8 wgrad.
        tokens = torch.arange(17).unsqueeze(0) % model.config.vocab_size
        output = model(input_ids=tokens, position_ids=torch.arange(17).unsqueeze(0), labels=tokens, use_cache=False)
        assert torch.isfinite(output.loss)
        output.loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert called == set(prepared), "Generated SwiGLU must call the converted Linear forwards."
    for name, parameter in model.named_parameters():
        assert isinstance(parameter.grad, DTensor) and parameter.grad.dtype == torch.float32, name
        assert torch.isfinite(parameter.grad.to_local()).all(), name
    for module in prepared.values():
        assert torch.count_nonzero(module.weight.grad.to_local()) > 0
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    after = _snapshot(model)
    assert all(not torch.equal(before[f"{name}.weight"], after[f"{name}.weight"]) for name in prepared)
    return after


def _loading_worker(root, all_gather):
    from transformers import Qwen3Config

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.arguments import AcceleratorConfig, MXFP8Config
    from veomni.distributed import mxfp8
    from veomni.distributed.mxfp8.tensors import MXFP8Linear
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
    from veomni.models import build_foundation_model
    from veomni.utils.device import get_device_type
    from veomni.utils.save_safetensor_utils import save_hf_safetensor

    assert get_device_type() == "cpu" and dist.get_backend() == dist.Backend.GLOO
    torch.set_num_threads(1)
    torch.manual_seed(42)
    init_parallel_state_from_config(AcceleratorConfig(), name="base")
    root = Path(root)
    source_path, export_path, disabled_path = root / "source", root / "export", root / "disabled"
    convert = mxfp8._convert_linears

    def emulate(model, recipe):
        return convert(model, recipe, emulate=True)

    try:
        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            use_cache=False,
        )
        if dist.get_rank() == 0:
            source = build_foundation_model(
                config, init_device="cpu", torch_dtype="float32", ops_implementation=make_eager_ops_config()
            )
            source.save_pretrained(source_path, safe_serialization=True)
            del source
        dist.barrier()
        incoming = load_file(source_path / "model.safetensors")
        assert any(not torch.equal(value, value.bfloat16().float()) for value in incoming.values())
        with (
            patch.object(mxfp8, "_validate_native_runtime", return_value=None),
            patch.object(mxfp8, "_convert_linears", side_effect=emulate),
        ):
            recipe = MXFP8Config(enable=True, enable_fsdp_all_gather=all_gather)
            model = _build_fsdp(source_path, recipe)
            _assert_prepared(model, all_gather)
            torch.testing.assert_close(_snapshot(model), incoming, atol=0, rtol=0)
            updated = _optimizer_step(model, source_path, all_gather)
            save_hf_safetensor(
                save_hf_safetensor_path=str(export_path),
                ckpt_manager="dcp",
                model=model,
                model_assets=[model.config],
            )
            torch.testing.assert_close(_snapshot(model), updated, atol=0, rtol=0)
            weight_map = json.loads((export_path / "model.safetensors.index.json").read_text())["weight_map"]
            exported = {}
            for filename in set(weight_map.values()):
                exported.update(load_file(export_path / filename))
            assert set(exported) == set(weight_map)
            assert all(value.dtype == torch.bfloat16 for value in exported.values())
            torch.testing.assert_close(
                exported, {name: value.bfloat16() for name, value in updated.items()}, atol=0, rtol=0
            )
            saved_recipe = json.loads((export_path / "config.json").read_text())["veomni_mxfp8"]
            assert saved_recipe["enable"] is True and saved_recipe["enable_fsdp_all_gather"] == all_gather

            # Default mxfp8=None must inherit the exported recipe, exactly as
            # the trainer's default MixedPrecisionConfig does.
            reloaded = _build_fsdp(export_path)
            _assert_prepared(reloaded, all_gather)
            rounded = {name: value.float() for name, value in exported.items()}
            torch.testing.assert_close(_snapshot(reloaded), rounded, atol=0, rtol=0)
            _optimizer_step(reloaded, export_path, all_gather)

            disabled = _build_fsdp(export_path, MXFP8Config(enable=False))
            assert not any(isinstance(module, MXFP8Linear) for module in disabled.modules())
            torch.testing.assert_close(_snapshot(disabled), rounded, atol=0, rtol=0)
            save_hf_safetensor(
                save_hf_safetensor_path=str(disabled_path),
                ckpt_manager="dcp",
                model=disabled,
                model_assets=[disabled.config],
            )
            assert json.loads((disabled_path / "config.json").read_text())["veomni_mxfp8"]["enable"] is False
            disabled_reload = _build_fsdp(disabled_path)
            assert not any(isinstance(module, MXFP8Linear) for module in disabled_reload.modules())
            torch.testing.assert_close(_snapshot(disabled_reload), rounded, atol=0, rtol=0)
    finally:
        clear_parallel_state()


@pytest.mark.parametrize("all_gather", [False, True], ids=["bf16_all_gather", "mxfp8_all_gather"])
def test_qwen3_mxfp8_cpu_meta_load_export_and_recipe_reload(tmp_path, monkeypatch, all_gather):
    # Spawn imports Torch only after these settings take effect, including
    # ordinary pytest on an accelerator host outside the CI case scheduler.
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    monkeypatch.setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    for variable in (
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "MLU_VISIBLE_DEVICES",
    ):
        monkeypatch.setenv(variable, "")
    torchrun(_loading_worker, 2, str(tmp_path), all_gather)

# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Physical two-GPU MXFP8 acceptance through the generated Qwen3 training path.

These tests require native MXFP8 GEMMs and NCCL. CPU numerical oracles are in
tests/ops/test_mxfp8_tensors.py; their results do not qualify these cases.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability, get_torch_device


_LAUNCH = LaunchSpec(nproc=2, device="accelerator")


def _check_rank_native_gemms():
    from veomni.distributed.mxfp8.tensors import MXFP8Linear

    error = None
    try:
        assert dist.get_backend() == dist.Backend.NCCL, "Physical CUDA/NCCL execution is required."
        assert get_gpu_compute_capability() >= 100, "The current rank requires SM100 or newer."
        assert get_torch_device().current_device() == dist.get_rank()
        # Test both backward GEMMs too, before any rank enters model sharding.
        layer = MXFP8Linear.from_float(torch.nn.Linear(64, 64, bias=False, device=get_device_type()), all_gather=False)
        inputs = torch.ones(32, 64, dtype=torch.bfloat16, device=get_device_type(), requires_grad=True)
        layer(inputs).sum().backward()
        get_torch_device().synchronize()
    except (AssertionError, RuntimeError, NotImplementedError) as exc:
        error = f"rank {dist.get_rank()}: {type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    assert not any(errors), f"Native MXFP8 readiness failed before model construction: {errors}"


def _build_fsdp(weights_path: Path, precision=None, *, skip_hf_load=False):
    from tests.tools.training_utils import unbuilt_runtime
    from veomni.arguments import MixedPrecisionConfig, ModelArguments
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_config, build_foundation_model

    config = build_config(str(weights_path))
    model = build_foundation_model(config, init_device="meta", torch_dtype="float32")
    assert ".generated." in type(model).__module__
    model = build_parallelize_model(
        model,
        weights_path=str(weights_path),
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(mxfp8=precision),
        enable_gradient_checkpointing=True,
        enable_reentrant=False,
        broadcast_model_weights_from_rank0=True,
        should_skip_hf_weight_load=skip_hf_load,
    )
    model.train()
    args = ModelArguments(model_path=str(weights_path))
    args.optimizer.lr = 1e-3
    runtime = unbuilt_runtime(args)
    runtime.model = model
    # Exercise the actual production optimizer construction, including fused
    # AdamW dispatch on tensor subclasses, rather than a hand-written fallback.
    runtime._build_optimizer()
    optimizer = runtime.optimizer
    assert isinstance(optimizer, torch.optim.AdamW)
    assert all(group["fused"] is True and group["foreach"] is False for group in optimizer.param_groups)
    return model, optimizer


def _plain_cpu(tensor):
    from torch.distributed.tensor import DTensor

    from veomni.distributed.mxfp8 import unwrap_mxfp8_tensor

    plain = unwrap_mxfp8_tensor(tensor)
    local = plain.to_local() if isinstance(plain, DTensor) else plain
    assert type(local) is torch.Tensor
    return local.detach().cpu().clone()


def _master_snapshot(model):
    from torch.distributed.tensor import DTensor

    result = {}
    for name, parameter in model.named_parameters():
        assert isinstance(parameter, DTensor) and parameter.dtype == torch.float32, name
        result[name] = _plain_cpu(parameter)
    return result


def _optimizer_snapshot(model, optimizer):
    return {
        name: {key: _plain_cpu(value) for key, value in optimizer.state[parameter].items()}
        for name, parameter in model.named_parameters()
    }


def _assert_recipe(model, all_gather):
    from veomni.distributed.mxfp8.tensors import MXFP8Linear, MXFP8Weight

    prepared = [module for module in model.modules() if isinstance(module, MXFP8Linear)]
    assert len(prepared) == 3 * model.config.num_hidden_layers
    assert all(isinstance(module.weight.to_local(), MXFP8Weight) == all_gather for module in prepared)


def _step(model, optimizer, step):
    from torch.distributed.tensor import DTensor

    from veomni.distributed.mxfp8.tensors import MXFP8Linear

    prepared = {name: module for name, module in model.named_modules() if isinstance(module, MXFP8Linear)}
    assert len(prepared) == 3 * model.config.num_hidden_layers
    assert type(model.lm_head) is torch.nn.Linear
    assert all(not module._emulate for module in prepared.values())
    before = {name: _plain_cpu(module.weight) for name, module in prepared.items()}
    called = set()
    handles = [
        module.register_forward_hook(lambda _module, _inputs, _output, name=name: called.add(name))
        for name, module in prepared.items()
    ]
    optimizer.zero_grad(set_to_none=True)
    losses = []
    try:
        # Ragged token counts require distinct padded wgrad reduction layouts.
        for microbatch, length in enumerate((17, 33)):
            generator = torch.Generator().manual_seed(1000 + 100 * step + 10 * dist.get_rank() + microbatch)
            tokens = torch.randint(0, model.config.vocab_size, (1, length), generator=generator)
            tokens = tokens.to(get_device_type())
            positions = torch.arange(length, device=tokens.device).unsqueeze(0)
            output = model(input_ids=tokens, position_ids=positions, labels=tokens, use_cache=False)
            loss = output.loss / 2
            assert torch.isfinite(loss)
            loss.backward()
            losses.append(loss.detach().cpu())
    finally:
        for handle in handles:
            handle.remove()
    assert called == set(prepared), "The generated MLP must invoke the MXFP8 Linear forwards."
    gradients = {}
    for name, parameter in model.named_parameters():
        assert isinstance(parameter.grad, DTensor) and parameter.grad.dtype == torch.float32, name
        gradients[name] = _plain_cpu(parameter.grad)
        assert torch.isfinite(gradients[name]).all(), name
    for name in prepared:
        assert torch.count_nonzero(gradients[f"{name}.weight"]) > 0, name
    optimizer.step()
    assert all(not torch.equal(_plain_cpu(module.weight), before[name]) for name, module in prepared.items())
    optimizer.zero_grad(set_to_none=True)
    return {"losses": torch.stack(losses), "gradients": gradients}


def _write_source(path):
    from transformers import Qwen3Config

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.models import build_foundation_model

    if dist.get_rank() == 0:
        torch.manual_seed(42)
        config = Qwen3Config(
            architectures=["Qwen3ForCausalLM"],
            vocab_size=128,
            hidden_size=128,
            intermediate_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=32,
            attention_dropout=0.0,
            tie_word_embeddings=False,
            use_cache=False,
        )
        source = build_foundation_model(
            config, init_device="cpu", torch_dtype="float32", ops_implementation=make_eager_ops_config()
        )
        source.save_pretrained(path, safe_serialization=True)
    dist.barrier()


def _fsdp_worker(root, all_gather, phase):
    from safetensors import safe_open

    from veomni.arguments import AcceleratorConfig, MXFP8Config
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.mxfp8 import unwrap_mxfp8_tensor
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
    from veomni.utils.save_safetensor_utils import save_hf_safetensor

    _check_rank_native_gemms()
    root = Path(root)
    base_path, dcp_path, hf_path = root / "base", root / "dcp", root / "hf"
    expected_path = root / f"expected_rank_{dist.get_rank()}.pt"
    precision = MXFP8Config(enable=True, enable_fsdp_all_gather=all_gather)
    init_parallel_state_from_config(AcceleratorConfig(), name="base")
    try:
        if phase == "save":
            _write_source(base_path)
            model, optimizer = _build_fsdp(base_path, precision)
            _assert_recipe(model, all_gather)
            initial = _master_snapshot(model)
            with safe_open(base_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
                assert set(checkpoint.keys()) == set(initial)
                for name, parameter in model.named_parameters():
                    loaded = unwrap_mxfp8_tensor(parameter).full_tensor().cpu()
                    torch.testing.assert_close(loaded, checkpoint.get_tensor(name), atol=0, rtol=0)
            _step(model, optimizer, 0)
            saved_weights = _master_snapshot(model)
            saved_optimizer = _optimizer_snapshot(model, optimizer)
            assert all(state for state in saved_optimizer.values())
            DistributedCheckpointer.save(str(dcp_path), {"model": model, "optimizer": optimizer})
            next_step = _step(model, optimizer, 1)
            torch.save(
                {
                    "saved_weights": saved_weights,
                    "saved_optimizer": saved_optimizer,
                    "next_step": next_step,
                    "next_weights": _master_snapshot(model),
                    "next_optimizer": _optimizer_snapshot(model, optimizer),
                },
                expected_path,
            )
        else:
            assert phase == "resume"
            expected = torch.load(expected_path, weights_only=True, map_location="cpu")
            model, optimizer = _build_fsdp(base_path, precision, skip_hf_load=True)
            _assert_recipe(model, all_gather)
            DistributedCheckpointer.load(str(dcp_path), {"model": model, "optimizer": optimizer})
            torch.testing.assert_close(_master_snapshot(model), expected["saved_weights"], atol=0, rtol=0)
            torch.testing.assert_close(
                _optimizer_snapshot(model, optimizer), expected["saved_optimizer"], atol=0, rtol=0
            )
            torch.testing.assert_close(_step(model, optimizer, 1), expected["next_step"], atol=0, rtol=0)
            torch.testing.assert_close(_master_snapshot(model), expected["next_weights"], atol=0, rtol=0)
            torch.testing.assert_close(
                _optimizer_snapshot(model, optimizer), expected["next_optimizer"], atol=0, rtol=0
            )

            # Production HF exports intentionally round FP32 masters to BF16.
            # MXFP8 transport payloads must never become the exported weights.
            save_hf_safetensor(
                save_hf_safetensor_path=str(hf_path),
                ckpt_manager="dcp",
                model=model,
                model_assets=[model.config],
            )
            torch.testing.assert_close(_master_snapshot(model), expected["next_weights"], atol=0, rtol=0)
            # Omitted configuration must retain the exported recipe; this is
            # intentionally different from explicitly disabling MXFP8.
            reloaded, reloaded_optimizer = _build_fsdp(hf_path)
            _assert_recipe(reloaded, all_gather)
            rounded = {name: value.bfloat16().float() for name, value in expected["next_weights"].items()}
            torch.testing.assert_close(_master_snapshot(reloaded), rounded, atol=0, rtol=0)
            _step(reloaded, reloaded_optimizer, 2)
    finally:
        clear_parallel_state()


def _require_two_native_devices():
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or runner_device_count() < _LAUNCH.nproc:
        pytest.skip("Requires two NVIDIA CUDA GPUs; no CPU/Gloo substitute.")
    if not all(get_gpu_compute_capability(rank) >= 100 for rank in range(_LAUNCH.nproc)):
        pytest.skip("Requires native MXFP8 on both allocated GPUs (SM100 or newer).")
    pytest.importorskip("torchao")


@pytest.mark.launch_plan(provider=_LAUNCH)
@pytest.mark.parametrize("all_gather", [False, True], ids=["bf16_all_gather", "mxfp8_all_gather"])
def test_qwen3_mxfp8_two_gpu_fsdp2_checkpoint_resume(tmp_path, monkeypatch, all_gather):
    _require_two_native_devices()
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torchrun(_fsdp_worker, _LAUNCH.nproc, str(tmp_path), all_gather, "save")
    # Restart both ranks: restore must not rely on cached payloads, state-dict
    # hooks from the saved model, or the original optimizer's in-memory state.
    torchrun(_fsdp_worker, _LAUNCH.nproc, str(tmp_path), all_gather, "resume")


def _transport_oracle_worker(root):
    from contextlib import ExitStack
    from unittest.mock import patch

    from veomni.arguments import AcceleratorConfig, MXFP8Config
    from veomni.distributed.mxfp8.tensors import MXFP8GatheredWeight, MXFP8Weight, _quantize
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config

    _check_rank_native_gemms()
    init_parallel_state_from_config(AcceleratorConfig(), name="base")
    original_pre = MXFP8Weight.fsdp_pre_all_gather
    calls = {"candidate": 0, "reference": 0}

    def candidate_pre(weight, *args, **kwargs):
        payload, metadata = original_pre(weight, *args, **kwargs)
        assert len(payload) == 4 and all(part.dtype == torch.uint8 for part in payload)
        calls["candidate"] += 1
        return payload, metadata

    def reference_pre(weight, *args, **kwargs):
        # Gather the original master without any rank-local quantization.
        assert weight._master.dtype == torch.float32
        calls["reference"] += 1
        return (weight._master,), ()

    def reference_post(weight, outputs, metadata, param_dtype, *, out=None):
        assert len(outputs) == 1 and outputs[0].dtype == torch.float32
        # Independently quantize both orientations from the full original FP32
        # master. This must match rank-local blocks only on supported layouts.
        row = _quantize(outputs[0], emulate=False)
        col = _quantize(outputs[0].t(), emulate=False)
        payload = (
            row.qdata.view(torch.uint8),
            row.scale.view(torch.uint8),
            col.qdata.t().contiguous().view(torch.uint8),
            col.scale.t().contiguous().view(torch.uint8),
        )
        if out is not None:
            for destination, source in zip(out.payload, payload, strict=True):
                destination.copy_(source)
            return None
        return MXFP8GatheredWeight(*payload, dtype=param_dtype), payload

    try:
        source_path = Path(root) / "base"
        _write_source(source_path)
        precision = MXFP8Config(enable=True, enable_fsdp_all_gather=True)
        results = []
        for reference in (False, True):
            model, optimizer = _build_fsdp(source_path, precision)
            with ExitStack() as stack:
                stack.enter_context(
                    patch.object(MXFP8Weight, "fsdp_pre_all_gather", reference_pre if reference else candidate_pre)
                )
                if reference:
                    stack.enter_context(patch.object(MXFP8Weight, "fsdp_post_all_gather", reference_post))
                steps = []
                for step in range(2):
                    steps.append(
                        {
                            "step": _step(model, optimizer, step),
                            "masters": _master_snapshot(model),
                            "optimizer": _optimizer_snapshot(model, optimizer),
                        }
                    )
                results.append(steps)
            del optimizer, model
        assert calls["candidate"] == calls["reference"] and calls["candidate"] > 0
        # Identical native GEMMs, but independently constructed transport
        # payloads/scales; not a comparison with BF16 training or convergence.
        torch.testing.assert_close(results[0], results[1], atol=0, rtol=0)
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_LAUNCH)
def test_mxfp8_shard_quantization_matches_fp32_transport_oracle(tmp_path, monkeypatch):
    _require_two_native_devices()
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torchrun(_transport_oracle_worker, _LAUNCH.nproc, str(tmp_path))

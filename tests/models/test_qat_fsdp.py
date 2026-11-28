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

"""Two-GPU Qwen3 training with torchao FP4 or MXFP8 and ordinary FSDP2 AllGather.

Exercise accumulated updates, DCP continuation and HF export/reload on NCCL.
"""

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


_LAUNCH = LaunchSpec(nproc=2)


def _rows(selection):
    return {"linear_implementation": selection, "gated_mlp_implementation": selection}


def _qat_ops(selection):
    from tests.tools.training_utils import make_eager_ops_config

    return make_eager_ops_config(**_rows(selection))


def _build_fsdp(weights_path: Path, selection: dict, *, skip_hf_load: bool = False):
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_config, build_foundation_model
    from veomni.ops import OpsSelection

    expected = OpsSelection(**selection)
    config = build_config(str(weights_path))
    # The trainer passes its own ops instance, which replaces the recorded rows.
    assert config.ops_config.linear_implementation == expected
    assert config.ops_config.gated_mlp_implementation == expected
    model = build_foundation_model(
        config, init_device="meta", torch_dtype="float32", ops_implementation=_qat_ops(selection)
    )
    assert model.config.ops_config.linear_implementation == expected
    assert type(model).__module__ == "veomni.models.transformers.qwen3.modeling_qwen3"
    model = build_parallelize_model(
        model,
        weights_path=str(weights_path),
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
        enable_gradient_checkpointing=True,
        enable_reentrant=False,
        broadcast_model_weights_from_rank0=True,
        should_skip_hf_weight_load=skip_hf_load,
    )
    model.train()
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)


def _master_snapshot(model):
    from torch.distributed.tensor import DTensor

    result = {}
    for name, param in model.named_parameters():
        assert isinstance(param, DTensor), name
        assert param.dtype == torch.float32, name
        result[name] = param.to_local().detach().cpu().clone()
    return result


def _optimizer_snapshot(model, optimizer):
    from torch.distributed.tensor import DTensor

    return {
        name: {
            key: (value.to_local() if isinstance(value, DTensor) else value).detach().cpu().clone()
            for key, value in optimizer.state[param].items()
        }
        for name, param in model.named_parameters()
    }


@contextlib.contextmanager
def _record_quantizer_calls(dtype):
    """Count FP4 operand rounding or MXFP8 projections for each selected row."""
    import veomni.ops.kernels.gated_mlp.standard.torchao as mlp_row
    import veomni.ops.kernels.linear.standard.torchao as linear_row

    calls = {"linear": 0, "gated_mlp": 0}

    def counted(family, fn):
        def wrapped(*args, **kwargs):
            calls[family] += 1
            return fn(*args, **kwargs)

        return wrapped

    def fp4_factory(family, factory):
        return lambda dtype: counted(family, factory(dtype))

    with contextlib.ExitStack() as stack:
        for family, row in (("linear", linear_row), ("gated_mlp", mlp_row)):
            if dtype == "mxfp8":
                fn = counted(family, row.mxfp8_model_projection)
                stack.enter_context(patch.object(row, "mxfp8_model_projection", fn))
            else:
                factory = fp4_factory(family, row.torchao_fake_quant)
                stack.enter_context(patch.object(row, "torchao_fake_quant", factory))
        yield calls


def _step(model, optimizer, step):
    from torch.distributed.tensor import DTensor

    before = {name: param.to_local().detach().clone() for name, param in model.named_parameters()}
    optimizer.zero_grad(set_to_none=True)
    losses = []
    dtype = model.config.ops_config.linear_implementation.kwargs["dtype"]
    with _record_quantizer_calls(dtype) as calls:
        # Unequal microbatch lengths exercise per-forward dynamic MX quantization.
        for microbatch, length in enumerate((17, 31)):
            generator = torch.Generator().manual_seed(1000 + 100 * step + 10 * dist.get_rank() + microbatch)
            tokens = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to(get_device_type())
            positions = torch.arange(length, device=tokens.device).unsqueeze(0)
            output = model(input_ids=tokens, position_ids=positions, labels=tokens, use_cache=False)
            # The eager loss is already a token mean; average the two microbatches.
            loss = output.loss / 2
            assert torch.isfinite(loss)
            loss.backward()
            losses.append(loss.detach().cpu())

    # Four attention and three MLP projections per layer. The FP4 rows round
    # eight attention operands and five MLP operands. Recompute repeats each forward.
    passes = model.config.num_hidden_layers * 2 * 2
    assert calls == {
        "linear": (4 if dtype == "mxfp8" else 8) * passes,
        "gated_mlp": (3 if dtype == "mxfp8" else 5) * passes,
    }, calls

    for name, param in model.named_parameters():
        grad = param.grad
        assert grad is not None, name
        assert isinstance(grad, DTensor) and grad.dtype == torch.float32, name
        assert torch.isfinite(grad.to_local()).all(), name
    optimizer.step()
    assert all(not torch.equal(param.to_local(), before[name]) for name, param in model.named_parameters())
    optimizer.zero_grad(set_to_none=True)
    return torch.stack(losses)


def _fsdp_worker(root: str, selection: dict):
    from safetensors import safe_open
    from transformers import Qwen3Config

    from veomni.arguments import AcceleratorConfig, FSDPConfig
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
    from veomni.models import build_foundation_model
    from veomni.ops import OpsSelection
    from veomni.utils.save_safetensor_utils import save_hf_safetensor

    assert dist.get_backend() == dist.Backend.NCCL, "This acceptance case requires physical CUDA/NCCL execution."
    root = Path(root)
    base_path, dcp_path, hf_path = root / "base", root / "dcp", root / "hf"
    init_parallel_state_from_config(AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="fsdp2")), name="base")
    try:
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
            config.ops_implementation = _rows(selection)
            source = build_foundation_model(
                config, init_device="cpu", torch_dtype="float32", ops_implementation=_qat_ops(selection)
            )
            source.save_pretrained(base_path, safe_serialization=True)
            del source
        dist.barrier()

        model, optimizer = _build_fsdp(base_path, selection)
        initial = _master_snapshot(model)
        with safe_open(base_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            assert set(checkpoint.keys()) == set(initial)
            for name, param in model.named_parameters():
                torch.testing.assert_close(param.full_tensor().cpu(), checkpoint.get_tensor(name), atol=0, rtol=0)
        _step(model, optimizer, 0)
        saved_weights = _master_snapshot(model)
        assert initial.keys() == saved_weights.keys()
        saved_optimizer = _optimizer_snapshot(model, optimizer)
        assert all(state for state in saved_optimizer.values())
        DistributedCheckpointer.save(str(dcp_path), {"model": model, "optimizer": optimizer})

        expected_losses = _step(model, optimizer, 1)
        expected_weights = _master_snapshot(model)
        expected_optimizer = _optimizer_snapshot(model, optimizer)
        del model, optimizer

        resumed, resumed_optimizer = _build_fsdp(base_path, selection, skip_hf_load=True)
        DistributedCheckpointer.load(str(dcp_path), {"model": resumed, "optimizer": resumed_optimizer})
        torch.testing.assert_close(_master_snapshot(resumed), saved_weights, atol=0, rtol=0)
        torch.testing.assert_close(_optimizer_snapshot(resumed, resumed_optimizer), saved_optimizer, atol=0, rtol=0)
        torch.testing.assert_close(_step(resumed, resumed_optimizer, 1), expected_losses, atol=0, rtol=0)
        torch.testing.assert_close(_master_snapshot(resumed), expected_weights, atol=0, rtol=0)
        torch.testing.assert_close(_optimizer_snapshot(resumed, resumed_optimizer), expected_optimizer, atol=0, rtol=0)

        # The existing HF writer rounds FP32 masters to BF16 without changing them.
        save_hf_safetensor(
            save_hf_safetensor_path=str(hf_path),
            ckpt_manager="dcp",
            model=resumed,
            model_assets=[resumed.config],
        )
        torch.testing.assert_close(_master_snapshot(resumed), expected_weights, atol=0, rtol=0)
        reloaded, _ = _build_fsdp(hf_path, selection)
        expected = OpsSelection(**selection)
        assert reloaded.config.ops_config.linear_implementation == expected
        assert reloaded.config.ops_config.gated_mlp_implementation == expected
        rounded_weights = {name: weight.bfloat16().float() for name, weight in expected_weights.items()}
        torch.testing.assert_close(_master_snapshot(reloaded), rounded_weights, atol=0, rtol=0)
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_LAUNCH)
@pytest.mark.parametrize(
    ("dtype", "emulated"),
    [
        pytest.param("nvfp4", None, id="nvfp4"),
        pytest.param("mxfp4", None, id="mxfp4"),
        pytest.param("mxfp8", True, id="mxfp8-emulated"),
        pytest.param("mxfp8", False, id="mxfp8-native"),
    ],
)
def test_qwen3_torchao_qat_two_gpu_fsdp2_checkpoint_resume(tmp_path, monkeypatch, dtype, emulated):
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or runner_device_count() < _LAUNCH.nproc:
        pytest.skip("Requires two NVIDIA CUDA GPUs; no CPU/Gloo substitute.")
    if not all(get_gpu_compute_capability(rank) >= 80 for rank in range(_LAUNCH.nproc)):
        pytest.skip("Requires CUDA GPUs with native BF16 support.")
    if emulated is False and any(get_gpu_compute_capability(rank) < 100 for rank in range(_LAUNCH.nproc)):
        pytest.skip("Native MXFP8 requires two NVIDIA SM100+ devices.")
    pytest.importorskip("torchao")
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    kwargs = {"dtype": dtype} if emulated is None else {"dtype": dtype, "emulated": emulated}
    torchrun(_fsdp_worker, _LAUNCH.nproc, str(tmp_path), {"name": "torchao", "kwargs": kwargs})

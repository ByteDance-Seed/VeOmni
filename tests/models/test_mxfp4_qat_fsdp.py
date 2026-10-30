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

"""Physical two-GPU acceptance for the generated Qwen3 MXFP4 QAT path.

No CPU collective substitute: this case requires two CUDA devices and NCCL.
It covers accumulated updates, DCP continuation and ordinary HF export/reload.
"""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


_LAUNCH = LaunchSpec(nproc=2)
_TARGETS = ["model.layers.*.self_attn.*_proj", "model.layers.*.mlp.*_proj"]


def _build_fsdp(weights_path: Path, *, skip_hf_load: bool = False):
    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_config, build_foundation_model

    config = build_config(str(weights_path))
    model = build_foundation_model(config, init_device="meta", torch_dtype="float32")
    assert ".generated." in type(model).__module__
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


def _step(model, optimizer, step):
    optimizer.zero_grad(set_to_none=True)
    losses = []
    # Unequal microbatch lengths exercise per-forward dynamic MX quantization.
    for microbatch, length in enumerate((17, 31)):
        generator = torch.Generator().manual_seed(1000 + 100 * step + 10 * dist.get_rank() + microbatch)
        tokens = torch.randint(0, model.config.vocab_size, (1, length), generator=generator).to(get_device_type())
        positions = torch.arange(length, device=tokens.device).unsqueeze(0)
        output = model(input_ids=tokens, position_ids=positions, labels=tokens, use_cache=False)
        # The VeOmni loss is per-token; the final causal target is ignored.
        loss = output.loss.sum() / (2 * (length - 1))
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(loss.detach().cpu())
    assert all(torch.isfinite(param.grad.to_local()).all() for param in model.parameters() if param.grad is not None)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return torch.stack(losses)


def _fsdp_worker(root: str):
    from safetensors import safe_open
    from transformers import Qwen3Config

    from tests.tools.training_utils import make_eager_ops_config
    from veomni.arguments import AcceleratorConfig, FSDPConfig
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
    from veomni.models import build_foundation_model
    from veomni.utils.save_safetensor_utils import save_hf_safetensor

    assert dist.get_backend() == dist.Backend.NCCL, "This acceptance case requires physical CUDA/NCCL execution."
    root = Path(root)
    base_path, dcp_path, hf_path = root / "base", root / "dcp", root / "hf"
    init_parallel_state_from_config(AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="fsdp2")), name="base")
    try:
        if dist.get_rank() == 0:
            torch.manual_seed(42)
            config = Qwen3Config(
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
            ops = make_eager_ops_config(qat_implementation="torchao_mxfp4", qat_target_modules=_TARGETS)
            source = build_foundation_model(config, init_device="cpu", torch_dtype="float32", ops_implementation=ops)
            source.save_pretrained(base_path, safe_serialization=True)
            del source
        dist.barrier()

        model, optimizer = _build_fsdp(base_path)
        initial = _master_snapshot(model)
        with safe_open(base_path / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            assert set(checkpoint.keys()) == set(initial)
            for name, param in model.named_parameters():
                torch.testing.assert_close(param.full_tensor().cpu(), checkpoint.get_tensor(name), atol=0, rtol=0)
        _step(model, optimizer, 0)
        saved_weights = _master_snapshot(model)
        assert initial.keys() == saved_weights.keys()
        assert any(not torch.equal(initial[name], saved_weights[name]) for name in saved_weights if ".mlp." in name)
        saved_optimizer = _optimizer_snapshot(model, optimizer)
        assert all(state for state in saved_optimizer.values())
        DistributedCheckpointer.save(str(dcp_path), {"model": model, "optimizer": optimizer})

        expected_losses = _step(model, optimizer, 1)
        expected_weights = _master_snapshot(model)
        expected_optimizer = _optimizer_snapshot(model, optimizer)
        del model, optimizer

        resumed, resumed_optimizer = _build_fsdp(base_path, skip_hf_load=True)
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
        reloaded, _ = _build_fsdp(hf_path)
        assert reloaded.config.ops_config.qat_implementation == "torchao_mxfp4"
        assert reloaded.config.ops_config.qat_target_modules == _TARGETS
        rounded_weights = {name: weight.bfloat16().float() for name, weight in expected_weights.items()}
        torch.testing.assert_close(_master_snapshot(reloaded), rounded_weights, atol=0, rtol=0)
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_LAUNCH)
def test_qwen3_mxfp4_qat_two_gpu_fsdp2_checkpoint_resume(tmp_path, monkeypatch):
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or runner_device_count() < _LAUNCH.nproc:
        pytest.skip("Requires two NVIDIA CUDA GPUs; no CPU/Gloo substitute.")
    if not all(get_gpu_compute_capability(rank) >= 80 for rank in range(_LAUNCH.nproc)):
        pytest.skip("Requires CUDA GPUs with native BF16 support.")
    pytest.importorskip("torchao")
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torchrun(_fsdp_worker, _LAUNCH.nproc, str(tmp_path))

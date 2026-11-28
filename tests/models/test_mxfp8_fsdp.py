# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Exercise native MXFP8 FSDP2 transport through normal Qwen3 training and checkpoint calls."""

from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from scripts.ci.accelerator.launch_plan import LaunchSpec, runner_device_count
from tests.models.test_qat_fsdp import _master_snapshot, _optimizer_snapshot
from tests.tools.launch_utils import torchrun
from tests.tools.training_utils import make_eager_ops_config
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_gpu_compute_capability


_LAUNCH = LaunchSpec(nproc=2)
_MXFP8 = {"name": "torchao", "kwargs": {"dtype": "mxfp8"}}
_ROWS = {
    "linear_implementation": {"name": "eager", "overrides": [{"site": r"layers\.0\.v_proj", **_MXFP8}]},
    "gated_mlp_implementation": _MXFP8,
}
_SELECTED = {
    f"model.layers.0.{site}.weight" for site in ("self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
}


def _build(path, *, gather, resume=False):
    from torch.distributed.fsdp import FSDPModule

    from veomni.arguments import MixedPrecisionConfig
    from veomni.distributed.torch_parallelize import build_parallelize_model
    from veomni.models import build_config, build_foundation_model

    model = build_foundation_model(
        build_config(str(path)),
        init_device="meta",
        torch_dtype="float32",
        ops_implementation=make_eager_ops_config(**_ROWS),
    )
    model = build_parallelize_model(
        model,
        weights_path=str(path),
        init_device="meta",
        mixed_precision=MixedPrecisionConfig(param_dtype="bfloat16", reduce_dtype="float32"),
        enable_gradient_checkpointing=True,
        enable_reentrant=False,
        broadcast_model_weights_from_rank0=True,
        should_skip_hf_weight_load=resume,
        mxfp8_all_gather=gather,
    ).train()
    names = {id(param): name for name, param in model.named_parameters()}
    hooked = set()
    for module in model.modules():
        if isinstance(module, FSDPModule) and (group := module._get_fsdp_state()._fsdp_param_group):
            for param in group.fsdp_params:
                local = param._sharded_local_tensor
                name = names[id(param.sharded_param)]
                assert type(local) is torch.Tensor and local.dtype == torch.float32, name
                if hasattr(local, "fsdp_pre_all_gather"):
                    assert hasattr(local, "fsdp_post_all_gather"), name
                    hooked.add(name)
    assert hooked == (_SELECTED if gather else set())
    return model, torch.optim.AdamW(model.parameters(), lr=1e-3, foreach=False)


def _step(model, optimizer, step, *, inspect_payload=False):
    from veomni.distributed.fsdp2.mxfp8_weight import MXFP8GatheredWeight
    from veomni.ops.torchao import _quantize_mxfp8

    seen = set()
    hooks = []
    if inspect_payload:
        # The ordinary FSDP mixed-precision path casts each FP32 master to BF16 before quantization.
        # Quantize the full matrix in each direction independently, including blocks from both ranks.
        expected = {
            name: param.full_tensor().detach().bfloat16()
            for name, param in model.named_parameters()
            if name in _SELECTED
        }

        def check_payload(_module, _args):
            for name in expected.keys() - seen:
                weight = model.get_parameter(name)
                if not isinstance(weight, MXFP8GatheredWeight):
                    continue
                for transposed in (False, True):
                    master = expected[name]
                    reference = _quantize_mxfp8(master.t() if transposed else master, emulated=False)
                    actual = weight.mx(transposed=transposed, emulated=False)
                    for field in ("qdata", "scale"):
                        assert torch.equal(
                            getattr(actual, field).view(torch.uint8), getattr(reference, field).view(torch.uint8)
                        ), (name, field, transposed)
                seen.add(name)

        layer = model.model.layers[0]
        hooks = [
            layer.self_attn.register_forward_pre_hook(check_payload),
            layer.mlp.register_forward_pre_hook(check_payload),
        ]

    generator = torch.Generator().manual_seed(1100 + 10 * step + dist.get_rank())
    tokens = torch.randint(0, model.config.vocab_size, (1, 17 + 14 * step), generator=generator).to(get_device_type())
    try:
        loss = model(
            input_ids=tokens,
            position_ids=torch.arange(tokens.shape[1], device=tokens.device)[None],
            labels=tokens,
            use_cache=False,
        ).loss
        assert torch.isfinite(loss)
        loss.backward()
    finally:
        for hook in hooks:
            hook.remove()
    if inspect_payload:
        assert seen == _SELECTED
    grads = {name: param.grad.to_local().detach().cpu().clone() for name, param in model.named_parameters()}
    assert all(grad.dtype == torch.float32 and grad.isfinite().all() for grad in grads.values())
    assert all(torch.count_nonzero(grads[name]) > 0 for name in _SELECTED)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return loss.detach().cpu(), grads


def _worker(root):
    from transformers import Qwen3Config

    from veomni.arguments import AcceleratorConfig, FSDPConfig
    from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
    from veomni.distributed.parallel_state import clear_parallel_state, init_parallel_state_from_config
    from veomni.models import build_foundation_model
    from veomni.utils.save_safetensor_utils import save_hf_safetensor

    assert dist.get_backend() == dist.Backend.NCCL
    root = Path(root)
    base, dcp, hf = root / "base", root / "dcp", root / "hf"
    exact = {"rtol": 0, "atol": 0}
    init_parallel_state_from_config(AcceleratorConfig(fsdp_config=FSDPConfig(fsdp_mode="fsdp2")), name="base")
    try:
        if dist.get_rank() == 0:
            torch.manual_seed(42)
            config = Qwen3Config(
                architectures=["Qwen3ForCausalLM"],
                vocab_size=128,
                hidden_size=128,
                intermediate_size=256,
                num_hidden_layers=1,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=32,
                attention_dropout=0.0,
                tie_word_embeddings=False,
                use_cache=False,
            )
            config.ops_implementation = _ROWS
            source = build_foundation_model(
                config, init_device="cpu", torch_dtype="float32", ops_implementation=make_eager_ops_config(**_ROWS)
            )
            source.save_pretrained(base)
        dist.barrier()
        model, optimizer = _build(base, gather=True)
        ordinary, ordinary_optimizer = _build(base, gather=False)
        initial = _master_snapshot(model)
        torch.testing.assert_close(initial, _master_snapshot(ordinary), **exact)
        for step in range(2):
            result = _step(model, optimizer, step, inspect_payload=True)
            torch.testing.assert_close(result, _step(ordinary, ordinary_optimizer, step), **exact)
            updated = _master_snapshot(model)
            torch.testing.assert_close(updated, _master_snapshot(ordinary), **exact)
            if step == 0:
                assert all(not torch.equal(updated[name], initial[name]) for name in _SELECTED)
                DistributedCheckpointer.save(str(dcp), {"model": model, "optimizer": optimizer})

        expected = _master_snapshot(model)
        resumed, resumed_optimizer = _build(base, gather=True, resume=True)
        DistributedCheckpointer.load(str(dcp), {"model": resumed, "optimizer": resumed_optimizer})
        torch.testing.assert_close(_step(resumed, resumed_optimizer, 1, inspect_payload=True), result, **exact)
        torch.testing.assert_close(_master_snapshot(resumed), expected, **exact)
        torch.testing.assert_close(
            _optimizer_snapshot(resumed, resumed_optimizer), _optimizer_snapshot(model, optimizer), **exact
        )
        save_hf_safetensor(
            save_hf_safetensor_path=str(hf), ckpt_manager="dcp", model=resumed, model_assets=[resumed.config]
        )
        reloaded, _ = _build(hf, gather=True)
        torch.testing.assert_close(
            _master_snapshot(reloaded), {n: t.bfloat16().float() for n, t in expected.items()}, **exact
        )
    finally:
        clear_parallel_state()


@pytest.mark.launch_plan(provider=_LAUNCH)
def test_mxfp8_all_gather_two_gpu_checkpoint_resume(tmp_path, monkeypatch):
    if torch.version.hip is not None or not IS_CUDA_AVAILABLE or runner_device_count() < _LAUNCH.nproc:
        pytest.skip("Requires two NVIDIA GPUs.")
    if any(get_gpu_compute_capability(rank) < 100 for rank in range(_LAUNCH.nproc)):
        pytest.skip("Requires native MXFP8 on SM100+.")
    pytest.importorskip("torchao")
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    torchrun(_worker, _LAUNCH.nproc, str(tmp_path))

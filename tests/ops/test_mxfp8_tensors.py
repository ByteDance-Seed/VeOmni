# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""CPU numerical oracles for the MXFP8 transport representation.

These deliberately emulate GEMMs; native accelerator coverage is separate. The
transport payloads, both independent weight quantizers, autograd and master I/O
are the production code paths. No hardware-performance claim follows from them.
"""

import copy
import io
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import nn


pytest.importorskip("torchao")

from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common import KernelPreference

from scripts.ci.accelerator.launch_plan import LaunchSpec, from_parameter
from veomni.distributed.mxfp8 import _convert_linears, unwrap_mxfp8_tensor
from veomni.distributed.mxfp8.tensors import MXFP8Linear, MXFP8Weight


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _reference_quantize(tensor):
    return MXTensor.to_mx(
        tensor.contiguous(),
        torch.float8_e4m3fn,
        block_size=32,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED,
    )


def _gather(weight, world_size=2, dtype=torch.bfloat16):
    masters = [MXFP8Weight(shard, emulate=True) for shard in weight.chunk(world_size)]
    mesh = SimpleNamespace(size=lambda: world_size)
    policy = SimpleNamespace(param_dtype=dtype)
    results = [master.fsdp_pre_all_gather(mesh, weight.shape, weight.stride(), None, policy) for master in masters]
    payloads = [result[0] for result in results]
    gathered = tuple(torch.cat(parts, dim=0) for parts in zip(*payloads, strict=True))
    tensor, _ = masters[0].fsdp_post_all_gather(gathered, results[0][1], dtype)
    return tensor, payloads


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_sharded_quantization_matches_both_full_master_orientations(world_size):
    torch.manual_seed(52)
    weight = torch.randn(128, 96) * torch.linspace(0.1, 10, 128).unsqueeze(1)
    gathered, payloads = _gather(weight, world_size)
    row, col = _reference_quantize(weight), _reference_quantize(weight.t())
    torch.testing.assert_close(gathered.row_data, row.qdata.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(gathered.row_scale, row.scale.view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(gathered.col_data, col.qdata.t().contiguous().view(torch.uint8), rtol=0, atol=0)
    torch.testing.assert_close(gathered.col_scale, col.scale.t().contiguous().view(torch.uint8), rtol=0, atol=0)
    expected_dtype = torch.float32 if world_size == 1 else torch.uint8
    assert all(part.dtype == expected_dtype for payload in payloads for part in payload)
    total_bytes = sum(part.numel() * part.element_size() for payload in payloads for part in payload)
    assert total_bytes == weight.numel() * (4 if world_size == 1 else 2 + 2 / 32)
    # A transpose/requantization of the forward representation is not the
    # backward quantizer: this fixture would catch that tempting shortcut.
    double_quantized = _reference_quantize(row.dequantize().t())
    assert not torch.equal(col.qdata.view(torch.uint8), double_quantized.qdata.view(torch.uint8))


@pytest.mark.parametrize("tokens", [1, 17, 32, 65])
def test_gathered_linear_forward_and_gradients_match_independent_quantized_oracle(tokens):
    torch.manual_seed(31)
    master = torch.randn(128, 64)
    gathered, _ = _gather(master)
    layer = MXFP8Linear(64, 128, bias=True)
    layer._emulate = True
    layer.weight = nn.Parameter(gathered)
    layer.bias = nn.Parameter(torch.randn(128, dtype=torch.bfloat16))
    x = torch.randn(2, tokens, 64, dtype=torch.bfloat16, requires_grad=True)
    dy = torch.randn(2, tokens, 128, dtype=torch.bfloat16)
    output = layer(x)
    output.backward(dy)

    x2d = x.detach().flatten(0, 1)
    dy2d = dy.flatten(0, 1)
    pad = (-x2d.shape[0]) % 32
    xp, gp = F.pad(x2d, (0, 0, 0, pad)), F.pad(dy2d, (0, 0, 0, pad))
    # Explicitly dequantize independently derived operands: the oracle does not
    # call either the bridge's matmul or autograd implementation.
    row_weight = _reference_quantize(master).dequantize(torch.bfloat16)
    col_weight_t = _reference_quantize(master.t()).dequantize(torch.bfloat16)
    expected = _reference_quantize(xp).dequantize() @ row_weight.t()
    expected = expected[: x2d.shape[0]].reshape_as(output) + layer.bias.detach()
    dx = _reference_quantize(gp).dequantize() @ col_weight_t.t()
    dw = _reference_quantize(gp.t()).dequantize() @ _reference_quantize(xp.t()).dequantize().t()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, dx[: x2d.shape[0]].reshape_as(x), rtol=0, atol=0)
    torch.testing.assert_close(layer.weight.grad, dw, rtol=0, atol=0)
    torch.testing.assert_close(layer.bias.grad, dy.sum((0, 1)), rtol=0, atol=0)


def test_post_all_gather_reuses_parameter_and_payload_after_master_update():
    torch.manual_seed(7)
    master = torch.randn(128, 64)
    gathered, _ = _gather(master)
    parameter = nn.Parameter(torch.as_strided(gathered, master.shape, master.stride(), 0))
    identities = [part.data_ptr() for part in parameter.payload]
    updated, _ = _gather(master + 0.125)
    MXFP8Weight(master[:64], emulate=True).fsdp_post_all_gather(updated.payload, (), torch.bfloat16, out=parameter)
    assert [part.data_ptr() for part in parameter.payload] == identities
    for actual, expected in zip(parameter.payload, updated.payload, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("shape,world_size", [((64, 48), 2), ((96, 64), 2), ((32, 64), 2), ((64, 64), 3)])
def test_all_gather_rejects_block_splitting_or_uneven_shards(shape, world_size):
    master = torch.randn(*shape)
    wrapper = MXFP8Weight(master.chunk(world_size)[0], emulate=True)
    with pytest.raises(ValueError, match="32"):
        wrapper.fsdp_pre_all_gather(
            SimpleNamespace(size=lambda: world_size),
            master.shape,
            master.stride(),
            None,
            SimpleNamespace(param_dtype=torch.bfloat16),
        )


def test_master_optimizer_updates_and_tensor_serialization_keep_fp32():
    weight = nn.Parameter(MXFP8Weight(torch.arange(64 * 64, dtype=torch.float32).reshape(64, 64), emulate=True))
    before = unwrap_mxfp8_tensor(weight).clone()
    weight.grad = torch.full_like(before, 0.125)
    torch.optim.SGD([weight], lr=0.5).step()
    expected = before - 0.0625
    torch.testing.assert_close(unwrap_mxfp8_tensor(weight), expected, rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(weight, stream)
    stream.seek(0)
    loaded = torch.load(stream, weights_only=True)
    assert isinstance(loaded, MXFP8Weight)
    assert loaded.dtype == torch.float32
    torch.testing.assert_close(unwrap_mxfp8_tensor(loaded), expected, rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["single_tensor", "foreach", "fused"])
def test_adamw_updates_master_and_moments_identically_to_plain_fp32(mode):
    # Exercise the actual in-place optimizer dispatch on the tensor subclass,
    # including the fused mode used by VeOmniModelRuntime.
    generator = torch.Generator().manual_seed(17)
    initial = torch.randn(64, 64, generator=generator)
    plain = nn.Parameter(initial.clone())
    wrapped = nn.Parameter(MXFP8Weight(initial.clone(), emulate=True))
    kwargs = {"foreach": mode == "foreach", "fused": mode == "fused"}
    optimizers = [torch.optim.AdamW([parameter], lr=0.003, **kwargs) for parameter in (plain, wrapped)]
    for _ in range(3):
        gradient = torch.randn(initial.shape, generator=generator) * 0.01
        for parameter, optimizer in zip((plain, wrapped), optimizers, strict=True):
            parameter.grad = gradient.clone()
            optimizer.step()
        assert isinstance(wrapped, MXFP8Weight) and wrapped.dtype == torch.float32
        torch.testing.assert_close(unwrap_mxfp8_tensor(wrapped), plain, rtol=0, atol=0)
        for key in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                unwrap_mxfp8_tensor(optimizers[1].state[wrapped][key]),
                optimizers[0].state[plain][key],
                rtol=0,
                atol=0,
            )


@pytest.mark.parametrize("gather", [False, True])
def test_meta_materialization_and_state_dict_preserve_exact_masters(gather):
    torch.manual_seed(3)
    original = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    expected = copy.deepcopy(original.state_dict())
    with torch.device("meta"):
        model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
    config = SimpleNamespace(enable_fsdp_all_gather=gather, target_modules=["0", "2"])
    model = _convert_linears(model, config, emulate=True)
    model.to_empty(device="cpu")
    model.load_state_dict(expected)
    state = model.state_dict()
    assert state.keys() == expected.keys()
    for name, value in state.items():
        assert type(value) is torch.Tensor and value.dtype == torch.float32
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    original.load_state_dict(torch.load(stream, weights_only=True))
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_export_rejects_lossy_compute_representation():
    gathered, _ = _gather(torch.randn(128, 64))
    with pytest.raises(ValueError, match="not masters"):
        unwrap_mxfp8_tensor(gathered)


@pytest.mark.parametrize("gathered", [False, True])
def test_mxfp8_aot_autograd_matches_eager_for_ragged_tokens(gathered):
    torch.manual_seed(81)
    model = MXFP8Linear.from_float(nn.Linear(64, 128), all_gather=False, emulate=True)
    if gathered:
        weight, _ = _gather(model.weight.detach())
        model.weight = nn.Parameter(weight)
    compiled = torch.compile(copy.deepcopy(model), backend="aot_eager", fullgraph=True)
    source = torch.randn(17, 64, dtype=torch.bfloat16)
    eager_input = source.clone().requires_grad_()
    compiled_input = source.clone().requires_grad_()
    eager_output = model(eager_input)
    compiled_output = compiled(compiled_input)
    eager_output.square().sum().backward()
    compiled_output.square().sum().backward()
    torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)
    torch.testing.assert_close(compiled_input.grad, eager_input.grad, rtol=0, atol=0)
    for actual, expected in zip(compiled.parameters(), model.parameters(), strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=0, atol=0)


def _fsdp_checkpoint_worker(rank, world_size, rendezvous, checkpoint, gather):
    from torch.distributed.checkpoint import load, save
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    from torch.distributed.tensor import DTensor

    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cpu", (world_size,))

        def build():
            torch.manual_seed(12)
            model = nn.Sequential(nn.Linear(64, 128), nn.GELU(), nn.Linear(128, 64))
            config = SimpleNamespace(enable_fsdp_all_gather=gather, target_modules=["0", "2"])
            model = _convert_linears(model, config, emulate=True)
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
            for layer in (model[0], model[2]):
                fully_shard(layer, mesh=mesh, reshard_after_forward=True, mp_policy=policy)
            fully_shard(model, mesh=mesh, mp_policy=policy)
            return model, torch.optim.AdamW(model.parameters(), lr=0.01, fused=True)

        def step(model, optimizer, index):
            optimizer.zero_grad()
            outputs = []
            for microbatch in range(2):
                generator = torch.Generator().manual_seed(31 + rank * 10 + index * 2 + microbatch)
                output = model(torch.randn(17, 64, generator=generator))
                (output.float().square().mean() / 2).backward()
                outputs.append(output.detach())
            optimizer.step()
            assert all(parameter.dtype == torch.float32 for parameter in model.parameters())
            return outputs

        model, optimizer = build()
        step(model, optimizer, 0)
        model_state, optimizer_state = get_state_dict(model, optimizer)
        for value in model_state.values():
            assert isinstance(value, DTensor)
            assert type(value.to_local()) is torch.Tensor
            assert value.dtype == torch.float32
        save({"model": model_state, "optimizer": optimizer_state}, checkpoint_id=checkpoint)
        expected_output = step(model, optimizer, 1)
        expected_state = {name: value.to_local().clone() for name, value in model.state_dict().items()}

        restored, restored_optimizer = build()
        restored_model_state, restored_optimizer_state = get_state_dict(restored, restored_optimizer)
        state = {"model": restored_model_state, "optimizer": restored_optimizer_state}
        load(state, checkpoint_id=checkpoint)
        set_state_dict(
            restored,
            restored_optimizer,
            model_state_dict=state["model"],
            optim_state_dict=state["optimizer"],
        )
        actual_output = step(restored, restored_optimizer, 1)
        torch.testing.assert_close(actual_output, expected_output, rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value.to_local(), expected_state[name], rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size,gather", [(1, True), (2, True), (2, False)])
@pytest.mark.launch_plan(provider=from_parameter("world_size", device="cpu"))
def test_real_cpu_fsdp_reshard_fused_adamw_and_dcp_resume(tmp_path, world_size, gather):
    # Real FSDP2 hooks, Gloo collectives, accumulated gradients and DCP; only the
    # GEMM is emulated. This does not establish native GPU kernel/NCCL acceptance.
    mp.spawn(
        _fsdp_checkpoint_worker,
        args=(world_size, str(tmp_path / "rendezvous"), str(tmp_path / "checkpoint"), gather),
        nprocs=world_size,
        join=True,
    )

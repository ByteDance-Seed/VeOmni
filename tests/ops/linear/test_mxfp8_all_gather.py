# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 shard packing, gathered GEMMs, and FSDP2 buffer reuse on CPU."""

from unittest.mock import Mock

import pytest
import torch

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.ops.linear.test_mxfp8 import _reference, _rounded
from veomni.distributed.fsdp2.mxfp8_weight import MXFP8AllGather
from veomni.ops.torchao import torchao_mxfp8_linear


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _gather_inputs(hooks, master, world):
    mesh = Mock()
    mesh.size.return_value = world
    inputs = []
    for shard in master.chunk(world):
        payload, metadata = hooks.pre(shard, mesh, master.shape, master.stride(), None, None)
        assert metadata is (world == 1)
        if world == 1:
            assert len(payload) == 1 and payload[0].dtype == torch.bfloat16
        else:
            assert len(payload) == 4
            assert all(p.dtype == torch.uint8 and p.is_contiguous() for p in payload)
            assert sum(p.nbytes for p in payload) == 2 * (shard.numel() + shard.numel() // 32)
        inputs.append(payload)
    # Native FSDP2 concatenates rank inputs for each output and restores its logical shape.
    return tuple(torch.cat(parts) for parts in zip(*inputs, strict=True)), metadata


@pytest.mark.parametrize(("world", "master_dtype"), [(1, torch.bfloat16), (2, torch.float32)])
def test_gathered_mxfp8_layout_and_storage_reuse(world, master_dtype):
    pytest.importorskip("torchao")
    generator = torch.Generator().manual_seed(472)
    hooks = MXFP8AllGather(emulated=True)
    master = torch.randn(128, 96, generator=generator, dtype=master_dtype)
    outputs, metadata = _gather_inputs(hooks, master, world)
    weight, inner_tensors = hooks.post(master, outputs, metadata, torch.bfloat16)
    assert weight.dtype == torch.bfloat16 and weight.shape == master.shape
    assert tuple(t.shape for t in weight.payload) == ((128, 96), (128, 3), (128, 96), (4, 96))
    if world == 1:
        assert all(a is b for a, b in zip(weight.payload, inner_tensors, strict=True))
    else:
        assert not inner_tensors
        assert all(a.data_ptr() == b.data_ptr() for a, b in zip(weight.payload, outputs, strict=True))

    def check_values():
        for transposed in (False, True):
            mx = weight.mx(transposed=transposed, emulated=True)
            operand = master.T if transposed else master
            torch.testing.assert_close(mx.dequantize(), _rounded(operand.bfloat16()), rtol=0, atol=0)

    check_values()
    # Reshard frees the native gather outputs and any extra storage returned by post().
    # The same parameter and its aliases must see the next optimizer update on unshard.
    param = torch.nn.Parameter(torch.as_strided(weight, master.shape, master.stride(), 0))
    managed = (*outputs, *inner_tensors)
    for tensor in managed:
        tensor.untyped_storage().resize_(0)
    assert all(t.untyped_storage().nbytes() == 0 for t in param.payload)
    master.normal_(generator=generator)
    new_outputs, metadata = _gather_inputs(hooks, master, world)
    for tensor in managed:
        tensor.untyped_storage().resize_(tensor.nbytes)
    for dest, src in zip(outputs, new_outputs, strict=True):
        dest.copy_(src)
    assert hooks.post(master, outputs, metadata, torch.bfloat16, out=param) is None
    check_values()


@pytest.mark.parametrize("tokens", [0, 37])
def test_gathered_mxfp8_forward_and_gradients(tokens):
    pytest.importorskip("torchao")
    generator = torch.Generator().manual_seed(138)
    master = torch.randn(128, 96, generator=generator)
    hooks = MXFP8AllGather(emulated=True)
    outputs, metadata = _gather_inputs(hooks, master, 2)
    weight, _ = hooks.post(master, outputs, metadata, torch.bfloat16)
    weight = torch.nn.Parameter(weight)
    x = torch.randn(2, tokens, 96, generator=generator, requires_grad=True)
    bias = torch.randn(128, generator=generator, requires_grad=True)
    dy = torch.randn(2, tokens, 128, generator=generator, dtype=torch.bfloat16)
    output = torchao_mxfp8_linear(x, weight, bias, emulated=True)
    output.backward(dy)
    if tokens:
        reference = _reference(x.detach(), master.bfloat16(), bias.detach(), dy)
    else:
        reference = (dy, torch.zeros_like(x), torch.zeros_like(master, dtype=weight.dtype), torch.zeros_like(bias))
    for actual, expected in zip((output, x.grad, weight.grad, bias.grad), reference, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

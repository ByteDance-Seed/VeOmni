import copy
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from veomni.arguments import AcceleratorConfig, FSDPConfig, MixedPrecisionConfig, TrainingArguments
from veomni.distributed.fsdp2 import reduce_scatter as reduce_scatter_module
from veomni.distributed.fsdp2.reduce_scatter import (
    BF16ReduceScatterWithFP32Accumulation,
    register_bf16_reduce_scatter_with_fp32_accumulation,
)
from veomni.utils.device import IS_CUDA_AVAILABLE


@pytest.mark.parametrize(
    ("op", "premul_sum_factor", "expected_scale"),
    [
        (dist.ReduceOp.SUM, None, 1.0),
        (dist.ReduceOp.AVG, None, 0.5),
        (dist._make_nccl_premul_sum(0.25), 0.25, 0.25),
    ],
)
def test_bf16_reduce_scatter_accumulates_in_fp32(monkeypatch, op, premul_sum_factor, expected_scale):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist,
        "all_to_all_single",
        lambda output, input, group, async_op: output.copy_(input),
    )

    input_tensor = torch.tensor([256.0, 1.0, 1.0, 1.0], dtype=torch.bfloat16)
    output_tensor = torch.empty(2, dtype=torch.bfloat16)
    comm = BF16ReduceScatterWithFP32Accumulation(premul_sum_factor=premul_sum_factor)

    result = comm(output_tensor, input_tensor, object(), op)

    assert result is None
    expected = input_tensor.view(2, -1).float().sum(dim=0).mul(expected_scale).bfloat16()
    torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)


def test_bf16_reduce_scatter_rejects_async():
    with pytest.raises(NotImplementedError, match="async_op=True"):
        BF16ReduceScatterWithFP32Accumulation()(
            torch.empty(2, dtype=torch.bfloat16),
            torch.empty(4, dtype=torch.bfloat16),
            object(),
            dist.ReduceOp.SUM,
            async_op=True,
        )


def test_non_bf16_input_uses_native_reduce_scatter(monkeypatch):
    expected_work = object()
    calls = []

    def fake_reduce_scatter(output, input, *, group, op, async_op):
        calls.append((output, input, group, op, async_op))
        return expected_work

    monkeypatch.setattr(dist, "reduce_scatter_tensor", fake_reduce_scatter)
    output_tensor = torch.empty(2, dtype=torch.float32)
    input_tensor = torch.empty(4, dtype=torch.float32)
    group = object()

    work = BF16ReduceScatterWithFP32Accumulation()(
        output_tensor,
        input_tensor,
        group,
        dist.ReduceOp.AVG,
        async_op=True,
    )

    assert work is expected_work
    assert calls == [(output_tensor, input_tensor, group, dist.ReduceOp.AVG, True)]


def test_bf16_reduce_scatter_validates_contract(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    comm = BF16ReduceScatterWithFP32Accumulation()

    with pytest.raises(TypeError, match="BF16 output"):
        comm(torch.empty(2), torch.empty(4, dtype=torch.bfloat16), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="one equally sized output shard"):
        comm(torch.empty(3, dtype=torch.bfloat16), torch.empty(4, dtype=torch.bfloat16), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="premul_sum_factor"):
        comm(
            torch.empty(2, dtype=torch.bfloat16),
            torch.empty(4, dtype=torch.bfloat16),
            object(),
            dist._make_nccl_premul_sum(0.5),
        )


def test_registers_every_fsdp_module(monkeypatch):
    class FakeFSDPModule:
        def __init__(self) -> None:
            self.comms = []

        def set_custom_reduce_scatter(self, comm) -> None:
            self.comms.append(comm)

    class FakeModel:
        def __init__(self) -> None:
            self.fsdp1 = FakeFSDPModule()
            self.unwrapped = object()
            self.fsdp2 = FakeFSDPModule()

        def modules(self):
            return [self, self.fsdp1, self.unwrapped, self.fsdp2]

    monkeypatch.setattr(reduce_scatter_module, "FSDPModule", FakeFSDPModule)
    model = FakeModel()

    count = register_bf16_reduce_scatter_with_fp32_accumulation(
        model,
        premul_sum_factors={model.fsdp1: 0.25},
    )

    assert count == 2
    assert len(model.fsdp1.comms) == 1
    assert model.fsdp1.comms[0] is not model.fsdp2.comms[0]
    assert model.fsdp1.comms[0]._premul_sum_factor == 0.25
    assert model.fsdp2.comms[0]._premul_sum_factor is None


def test_reduce_scatter_config_requires_fsdp2_bf16_reduction():
    with pytest.raises(ValueError, match="fsdp_mode='fsdp2'"):
        FSDPConfig(
            fsdp_mode="ddp",
            mixed_precision=MixedPrecisionConfig(reduce_dtype="bfloat16"),
            reduce_scatter_with_fp32_accumulation=True,
        )
    with pytest.raises(ValueError, match="reduce_dtype='bfloat16'"):
        FSDPConfig(reduce_scatter_with_fp32_accumulation=True)

    config = FSDPConfig(
        mixed_precision=MixedPrecisionConfig(reduce_dtype="bfloat16"),
        reduce_scatter_with_fp32_accumulation=True,
    )
    assert config.reduce_scatter_with_fp32_accumulation


def test_reduce_scatter_config_rejects_hsdp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "4")
    fsdp_config = FSDPConfig(
        mixed_precision=MixedPrecisionConfig(reduce_dtype="bfloat16"),
        reduce_scatter_with_fp32_accumulation=True,
    )

    with pytest.raises(ValueError, match="does not support HSDP"):
        TrainingArguments(
            accelerator=AcceleratorConfig(
                dp_replicate_size=2,
                dp_shard_size=2,
                fsdp_config=fsdp_config,
            )
        )


def _run_reduce_scatter_nccl() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    shard_numel = 4096
    values = torch.arange(world_size * shard_numel, device=device, dtype=torch.float32)
    input_tensor = (values.remainder(31) + rank * 0.25).bfloat16()

    for op, scale in (
        (dist.ReduceOp.SUM, 1.0),
        (dist.ReduceOp.AVG, 1.0 / world_size),
        (dist._make_nccl_premul_sum(1.0 / world_size), 1.0 / world_size),
    ):
        reference = torch.empty(shard_numel, device=device, dtype=torch.float32)
        dist.reduce_scatter_tensor(reference, input_tensor.float(), group=dist.group.WORLD, op=dist.ReduceOp.SUM)
        expected = reference.mul(scale).bfloat16()

        output = torch.empty(shard_numel, device=device, dtype=torch.bfloat16)
        comm = BF16ReduceScatterWithFP32Accumulation(premul_sum_factor=1.0 / world_size)
        comm(output, input_tensor, dist.group.WORLD, op)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_fsdp2_optimizer_step() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    class RecordingReduceScatter(BF16ReduceScatterWithFP32Accumulation):
        def __init__(self, premul_sum_factor):
            super().__init__(premul_sum_factor)
            self.calls = []

        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            self.calls.append((input_tensor.dtype, op))
            return super().__call__(output_tensor, input_tensor, group, op, async_op)

    for gradient_divide_factor in (None, world_size * 2):
        torch.manual_seed(1234)
        baseline = nn.Linear(32, 16, bias=False, device=device)
        custom = copy.deepcopy(baseline)
        baseline_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        custom_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)
        fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
        fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
        if gradient_divide_factor is not None:
            baseline.set_gradient_divide_factor(gradient_divide_factor)
            custom.set_gradient_divide_factor(gradient_divide_factor)
        premul_sum_factor = None if gradient_divide_factor is None else 1.0 / gradient_divide_factor
        comm = RecordingReduceScatter(premul_sum_factor)
        custom.set_custom_reduce_scatter(comm)

        torch.manual_seed(9000 + rank)
        inputs = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
        baseline(inputs).float().square().sum().backward()
        custom(inputs).float().square().sum().backward()

        baseline_grad = baseline.weight.grad.to_local()
        custom_grad = custom.weight.grad.to_local()
        assert baseline_grad.dtype == torch.float32
        assert custom_grad.dtype == torch.float32
        assert comm.calls
        assert all(dtype == torch.bfloat16 for dtype, _ in comm.calls)
        expected_op = dist.ReduceOp.AVG if gradient_divide_factor is None else dist.ReduceOp.PREMUL_SUM
        assert all(op == expected_op for _, op in comm.calls)
        torch.testing.assert_close(custom_grad, baseline_grad.bfloat16().float(), rtol=0, atol=0)

        baseline_optim = torch.optim.SGD(baseline.parameters(), lr=1e-3)
        custom_optim = torch.optim.SGD(custom.parameters(), lr=1e-3)
        baseline_optim.step()
        custom_optim.step()
        assert torch.isfinite(custom.weight.to_local()).all()


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires two CUDA devices")
def test_reduce_scatter_fp32_accumulation_nccl():
    from ..tools.launch_utils import torchrun

    torchrun(_run_reduce_scatter_nccl, world_size=2)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires two CUDA devices")
def test_reduce_scatter_fp32_accumulation_fsdp2_step():
    from ..tools.launch_utils import torchrun

    torchrun(_run_fsdp2_optimizer_step, world_size=2)

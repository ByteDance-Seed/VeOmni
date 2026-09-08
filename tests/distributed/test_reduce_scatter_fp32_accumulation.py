import copy
import gc
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
    BF16FP16ReduceScatterWithFP32Accumulation,
    register_bf16_fp16_reduce_scatter_with_fp32_accumulation,
)
from veomni.distributed.torch_parallelize import _configure_fsdp_gradient_reduction
from veomni.utils.device import IS_CUDA_AVAILABLE


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("reduction_scale", [1.0, 0.5, 1.0 / 3.0])
def test_low_precision_reduce_scatter_accumulates_and_scales_in_fp32(monkeypatch, dtype, reduction_scale):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist,
        "all_to_all_single",
        lambda output, input, group, async_op: output.copy_(input),
    )

    input_tensor = torch.tensor([256.0, 1.0, 1.0, 1.0], dtype=dtype)
    output_tensor = torch.empty(2, dtype=dtype)
    comm = BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=reduction_scale)

    result = comm(output_tensor, input_tensor, object(), dist.ReduceOp.SUM)

    assert result is None
    expected = input_tensor.view(2, -1).float().sum(dim=0).mul(reduction_scale).to(dtype)
    torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_rejects_async(dtype):
    with pytest.raises(NotImplementedError, match="async_op=True"):
        BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=0.5)(
            torch.empty(2, dtype=dtype),
            torch.empty(4, dtype=dtype),
            object(),
            dist.ReduceOp.SUM,
            async_op=True,
        )


def test_non_low_precision_input_is_rejected():
    with pytest.raises(TypeError, match="BF16 or FP16 input"):
        BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=0.5)(
            torch.empty(2),
            torch.empty(4),
            object(),
            dist.ReduceOp.SUM,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_validates_contract(monkeypatch, dtype):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    comm = BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=0.5)

    with pytest.raises(TypeError, match="matching input and output dtypes"):
        comm(torch.empty(2), torch.empty(4, dtype=dtype), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="one equally sized output shard"):
        comm(torch.empty(3, dtype=dtype), torch.empty(4, dtype=dtype), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="requires SUM"):
        comm(
            torch.empty(2, dtype=dtype),
            torch.empty(4, dtype=dtype),
            object(),
            dist.ReduceOp.AVG,
        )


def test_registers_only_selected_fsdp_modules_and_moves_scaling_into_hook(monkeypatch):
    class FakeFSDPModule:
        def __init__(self) -> None:
            self.comms = []
            self.gradient_divide_factors = []
            self.force_sum_reductions = []

        def set_gradient_divide_factor(self, factor) -> None:
            self.gradient_divide_factors.append(factor)

        def set_force_sum_reduction_for_comms(self, enable) -> None:
            self.force_sum_reductions.append(enable)

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

    count = register_bf16_fp16_reduce_scatter_with_fp32_accumulation(
        model,
        reduction_scales={model.fsdp1: 0.25},
    )

    assert count == 1
    assert len(model.fsdp1.comms) == 1
    assert model.fsdp1.comms[0]._reduction_scale == 0.25
    assert model.fsdp1.gradient_divide_factors == [1.0]
    assert model.fsdp1.force_sum_reductions == [True]
    assert model.fsdp2.comms == []
    assert model.fsdp2.gradient_divide_factors == []
    assert model.fsdp2.force_sum_reductions == []


@pytest.mark.parametrize("reduce_scatter_group_size", [1, 2])
def test_extra_parallel_single_rank_group_keeps_native_gradient_scaling(reduce_scatter_group_size):
    class FakeFSDPModule:
        def __init__(self) -> None:
            self.gradient_divide_factors = []

        def set_gradient_divide_factor(self, factor) -> None:
            self.gradient_divide_factors.append(factor)

    module = FakeFSDPModule()
    reduction_scales = {}
    _configure_fsdp_gradient_reduction(
        module,
        gradient_divide_factor=8.0,
        reduce_scatter_group_size=reduce_scatter_group_size,
        low_precision_reduction_scales=reduction_scales,
    )

    if reduce_scatter_group_size == 1:
        assert module.gradient_divide_factors == [8.0]
        assert reduction_scales == {}
    else:
        assert module.gradient_divide_factors == []
        assert reduction_scales == {module: 0.125}


def test_reduce_scatter_config_requires_fsdp2_low_precision_reduction():
    with pytest.raises(ValueError, match="fsdp_mode='fsdp2'"):
        FSDPConfig(
            fsdp_mode="ddp",
            mixed_precision=MixedPrecisionConfig(reduce_dtype="bfloat16"),
            reduce_scatter_with_fp32_accumulation=True,
        )
    with pytest.raises(ValueError, match="reduce_dtype='bfloat16' or 'float16'"):
        FSDPConfig(reduce_scatter_with_fp32_accumulation=True)

    for reduce_dtype in ("bfloat16", "float16"):
        config = FSDPConfig(
            mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
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
    for dtype in (torch.bfloat16, torch.float16):
        input_tensor = (values.remainder(31) + rank * 0.25).to(dtype)
        for scale in (1.0, 1.0 / world_size, 1.0 / 3.0):
            reference = torch.empty(shard_numel, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(reference, input_tensor.float(), group=dist.group.WORLD, op=dist.ReduceOp.SUM)
            expected = reference.mul(scale).to(dtype)

            output = torch.empty(shard_numel, device=device, dtype=dtype)
            comm = BF16FP16ReduceScatterWithFP32Accumulation(reduction_scale=scale)
            comm(output, input_tensor, dist.group.WORLD, dist.ReduceOp.SUM)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_fsdp2_optimizer_step() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    class RecordingReduceScatter(BF16FP16ReduceScatterWithFP32Accumulation):
        def __init__(self, reduction_scale):
            super().__init__(reduction_scale)
            self.calls = []

        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            self.calls.append((input_tensor.dtype, op))
            return super().__call__(output_tensor, input_tensor, group, op, async_op)

    for reduce_dtype in (torch.bfloat16, torch.float16):
        for gradient_divide_factor in (float(world_size), 3.0):
            torch.manual_seed(1234)
            baseline = nn.Linear(32, 16, bias=False, device=device)
            custom = copy.deepcopy(baseline)
            baseline_policy = MixedPrecisionPolicy(param_dtype=reduce_dtype, reduce_dtype=torch.float32)
            custom_policy = MixedPrecisionPolicy(param_dtype=reduce_dtype, reduce_dtype=reduce_dtype)
            fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
            fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
            baseline.set_gradient_divide_factor(gradient_divide_factor)
            custom.set_gradient_divide_factor(1.0)
            custom.set_force_sum_reduction_for_comms(True)
            comm = RecordingReduceScatter(1.0 / gradient_divide_factor)
            custom.set_custom_reduce_scatter(comm)

            torch.manual_seed(9000 + rank)
            inputs = torch.randn(8, 32, device=device, dtype=reduce_dtype)
            baseline(inputs).float().square().sum().backward()
            custom(inputs).float().square().sum().backward()

            baseline_grad = baseline.weight.grad.to_local()
            custom_grad = custom.weight.grad.to_local()
            assert baseline_grad.dtype == torch.float32
            assert custom_grad.dtype == torch.float32
            assert comm.calls
            assert all(dtype == reduce_dtype for dtype, _ in comm.calls)
            assert all(op == dist.ReduceOp.SUM for _, op in comm.calls)
            expected_grad = baseline_grad.to(reduce_dtype).float()
            torch.testing.assert_close(custom_grad, expected_grad, rtol=5e-3, atol=5e-3)

            baseline_optim = torch.optim.SGD(baseline.parameters(), lr=1e-3)
            custom_optim = torch.optim.SGD(custom.parameters(), lr=1e-3)
            baseline_optim.step()
            custom_optim.step()
            assert torch.isfinite(custom.weight.to_local()).all()
            del baseline_optim, custom_optim, baseline_grad, custom_grad, inputs, comm, baseline, custom
            gc.collect()
            dist.barrier(device_ids=[device.index])

    baseline = nn.Linear(4, 1, bias=False, device=device)
    custom = copy.deepcopy(baseline)
    baseline_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    custom_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float16)
    fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
    fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
    custom.set_gradient_divide_factor(1.0)
    custom.set_force_sum_reduction_for_comms(True)
    comm = RecordingReduceScatter(1.0 / world_size)
    custom.set_custom_reduce_scatter(comm)

    inputs = torch.full((1, 4), 65504.0, device=device, dtype=torch.float16)
    baseline(inputs).sum().backward()
    custom(inputs).sum().backward()
    baseline_grad = baseline.weight.grad.to_local()
    custom_grad = custom.weight.grad.to_local()
    assert torch.isfinite(custom_grad).all()
    torch.testing.assert_close(custom_grad, baseline_grad.to(torch.float16).float(), rtol=0, atol=0)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_nccl():
    from ..tools.launch_utils import torchrun

    torchrun(_run_reduce_scatter_nccl, world_size=4)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_fsdp2_step():
    from ..tools.launch_utils import torchrun

    torchrun(_run_fsdp2_optimizer_step, world_size=4)

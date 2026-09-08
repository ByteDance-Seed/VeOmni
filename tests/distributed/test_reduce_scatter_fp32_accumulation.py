import copy
import gc
import os

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.device_mesh import init_device_mesh

from veomni.arguments import FSDPConfig, MixedPrecisionConfig
from veomni.distributed import torch_parallelize
from veomni.distributed.fsdp2 import reduce_scatter as reduce_scatter_module
from veomni.distributed.fsdp2.reduce_scatter import (
    FP32ReduceScatterWithLowPrecisionTransport,
    register_fp32_reduce_scatter_with_low_precision_transport,
)
from veomni.distributed.torch_parallelize import (
    _configure_fsdp_gradient_reduction,
    _reduce_scatter_group_size,
    _uses_low_precision_reduce_scatter_transport,
)
from veomni.utils.device import IS_CUDA_AVAILABLE


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("reduction_scale", [1.0, 0.5, 1.0 / 3.0])
def test_low_precision_transport_accumulates_and_outputs_fp32(monkeypatch, dtype, reduction_scale):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(
        dist,
        "all_to_all_single",
        lambda output, input, group, async_op: output.copy_(input),
    )

    input_tensor = torch.tensor([256.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    output_tensor = torch.empty(2, dtype=torch.float32)
    comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale)

    result = comm(output_tensor, input_tensor, object(), dist.ReduceOp.SUM)

    assert result is None
    expected = input_tensor.to(dtype).view(2, -1).float().sum(dim=0).mul(reduction_scale)
    torch.testing.assert_close(output_tensor, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_rejects_async(dtype):
    with pytest.raises(NotImplementedError, match="async_op=True"):
        FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=0.5)(
            torch.empty(2),
            torch.empty(4),
            object(),
            dist.ReduceOp.SUM,
            async_op=True,
        )


def test_non_fp32_reduction_buffers_are_rejected():
    with pytest.raises(TypeError, match="requires FP32"):
        FP32ReduceScatterWithLowPrecisionTransport(torch.bfloat16, reduction_scale=0.5)(
            torch.empty(2, dtype=torch.bfloat16),
            torch.empty(4, dtype=torch.bfloat16),
            object(),
            dist.ReduceOp.SUM,
        )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_low_precision_reduce_scatter_validates_contract(monkeypatch, dtype):
    monkeypatch.setattr(dist, "get_world_size", lambda group: 2)
    comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=0.5)

    with pytest.raises(ValueError, match="one equally sized output shard"):
        comm(torch.empty(3), torch.empty(4), object(), dist.ReduceOp.SUM)
    with pytest.raises(ValueError, match="requires SUM"):
        comm(
            torch.empty(2),
            torch.empty(4),
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

    count = register_fp32_reduce_scatter_with_low_precision_transport(
        model,
        transport_dtype=torch.bfloat16,
        reduction_scales={model.fsdp1: 0.25},
    )

    assert count == 1
    assert len(model.fsdp1.comms) == 1
    assert model.fsdp1.comms[0]._transport_dtype == torch.bfloat16
    assert model.fsdp1.comms[0]._reduction_scale == 0.25
    assert model.fsdp1.gradient_divide_factors == [1.0]
    assert model.fsdp1.force_sum_reductions == [True]
    assert model.fsdp2.comms == []
    assert model.fsdp2.gradient_divide_factors == []
    assert model.fsdp2.force_sum_reductions == []


@pytest.mark.parametrize(
    "reduce_scatter_group_size",
    [1, 2],
)
def test_fsdp_gradient_scaling_uses_custom_path_only_when_needed(reduce_scatter_group_size):
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
        transport_reduction_scales=reduction_scales,
    )

    if reduce_scatter_group_size == 1:
        assert module.gradient_divide_factors == [8.0]
        assert reduction_scales == {}
    else:
        assert module.gradient_divide_factors == []
        assert reduction_scales == {module: 0.125}


@pytest.mark.parametrize(
    ("mesh_dim_names", "sizes", "expected"),
    [
        (("dp_shard",), {"dp_shard": 8}, 8),
        (("dp_replicate", "dp_shard"), {"dp_replicate": 2, "dp_shard": 4}, 4),
    ],
)
def test_reduce_scatter_group_size_uses_last_mesh_dimension(mesh_dim_names, sizes, expected):
    class FakeMeshDimension:
        def __init__(self, size):
            self._size = size

        def size(self):
            return self._size

    class FakeMesh:
        def __init__(self):
            self.mesh_dim_names = mesh_dim_names

        def __getitem__(self, name):
            return FakeMeshDimension(sizes[name])

    assert _reduce_scatter_group_size(FakeMesh()) == expected


def test_reduce_scatter_transport_config_and_native_fallback():
    with pytest.raises(ValueError, match="fsdp_mode='fsdp2'"):
        FSDPConfig(
            fsdp_mode="ddp",
            reduce_scatter_transport_dtype="bfloat16",
        )
    for reduce_dtype, transport_dtype in (("bfloat16", "float16"), ("float16", "bfloat16")):
        with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
            FSDPConfig(
                mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
                reduce_scatter_transport_dtype=transport_dtype,
            )
    with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
        FSDPConfig(
            mixed_precision=MixedPrecisionConfig(enable=False, reduce_dtype="float32"),
            reduce_scatter_transport_dtype="bfloat16",
        )
    with pytest.raises(ValueError, match="must be one of"):
        FSDPConfig(reduce_scatter_transport_dtype="float8")

    for transport_dtype in ("bfloat16", "float16"):
        config = FSDPConfig(
            reduce_scatter_transport_dtype=transport_dtype,
        )
        assert config.reduce_scatter_transport_dtype == transport_dtype

    for reduce_dtype in ("bfloat16", "float16", "float32"):
        config = FSDPConfig(
            mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
            reduce_scatter_transport_dtype=reduce_dtype,
        )
        assert config.reduce_scatter_transport_dtype == reduce_dtype

    native_config = FSDPConfig(
        fsdp_mode="ddp",
        mixed_precision=MixedPrecisionConfig(enable=False, reduce_dtype="float32"),
        reduce_scatter_transport_dtype="float32",
    )
    assert native_config.reduce_scatter_transport_dtype == native_config.mixed_precision.reduce_dtype


@pytest.mark.parametrize("dtype", ["bfloat16", "float16", "float32"])
def test_matching_transport_and_reduce_dtype_uses_native_path(dtype):
    assert not _uses_low_precision_reduce_scatter_transport(None, dtype)
    assert not _uses_low_precision_reduce_scatter_transport(dtype, dtype)
    assert _uses_low_precision_reduce_scatter_transport("bfloat16", "float32")


@pytest.mark.parametrize(
    ("reduce_dtype", "transport_dtype"),
    [("bfloat16", "float16"), ("float16", "bfloat16")],
)
def test_parallelize_rejects_cross_16bit_transport_before_backend_check(
    monkeypatch,
    reduce_dtype,
    transport_dtype,
):
    monkeypatch.setattr(torch_parallelize, "get_parallel_state", object)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")

    with pytest.raises(ValueError, match="supports only.*reduce_dtype='float32'"):
        torch_parallelize.parallelize_model_fsdp2(
            nn.Linear(2, 2),
            mixed_precision=MixedPrecisionConfig(reduce_dtype=reduce_dtype),
            reduce_scatter_transport_dtype=transport_dtype,
        )


def test_matching_transport_dtype_does_not_register_custom_collective(monkeypatch):
    class ParallelState:
        any_extra_parallel_enabled = False
        extra_parallel_names = []
        fsdp_mesh = None

    monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: ParallelState())
    monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")
    monkeypatch.setattr(torch_parallelize, "_materialize_and_load_weights", lambda *args, **kwargs: None)

    def fail_registration(*args, **kwargs):
        raise AssertionError("matching dtypes must not register a custom ReduceScatter")

    monkeypatch.setattr(
        torch_parallelize,
        "register_fp32_reduce_scatter_with_low_precision_transport",
        fail_registration,
    )

    model = nn.Linear(2, 2)
    result = torch_parallelize.parallelize_model_fsdp2(
        model,
        mixed_precision=MixedPrecisionConfig(enable=False, reduce_dtype="float32"),
        reduce_scatter_transport_dtype="float32",
        init_device="meta",
    )

    assert result is model


def _run_reduce_scatter_nccl() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    shard_numel = 4096
    values = torch.arange(world_size * shard_numel, device=device, dtype=torch.float32)
    for dtype in (torch.bfloat16, torch.float16):
        input_tensor = (values.remainder(31) + rank * 0.25).to(dtype).float()
        for scale in (1.0, 1.0 / world_size, 1.0 / 3.0):
            reference = torch.empty(shard_numel, device=device, dtype=torch.float32)
            dist.reduce_scatter_tensor(reference, input_tensor, group=dist.group.WORLD, op=dist.ReduceOp.SUM)
            expected = reference.mul(scale)

            output = torch.empty(shard_numel, device=device, dtype=torch.float32)
            comm = FP32ReduceScatterWithLowPrecisionTransport(dtype, reduction_scale=scale)
            comm(output, input_tensor, dist.group.WORLD, dist.ReduceOp.SUM)
            torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_fsdp2_optimizer_step() -> None:
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", rank)))
    mesh = init_device_mesh("cuda", (world_size,), mesh_dim_names=("dp_shard",))

    class RecordingReduceScatter(FP32ReduceScatterWithLowPrecisionTransport):
        def __init__(self, transport_dtype, reduction_scale):
            super().__init__(transport_dtype, reduction_scale)
            self.calls = []

        def __call__(self, output_tensor, input_tensor, group, op, async_op=False):
            self.calls.append((input_tensor.dtype, op))
            return super().__call__(output_tensor, input_tensor, group, op, async_op)

    for transport_dtype in (torch.bfloat16, torch.float16):
        for gradient_divide_factor in (float(world_size), 3.0):
            torch.manual_seed(1234)
            baseline = nn.Linear(32, 16, bias=False, device=device)
            custom = copy.deepcopy(baseline)
            baseline_policy = MixedPrecisionPolicy(param_dtype=transport_dtype, reduce_dtype=torch.float32)
            custom_policy = MixedPrecisionPolicy(param_dtype=transport_dtype, reduce_dtype=torch.float32)
            fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
            fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
            baseline.set_gradient_divide_factor(gradient_divide_factor)
            custom.set_gradient_divide_factor(1.0)
            custom.set_force_sum_reduction_for_comms(True)
            comm = RecordingReduceScatter(transport_dtype, 1.0 / gradient_divide_factor)
            custom.set_custom_reduce_scatter(comm)

            torch.manual_seed(9000 + rank)
            inputs = torch.randn(8, 32, device=device, dtype=transport_dtype)
            baseline(inputs).float().square().sum().backward()
            custom(inputs).float().square().sum().backward()

            baseline_grad = baseline.weight.grad.to_local()
            custom_grad = custom.weight.grad.to_local()
            assert baseline_grad.dtype == torch.float32
            assert custom_grad.dtype == torch.float32
            assert comm.calls
            assert all(dtype == torch.float32 for dtype, _ in comm.calls)
            assert all(op == dist.ReduceOp.SUM for _, op in comm.calls)
            torch.testing.assert_close(custom_grad, baseline_grad, rtol=5e-6, atol=5e-6)

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
    custom_policy = MixedPrecisionPolicy(param_dtype=torch.float16, reduce_dtype=torch.float32)
    fully_shard(baseline, mesh=mesh, mp_policy=baseline_policy)
    fully_shard(custom, mesh=mesh, mp_policy=custom_policy)
    custom.set_gradient_divide_factor(1.0)
    custom.set_force_sum_reduction_for_comms(True)
    comm = RecordingReduceScatter(torch.float16, 1.0 / world_size)
    custom.set_custom_reduce_scatter(comm)

    inputs = torch.full((1, 4), 65504.0, device=device, dtype=torch.float16)
    baseline(inputs).sum().backward()
    custom(inputs).sum().backward()
    baseline_grad = baseline.weight.grad.to_local()
    custom_grad = custom.weight.grad.to_local()
    assert torch.isfinite(custom_grad).all()
    torch.testing.assert_close(custom_grad, baseline_grad, rtol=0, atol=0)

    hsdp_mesh = init_device_mesh(
        "cuda",
        (2, 2),
        mesh_dim_names=("dp_replicate", "dp_shard"),
    )
    torch.manual_seed(5678)
    baseline = nn.Linear(32, 16, bias=False, device=device)
    custom = copy.deepcopy(baseline)
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    fully_shard(baseline, mesh=hsdp_mesh, mp_policy=policy)
    fully_shard(custom, mesh=hsdp_mesh, mp_policy=policy)
    custom.set_gradient_divide_factor(1.0)
    custom.set_force_sum_reduction_for_comms(True)
    comm = RecordingReduceScatter(torch.bfloat16, 1.0 / world_size)
    custom.set_custom_reduce_scatter(comm)

    torch.manual_seed(12000 + rank)
    inputs = torch.randn(8, 32, device=device, dtype=torch.bfloat16)
    baseline(inputs).float().square().sum().backward()
    all_reduce_dtypes = []
    original_all_reduce = dist.all_reduce

    def recording_all_reduce(tensor, *args, **kwargs):
        all_reduce_dtypes.append(tensor.dtype)
        return original_all_reduce(tensor, *args, **kwargs)

    dist.all_reduce = recording_all_reduce
    try:
        custom(inputs).float().square().sum().backward()
    finally:
        dist.all_reduce = original_all_reduce

    assert all_reduce_dtypes == [torch.float32]
    torch.testing.assert_close(custom.weight.grad.to_local(), baseline.weight.grad.to_local(), rtol=5e-6, atol=5e-6)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_nccl():
    from ..tools.launch_utils import torchrun

    torchrun(_run_reduce_scatter_nccl, world_size=4)


@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="requires four CUDA devices")
def test_reduce_scatter_fp32_accumulation_fsdp2_step():
    from ..tools.launch_utils import torchrun

    torchrun(_run_fsdp2_optimizer_step, world_size=4)

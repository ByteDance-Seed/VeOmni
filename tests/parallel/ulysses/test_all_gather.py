from datetime import timedelta
from itertools import product
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss
from veomni.distributed.sequence_parallel.ulysses import _all_gather, _Gather
from veomni.utils.device import (
    IS_CUDA_AVAILABLE,
    IS_NPU_AVAILABLE,
    get_device_type,
    get_dist_comm_backend,
    get_torch_device,
)
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed


_HAS_ACCELERATOR_BACKEND = (
    get_device_type() != "cpu" and dist.is_available() and dist.is_backend_available(get_dist_comm_backend())
)
_DEVICE_COUNT = get_torch_device().device_count() if _HAS_ACCELERATOR_BACKEND else 0
_GATHER_BACKWARD_BACKENDS = [
    pytest.param("gloo", marks=pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")),
    pytest.param(
        dist.Backend.NCCL,
        marks=pytest.mark.skipif(
            not IS_CUDA_AVAILABLE or not dist.is_nccl_available() or get_torch_device().device_count() < 2,
            reason="Two CUDA devices and NCCL required",
        ),
    ),
]
_SHARED_GRADIENT_BACKENDS = [
    *_GATHER_BACKWARD_BACKENDS,
    pytest.param(
        "hccl",
        marks=pytest.mark.skipif(
            not IS_NPU_AVAILABLE
            or get_device_type() != "npu"
            or not get_torch_device().is_available()
            or not dist.is_available()
            or not dist.is_backend_available("hccl")
            or get_torch_device().device_count() < 2,
            reason="Two NPU devices and HCCL required",
        ),
    ),
]

if _HAS_ACCELERATOR_BACKEND:
    from .utils import SequenceParallelTest
else:
    # Keep CPU/Gloo regressions collectible without the accelerator-only harness.
    from unittest import TestCase as SequenceParallelTest


class AllToAllCommTest(SequenceParallelTest):
    @staticmethod
    def _get_even_input_data():
        S = 20
        H = 8
        input_ = torch.randn(S, H).to(get_device_type())
        dist.broadcast(input_, src=0)
        return input_

    @staticmethod
    def _get_uneven_input_data():
        B = 2
        S = 20
        H = 80
        input_ = torch.randn(B, S, H).to(get_device_type())
        dist.broadcast(input_, src=0)
        dim_size_list = list(range(1, dist.get_world_size()))
        dim_size_list.append(S - sum(dim_size_list))
        return input_, dim_size_list

    @pytest.mark.skipif(_DEVICE_COUNT < 4, reason="device_count should be >= 4")
    def test_even_input(self):
        group = self._get_process_group()
        input_ = self._get_even_input_data()
        test_input = slice_input_tensor(input_.clone(), 0, False, group=group)
        test_input_final = gather_outputs(test_input, gather_dim=0, group=group)

        torch.allclose(input_, test_input_final)

    @pytest.mark.skipif(_DEVICE_COUNT < 4, reason="device_count should be >= 4")
    def test_uneven_input(self):
        group = self._get_process_group()
        input_, dim_size_list = self._get_uneven_input_data()
        test_input = input_.clone().split(dim_size_list, dim=1)[dist.get_rank()].contiguous()
        test_input_final = gather_outputs(test_input, gather_dim=1, group=group)

        torch.allclose(input_, test_input_final)

    @pytest.mark.skipif(_DEVICE_COUNT < 2, reason="device_count should be >= 2")
    def test_all_gather_shapes_stay_on_host(self):
        group = self._get_process_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        local = torch.full((rank + 1, 3), float(rank), device=get_device_type())

        tensor_list, size_list = _all_gather(local, group=group)

        # Plain ints, not device tensors: reading a shape back one dimension at a time
        # syncs the device on every gather, and every layer gathers.
        assert size_list == [[i + 1, 3] for i in range(world_size)]
        for i, tensor in enumerate(tensor_list):
            assert torch.equal(tensor, torch.full_like(tensor, float(i)))

    @staticmethod
    def _run_forward(x):
        return x * (3.1 / 1.7) + 0.1 * (x / 2.3).pow(2)

    @staticmethod
    def _run_loss_grad_sp(group, shard_value):
        local_x = torch.tensor([[float(shard_value)]], device=get_device_type(), requires_grad=True)
        local_y = gather_outputs(local_x, gather_dim=1, scale_grad=False, group=group)
        local_y = local_y.flip(dims=(1,))
        local_y = slice_input_tensor(local_y, dim=1, group=group)
        local_out = AllToAllCommTest._run_forward(local_y)
        local_loss = local_out.mean()
        num_valid_tokens = torch.tensor(1.0, dtype=local_loss.dtype, device=local_loss.device)
        reduced_loss = reduce_sequence_parallel_loss(local_loss, num_valid_tokens)
        reduced_loss.backward()
        return reduced_loss.item(), local_x.grad.item()

    @staticmethod
    def _run_loss_grad_ref(shard_values, rank):
        global_x = torch.tensor([[float(v) for v in shard_values]], dtype=torch.float32, requires_grad=True)
        global_y = global_x.flip(dims=(1,))
        global_out = AllToAllCommTest._run_forward(global_y)
        global_loss = global_out.mean()
        global_loss.backward()
        return global_loss.item(), global_x.grad[0, rank].item()

    @pytest.mark.skipif(_DEVICE_COUNT < 2, reason="device_count should be >= 2")
    def test_grad_aligned(self):
        group = self._get_process_group()
        rank = dist.get_rank(group)
        world_size = dist.get_world_size(group)
        shard_values = torch.rand(world_size).tolist()
        shard_value = shard_values[rank]

        loss_ref, grad_ref = self._run_loss_grad_ref(shard_values, rank)
        loss_sp, grad_sp = self._run_loss_grad_sp(group, shard_value)

        torch.testing.assert_close(loss_ref, loss_sp, rtol=1e-8, atol=1e-8)
        torch.testing.assert_close(grad_ref, grad_sp, rtol=1e-8, atol=1e-8)


def _check_gather_backward(rank, init_method, backend):
    device = "cpu"
    if backend in (dist.Backend.NCCL, "hccl"):
        get_torch_device().set_device(rank)
        device = get_device_type()
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        for layout, sum_grad, scale_grad in product(
            ("contiguous", "transposed", "narrowed", "expanded"), (False, True), (False, True)
        ):
            case = f"{backend=}, {rank=}, {layout=}, {sum_grad=}, {scale_grad=}"
            local = torch.full((2, 3), float(rank), device=device, requires_grad=True)
            gathered = _Gather.apply(dist.group.WORLD, local, 0, scale_grad, sum_grad)
            values = torch.arange(1, 13, dtype=torch.float32, device=device).reshape(4, 3) + rank * 10
            if layout == "contiguous":
                upstream = values.clone()
            elif layout == "transposed":
                upstream = values.T.contiguous().T
            elif layout == "narrowed":
                backing = torch.zeros((4, 5), device=device)
                backing[:, 1:4] = values
                upstream = backing[:, 1:4]
            else:
                upstream = torch.tensor(float(rank + 1), device=device).expand(4, 3)
            original = upstream.clone()
            # NCCL requires a contiguous reference buffer; upstream keeps its layout.
            expected = original.clone(memory_format=torch.contiguous_format)
            if sum_grad:
                dist.all_reduce(expected, group=dist.group.WORLD)
            if scale_grad:
                expected = expected * 2

            # AddBackward shares its incoming gradient with both branches. A gather
            # that reduces it in place also changes the unrelated bias gradient.
            bias = torch.zeros_like(gathered, requires_grad=True)
            local_grad, bias_grad = torch.autograd.grad(gathered + bias, (local, bias), grad_outputs=upstream)

            torch.testing.assert_close(local_grad, expected[rank * 2 : (rank + 1) * 2], rtol=0, atol=0, msg=case)
            torch.testing.assert_close(bias_grad, original, rtol=0, atol=0, msg=case)
            torch.testing.assert_close(upstream, original, rtol=0, atol=0, msg=case)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", _SHARED_GRADIENT_BACKENDS)
def test_gather_backward_preserves_shared_gradients(tmp_path, backend):
    mp.spawn(
        _check_gather_backward,
        args=((tmp_path / "rendezvous").as_uri(), backend),
        nprocs=2,
    )


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("shape,dim", [((8, 3), 0), ((1, 8, 3), 1), ((1, 8, 3), -2)])
@pytest.mark.parametrize("sizes", [(4, 4), (2, 6)])
def test_gather_backward_scales_only_local_storage(rank, shape, dim, sizes):
    upstream = torch.arange(24, dtype=torch.float32).reshape(shape)
    original = upstream.clone()
    ctx = SimpleNamespace(
        group=None, rank=rank, dim=dim, dim_size_list=sizes, seq_world_size=2, sum_grad=False, grad_scale=True
    )

    result = _Gather.backward(ctx, upstream)[1]

    torch.testing.assert_close(result, original.split(sizes, dim=dim)[rank] * 2, rtol=0, atol=0)
    torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    assert result.untyped_storage().nbytes() == result.numel() * result.element_size()


def _check_gather_backward_edges(rank, init_method, backend):
    device = "cpu"
    if backend == dist.Backend.NCCL:
        get_torch_device().set_device(rank)
        device = get_device_type()
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    # Exercise the actual backward separately: Gloo's forward all_gather does
    # not support uneven input sizes, while the backward still has valid sums.
    cases = [
        ((4, 3), (2, 2), 0, torch.float32, None),
        ((2, 4, 3), (2, 2), 1, torch.float32, None),
        ((2, 3, 4), (2, 2), -1, torch.float32, None),
        ((4, 3), (1, 3), 0, torch.float32, None),
        ((2, 4, 3), (1, 3), 1, torch.float32, None),
        ((2, 3, 4), (1, 3), -1, torch.float32, None),
        ((4, 3), (0, 4), 0, torch.float32, None),
        ((2, 4, 3), (2, 2), 1, torch.float16, None),
        ((2, 4, 3), (2, 2), 1, torch.bfloat16, None),
        ((2, 4, 3), (2, 2), 1, torch.complex64, None),
        ((2, 4, 3), (1, 3), 1, torch.complex64, None),
        ((4, 3), (2, 2), 0, torch.complex64, "conjugate"),
        ((4, 3), (2, 2), 0, torch.float32, "negative_rank0"),
        ((4, 3), (1, 3), 0, torch.float32, "negative_rank0"),
        ((4, 3), (2, 2), 0, torch.float32, "negative_rank1"),
        ((4, 3), (1, 3), 0, torch.float32, "negative_rank1"),
        ((4, 3), (2, 2), 0, torch.float16, "overflow"),
        ((4, 0), (2, 2), 0, torch.float32, None),
        ((4, 0), (1, 3), 0, torch.float32, None),
    ]
    try:
        for case, layout, summed, scaled in product(
            cases, ("contiguous", "transposed", "narrowed", "expanded"), (False, True), (False, True)
        ):
            shape, sizes, dim, dtype, special = case
            upstream = torch.arange(1, 1 + torch.Size(shape).numel(), dtype=torch.float32, device=device).reshape(
                shape
            )
            upstream = (upstream + 10 * rank).to(dtype)
            if dtype.is_complex:
                upstream = upstream + 1j * (upstream * 2 + rank)
            if layout == "transposed":
                upstream = upstream.transpose(0, -1).contiguous().transpose(0, -1)
            elif layout == "narrowed":
                backing_shape = list(shape)
                backing_shape[-1] += 2
                backing = torch.zeros(backing_shape, dtype=dtype, device=device)
                backing[..., 1:-1] = upstream
                upstream = backing[..., 1:-1]
            elif layout == "expanded":
                upstream = torch.tensor(float(rank + 1), dtype=dtype, device=device).expand(shape)
            if special == "conjugate":
                upstream = upstream.conj()
            elif special == f"negative_rank{rank}":
                upstream = torch._neg_view(upstream)
            elif special == "overflow":
                upstream = torch.full_like(upstream, 40000 if rank == 0 else -40000)
            original = upstream.clone()
            expected = original.clone(memory_format=torch.contiguous_format)
            # Multiplication and sum are not interchangeable in low precision.
            if scaled:
                expected.mul_(2)
            if summed:
                dist.all_reduce(expected)
            ctx = SimpleNamespace(
                group=dist.group.WORLD,
                rank=rank,
                dim=dim,
                dim_size_list=sizes,
                seq_world_size=2,
                sum_grad=summed,
                grad_scale=scaled,
            )
            result = _Gather.backward(ctx, upstream)[1]
            reference_local = expected.split(sizes, dim=dim)[rank].contiguous()
            torch.testing.assert_close(result, reference_local, rtol=0, atol=0, equal_nan=True)
            torch.testing.assert_close(upstream, original, rtol=0, atol=0, equal_nan=True)
            if backend == dist.Backend.NCCL and summed and not dtype.is_complex and all(sizes) and upstream.numel():
                assert result.untyped_storage().nbytes() <= reference_local.untyped_storage().nbytes()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("backend", _GATHER_BACKWARD_BACKENDS)
def test_gather_backward_edge_cases(tmp_path, backend):
    mp.spawn(_check_gather_backward_edges, args=((tmp_path / "rendezvous").as_uri(), backend), nprocs=2)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()

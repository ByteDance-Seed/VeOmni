from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.distributed.sequence_parallel.loss import reduce_sequence_parallel_loss
from veomni.distributed.sequence_parallel.ulysses import _all_gather, _Gather
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type, get_dist_comm_backend, get_torch_device
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed


_HAS_ACCELERATOR_BACKEND = (
    get_device_type() != "cpu" and dist.is_available() and dist.is_backend_available(get_dist_comm_backend())
)
_DEVICE_COUNT = get_torch_device().device_count() if _HAS_ACCELERATOR_BACKEND else 0

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


def _check_gather_backward(rank, init_method, backend, layout, sum_grad, scale_grad):
    device = "cpu"
    if backend == "nccl":
        get_torch_device().set_device(rank)
        device = get_device_type()
    dist.init_process_group(backend, init_method=init_method, rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
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

        torch.testing.assert_close(local_grad, expected[rank * 2 : (rank + 1) * 2], rtol=0, atol=0)
        torch.testing.assert_close(bias_grad, original, rtol=0, atol=0)
        torch.testing.assert_close(upstream, original, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "backend",
    [
        pytest.param("gloo", marks=pytest.mark.skipif(not dist.is_gloo_available(), reason="Gloo required")),
        pytest.param(
            "nccl",
            marks=pytest.mark.skipif(
                not IS_CUDA_AVAILABLE or not dist.is_nccl_available() or get_torch_device().device_count() < 2,
                reason="Two CUDA devices and NCCL required",
            ),
        ),
    ],
)
@pytest.mark.parametrize("layout", ["contiguous", "transposed", "narrowed", "expanded"])
@pytest.mark.parametrize("sum_grad", [False, True])
@pytest.mark.parametrize("scale_grad", [False, True])
def test_gather_backward_preserves_shared_gradients(tmp_path, backend, layout, sum_grad, scale_grad):
    mp.spawn(
        _check_gather_backward,
        args=((tmp_path / "rendezvous").as_uri(), backend, layout, sum_grad, scale_grad),
        nprocs=2,
    )


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()

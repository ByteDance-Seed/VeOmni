import sys

import torch
import torch.distributed as c10d

from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device


if not c10d.is_available() or not c10d.is_backend_available(get_dist_comm_backend()):
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import pytest
import torch.distributed as dist
from torch.testing._internal.common_utils import run_tests

from veomni.distributed.sequence_parallel.data import gather_outputs, slice_input_tensor
from veomni.utils.helper import enable_high_precision_for_bf16, set_seed

from .utils import (
    SequenceParallelTest,
)


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

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_even_input(self):
        group = self._get_process_group()
        input_ = self._get_even_input_data()
        test_input = slice_input_tensor(input_.clone(), 0, False, group=group)
        test_input_final = gather_outputs(test_input, gather_dim=0, scale_grad=False, group=group)

        torch.allclose(input_, test_input_final)

    @pytest.mark.skipif(get_torch_device().device_count() < 4, reason="device_count should be >= 4")
    def test_uneven_input(self):
        group = self._get_process_group()
        input_, dim_size_list = self._get_uneven_input_data()
        test_input = input_.clone().split(dim_size_list, dim=1)[dist.get_rank()].contiguous()
        test_input_final = gather_outputs(test_input, gather_dim=1, scale_grad=False, group=group)

        torch.allclose(input_, test_input_final)


if __name__ == "__main__":
    assert not get_torch_device()._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    set_seed(seed=0, full_determinism=True)
    enable_high_precision_for_bf16()
    run_tests()

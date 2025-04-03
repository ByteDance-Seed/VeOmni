import logging
import os
import sys

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_nccl_available():
    logging.error("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed as dist
from torch.testing._internal.common_distributed import MultiProcessTestCase

from veomni.distributed.sequence_parallel import set_ulysses_sequence_parallel_group


def sync_tensor(variable, dim=1):
    def all_gather(tensor, dim):
        if dim != 0:
            tensor_t = tensor.transpose(0, dim).contiguous()
        else:
            tensor_t = tensor.contiguous()
        dim_size = list(tensor_t.size())
        dim_size[0] = dim_size[0] * dist.get_world_size()
        output = torch.empty(dim_size, dtype=tensor.dtype, device=torch.cuda.current_device())

        dist.all_gather_into_tensor(output, tensor_t.contiguous())
        if dim != 0:
            return output.transpose(0, dim).contiguous()
        else:
            return output

    output = all_gather(variable, dim)
    return output


class CommonDistributedDataParallelTest:
    def tearDown(self):
        # DistributedDataParallel test doesn't seem to call FileStore destructor
        # Use this hack to remove files for that test
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    def _get_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        raise NotImplementedError("To be implemented by child class")


class SequenceParallelTest(CommonDistributedDataParallelTest, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        torch.cuda.set_device(self.rank)
        c10d.init_process_group("nccl", store=store, rank=self.rank, world_size=self.world_size)
        group = c10d.distributed_c10d._get_default_group()
        set_ulysses_sequence_parallel_group(group)
        self.rank = dist.get_rank(group)
        return group

    @staticmethod
    def _sync_model(state_dict, rank):
        for key, value in state_dict.items():
            dist.broadcast(value, src=0)
            if rank != 0:
                state_dict[key] = value
        return state_dict

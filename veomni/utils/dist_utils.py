# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Callable, List, Literal, Optional, Union

import torch
from torch import distributed as dist

from ..utils.device import get_device_type


if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


def all_gather(tensor: "torch.Tensor", world_size: int) -> "torch.Tensor":
    """
    Gathers the tensor from all ranks and concats them along the first dim.
    """
    output_tensor = torch.empty(world_size * tensor.numel(), dtype=tensor.dtype, device=get_device_type())
    dist.all_gather_into_tensor(output_tensor, tensor)
    return output_tensor.view(-1, *tensor.size()[1:])


def all_reduce(
    data: Union[int, float, List[Union[int, float]], "torch.Tensor"],
    op: Literal["mean", "sum", "max", "min"] = "mean",
    group: Optional["ProcessGroup"] = None,
) -> Union[int, float, List[Union[int, float]]]:
    """
    Performs all reduce in the given process group.
    """
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment is not initialized.")

    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float, device=get_device_type())

    reduce_ops = {
        "mean": dist.ReduceOp.SUM,
        "sum": dist.ReduceOp.SUM,
        "max": dist.ReduceOp.MAX,
        "min": dist.ReduceOp.MIN,
    }
    dist.all_reduce(data, op=reduce_ops[op], group=group)
    if op == "mean":  # ReduceOp.AVG is not supported by the NPU backend
        data /= dist.get_world_size(group=group)

    if data.numel() == 1:
        return data.item()
    else:
        return data.tolist()


def any_rank_failed(failed: bool, group: Optional[Any] = None) -> bool:
    """Whether *any* rank hit an error, so every rank can agree on what to do next.

    MAX rather than SUM: one failure is enough, and SUM would overflow int32 on a
    large enough group. ``group`` defaults to the training group; a gloo group
    reduces on CPU, so the device follows its backend.
    """
    if not dist.is_initialized():
        return failed
    device = "cpu" if group is not None and dist.get_backend(group) == "gloo" else get_device_type()
    flag = torch.tensor([1 if failed else 0], dtype=torch.int32, device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group)
    return bool(flag.item())


def raise_if_any_rank_failed(error: Optional[BaseException], what: str, group: Optional[Any] = None) -> None:
    """Turn one rank's error into an error on every rank.

    Work that is split across ranks -- one rank writes a replicated file, one
    leader per node copies a directory, one rank's async save fails -- starts out
    visible to a single rank. Raising only there leaves the peers to walk into
    the next collective alone, which hangs until the backend times out instead of
    failing the step.

    The reduction inside is itself the synchronization point, and every rank
    reaches it on every path, so it replaces rather than accompanies a barrier.

    ``group`` goes straight to :func:`any_rank_failed`. Pass the group the work
    itself ran on when ranks can reach this far apart: they wait out that skew
    inside the reduction, and on the training backend a long enough wait is what
    the NCCL watchdog aborts the process over.
    """
    try:
        failed = any_rank_failed(error is not None, group=group)
    except BaseException as group_error:
        # The group can be the very thing that broke -- a gloo group whose peers
        # timed out is how ``work`` failed in the first place. Keep this rank's
        # own error as the cause, or the log shows the connection closing and
        # not what closed it.
        if error is not None:
            raise group_error from error
        raise
    if failed:
        raise error or RuntimeError(f"{what} failed on another rank")


@contextmanager
def main_process_first(local_only: bool = True) -> None:
    """
    A context manager for torch distributed environment to do something on the main process firstly.
    """
    if int(os.getenv("WORLD_SIZE", "1")) > 1:
        is_main_process = int(os.getenv("LOCAL_RANK")) == 0 if local_only else int(os.getenv("RANK")) == 0
        try:
            if not is_main_process:
                dist.barrier()
            yield
        finally:
            if is_main_process:
                dist.barrier()
    else:
        yield


def execute_in_order(task: Callable, *, local_only: bool = True, **kwargs) -> Any:
    """
    Executes the task in the order of rank.
    """
    world_size = int(os.getenv("LOCAL_WORLD_SIZE", "1") if local_only else os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("LOCAL_RANK", "1") if local_only else os.getenv("RANK", "1"))
    if world_size > 1:
        dist.barrier()
        for i in range(world_size):
            if rank == i:
                result = task(**kwargs)
                dist.barrier()
            else:
                dist.barrier()

        return result
    else:
        return task(**kwargs)

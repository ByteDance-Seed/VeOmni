import torch
from torch.distributed.distributed_c10d import ReduceOp


def hccl_premul_sum_wrapper(op, output_name):
    def wrapper(*args, **kwargs):
        factor = None
        if kwargs["op"] == ReduceOp.PREMUL_SUM:
            factor = kwargs["op"].__getstate__()[1]
            kwargs["op"] = ReduceOp.SUM
        handle = op(args, kwargs)
        if handle is not None:
            handle.wait()
        if factor is not None:
            output = args[0] if len(args) > 0 else kwargs[output_name]
            output.data.mul_(factor)
        return handle

    return wrapper


def apply_hccl_premul_sum_patch():
    torch.distributed.all_reduce = hccl_premul_sum_wrapper(torch.distributed.all_reduce, "tensor")
    torch.distributed.reduce_scatter = hccl_premul_sum_wrapper(torch.distributed.reduce_scatter, "output")
    torch.distributed.reduce_scatter_tensor = hccl_premul_sum_wrapper(
        torch.distributed.reduce_scatter_tensor, "output"
    )

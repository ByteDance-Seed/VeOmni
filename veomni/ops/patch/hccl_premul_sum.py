import torch
from torch.distributed.distributed_c10d import ReduceOp


def hccl_premul_sum_wrapper(op, output_name):
    """
    A wrapper for distributed operations to handle ReduceOp.PREMUL_SUM which is not supported in Huawei HCCL.
    This wrapper intercepts operations using ReduceOp.PREMUL_SUM and converts them into equivalent
    ReduceOp.SUM operations followed by scalar multiplication.
    """

    def wrapper(*args, **kwargs):
        # Note:Although the sequence of operations(ReduceOp.SUM followed by multiplication) may differ from semantics,
        # we have verified that there is no problem with the performance and accuracy of this sequence.
        factor = None
        if kwargs["op"] == ReduceOp.PREMUL_SUM:
            factor = kwargs["op"].__getstate__()[1]
            kwargs["op"] = ReduceOp.SUM
        handle = op(*args, **kwargs)
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

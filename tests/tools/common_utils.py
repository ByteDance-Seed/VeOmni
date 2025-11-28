import torch.distributed as dist

from veomni.utils.device import get_device_type, get_torch_device


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def print_device_mem_info(prefix_info=""):
    if get_device_type() == "cuda" or get_device_type() == "npu":
        current_memory_allocated = get_torch_device().memory_allocated() / (1024**2)
        memory_reserved = get_torch_device().memory_reserved() / (1024**2)
        max_memory_allocated = get_torch_device().max_memory_allocated() / (1024**2)
    else:
        current_memory_allocated = 0.0
        memory_reserved = 0.0
        max_memory_allocated = 0.0

    print(
        f"{prefix_info} current_memory:{current_memory_allocated:.2f} MB | memory_reserved:{memory_reserved:.2f} MB | max_memory:{max_memory_allocated:.2f} MB"
    )

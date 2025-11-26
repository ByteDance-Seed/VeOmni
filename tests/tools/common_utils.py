import torch
import torch.distributed as dist


def get_device_type():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_world_size():
    return dist.get_world_size()


def get_rank():
    return dist.get_rank()


def print_device_mem_info(prefix_info=""):
    current_memory_allocated = torch.cuda.memory_allocated() / (1024**2)
    memory_reserved = torch.cuda.memory_reserved() / (1024**2)
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024**2)

    print(
        f"{prefix_info} current_memory:{current_memory_allocated:.2f} MB | memory_reserved:{memory_reserved:.2f} MB | max_memory:{max_memory_allocated:.2f} MB"
    )

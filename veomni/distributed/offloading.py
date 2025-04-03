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


import enum
from contextlib import nullcontext
from typing import Tuple, Union

import torch
from torch.autograd.graph import saved_tensors_hooks


class OffloadPolicy(enum.Enum):
    OFFLOAD = 0
    KEEP_ON_GPU = 1
    IGNORE = 2


class custom_save_on_cpu(saved_tensors_hooks):
    def __init__(self, gpu_limit_in_gb: float = 0, pin_memory: bool = False, min_offload_size: int = 1024) -> None:
        self.cur_gpu_ram_in_mb = 0.0

        def pack_to_cpu(tensor: torch.Tensor) -> Tuple[OffloadPolicy, torch.device, torch.Tensor]:
            tensor_num_bytes = tensor.element_size() * tensor.nelement()
            # heuristic to skip nn.Linear.weight
            if type(tensor.grad_fn).__name__ == "TBackward0" or tensor_num_bytes <= min_offload_size:
                return (OffloadPolicy.IGNORE, tensor.device, tensor)

            if self.cur_gpu_ram_in_mb < gpu_limit_in_gb * 1024:
                self.cur_gpu_ram_in_mb += tensor_num_bytes / 1024 / 1024
                return (OffloadPolicy.KEEP_ON_GPU, tensor.device, tensor)

            if not pin_memory:
                return (OffloadPolicy.OFFLOAD, tensor.device, tensor.cpu())

            packed = torch.empty(
                tensor.size(),
                dtype=tensor.dtype,
                layout=tensor.layout,
                pin_memory=(not tensor.is_sparse),
            )
            packed.copy_(tensor)
            return (OffloadPolicy.OFFLOAD, tensor.device, packed)

        def unpack_from_cpu(packed: Tuple[OffloadPolicy, torch.device, torch.Tensor]) -> torch.Tensor:
            offload_policy, device, tensor = packed

            if offload_policy == OffloadPolicy.IGNORE:
                return tensor
            elif offload_policy == OffloadPolicy.KEEP_ON_GPU:
                tensor_num_bytes = tensor.element_size() * tensor.nelement()
                self.cur_gpu_ram_in_mb -= tensor_num_bytes / 1024 / 1024
                return tensor
            else:
                return tensor.to(device, non_blocking=pin_memory)

        super().__init__(pack_to_cpu, unpack_from_cpu)


def build_activation_offloading_context(
    enable_activation_offload: bool = False,
    enable_gradient_checkpointing: bool = False,
    activation_gpu_limit: float = 0.0,
) -> Tuple[Union["saved_tensors_hooks", "nullcontext"], Union["saved_tensors_hooks", "nullcontext"]]:
    model_fwd_context, model_bwd_context = nullcontext(), nullcontext()
    if enable_activation_offload:
        # pin_memory=False since CachingHostAllocator caches pinned memory aggressively.
        # torch._C._host_emptyCache() can be used after version 2.5.
        if enable_gradient_checkpointing:
            # inter-layer activations are always offloaded when enabling gradient checkpointing to avoid potential thrashing
            model_fwd_context = custom_save_on_cpu(gpu_limit_in_gb=0.0, pin_memory=False)
            model_bwd_context = custom_save_on_cpu(gpu_limit_in_gb=activation_gpu_limit, pin_memory=False)
        else:
            model_fwd_context = custom_save_on_cpu(gpu_limit_in_gb=activation_gpu_limit, pin_memory=False)

    return model_fwd_context, model_bwd_context

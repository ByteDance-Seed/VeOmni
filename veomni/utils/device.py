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

# Following codes are inspired from https://github.com/volcengine/verl/blob/main/verl/utils/device.py

from typing import Any

import torch

from . import logging
from .import_utils import is_torch_npu_available


logger = logging.get_logger(__name__)


IS_CUDA_AVAILABLE = torch.cuda.is_available()
IS_NPU_AVAILABLE = is_torch_npu_available()

if IS_NPU_AVAILABLE:
    torch.npu.config.allow_internal_format = False


def get_device_type() -> str:
    """Get device type based on current machine, currently only support CPU, CUDA, NPU."""
    if IS_CUDA_AVAILABLE:
        device = "cuda"
    elif IS_NPU_AVAILABLE:
        device = "npu"
    else:
        device = "cpu"

    return device


def get_device_name() -> str:
    """Get real device name, e.g. A100, H100"""
    return get_torch_device().get_device_name()


def get_torch_device() -> Any:
    """Get torch attribute based on device type, e.g. torch.cuda or torch.npu"""
    device_name = get_device_type()

    try:
        return getattr(torch, device_name)
    except AttributeError:
        logger.warning(f"Device namespace '{device_name}' not found in torch, try to load 'torch.cuda'.")
        return torch.cuda


def get_device_id() -> int:
    """Get current device id based on device type."""
    return get_torch_device().current_device()


def get_dist_comm_backend() -> str:
    """Return distributed communication backend type based on device type."""
    if IS_CUDA_AVAILABLE:
        return "nccl"
    elif IS_NPU_AVAILABLE:
        return "hccl"
    else:
        raise RuntimeError(f"No available distributed communication backend found on device type {get_device_type()}.")


def synchronize() -> None:
    """Execute torch synchronize operation."""
    get_torch_device().synchronize()


def stream_synchronize() -> None:
    """Execute device stream synchronize operation."""
    if IS_CUDA_AVAILABLE:
        torch.cuda.current_stream().synchronize()
    elif IS_NPU_AVAILABLE:
        torch.npu.current_stream().synchronize()
    else:
        synchronize()


def empty_cache() -> None:
    """Execute torch empty cache operation."""
    get_torch_device().empty_cache()


def set_device(device: torch.types.Device) -> None:
    """Execute set device operation."""
    get_torch_device().set_device(device)


def is_nccl_backend() -> bool:
    """Check if the distributed communication backend is NCCL."""
    return get_dist_comm_backend() == "nccl"


def is_hccl_backend() -> bool:
    """Check if the distributed communication backend is HCCL."""
    return get_dist_comm_backend() == "hccl"

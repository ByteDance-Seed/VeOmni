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

"""Opt-in HCCL options for the existing expert-parallel mesh dimension."""


def validate_ep_hccl_options(buffer_mib: int | None, expansion_mode: str | None) -> None:
    """Reject invalid values before constructing any process groups."""
    if buffer_mib is not None and (type(buffer_mib) is not int or buffer_mib <= 0):
        raise ValueError("npu_ep_hccl_buffer_mib must be a positive integer or None")
    if expansion_mode not in (None, "host", "device"):
        raise ValueError("npu_ep_hccl_expansion_mode must be host, device, or None")


def ep_backend_override(device_type: str, buffer_mib: int | None, expansion_mode: str | None) -> dict:
    """Build public DeviceMesh backend options, leaving other mesh dimensions unchanged."""
    validate_ep_hccl_options(buffer_mib, expansion_mode)
    if buffer_mib is None and expansion_mode is None:
        return {}
    if device_type != "npu":
        raise ValueError("HCCL expert-parallel options require device_type=npu")
    import torch_npu

    options = torch_npu._C._distributed_c10d.ProcessGroupHCCL.Options()
    config = {}
    if buffer_mib is not None:
        config["hccl_buffer_size"] = buffer_mib
    if expansion_mode is not None:
        config["hccl_op_expansion_mode"] = {"host": 1, "device": 2}[expansion_mode]
    options.hccl_config = config
    return {"ep": ("hccl", options)}

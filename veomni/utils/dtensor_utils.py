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

"""DTensor helpers shared by the optimizer and checkpoint code."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.distributed.tensor._utils import compute_global_tensor_info


def rewrap_dtensor_local(
    local: torch.Tensor,
    *,
    mesh: DeviceMesh,
    placements: Sequence,
    shape: Optional[torch.Size] = None,
    stride: Optional[tuple] = None,
) -> DTensor:
    """Wrap ``local`` as a DTensor on ``mesh`` without moving it to the device.

    ``DTensor.from_local`` copies ``local`` to the mesh device just to relabel it;
    this builds the same DTensor but leaves the storage and device untouched, so a
    host-resident local (the swap optimizer's states) stays on the host.

    ``shape``/``stride`` are the global tensor's and are optional: when omitted the
    global metadata is inferred from the local shard, mesh and placements (the same
    arithmetic ``from_local`` uses), which is required when the target mesh differs
    from the source one. Pass them only when the global shape is known and the local
    may be unevenly sharded.
    """
    if local.device.type == mesh.device_type:
        if shape is None or stride is None:
            return DTensor.from_local(local, device_mesh=mesh, placements=list(placements), run_check=False)
        return DTensor.from_local(
            local, device_mesh=mesh, placements=list(placements), run_check=False, shape=shape, stride=stride
        )

    if shape is None or stride is None:
        global_shape, global_stride = compute_global_tensor_info(local, mesh, tuple(placements))
        shape, stride = torch.Size(global_shape), tuple(global_stride)

    spec = DTensorSpec(mesh, tuple(placements), tensor_meta=TensorMeta(shape, stride, local.dtype))
    return DTensor(local.view_as(local), spec, requires_grad=False)


__all__ = ["rewrap_dtensor_local"]

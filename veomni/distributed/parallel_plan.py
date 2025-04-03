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


from dataclasses import dataclass
from typing import Dict, Union

import torch
import torch.nn as nn
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard

from ..utils import logging
from .utils import check_fqn_match, get_module_from_path, set_module_from_path


logger = logging.get_logger(__name__)


@dataclass
class SpecInfo:
    mesh: DeviceMesh
    placement: Union[Shard, Replicate]
    fqn: str


class ParallelPlan:
    def __init__(self, ep_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan
        self.ep_param_suffix = {k.split(".")[-1] for k in ep_plan.keys()}
        self.fsdp_no_shard_module = {".".join(list(ep_plan.keys())[0].split(".")[:-1])}

    def apply(self, model: nn.Module, ep_mesh: DeviceMesh):
        """
        ep_mesh: [replicate, replicate, ... , shard]
        """
        # ep_plan
        fqn2spec_info = {}
        if self.ep_plan:
            ep_size = ep_mesh.size(-1)
            ep_replicate = [Replicate() for _ in range(ep_mesh.ndim)]

            for fqn, param in model.named_parameters():
                for fqn_pattern, shard in self.ep_plan.items():
                    if check_fqn_match(fqn_pattern, fqn):
                        assert param.size(shard.dim) % ep_size == 0
                        ep_placement = ep_replicate[:-1] + [shard]
                        logger.info_rank0(f"EP sharding info: {fqn}, {shard}, {ep_placement}")
                        dtensor = DTensor.from_local(
                            local_tensor=param.data, device_mesh=ep_mesh, placements=ep_replicate
                        )
                        dtensor = dtensor.redistribute(device_mesh=ep_mesh, placements=ep_placement)
                        local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                        local_chunk.spec_info = SpecInfo(mesh=ep_mesh, placement=shard, fqn=fqn)
                        set_module_from_path(model, fqn, local_chunk)
                        fqn2spec_info[fqn] = SpecInfo(mesh=ep_mesh, placement=shard, fqn=fqn)
                        break
                if fqn not in fqn2spec_info:  # not sharded
                    param.spec_info = SpecInfo(mesh=ep_mesh, placement=Replicate(), fqn=fqn)
                    fqn2spec_info[fqn] = SpecInfo(mesh=ep_mesh, placement=Replicate(), fqn=fqn)

        for param in model.parameters():
            assert hasattr(param, "spec_info"), f"Internal Error: {param} is omitted"

        return fqn2spec_info

    def get_fsdp_no_shard_info(self, model: nn.Module):
        if self.fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, param in model.named_modules():
            for no_shard_pattern in self.fsdp_no_shard_module:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, "no module in model match `fsdp_no_shard_module`"

        return fsdp_no_shard_states_fqn_to_module

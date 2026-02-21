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
    placement: Union[Shard, Replicate]
    fqn: str
    ep_fsdp_mesh: DeviceMesh = None
    emb_fsdp_mesh: DeviceMesh = None

    @property
    def ep_mesh(self):
        if self.ep_fsdp_mesh is not None:
            return self.ep_fsdp_mesh["ep"]
        else:
            return None

    @property
    def emb_mesh(self):
        if self.emb_fsdp_mesh is not None:
            return self.emb_fsdp_mesh["emb"]
        else:
            return None

    @property
    def rowshard_mesh(self):
        return self.emb_mesh if self.emb_mesh else self.ep_mesh

    @property
    def rowshard_fsdp_mesh(self):
        return self.emb_fsdp_mesh if self.emb_fsdp_mesh else self.ep_fsdp_mesh


class ParallelPlan:
    def __init__(self, ep_plan: Dict[str, Shard], emb_plan: Dict[str, Shard]):
        self.ep_plan = ep_plan
        self.emb_plan = emb_plan
        self.ep_param_suffix = {k.split(".")[-1] for k in ep_plan.keys()}
        self.emb_param_suffix = {k.split(".")[-1] for k in emb_plan.keys()}
        self.ep_fsdp_no_shard_module = {".".join(list(ep_plan.keys())[0].split(".")[:-1])}
        self.emb_fsdp_no_shard_module = {".".join(list(emb_plan.keys())[0].split(".")[:-1])}

    def apply(self, model: nn.Module, ep_fsdp_mesh: DeviceMesh, emb_fsdp_mesh: DeviceMesh):
        """
        ep_fsdp_mesh: [replicate, replicate, ... , shard]
        """
        ep_mesh = ep_fsdp_mesh["ep"] if ep_fsdp_mesh is not None else None
        emb_mesh = emb_fsdp_mesh["emb"] if emb_fsdp_mesh is not None else None
        # ep_plan
        fqn2spec_info = {}
        if self.ep_plan and ep_mesh is not None:
            ep_size = ep_mesh.size(-1)
            ep_replicate = [Replicate() for _ in range(ep_mesh.ndim)]
            for fqn, param in model.named_parameters():
                for fqn_pattern, shard in self.ep_plan.items():
                    if check_fqn_match(fqn_pattern, fqn):
                        assert param.size(shard.dim) % ep_size == 0
                        ep_placement = ep_replicate[:-1] + [shard]
                        logger.info_rank0(
                            f"EP sharding: slicing param {fqn} along ep_mesh with placement {ep_placement}"
                        )
                        dtensor = DTensor.from_local(
                            local_tensor=param.data, device_mesh=ep_mesh, placements=ep_replicate
                        )
                        dtensor = dtensor.redistribute(device_mesh=ep_mesh, placements=ep_placement)
                        local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                        local_chunk.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                        set_module_from_path(model, fqn, local_chunk)
                        fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=shard, fqn=fqn)
                        break
                if fqn not in fqn2spec_info:  # not sharded
                    param.spec_info = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)
                    fqn2spec_info[fqn] = SpecInfo(ep_fsdp_mesh=ep_fsdp_mesh, placement=Replicate(), fqn=fqn)

        if self.emb_plan and emb_mesh is not None:
            emb_size = emb_mesh.size(-1)
            emb_replicate = [Replicate() for _ in range(emb_mesh.ndim)]
            for fqn, param in model.named_parameters():
                for fqn_pattern, shard in self.emb_plan.items():
                    if check_fqn_match(fqn_pattern, fqn):
                        assert param.size(shard.dim) % emb_size == 0, f"param.size: {param.size()}"
                        emb_placement = emb_replicate[:-1] + [shard]
                        logger.info_rank0(
                            f"EMB sharding: slicing param {fqn} along emb_mesh with placement {emb_placement}"
                        )
                        dtensor = DTensor.from_local(
                            local_tensor=param.data, device_mesh=emb_mesh, placements=emb_replicate
                        )
                        dtensor = dtensor.redistribute(device_mesh=emb_mesh, placements=emb_placement)
                        local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                        local_chunk.spec_info = SpecInfo(emb_fsdp_mesh=emb_fsdp_mesh, placement=shard, fqn=fqn)
                        set_module_from_path(model, fqn, local_chunk)
                        fqn2spec_info[fqn] = SpecInfo(emb_fsdp_mesh=emb_fsdp_mesh, placement=shard, fqn=fqn)
                        break
                if fqn not in fqn2spec_info:  # not sharded
                    param.spec_info = SpecInfo(emb_fsdp_mesh=emb_fsdp_mesh, placement=Replicate(), fqn=fqn)
                    fqn2spec_info[fqn] = SpecInfo(emb_fsdp_mesh=emb_fsdp_mesh, placement=Replicate(), fqn=fqn)

        for fqn, param in model.named_parameters():
            assert hasattr(param, "spec_info"), f"Internal Error: {fqn=} with {param=} is omitted"

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

    def get_ep_fsdp_no_shard_info(self, model: nn.Module):
        if self.ep_fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, param in model.named_modules():
            for no_shard_pattern in self.ep_fsdp_no_shard_module:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, "no module in model match `fsdp_no_shard_module`"

        return fsdp_no_shard_states_fqn_to_module

    def get_emb_fsdp_no_shard_info(self, model: nn.Module):
        if self.emb_fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, param in model.named_modules():
            for no_shard_pattern in self.emb_fsdp_no_shard_module:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, "no module in model match `fsdp_no_shard_module`"

        return fsdp_no_shard_states_fqn_to_module

    def update_prefix(self, prefix: str):
        """
        Update ep_plan when model is wrappered.
        """
        self.ep_plan = {prefix + "." + k: v for k, v in self.ep_plan.items()}
        self.emb_plan = {prefix + "." + k: v for k, v in self.emb_plan.items()}
        self.ep_param_suffix = {k.split(".")[-1] for k in self.ep_plan.keys()}
        self.emb_param_suffix = {k.split(".")[-1] for k in self.emb_plan.keys()}
        self.fsdp_no_shard_module = {prefix + "." + k for k in self.fsdp_no_shard_module}
        self.ep_fsdp_no_shard_module = {".".join(list(self.ep_plan.keys())[0].split(".")[:-1])}
        self.emb_fsdp_no_shard_module = {".".join(list(self.emb_plan.keys())[0].split(".")[:-1])}

    def shard_tensor(self, tensor: "torch.Tensor", full_param_name: str, target_shape: tuple) -> "torch.Tensor":
        """
        Shard tensor for expert parallelism if needed.
        In the future, we may add other tensor slicing in this function to determine TP parameter and its sharding.

        Args:
            tensor: The tensor to potentially shard
            full_param_name: The full parameter name (e.g., "model.layers.0.mlp.experts.gate_proj.weight")
            target_shape: The expected shape of the target parameter

        Returns:
            The original tensor or a sliced version for EP
        """
        shard_group = self._get_shard_parameter_groupname(full_param_name)
        if shard_group:
            return self._slice_shard_tensor(tensor, full_param_name, target_shape, shard_group)
        return tensor

    def _get_shard_parameter_groupname(self, parameter_name: str) -> bool:
        if not self.ep_plan and not self.emb_plan:
            return None
            # Check if this parameter matches any pattern in the EP plan
        for fqn_pattern in self.ep_plan.keys():
            if check_fqn_match(fqn_pattern, parameter_name):
                return "ep"

        for fqn_pattern in self.emb_plan.keys():
            if check_fqn_match(fqn_pattern, parameter_name):
                return "emb"
        return False

    def _slice_shard_tensor(
        self, tensor: "torch.Tensor", parameter_name: str, target_shape: tuple, shard_group: str
    ) -> "torch.Tensor":
        """Slice shard tensor for expert/embed parallelism."""
        try:
            from .parallel_state import get_parallel_state

            parallel_state = get_parallel_state()

            # Check if we need to slice based on tensor vs target shape mismatch
            if len(tensor.shape) >= 1 and len(target_shape) >= 1:
                tensor_experts = tensor.shape[0]
                target_experts = target_shape[0]

                # If tensor has more experts than target, we need to slice
                if tensor_experts > target_experts and tensor_experts % target_experts == 0:
                    ep_size = tensor_experts // target_experts
                    if shard_group == "ep":
                        ep_rank = parallel_state.ep_rank if parallel_state.ep_enabled else 0
                    else:
                        ep_rank = parallel_state.emb_rank if parallel_state.emb_enabled else 0
                    start_idx = ep_rank * target_experts
                    end_idx = start_idx + target_experts

                    sliced_tensor = tensor[start_idx:end_idx]

                    logger.info_rank0(
                        f"Expert parameter {parameter_name}: sliced {tensor.shape} -> {sliced_tensor.shape} "
                        f"for EP rank {ep_rank}/{ep_size}"
                    )

                    return sliced_tensor

            # No slicing needed
            return tensor

        except Exception as e:
            # Fallback: if anything fails, return original tensor
            logger.warning(f"Failed to slice expert tensor {parameter_name}: {e}")
            return tensor

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
    para_name: str  # name of the ExtraParallel this fqn belongs to
    placement: Union[Shard, Replicate]
    fqn: str
    para_fsdp_mesh: DeviceMesh

    @property
    def para_mesh(self):
        if self.para_fsdp_mesh is not None:
            return self.para_fsdp_mesh[self.para_name]
        else:
            return None


class ParallelPlan:
    def __init__(self, extra_parallel_plan: Dict[str, Dict[str, Shard]]):
        self.extra_parallel_plan = extra_parallel_plan
        self.extra_parallel_fsdp_no_shard_module = {
            para_name: {".".join(list(plan.keys())[0].split(".")[:-1])}
            for para_name, plan in self.extra_parallel_plan.items()
        }

    def apply(self, model: nn.Module, extra_parallel_fsdp_device_mesh: Dict[str, DeviceMesh]):
        """
        xxx_fsdp_mesh: [replicate, replicate, ... , shard]
        """
        extra_parallel_mesh = {
            para: para_fsdp_mesh[para] if para_fsdp_mesh is not None else None
            for para, para_fsdp_mesh in extra_parallel_fsdp_device_mesh.items()
        }

        fqn2spec_info = {}
        for para, para_plan in self.extra_parallel_plan.items():
            para_mesh = extra_parallel_mesh[para]
            para_fsdp_mesh = extra_parallel_fsdp_device_mesh[para]
            if para_plan and para_mesh is not None:
                para_size = para_mesh.size(-1)
                para_replicate = [Replicate() for _ in range(para_mesh.ndim)]
                for fqn, param in model.named_parameters():
                    for fqn_pattern, shard in para_plan.items():
                        if check_fqn_match(fqn_pattern, fqn):
                            assert param.size(shard.dim) % para_size == 0
                            para_placement = para_replicate[:-1] + [shard]
                            logger.info_rank0(
                                f"{para} sharding: slicing param {fqn} along {para}_mesh with placement {para_placement}"
                            )
                            dtensor = DTensor.from_local(
                                local_tensor=param.data, device_mesh=para_mesh, placements=para_replicate
                            )
                            dtensor = dtensor.redistribute(device_mesh=para_mesh, placements=para_placement)
                            local_chunk = torch.nn.Parameter(dtensor.to_local(), requires_grad=param.requires_grad)
                            local_chunk.spec_info = SpecInfo(
                                para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=shard, fqn=fqn
                            )
                            set_module_from_path(model, fqn, local_chunk)
                            fqn2spec_info[fqn] = SpecInfo(
                                para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=shard, fqn=fqn
                            )
                            break
                    if fqn not in fqn2spec_info:  # not sharded
                        param.spec_info = SpecInfo(
                            para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=Replicate(), fqn=fqn
                        )
                        fqn2spec_info[fqn] = SpecInfo(
                            para_name=para, para_fsdp_mesh=para_fsdp_mesh, placement=Replicate(), fqn=fqn
                        )

        for fqn, param in model.named_parameters():
            assert hasattr(param, "spec_info"), f"Internal Error: {fqn=} with {param=} is omitted"

        return fqn2spec_info

    def get_fsdp_no_shard_info(self, model: nn.Module):
        if self.extra_parallel_fsdp_no_shard_module is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        fsdp_no_shard_states_fqn_to_para = {}
        for fqn, _param in model.named_modules():
            for para, no_shard_patterns in self.extra_parallel_fsdp_no_shard_module.items():
                for no_shard_pattern in no_shard_patterns:
                    if check_fqn_match(no_shard_pattern, fqn):
                        fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
                        fsdp_no_shard_states_fqn_to_para[fqn] = para
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, (
            "no module in model match `extra_parallel_fsdp_no_shard_module`"
        )

        return fsdp_no_shard_states_fqn_to_module, fsdp_no_shard_states_fqn_to_para

    def get_extra_parallel_fsdp_no_shard_info(self, model: nn.Module, para_name: str):
        if self.extra_parallel_fsdp_no_shard_module[para_name] is None:
            return None

        fsdp_no_shard_states_fqn_to_module = {}
        for fqn, _param in model.named_modules():
            for no_shard_pattern in self.extra_parallel_fsdp_no_shard_module[para_name]:
                if check_fqn_match(no_shard_pattern, fqn):
                    fsdp_no_shard_states_fqn_to_module[fqn] = get_module_from_path(model, fqn)
        assert len(fsdp_no_shard_states_fqn_to_module) > 0, (
            "no module in model match `extra_parallel_fsdp_no_shard_module`"
        )

        return fsdp_no_shard_states_fqn_to_module

    def update_prefix(self, prefix: str):
        """
        Update extra_parallel_plan when model is wrappered.
        """
        self.extra_parallel_plan = {
            para_name: {prefix + "." + k: v for k, v in plan.items()}
            for para_name, plan in self.extra_parallel_plan.items()
        }
        self.extra_parallel_fsdp_no_shard_module = {
            para_name: {prefix + "." + no_shard_pattern for no_shard_pattern in para_fsdp_no_shard_module}
            for para_name, para_fsdp_no_shard_module in self.extra_parallel_fsdp_no_shard_module.items()
        }

    def shard_tensor(self, tensor: "torch.Tensor", full_param_name: str, target_shape: tuple) -> "torch.Tensor":
        """
        Shard tensor for one extra_parallel parallelism if needed.
        In the future, we may add other tensor slicing in this function to determine TP parameter and its sharding.

        Args:
            tensor: The tensor to potentially shard
            full_param_name: The full parameter name (e.g., "model.layers.0.mlp.experts.gate_proj.weight")
            target_shape: The expected shape of the target parameter

        Returns:
            The original tensor or a sliced version for one extra_parallel parallelism
        """
        shard_group = self._get_shard_parameter_groupname(full_param_name)
        if shard_group:
            return self._slice_shard_tensor(tensor, full_param_name, target_shape, shard_group)
        return tensor

    def _get_shard_parameter_groupname(self, parameter_name: str) -> bool:
        # note that parameter_name should be full name
        for para_name, para_plan in self.extra_parallel_plan.items():
            for fqn_pattern in para_plan.keys():
                if check_fqn_match(fqn_pattern, parameter_name):
                    return para_name
        return None

    def _slice_shard_tensor(
        self, tensor: "torch.Tensor", parameter_name: str, target_shape: tuple, shard_group: str
    ) -> "torch.Tensor":
        """Slice shard tensor for extra_parallel parallelism."""
        try:
            from .parallel_state import get_parallel_state

            parallel_state = get_parallel_state()

            # Check if we need to slice based on tensor vs target shape mismatch
            if len(tensor.shape) >= 1 and len(target_shape) >= 1:
                # If tensor has more feature than target, we need to slice
                if tensor.shape[0] > target_shape[0] and tensor.shape[0] % target_shape[0] == 0:
                    para_size = tensor.shape[0] // target_shape[0]
                    para_rank = (
                        parallel_state.extra_parallel_rank(shard_group)
                        if parallel_state.extra_parallel_enabled(shard_group)
                        else 0
                    )

                    start_idx = para_rank * target_shape[0]
                    end_idx = start_idx + target_shape[0]

                    sliced_tensor = tensor[start_idx:end_idx]

                    logger.info_rank0(
                        f"{shard_group} parameter {parameter_name}: sliced {tensor.shape} -> {sliced_tensor.shape} "
                        f"for {shard_group} rank {para_rank}/{para_size}"
                    )

                    return sliced_tensor

            # No slicing needed
            return tensor

        except Exception as e:
            # Fallback: if anything fails, return original tensor
            logger.warning(f"Failed to slice extra_parallel tensor {parameter_name}: {e}")
            return tensor

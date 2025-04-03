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


import copy
from functools import partial
from typing import Any, Dict, List, Tuple, Union

import torch
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._fsdp_extensions import FSDPExtensions
from torch.distributed.fsdp._optim_utils import (
    FSDPParamInfo,
    _OptimStateKey,
    _unflatten_optim_state,
    sorted_items,
)

from ..distributed.parallel_plan import SpecInfo
from ..utils import logging


logger = logging.get_logger(__name__)

OPTIM_STATE_NO_SHARD_KEY = ["step"]
orig_optim_state_dict = FSDP.optim_state_dict
orig_optim_state_dict_to_load = FSDP.optim_state_dict_to_load


def _shard_tensor(orgin_tensor: torch.Tensor, device_mesh: DeviceMesh, shard: Shard = Shard(0)):
    """
    Shard Tensor to DTensor.

    args:
        orgin_tensor (torch.Tensor): The orgin tensor.
        device_mesh (DeviceMesh): The ep device mesh.
        shard (Shard): The shard info, default Shard(0).

    """
    assert isinstance(
        orgin_tensor, torch.Tensor
    ), f"Only support torch.Tensor, got {type(orgin_tensor)}, for DTensor, use _shard_tensor instead."

    dtensor = DTensor.from_local(orgin_tensor, device_mesh=device_mesh, placements=[shard])

    return dtensor


def _shard_dtensor(orgin_dtensor: DTensor, device_mesh: DeviceMesh, shard: Shard = Shard(0)):
    """
    Convert DTensor to local Tensor

    args:
        orgin_dtensor (torch.Tensor): The orgin tensor.
        device_mesh (DeviceMesh): The ep device mesh.
        shard (Shard): The shard info, default Shard(0).

    """
    assert isinstance(
        orgin_dtensor, DTensor
    ), f"Only support DTensor, got {type(orgin_dtensor)}, for torch.Tensor, use _shard_dtensor instead."

    local_tensor = orgin_dtensor.to_local()

    return local_tensor


def check_any_unflat_param_names_match(unflat_param_name: str, fqn2spec_info: Dict[str, SpecInfo], prefix: str = None):
    assert isinstance(unflat_param_name, str), f"unflat_param_name must be a str, got {type(unflat_param_name)}"

    if prefix:
        assert unflat_param_name.startswith(
            prefix
        ), f"unflat_param_name {unflat_param_name} must start with prefix {prefix}"
        unflat_param_name = unflat_param_name[len(prefix) :].lstrip(".")

    if unflat_param_name not in fqn2spec_info:
        logger.warning_rank0(f"unflat_param_name {unflat_param_name} not in fqn2spec_info.")
        return False

    if isinstance(fqn2spec_info[unflat_param_name].placement, Shard):
        return True

    return False


def check_all_unflat_param_names_match(unflat_param_names: Tuple[str], fqn2spec_info: Dict[str, SpecInfo]):
    """
    Check
    """
    assert isinstance(
        unflat_param_names, (list, tuple)
    ), f"unflat_param_names must be a list or tuple, got {type(unflat_param_names)}"

    unflat_len = len(unflat_param_names)
    cnt = 0
    for names in unflat_param_names:
        assert (
            names in fqn2spec_info
        ), f"unflat_param_names {unflat_param_names} must be in fqn2spec_info {fqn2spec_info}"
        if isinstance(fqn2spec_info[names].placement, Shard):
            cnt += 1
    assert cnt == 0 or cnt == unflat_len, f"unflat_param_names {unflat_param_names} must be all shard or all not shard"

    return cnt == unflat_len


class CheckpointExtensions(FSDPExtensions):
    def __init__(self, save_hook_mesh: DeviceMesh, fqn2spec_info: Dict[str, SpecInfo]):
        super().__init__()
        self.ep_mesh = save_hook_mesh
        self.fqn2spec_info = fqn2spec_info

    def chunk_dtensor(self, tensor: torch.Tensor, rank: int, device_mesh: DeviceMesh) -> torch.Tensor:
        """Shards a tensor/DTensor to DTensor and returns the local DTensor."""
        # We need to explicitly call .detach() to return a new tensor detached from the current graph.
        tensor = tensor.clone().detach()
        fsdp_size = device_mesh.size(-1)
        dimlens = tuple(tensor.size())
        # by default we use the max-len dimension for sharding
        selected_dim = dimlens.index(max(dimlens))
        for dim, dimlen in enumerate(dimlens):
            if dimlen % fsdp_size == 0:
                selected_dim = dim
                break
        # HSDP placements: [Replicate(), ..., Shard(selected_dim)]
        replicate_placements = [Replicate() for _ in range(device_mesh.ndim)]
        shard_placements = [Replicate() for _ in range(device_mesh.ndim)]
        shard_placements[-1] = Shard(selected_dim)  # type: ignore[call-overload]
        dtensor = DTensor.from_local(tensor, device_mesh, replicate_placements, run_check=False).redistribute(
            placements=shard_placements,
        )

        return dtensor

    def chunk_tensor(self, tensor, rank, world_size, num_devices_per_node, pg, device=None):
        # use default
        raise NotImplementedError("Please init FSDP with device mesh")
        from torch.distributed.fsdp._fsdp_extensions import _ext_chunk_tensor

        return _ext_chunk_tensor(tensor, rank, world_size, num_devices_per_node, pg)

    def pre_flatten_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_flatten_transform

        return _ext_pre_flatten_transform(tensor)

    def pre_load_state_dict_transform(self, tensor):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_pre_load_state_dict_transform

        return _ext_pre_load_state_dict_transform(tensor)

    def post_unflatten_transform(self, tensor, param_extension):
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_post_unflatten_transform

        return _ext_post_unflatten_transform(tensor, param_extension)

    def all_gather_dtensor(self, tensor: DTensor, parent_mesh):
        # this is required during loading checkpoint (model.load_state_dict)
        # use default
        from torch.distributed.fsdp._fsdp_extensions import _ext_all_gather_dtensor

        return _ext_all_gather_dtensor(tensor, None)

    @torch.no_grad()
    def state_dict_post_hook(
        self, module, state_dict, prefix, local_metadata, fqn2spec_info: Dict[str, SpecInfo] = None
    ):
        """
        Post state dict when calling `model.state_dict()` for EP cases.

        This will append EP placements to the FSDP DTensor state dicts
        """
        assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        if self.ep_mesh is None:
            return
        # [pp, ep_dp, ep, tp]
        global_device_mesh = self.ep_mesh._parent_mesh
        assert global_device_mesh.ndim == 4

        keys = list(state_dict.keys())
        for name in sorted(keys):
            if name in fqn2spec_info and isinstance(fqn2spec_info[name].placement, Shard):
                cur_spec_info = fqn2spec_info[name]
                tensor = state_dict[name]
                tensor = _shard_tensor(tensor, cur_spec_info.mesh, cur_spec_info.placement)
                state_dict[name] = tensor

    @torch.no_grad()
    def load_state_dict_pre_hook(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
        fqn2spec_info: Dict[str, SpecInfo] = None,
    ):
        """
        Pre load state dict when calling `model.load_state_dict()` for EP cases.

        This will shard Dtensor from ckpt to tensor state dicts
        """
        assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

        if self.ep_mesh is None:
            return
        # [pp, ep_dp, ep, tp]
        global_device_mesh = self.ep_mesh._parent_mesh
        assert global_device_mesh.ndim == 4

        keys = list(state_dict.keys())
        for name in sorted(keys):
            tensor = state_dict[name]
            if check_any_unflat_param_names_match(name, fqn2spec_info, "_fsdp_wrapped_module"):
                fqn = name.split("_fsdp_wrapped_module.")[-1]
                cur_spec_info = fqn2spec_info[fqn]
                tensor = _shard_dtensor(tensor, cur_spec_info.mesh, cur_spec_info.placement)
                state_dict[name] = tensor

    def patch_convert_state_with_flat_params(self):
        """ """

        # Modify from torch.distributed.fsdp._optim_utils._convert_state_with_flat_params
        def _convert_state_with_flat_params_patch(
            all_optim_state_keys: List[_OptimStateKey],
            optim_state_key_to_param_key: Dict[_OptimStateKey, Union[int, str]],
            fqn_to_fsdp_param_info: Dict[str, FSDPParamInfo],
            optim_state_dict: Dict[Union[str, int], Any],
            to_save: bool,
            shard_state: bool,
            cpu_offload: bool = True,
            fqn2spec_info: Dict[str, SpecInfo] = None,
        ) -> Dict[str, Any]:
            fsdp_osd_state: Dict[str, Any] = {}
            # Iterate in rank 0's flat parameter ID order to ensure aligned all-gathers
            # across ranks
            for optim_state_key in all_optim_state_keys:
                param_key: Union[str, int, None] = optim_state_key_to_param_key.get(optim_state_key, None)

                assert param_key is not None, (
                    "If use_orig_params is False, we must be able to find the "
                    f"corresponding param id. {optim_state_key} {param_key}"
                )

                if optim_state_key.is_fsdp_managed:
                    # If there are multiple unflat_param_names (not use_orig_params),
                    # they share the same FSDPParamInfo. So the first unflat_param_name
                    # is sufficient to fetch the FSDPParamInfo.
                    fqn = optim_state_key.unflat_param_names[0]
                    fsdp_param_info = fqn_to_fsdp_param_info[fqn]
                    if check_all_unflat_param_names_match(optim_state_key.unflat_param_names, fqn2spec_info):
                        unflat_state = _unflatten_optim_state(
                            fsdp_param_info,
                            optim_state_dict[param_key],
                            to_save,
                            False,
                            cpu_offload,
                        )
                    else:
                        unflat_state = _unflatten_optim_state(
                            fsdp_param_info,
                            optim_state_dict[param_key],
                            to_save,
                            shard_state,
                            cpu_offload,
                        )
                    if to_save:
                        assert len(unflat_state) == len(optim_state_key.unflat_param_names)
                        for unflat_param_name, unflat_param_state in zip(
                            optim_state_key.unflat_param_names,
                            unflat_state,
                        ):
                            fsdp_osd_state[unflat_param_name] = unflat_param_state
                elif to_save:
                    assert len(optim_state_key.unflat_param_names) == 1
                    unflat_param_name = optim_state_key.unflat_param_names[0]
                    fsdp_osd_state[unflat_param_name] = copy.copy(optim_state_dict[param_key])
                    if cpu_offload:
                        for state_name, value in sorted_items(fsdp_osd_state[unflat_param_name]):
                            if not torch.is_tensor(value):
                                continue
                            fsdp_osd_state[unflat_param_name][state_name] = value.cpu()

            return fsdp_osd_state

        # monkey patch
        torch.distributed.fsdp._optim_utils._convert_state_with_flat_params = partial(
            _convert_state_with_flat_params_patch, fqn2spec_info=self.fqn2spec_info
        )

    def patch_fsdp_optim_state_dict(self):
        """ """

        def fsdp_optim_state_post_patch_fn(
            model, optim, optim_state_dict=None, fqn2spec_info: Dict[str, SpecInfo] = None
        ):
            assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

            fsdp_mesh = model._device_mesh
            assert fsdp_mesh is not None, "Please init FSDP module with device_mesh"
            # NOTE we don't support diverse process group for different FSDP sub-modules
            fsdp_pg = model.process_group
            optim_state = orig_optim_state_dict(model, optim, optim_state_dict, fsdp_pg)
            if self.ep_mesh is None:
                return optim_state

            global_device_mesh = self.ep_mesh._parent_mesh
            assert global_device_mesh.ndim == 4

            # extend placements by adding EP placement
            for fqn in sorted(optim_state["state"].keys()):
                if fqn in fqn2spec_info and isinstance(fqn2spec_info[fqn].placement, Shard):
                    cur_spec_info = fqn2spec_info[fqn]
                    fqn_state = {}
                    for key, val in optim_state["state"][fqn].items():
                        # key in OPTIM_STATE_NO_SHARD_KEY in optim stat dict is scalar, like'step', should not be sharded
                        if key not in OPTIM_STATE_NO_SHARD_KEY:
                            val = _shard_tensor(val, cur_spec_info.mesh, cur_spec_info.placement)
                        fqn_state[key] = val
                    optim_state["state"][fqn] = fqn_state
            return optim_state

        # monkey patch
        FSDP.optim_state_dict = staticmethod(partial(fsdp_optim_state_post_patch_fn, fqn2spec_info=self.fqn2spec_info))

    def patch_fsdp_optim_state_dict_to_load(self):
        """
        post optimizer state dict hook when calling `FSDP.optim_state_dict(model, optimizer)`

        This will extend the DTensors in optimizer state dict with EP placements

        Args:
            fsdp_no_shard_param_names: List[str], like
        """

        def optim_state_dict_to_load_pre_patch_fn(
            model,
            optim,
            optim_state_dict,
            is_named_optimizer=False,
            load_directly=False,
            group=None,
            fqn2spec_info: Dict[str, SpecInfo] = None,
        ):
            """
            At this point, the `optim_state_dict` is correctly resharded to the current device mesh by `dcp.load`
            """
            assert fqn2spec_info is not None, "if fqn2spec_info is None it should not be patch"

            fsdp_mesh = model._device_mesh
            assert fsdp_mesh is not None, "Please init FSDP module with device_mesh"

            # NOTE we don't support diverse process group for different FSDP sub-modules
            if self.ep_mesh is not None:
                global_device_mesh = self.ep_mesh._parent_mesh
                assert global_device_mesh.ndim == 4
                for fqn in sorted(optim_state_dict["state"].keys()):
                    if check_any_unflat_param_names_match(fqn, fqn2spec_info):
                        fqn_state = {}
                        for key, val in optim_state_dict["state"][fqn].items():
                            # key in OPTIM_STATE_NO_SHARD_KEY in optim stat dict is scalar, like 'step', should not be sharded
                            if key not in OPTIM_STATE_NO_SHARD_KEY:
                                val = _shard_dtensor(val, self.ep_mesh)
                            fqn_state[key] = val
                        optim_state_dict["state"][fqn] = fqn_state

            fsdp_pg = model.process_group
            optim_state = orig_optim_state_dict_to_load(
                model, optim, optim_state_dict, is_named_optimizer, load_directly, fsdp_pg
            )
            return optim_state

        # monkey patch
        FSDP.optim_state_dict_to_load = staticmethod(
            partial(optim_state_dict_to_load_pre_patch_fn, fqn2spec_info=self.fqn2spec_info)
        )


def register_checkpoint_extension(
    fsdp_model: FSDP, device_mesh: DeviceMesh = None, fqn2spec_info: Dict[str, SpecInfo] = None
):
    """
    Register dtensor-based hooks for FSDP+EP

    This will:

    1. Customize the FSDP extension for save / load hooks in EP scenarios.
    """

    extension = CheckpointExtensions(device_mesh, fqn2spec_info)
    for fsdp_module in FSDP.fsdp_modules(fsdp_model):
        fsdp_module._fsdp_extension = extension
        fsdp_module._handle._fsdp_extension = extension
    # make sure the root module is also registered
    fsdp_model._fsdp_extension = extension
    fsdp_model._handle._fsdp_extension = extension

    # register load / save hook for ep
    if fqn2spec_info is not None:
        state_dict_post_hook_fn = partial(extension.state_dict_post_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_state_dict_hook(state_dict_post_hook_fn)

        load_state_dict_pre_hook_fn = partial(extension.load_state_dict_pre_hook, fqn2spec_info=fqn2spec_info)
        fsdp_model._register_load_state_dict_pre_hook(load_state_dict_pre_hook_fn)

        # patch load / save functino for ep
        extension.patch_convert_state_with_flat_params()
        extension.patch_fsdp_optim_state_dict()
        extension.patch_fsdp_optim_state_dict_to_load()

    return fsdp_model

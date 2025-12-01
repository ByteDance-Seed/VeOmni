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


import gc
from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from ..utils import logging
from ..utils.device import empty_cache, get_device_type
from .parallel_state import get_parallel_state


if TYPE_CHECKING:
    from torch import nn
    from vescale import DeviceMesh


logger = logging.get_logger(__name__)


def build_parallelize_model(
    model: "nn.Module",
    dp_mode: str,
    hf_weight_path: Optional[str] = None,
    enable_full_shard: bool = True,
    enable_fsdp_offload: bool = False,
    enable_mixed_precision: bool = True,
    enable_gradient_checkpointing: bool = True,
    basic_modules: Optional[List[str]] = None,
    enable_reentrant: bool = True,
    use_pin_mem_for_offload: bool = True,
) -> Tuple["nn.Module", "DeviceMesh"]:
    """
    Build a parallelized model with Vescale.
    """
    logger.info_rank0("Apply vescale parallel to the model.")
    parallel_state = get_parallel_state()

    assert dp_mode in ["fsdp2", "fsdp2-vescale"]
    params_stored_in_dtensor = dp_mode == "fsdp2"
    mesh = parallel_state.fsdp_mesh

    if enable_mixed_precision:
        model.float()

    module_init_fn = lambda sub_mod, *_: sub_mod  # noqa: E731
    if hf_weight_path is not None:
        from vescale.initialize.hf_utils import parallel_init_module_fn, parallel_load_safetensors

        shard_states = parallel_load_safetensors(hf_weight_path)
        module_init_fn = parallel_init_module_fn(model, shard_states)

    from vescale import CPUOffloadPolicy, MixedPrecisionPolicy, OffloadPolicy, fully_shard

    if enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        logger.info_rank0("Enable gradient checkpointing.")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": enable_reentrant})

    # mp policy
    mp_policy = MixedPrecisionPolicy()
    if enable_mixed_precision:
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
        )

    # cpu off load policy
    cpu_offload_policy = OffloadPolicy()
    if enable_fsdp_offload:
        cpu_offload_policy = CPUOffloadPolicy(pin_memory=use_pin_mem_for_offload)

    last_fsdp_module = None
    for module in model.modules():
        sub_mod_cls_name = module.__class__.__name__
        if (sub_mod_cls_name in basic_modules) or (sub_mod_cls_name in model._no_split_modules):
            module_init_fn(module)
            if enable_fsdp_offload:
                module.cpu()
                gc.collect()
                empty_cache()
            else:
                model.to(get_device_type())
            fully_shard(
                module,
                mesh=mesh,
                reshard_after_forward=enable_full_shard,
                mp_policy=mp_policy,
                params_stored_in_dtensor=params_stored_in_dtensor,
                offload_policy=cpu_offload_policy,
            )
            # explicit prefetch
            if last_fsdp_module is not None:
                last_fsdp_module.set_modules_to_forward_prefetch([module])
                module.set_modules_to_backward_prefetch([last_fsdp_module])
            last_fsdp_module = module

    module_init_fn(model)
    model = fully_shard(
        model,
        mesh=mesh,
        reshard_after_forward=enable_full_shard,
        mp_policy=mp_policy,
        params_stored_in_dtensor=params_stored_in_dtensor,
        offload_policy=cpu_offload_policy,
    )
    gc.collect()
    empty_cache()

    # NOTE: uncomment below for saving memory fragmentation
    model._set_unshard_async_op(True)

    # for root module, we don't need to reshard after backward since forward will imediately use it
    # model.set_reshard_after_backward(False, recurse=False)
    # NOTE: the above line is WRONG in torch-native fsdp2's senmantic, as resulting logic follows:
    # -) after backward, it is gradient clip to normalize model.parameters()'s grad
    # -) at this time, model.parameters is unsharded param, which has already moved .grad to shard_param.grad, so unshard param.grad is always None
    # -) then None grad disable gradient clip, which is WRONG!
    # -) Even if we have no clip gradient, the optimizer step gives updated weight, which is never used in the next forward; as optimizer step only updates sharded_param, not unshard param
    # -) but next forward of root is already in unsharded state, so never allgather from updated sharded param, which is WRONG again!

    if not hasattr(mesh, "ndevice"):
        # vescale ckpt use vescale device mesh, but here we have torch-native devicemesh, which does not have ndevice attribute
        ndevice_func = lambda self: torch.numel(self.mesh)  # noqa: E731
        mesh.__class__.ndevice = property(ndevice_func)

    return model, mesh

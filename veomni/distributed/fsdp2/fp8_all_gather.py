# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Native FSDP2 all-gather of DeepGEMM expert payloads and block scales."""

from types import MethodType

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule
from torch.distributed.tensor import DTensor

from ...ops.kernels.moe_experts.shared.fp8_weight import FP8ExpertWeight
from ...utils.device import IS_CUDA_AVAILABLE, get_device_type, get_torch_device


class _ExpertAllGather:
    def __init__(self, shape, shard_dim):
        self.shape = tuple(shape)
        self.shard_dim = shard_dim

    def pre(self, local, mesh, outer_size, outer_stride, module, mp_policy):
        from ...ops.kernels.moe_experts.shared.deepgemm_fp8 import quantize_weights

        qdata, scale, _, _ = quantize_weights(local, store_transpose=False, round_bf16=True)
        gathered = scale.new_empty((mesh.size(), *scale.shape))
        work = dist.all_gather_into_tensor(gathered.flatten(), scale.flatten(), group=mesh.get_group(), async_op=True)
        # Keep both buffers alive until completion. Compact scales cannot be a
        # second native extension output: Torch 2.11's Shard(1) copy-out reshapes
        # every output using the original parameter's (much larger) shard shape.
        # Byte views preserve the FP8 bits and avoid ATen's scalar float8-copy
        # conversion kernels in native FSDP packing and nonzero-dim copy-out.
        return (qdata.view(torch.uint8),), (gathered, scale, work)

    def post(self, local, outputs, metadata, param_dtype, *, out=None):
        gathered, local_scale, work = metadata
        work.wait()
        # Scale AG was launched on FSDP's copy-in stream. Its consumer runs on
        # the copy-out stream; prevent the allocator recycling that input early.
        stream = get_torch_device().current_stream(gathered.device)
        gathered.record_stream(stream)
        local_scale.record_stream(stream)
        scale = torch.cat(gathered.unbind(0), dim=self.shard_dim)
        qdata = outputs[0].view(torch.float8_e4m3fn).view(self.shape)
        if out is not None:
            target = out.to_local() if isinstance(out, DTensor) else out
            target.qdata.view(torch.uint8).copy_(qdata.view(torch.uint8))
            target.scale.copy_(scale)
            return None
        weight = FP8ExpertWeight(qdata, scale)
        # Native FSDP owns the qdata all-gather buffer; ask it to allocate/free
        # scales on the same unshard/reshard lifecycle too.
        return weight, (scale,)


def _deepgemm_experts(model):
    return (
        (name, module)
        for name, module in model.named_modules()
        if getattr(getattr(module, "veomni_moe", None), "impl", None) == "deepgemm_fp8"
    )


def resolve_fp8_all_gather(model, enabled, mixed_precision, *, fsdp_mode="fsdp2", offload=False):
    """Ignore the inherited policy without matching experts; validate it otherwise."""
    if not isinstance(enabled, bool):
        raise ValueError("expert_weight_fp8_all_gather must be a boolean.")
    if not enabled or next(_deepgemm_experts(model), None) is None:
        return False
    if (
        fsdp_mode != "fsdp2"
        or offload
        or mixed_precision is None
        or not mixed_precision.enable
        or mixed_precision.param_dtype != "bfloat16"
    ):
        raise ValueError("expert_weight_fp8_all_gather requires FSDP2, BF16 mixed precision and no CPU offload.")
    return True


def register_deepgemm_fp8_all_gather(model: torch.nn.Module) -> int:
    """Install hooks after FSDP wrapping/materialization and before first forward.

    Select only modules whose instance-local moe_experts handle uses DeepGEMM.
    Master shards remain ordinary tensors. Complete 128x128 blocks must fit
    inside each shard; uneven and block-splitting layouts fail before training.
    This integration targets the pinned Torch 2.11 eager FSDP extension API.
    Models without matching experts register no hooks. Selected experts must
    satisfy all shape, device and FSDP checks regardless of model scope.
    """
    selected = {}
    for name, module in _deepgemm_experts(model):
        if not getattr(module, "use_swiglu_mlp", True):
            raise ValueError(f"FP8 all-gather requires the SwiGLU DeepGEMM execution path: {name}.")
        params = list(module.named_parameters(recurse=False))
        if len(params) != 2 or any(p.ndim != 3 for _, p in params):
            raise ValueError(f"FP8 all-gather requires two merged expert weights: {name}.")
        for param_name, param in params:
            selected[id(param)] = f"{name}.{param_name}"
    if not selected:
        return 0
    pending = []
    for module in model.modules():
        if not isinstance(module, FSDPModule):
            continue
        state = module._get_fsdp_state()
        group = state._fsdp_param_group
        if group is None:
            continue
        for param in group.fsdp_params:
            if id(param.sharded_param) not in selected:
                continue
            name = selected[id(param.sharded_param)]
            if group._reset_sharded_params or hasattr(param, "_unsharded_param"):
                raise ValueError("Install FP8 all-gather before FSDP's first forward/unshard.")
            if param.offload_to_cpu or (
                param.post_forward_mesh_info is not None and param.post_forward_mesh_info is not param.mesh_info
            ):
                raise ValueError("FP8 all-gather does not support CPU offload or partial resharding.")
            if param.mp_policy.param_dtype != torch.bfloat16:
                raise ValueError("FP8 all-gather requires FSDP mixed_precision.param_dtype='bfloat16'.")
            local = param._sharded_local_tensor
            shape, dim, world = param._orig_size, param.fsdp_placement.dim, param.shard_mesh_from_root.size()
            if (
                not IS_CUDA_AVAILABLE
                or local.device.type != get_device_type()
                or torch.version.hip is not None
                or get_torch_device().get_device_capability(local.device) not in ((9, 0), (10, 0))
                or local.dtype not in (torch.float32, torch.bfloat16)
                or not local.is_contiguous()
                or min(local.shape) == 0
                or local.shape[1] % 128
                or local.shape[2] % 128
                or shape[dim] % world
            ):
                raise ValueError(
                    f"FP8 all-gather needs contiguous CUDA BF16/FP32, even shards and full 128x128 blocks: {name}, {tuple(local.shape)}."
                )
            if hasattr(local, "fsdp_pre_all_gather") or hasattr(local, "fsdp_post_all_gather"):
                raise ValueError(f"An all-gather extension is already installed on {name}.")
            pending.append((local, _ExpertAllGather(shape, dim)))
    if len(pending) != len(selected):
        raise ValueError("Every selected expert must be managed by native FSDP2.")
    # Validate all parameters before installing any hooks.
    for local, hook in pending:
        local.fsdp_pre_all_gather = MethodType(hook.pre, local)
        local.fsdp_post_all_gather = MethodType(hook.post, local)
    return len(pending)

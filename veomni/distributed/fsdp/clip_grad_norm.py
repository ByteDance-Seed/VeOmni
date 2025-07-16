import functools
import math
import warnings

import torch
import torch.distributed as dist
from torch.distributed._tensor import Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import _get_grad_norm

from ...utils.import_utils import is_torch_version_greater_than
from ..parallel_plan import SpecInfo


def clip_grad_norm_(fsdp_model: FSDP, max_norm, norm_type=2.0) -> torch.Tensor:
    extension = fsdp_model._fsdp_extension
    ep_mesh = extension.ep_mesh
    ep_group = None if ep_mesh is None else ep_mesh.get_group()

    if ep_group is None or dist.get_world_size(ep_group) in (1, dist.get_world_size()):
        return FSDP.clip_grad_norm_(fsdp_model, max_norm, norm_type)

    assert fsdp_model._is_root
    # use dict as ordered set to make param order consistent among
    # dp (hsdp) ranks to avoid gnorm difference due to reduction order
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    fsdp_managed_params = set()
    sharded_params_for_gnorm = {}
    ep_fsdp_sharded_params_for_gnorm = {}
    nonsharded_params_for_gnorm = {}
    grads_for_clip = []
    ep_fsdp_process_group = None

    for handle in fsdp_model._all_handles:
        assert handle.uses_sharded_strategy
        assert handle._use_orig_params, "tensor parallelism can only work with FSDP using `use_orig_params=True`"
        for param in handle.flat_param._params:
            assert hasattr(param, "spec_info")
            spec_info: SpecInfo = param.spec_info
            fsdp_managed_params.add(param)
            if param.grad is not None:
                grads_for_clip.append(param.grad)
            # ep param
            if isinstance(spec_info.placement, Shard):
                if ep_fsdp_process_group is None:
                    ep_fsdp_process_group = handle.process_group
                ep_fsdp_sharded_params_for_gnorm.setdefault(param, None)
            # fsdp param
            else:
                sharded_params_for_gnorm.setdefault(param, None)
    for param in fsdp_model.parameters():
        not_fsdp_managed = param not in fsdp_managed_params and param not in sharded_params_for_gnorm
        if not_fsdp_managed:
            assert hasattr(param, "_spec")
            raise NotImplementedError(f"param {param._spec.fqn} is not managed by FSDP")

    # Compute local norms (forced to be in FP32)
    if is_torch_version_greater_than("2.5.0"):
        grad_norm_kwargs = {
            "norm_type": norm_type,
            "zero": torch.tensor(0.0),
            "device": fsdp_model.compute_device,
        }
    else:
        grad_norm_kwargs = {
            "norm_type": norm_type,
        }

    local_sharded_norm = _get_grad_norm(sharded_params_for_gnorm, **grad_norm_kwargs).to(fsdp_model.compute_device)
    local_ep_fsdp_sharded_norm = (
        _get_grad_norm(ep_fsdp_sharded_params_for_gnorm, **grad_norm_kwargs).to(fsdp_model.compute_device)
        if ep_fsdp_sharded_params_for_gnorm
        else None
    )
    local_nonsharded_norm = (
        _get_grad_norm(nonsharded_params_for_gnorm, **grad_norm_kwargs).to(fsdp_model.compute_device)
        if nonsharded_params_for_gnorm
        else None
    )

    # Reconstruct the total gradient norm depending on the norm type
    if norm_type == math.inf:
        total_norm = (
            torch.maximum(local_sharded_norm, local_nonsharded_norm)
            if local_nonsharded_norm is not None
            else local_sharded_norm
        )
        dist.all_reduce(total_norm, op=torch.distributed.ReduceOp.MAX, group=fsdp_model.process_group)
        # allreduce across tp group
        dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=ep_group)
    else:
        total_norm = local_sharded_norm**norm_type
        dist.all_reduce(total_norm, group=fsdp_model.process_group)
        if local_ep_fsdp_sharded_norm is not None:
            total_ep_fsdp_sharded_norm = local_ep_fsdp_sharded_norm**norm_type
            dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_fsdp_process_group)
            dist.all_reduce(total_ep_fsdp_sharded_norm, group=ep_group)
            total_norm += total_ep_fsdp_sharded_norm

        # All-reducing the local non-sharded norm would count it an extra
        # world-size-many times
        if local_nonsharded_norm is not None:
            total_norm += local_nonsharded_norm**norm_type
        total_norm = total_norm ** (1.0 / norm_type)
    if fsdp_model.cpu_offload.offload_params:
        total_norm = total_norm.cpu()

    clip_coef = max_norm / (total_norm + 1e-6)
    # Multiplying by the clamped coefficient is meaningless when it is
    # equal to 1, but it avoids the host-device sync that would result from
    # `if clip_coef < 1`
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    for grad in grads_for_clip:
        grad.mul_(clip_coef_clamped.to(grad.device, grad.dtype))
    # Use the "largest" dtype by type promotion semantics to use the same
    # dtype as if we did not force local norm computation to be in FP32
    if len(grads_for_clip) == 0:
        # If this rank has no gradients, then we must default to FP32
        # unless we use additional communication, which we prefer to avoid
        # since `clip_grad_norm_()` is called in the training loop
        warnings.warn(
            f"Called FSDP.clip_grad_norm_() on rank {fsdp_model.rank} with no "
            "gradients -- returning the total norm in the default dtype "
            f"{total_norm.dtype}"
        )  # warn since this is generally unexpected
        return total_norm
    total_norm_dtype = functools.reduce(
        torch.promote_types,
        [grad.dtype for grad in grads_for_clip],
    )
    return total_norm.to(total_norm_dtype)


def _is_first_ep_rank(ep_group: dist.ProcessGroup):
    assert ep_group is not None
    return dist.get_rank(ep_group) == 0

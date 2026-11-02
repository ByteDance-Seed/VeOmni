# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""BF16-facing container for DeepGEMM block-quantized FSDP parameters."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


class FP8ExpertWeight(torch.Tensor):
    """Carry E4M3 payload and 128x128 dequantization scales without BF16 storage.

    Only metadata-preserving operations are supported. Arithmetic must consume
    ``qdata`` and ``scale`` explicitly through the DeepGEMM expert backend.
    Optimizers and checkpoints continue to operate on ordinary master shards.
    """

    @staticmethod
    def __new__(cls, qdata, scale):
        result = torch.Tensor._make_wrapper_subclass(
            cls, qdata.shape, strides=qdata.stride(), device=qdata.device, dtype=torch.bfloat16
        )
        result.qdata = qdata
        result.scale = scale
        return result

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        """Preserve the container across FSDP's parameter views and detaches."""
        kwargs = kwargs or {}
        weight = args[0]
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            result = FP8ExpertWeight(weight.qdata, weight.scale)
            return return_and_correct_aliasing(func, args, kwargs, result)
        if func == torch.ops.aten.as_strided.default:
            size, stride = args[1:3]
            offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
            if tuple(size) == tuple(weight.shape) and tuple(stride) == weight.stride() and offset == 0:
                result = FP8ExpertWeight(weight.qdata, weight.scale)
                return return_and_correct_aliasing(func, args, kwargs, result)
        if func == torch.ops.aten.new_empty.default:
            # Models use a weight to construct the empty split-weight sentinel.
            return torch.empty(
                args[1], device=kwargs.get("device", weight.device), dtype=kwargs.get("dtype", weight.dtype)
            )
        raise NotImplementedError(f"FP8ExpertWeight does not support {func}; use the deepgemm_fp8 MoE backend.")

    __torch_function__ = torch._C._disabled_torch_function_impl

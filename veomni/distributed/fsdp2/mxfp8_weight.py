# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""Compute views of MXFP8 weights gathered by native FSDP2."""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing


class MXFP8GatheredWeight(torch.Tensor):
    """BF16-facing weight backed by rowwise and independently columnwise MXFP8 blocks."""

    @staticmethod
    def __new__(cls, row_data, row_scale, col_data, col_scale):
        result = torch.Tensor._make_wrapper_subclass(cls, row_data.shape, device=row_data.device, dtype=torch.bfloat16)
        result.payload = (row_data, row_scale, col_data, col_scale)
        return result

    def mx(self, *, transposed, emulated):
        from torchao.prototype.mx_formats.mx_tensor import MXTensor
        from torchao.quantization.quantize_.common import KernelPreference

        from ...ops.torchao import MXFP8_BLOCK_SIZE

        row_data, row_scale, col_data, col_scale = self.payload
        data, scale = (col_data.t().contiguous(), col_scale.t().contiguous()) if transposed else (row_data, row_scale)
        return MXTensor.from_qdata_and_scales(
            data.view(torch.float8_e4m3fn),
            scale.view(torch.float8_e8m0fnu),
            torch.bfloat16,
            block_size=MXFP8_BLOCK_SIZE,
            kernel_preference=KernelPreference.EMULATED if emulated else KernelPreference.AUTO,
        )

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        weight = args[0]
        kwargs = kwargs or {}
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return return_and_correct_aliasing(func, args, kwargs, cls(*weight.payload))
        if func == torch.ops.aten.as_strided.default:
            size, stride = args[1:3]
            offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
            if tuple(size) == tuple(weight.shape) and tuple(stride) == weight.stride() and offset == 0:
                return return_and_correct_aliasing(func, args, kwargs, cls(*weight.payload))
        if func == torch.ops.aten.new_empty.default:
            # Empty bias sentinel used by gated MLP.
            return torch.empty(
                args[1], device=kwargs.get("device", weight.device), dtype=kwargs.get("dtype", weight.dtype)
            )
        raise NotImplementedError(f"Unsupported operation {func} on a gathered MXFP8 weight.")

    __torch_function__ = torch._C._disabled_torch_function_impl


class MXFP8AllGather:
    def __init__(self, emulated):
        self.emulated = emulated

    def _payload(self, local):
        from ...ops.torchao import _quantize_mxfp8

        master = local.to(torch.bfloat16)
        row = _quantize_mxfp8(master, self.emulated)
        col = _quantize_mxfp8(master.t(), self.emulated)
        return (
            row.qdata.view(torch.uint8),
            row.scale.view(torch.uint8),
            col.qdata.view(torch.uint8).t().contiguous(),
            col.scale.view(torch.uint8).t().contiguous(),
        )

    def pre(self, local, mesh, outer_size, outer_stride, module, mp_policy):
        # PyTorch 2.11's singleton shortcut only forwards the first input.
        if mesh.size() == 1:
            return (local.to(torch.bfloat16),), True
        return self._payload(local), False

    def post(self, local, outputs, metadata, param_dtype, *, out=None):
        payload = self._payload(outputs[0]) if metadata else outputs
        if out is not None:
            for target, source in zip(out.payload, payload, strict=True):
                if target.data_ptr() != source.data_ptr():
                    target.copy_(source)
            return None
        # Multi-rank payloads alias FSDP's native all-gather buffers. Singleton
        # payloads are allocated here and must follow the same reshard lifecycle.
        return MXFP8GatheredWeight(*payload), tuple(payload) if metadata else ()

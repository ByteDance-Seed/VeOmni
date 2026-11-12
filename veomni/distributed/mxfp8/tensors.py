# Copyright 2026 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0

"""MXFP8 dense Linear and exact, independent row/column weight quantization.

The persistent parameter is a high-precision master. The FSDP2 extension gathers
two independently quantized views of that master, never re-quantizing a lossy
forward representation for backward. Both views include E8M0 scales. This costs
2 + 2/32 bytes per parameter, slightly MORE than a single BF16 all-gather. It is
a correctness baseline for low-precision transport, not a bandwidth reduction.

Only ordinary 2-D weights evenly sharded along dim 0 at 32-row boundaries are
supported. This makes every columnwise quantization block rank-local. An uneven
or block-splitting layout is rejected rather than changing the quantizer.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils import _pytree as pytree
from torchao.prototype.mx_formats.config import ScaleCalculationMode
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchao.quantization.quantize_.common import KernelPreference


_BLOCK_SIZE = 32
_MASTER_PRESERVING_OPS = {
    torch.ops.aten.detach.default,
    torch.ops.aten.alias.default,
    torch.ops.aten.empty_like.default,
    torch.ops.aten.new_zeros.default,
    torch.ops.aten.slice.Tensor,
    torch.ops.aten.view.default,
    torch.ops.aten.as_strided.default,
    torch.ops.aten._to_copy.default,
    torch.ops.aten._pin_memory.default,
    torch.ops.aten.split.Tensor,
    torch.ops.aten.clone.default,
}


def _quantize(tensor: torch.Tensor, emulate: bool) -> MXTensor:
    """Use torchao's RCEIL quantizer; AUTO always selects its native MX GEMM."""
    quantized = MXTensor.to_mx(
        tensor.contiguous(),
        torch.float8_e4m3fn,
        block_size=_BLOCK_SIZE,
        scaling_mode=ScaleCalculationMode.RCEIL,
        kernel_preference=KernelPreference.EMULATED if emulate else KernelPreference.AUTO,
    )
    # The native torchao MX GEMM always accumulates to BF16. Keep that logical
    # compute dtype even when quantization starts from an FP32 master; do not
    # round the master to BF16 before deriving its qdata and block scales.
    return MXTensor.from_qdata_and_scales(
        quantized.qdata,
        quantized.scale,
        torch.bfloat16,
        block_size=_BLOCK_SIZE,
        kernel_preference=KernelPreference.EMULATED if emulate else KernelPreference.AUTO,
    )


def _from_payload(data: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, emulate: bool) -> MXTensor:
    return MXTensor.from_qdata_and_scales(
        data.view(torch.float8_e4m3fn),
        scale.view(torch.float8_e8m0fnu),
        dtype,
        block_size=_BLOCK_SIZE,
        kernel_preference=KernelPreference.EMULATED if emulate else KernelPreference.AUTO,
    )


class MXFP8Weight(torch.Tensor):
    """High-precision master with an FSDP2 quantize-before-gather extension.

    ``emulate`` is an internal numerical-test seam. The public configuration
    never sets it and never falls back to emulation on unsupported hardware.
    """

    @staticmethod
    def __new__(cls, tensor: torch.Tensor, emulate: bool = False):
        return torch.Tensor._make_wrapper_subclass(
            cls,
            tensor.shape,
            strides=tensor.stride(),
            storage_offset=tensor.storage_offset(),
            dtype=tensor.dtype,
            device=tensor.device,
            layout=tensor.layout,
            requires_grad=tensor.requires_grad,
        )

    def __init__(self, tensor: torch.Tensor, emulate: bool = False):
        self._master = tensor
        self._emulate = emulate

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        wrappers = [value for value in pytree.tree_leaves((args, kwargs)) if isinstance(value, cls)]
        emulate = wrappers[0]._emulate
        if any(value._emulate != emulate for value in wrappers):
            raise ValueError("Cannot mix native and emulated MXFP8 master weights.")
        unwrapped_args, unwrapped_kwargs = pytree.tree_map_only(cls, lambda value: value._master, (args, kwargs or {}))
        result = func(*unwrapped_args, **unwrapped_kwargs)
        if func == torch.ops.aten.copy_.default:
            return args[0]
        if func in _MASTER_PRESERVING_OPS:
            return pytree.tree_map_only(torch.Tensor, lambda value: cls(value, emulate), result)
        return result

    def __tensor_flatten__(self):
        return ["_master"], {"emulate": self._emulate}

    @classmethod
    def __tensor_unflatten__(cls, tensors, metadata, outer_size, outer_stride):
        return cls(tensors["_master"], metadata["emulate"])

    def __repr__(self):
        return f"MXFP8Weight(master={self._master}, emulate={self._emulate})"

    def fsdp_pre_all_gather(self, mesh, outer_size, outer_stride, module, mp_policy):
        del module
        world_size = mesh.size()
        if (
            len(outer_size) != 2
            or outer_stride != (outer_size[1], 1)
            or outer_size[0] % (world_size * _BLOCK_SIZE)
            or outer_size[1] % _BLOCK_SIZE
            or self._master.shape != (outer_size[0] // world_size, outer_size[1])
            or self._master.numel() == 0
        ):
            raise ValueError(
                "MXFP8 all-gather requires a contiguous 2-D weight evenly sharded on dim 0, "
                "with both the local row count and input features divisible by 32; "
                f"got global shape {tuple(outer_size)}, local shape {tuple(self._master.shape)}, "
                f"stride {outer_stride}, shard world size {world_size}."
            )
        if mp_policy.param_dtype != torch.bfloat16:
            raise ValueError("MXFP8 all-gather requires FSDP2 param_dtype=torch.bfloat16.")
        if world_size == 1:
            # PyTorch's singleton-mesh shortcut only consumes the first payload.
            # There is no collective here: preserve FP32 through that shortcut,
            # then build both compute views in post_all_gather.
            return (self._master,), ("singleton",)
        # Quantize BOTH orientations directly from the same master. In particular,
        # do not dequantize the rowwise weight to construct the columnwise weight.
        row = _quantize(self._master, self._emulate)
        col = _quantize(self._master.t(), self._emulate)
        # All payloads are rank-major along the first dimension. E8M0 does not
        # have portable collective support; uint8 transports its bits unchanged.
        payload = (
            row.qdata.view(torch.uint8),
            row.scale.view(torch.uint8),
            col.qdata.t().contiguous().view(torch.uint8),
            col.scale.t().contiguous().view(torch.uint8),
        )
        return payload, ()

    def fsdp_post_all_gather(self, all_gather_outputs, metadata, param_dtype, *, out=None):
        if metadata == ("singleton",):
            master = all_gather_outputs[0]
            row = _quantize(master, self._emulate)
            col = _quantize(master.t(), self._emulate)
            all_gather_outputs = (
                row.qdata.view(torch.uint8),
                row.scale.view(torch.uint8),
                col.qdata.t().contiguous().view(torch.uint8),
                col.scale.t().contiguous().view(torch.uint8),
            )
        if out is not None:
            if not isinstance(out, MXFP8GatheredWeight):
                raise TypeError(f"Expected MXFP8GatheredWeight, got {type(out).__name__}.")
            for target, source in zip(out.payload, all_gather_outputs, strict=True):
                if target.data_ptr() != source.data_ptr():
                    target.copy_(source)
            return None
        result = MXFP8GatheredWeight(*all_gather_outputs, dtype=param_dtype, emulate=self._emulate)
        return result, tuple(all_gather_outputs)


class MXFP8GatheredWeight(torch.Tensor):
    """Ephemeral FSDP2 compute parameter; never an export/checkpoint master."""

    @staticmethod
    def __new__(cls, row_data, row_scale, col_data, col_scale, *, dtype, emulate=False):
        return torch.Tensor._make_wrapper_subclass(
            cls, row_data.shape, dtype=dtype, device=row_data.device, requires_grad=False
        )

    def __init__(self, row_data, row_scale, col_data, col_scale, *, dtype, emulate=False):
        self.row_data = row_data
        self.row_scale = row_scale
        self.col_data = col_data
        self.col_scale = col_scale
        self._emulate = emulate

    @property
    def payload(self):
        return self.row_data, self.row_scale, self.col_data, self.col_scale

    def rowwise(self):
        return _from_payload(self.row_data, self.row_scale, self.dtype, self._emulate)

    def columnwise_transpose(self):
        return _from_payload(
            self.col_data.t().contiguous(), self.col_scale.t().contiguous(), self.dtype, self._emulate
        )

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        kwargs = kwargs or {}
        value = args[0]
        if func in (torch.ops.aten.detach.default, torch.ops.aten.alias.default):
            return cls(*value.payload, dtype=value.dtype, emulate=value._emulate)
        if func == torch.ops.aten.as_strided.default:
            size, stride = tuple(args[1]), tuple(args[2])
            offset = args[3] if len(args) > 3 else kwargs.get("storage_offset", 0)
            if size != tuple(value.shape) or stride != value.stride() or offset != 0:
                raise ValueError("MXFP8 gathered weights only support their original contiguous FSDP2 view.")
            return cls(*value.payload, dtype=value.dtype, emulate=value._emulate)
        if func == torch.ops.aten.clone.default:
            return cls(*(part.clone() for part in value.payload), dtype=value.dtype, emulate=value._emulate)
        if func == torch.ops.aten.copy_.default:
            source = args[1]
            if not isinstance(source, cls):
                raise TypeError("Cannot overwrite an MXFP8 gathered weight with a high-precision tensor.")
            for target, data in zip(value.payload, source.payload, strict=True):
                target.copy_(data)
            return value
        raise NotImplementedError(f"{func} is not supported for an ephemeral MXFP8 gathered weight.")

    def __tensor_flatten__(self):
        return ["row_data", "row_scale", "col_data", "col_scale"], {
            "dtype": self.dtype,
            "emulate": self._emulate,
        }

    @classmethod
    def __tensor_unflatten__(cls, tensors, metadata, outer_size, outer_stride):
        return cls(
            *(tensors[name] for name in ("row_data", "row_scale", "col_data", "col_scale")),
            **metadata,
        )

    def __repr__(self):
        return f"MXFP8GatheredWeight(shape={tuple(self.shape)}, dtype={self.dtype}, emulate={self._emulate})"


def _pad_rows(tensor):
    rows = tensor.shape[0]
    return F.pad(tensor, (0, 0, 0, (-rows) % _BLOCK_SIZE))


@torch._dynamo.allow_in_graph
class _MXFP8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, weight, emulate):
        original_shape = input_tensor.shape
        input_2d = input_tensor.reshape(-1, original_shape[-1]).contiguous()
        if input_2d.shape[0] == 0:
            raise ValueError("MXFP8 Linear requires at least one input token.")
        input_padded = _pad_rows(input_2d)
        if isinstance(weight, MXFP8GatheredWeight):
            row_weight = weight.rowwise()
        else:
            master = weight._master if isinstance(weight, MXFP8Weight) else weight
            row_weight = _quantize(master, emulate)
        output = torch.mm(_quantize(input_padded, emulate), row_weight.t())
        # Save the compute parameter rather than its payloads so FSDP2 can
        # reshard it after forward and restore it for backward through its hooks.
        ctx.save_for_backward(input_tensor, weight)
        ctx.emulate = emulate
        return output[: input_2d.shape[0]].reshape(*original_shape[:-1], weight.shape[0])

    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, weight = ctx.saved_tensors
        input_2d = input_tensor.reshape(-1, input_tensor.shape[-1]).contiguous()
        grad_2d = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        input_padded, grad_padded = _pad_rows(input_2d), _pad_rows(grad_2d)
        if isinstance(weight, MXFP8GatheredWeight):
            col_weight_t = weight.columnwise_transpose()
        else:
            master = weight._master if isinstance(weight, MXFP8Weight) else weight
            col_weight_t = _quantize(master.t(), ctx.emulate)
        grad_input = torch.mm(_quantize(grad_padded, ctx.emulate), col_weight_t.t())
        # Both weight-gradient operands are independently quantized along the
        # token reduction axis. Padding zeros makes a ragged final block explicit.
        grad_weight = torch.mm(_quantize(grad_padded.t(), ctx.emulate), _quantize(input_padded.t(), ctx.emulate).t())
        return (
            grad_input[: input_2d.shape[0]].reshape(input_tensor.shape).to(input_tensor.dtype),
            grad_weight.to(weight.dtype),
            None,
        )


class MXFP8Linear(nn.Linear):
    """Dense Linear using torchao MXTensor GEMMs for forward and backward."""

    @classmethod
    def from_float(cls, module: nn.Linear, *, all_gather: bool, emulate: bool = False):
        with torch.device("meta"):
            result = cls(module.in_features, module.out_features, bias=module.bias is not None)
        result.weight = (
            nn.Parameter(MXFP8Weight(module.weight.detach(), emulate), requires_grad=module.weight.requires_grad)
            if all_gather
            else module.weight
        )
        result.bias = module.bias
        result._emulate = emulate
        result.train(module.training)
        return result

    def forward(self, input_tensor):
        if torch.is_autocast_enabled(input_tensor.device.type):
            input_tensor = input_tensor.to(torch.get_autocast_dtype(input_tensor.device.type))
        output = _MXFP8LinearFunction.apply(input_tensor, self.weight, self._emulate)
        if self.bias is not None:
            output = output + self.bias.to(output.dtype)
        return output


torch.serialization.add_safe_globals([MXFP8Weight, MXFP8GatheredWeight])

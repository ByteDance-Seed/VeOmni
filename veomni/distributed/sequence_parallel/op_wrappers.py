# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""No-autograd wrappers for async Ulysses compound autograd.

The helpers resolve the same ``OpsImplementationConfig`` fields that bind
``OpSlot`` objects, but they expose a ``(output, saved)`` / ``backward`` pair
that can run inside a compound ``Function``. They do not mutate ``OpSlot``
instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

import torch
from torch import Tensor

from ...ops.config.singleton import get_ops_config
from ...utils.device import IS_NPU_AVAILABLE


class OpWrapper(Protocol):
    def forward(self, *args: Any, **kwargs: Any) -> tuple[Tensor, OpSavedState]: ...

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor, ...]: ...


@dataclass(frozen=True)
class OpSavedState:
    """Tensor payload and non-Tensor metadata required by an op backward."""

    tensors: tuple[Tensor, ...]
    metadata: Any


@dataclass(frozen=True)
class OpSavedRef:
    """Reference to an op state stored by a compound autograd Function."""

    indices: tuple[int, ...]
    metadata: Any


def pack_saved_state(saved_tensors: list[Tensor | None], state: OpSavedState) -> OpSavedRef:
    """Append an op's tensors for ``ctx.save_for_backward`` and return metadata."""

    indices = []
    for tensor in state.tensors:
        index = next((index for index, saved in enumerate(saved_tensors) if saved is tensor), None)
        if index is None:
            index = len(saved_tensors)
            saved_tensors.append(tensor)
        indices.append(index)
    return OpSavedRef(indices=tuple(indices), metadata=state.metadata)


def unpack_saved_state(saved_tensors: tuple[Tensor | None, ...], ref: OpSavedRef) -> OpSavedState:
    """Reconstruct an op state from tensors restored by autograd."""

    tensors = []
    for index in ref.indices:
        tensor = saved_tensors[index]
        assert tensor is not None
        tensors.append(tensor)
    return OpSavedState(tensors=tuple(tensors), metadata=ref.metadata)


def _batch_dims(tensor: Tensor) -> tuple[int, ...]:
    return tuple(range(tensor.ndim - 1))


@dataclass(frozen=True)
class _EagerRMSMetadata:
    empty: bool
    eps: float


@dataclass
class _EagerRMSNorm:
    offset: float = 0.0

    def forward(self, x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, OpSavedState]:
        scale = weight if self.offset == 0.0 else self.offset + weight
        if x.numel() == 0:
            return x * scale, OpSavedState((x, weight), _EagerRMSMetadata(True, eps))

        x_f = x.float()
        rstd = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
        normed = x_f * rstd
        if self.offset == 0.0:
            output = scale * normed.to(x.dtype)
        else:
            output = (scale.float() * normed).to(x.dtype)
        return output, OpSavedState((x, weight, rstd), _EagerRMSMetadata(False, eps))

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor, Tensor]:
        metadata = saved.metadata
        assert isinstance(metadata, _EagerRMSMetadata)
        x, weight, *optional_rstd = saved.tensors
        if metadata.empty:
            return torch.zeros_like(x), torch.zeros_like(weight)

        (rstd,) = optional_rstd
        x_f = x.float()
        scale = weight if self.offset == 0.0 else self.offset + weight
        n = x.shape[-1]
        if self.offset == 0.0:
            scaled_grad = (grad_output * scale).float()
            grad_weight = (grad_output * (x_f * rstd).to(x.dtype)).sum(dim=_batch_dims(x))
        else:
            scaled_grad = grad_output.float() * scale.float()
            grad_weight = (grad_output.float() * (x_f * rstd)).sum(dim=_batch_dims(x)).to(weight.dtype)
        grad_x = rstd * scaled_grad - (rstd.pow(3) / n) * x_f * (scaled_grad * x_f).sum(dim=-1, keepdim=True)
        return grad_x.to(x.dtype), grad_weight.to(weight.dtype)


@dataclass(frozen=True)
class _LigerRMSMetadata:
    empty: bool
    offset: float
    casting_mode: int
    block_size: int
    num_warps: int
    eps: float


@dataclass
class _LigerRMSNorm:
    offset: float
    casting_mode: str

    def forward(self, x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, OpSavedState]:
        if x.numel() == 0:
            output, _ = _EagerRMSNorm(self.offset).forward(x, weight, eps)
            metadata = _LigerRMSMetadata(True, self.offset, 0, 0, 0, eps)
            return output, OpSavedState((x, weight), metadata)

        from liger_kernel.ops.rms_norm import rms_norm_forward

        output, saved_x, rstd, block_size, num_warps, casting_mode = rms_norm_forward(
            x.contiguous(),
            weight.contiguous(),
            eps,
            self.offset,
            self.casting_mode,
            None,
        )
        return output, OpSavedState(
            (saved_x, weight.contiguous(), rstd),
            _LigerRMSMetadata(False, self.offset, casting_mode, block_size, num_warps, eps),
        )

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor, Tensor]:
        metadata = saved.metadata
        assert isinstance(metadata, _LigerRMSMetadata)
        x, weight, *optional_rstd = saved.tensors
        if metadata.empty:
            return _EagerRMSNorm(metadata.offset).backward(
                grad_output,
                OpSavedState((x, weight), _EagerRMSMetadata(True, metadata.eps)),
            )

        (rstd,) = optional_rstd

        from liger_kernel.ops.rms_norm import rms_norm_backward

        grad_x, grad_weight = rms_norm_backward(
            grad_output.contiguous(),
            x,
            weight,
            rstd,
            metadata.offset,
            metadata.casting_mode,
            metadata.block_size,
            metadata.num_warps,
            False,
            None,
        )
        return grad_x, grad_weight


@dataclass(frozen=True)
class _NpuRMSMetadata:
    empty: bool
    eps: float


@dataclass
class _NpuRMSNorm:
    offset: float = 0.0

    def forward(self, x: Tensor, weight: Tensor, eps: float) -> tuple[Tensor, OpSavedState]:
        if x.numel() == 0:
            output, _ = _EagerRMSNorm(self.offset).forward(x, weight, eps)
            return output, OpSavedState((x, weight), _NpuRMSMetadata(True, eps))

        import torch_npu

        scale = weight if self.offset == 0.0 else self.offset + weight
        output, rstd = torch_npu.npu_rms_norm(x, scale, eps)
        return output, OpSavedState((x, weight, rstd), _NpuRMSMetadata(False, eps))

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor, Tensor]:
        metadata = saved.metadata
        assert isinstance(metadata, _NpuRMSMetadata)
        x, weight, *optional_rstd = saved.tensors
        if metadata.empty:
            return _EagerRMSNorm(self.offset).backward(
                grad_output,
                OpSavedState((x, weight), _EagerRMSMetadata(True, metadata.eps)),
            )

        import torch_npu

        (rstd,) = optional_rstd
        scale = weight if self.offset == 0.0 else self.offset + weight
        return torch_npu.npu_rms_norm_backward(
            grad_output.contiguous(),
            x,
            scale,
            rstd,
        )


def _build_rms_norm(variant: str, impl_name: str) -> OpWrapper:
    if impl_name == "eager":
        return _EagerRMSNorm(offset=1.0 if variant == "qwen3_5" else 0.0)
    if impl_name == "liger_kernel":
        if variant == "qwen3_5":
            return _LigerRMSNorm(offset=1.0, casting_mode="gemma")
        return _LigerRMSNorm(offset=0.0, casting_mode="llama")
    if impl_name == "npu":
        if not IS_NPU_AVAILABLE:
            raise RuntimeError("rms_norm implementation 'npu' requires an NPU device.")
        return _NpuRMSNorm(offset=1.0 if variant == "qwen3_5" else 0.0)
    raise KeyError(f"No async RMSNorm wrapper for impl={impl_name!r}, variant={variant!r}")


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


@dataclass(frozen=True)
class _EagerRotaryMetadata:
    empty: bool


class _EagerRotary:
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, OpSavedState]:
        if x.numel() == 0:
            return x, OpSavedState((cos, sin), _EagerRotaryMetadata(True))
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        return (x * cos_u) + (_rotate_half(x) * sin_u), OpSavedState((cos, sin), _EagerRotaryMetadata(False))

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor]:
        metadata = saved.metadata
        assert isinstance(metadata, _EagerRotaryMetadata)
        if metadata.empty:
            return (grad_output,)
        cos, sin = saved.tensors
        cos_u = cos.unsqueeze(1)
        sin_u = sin.unsqueeze(1)
        return ((grad_output * cos_u) - _rotate_half(grad_output * sin_u),)


@dataclass(frozen=True)
class _LigerRotaryMetadata:
    empty: bool
    num_heads: int
    head_dim: int


class _LigerRotary:
    """Single-tensor packed RoPE. A dummy one-head K keeps Q A2A launch-first."""

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, OpSavedState]:
        if x.numel() == 0:
            metadata = _LigerRotaryMetadata(True, x.shape[1] if x.ndim == 3 else 0, x.shape[-1])
            return x, OpSavedState((cos, sin), metadata)

        from liger_kernel.ops.rope import rope_forward

        query = x.transpose(0, 1).unsqueeze(0).contiguous()
        dummy_key = query.new_empty(query.shape[0], 1, query.shape[2], query.shape[3])
        query_out, _, saved_cos, saved_sin = rope_forward(
            query,
            dummy_key,
            cos.unsqueeze(0).contiguous(),
            sin.unsqueeze(0).contiguous(),
        )
        return (
            query_out.squeeze(0).transpose(0, 1).contiguous(),
            OpSavedState((saved_cos, saved_sin), _LigerRotaryMetadata(False, x.shape[1], x.shape[-1])),
        )

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor]:
        metadata = saved.metadata
        assert isinstance(metadata, _LigerRotaryMetadata)
        if metadata.empty:
            return (grad_output,)

        from liger_kernel.ops.rope import rope_backward

        grad_query = grad_output.transpose(0, 1).unsqueeze(0).contiguous()
        dummy_grad_key = grad_query.new_empty(grad_query.shape[0], 1, grad_query.shape[2], grad_query.shape[3])
        cos, sin = saved.tensors
        grad_query_out, _ = rope_backward(grad_query, dummy_grad_key, cos, sin)
        return (grad_query_out.squeeze(0).transpose(0, 1).contiguous(),)


class _NpuRotary:
    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> tuple[Tensor, OpSavedState]:
        if x.numel() == 0:
            return x, OpSavedState((cos, sin), _EagerRotaryMetadata(True))

        import torch_npu

        query = x.transpose(0, 1).unsqueeze(0)
        cos_u = cos.unsqueeze(0).unsqueeze(1)
        sin_u = sin.unsqueeze(0).unsqueeze(1)
        output = torch_npu.npu_rotary_mul(query, cos_u, sin_u)
        return output.squeeze(0).transpose(0, 1).contiguous(), OpSavedState((cos, sin), _EagerRotaryMetadata(False))

    def backward(self, grad_output: Tensor, saved: OpSavedState) -> tuple[Tensor]:
        return _EagerRotary().backward(grad_output, saved)


def _build_rotary(variant: str, impl_name: str) -> OpWrapper:
    if variant != "full":
        raise KeyError(f"Async Ulysses only supports full RoPE, got variant={variant!r}")
    if impl_name == "eager":
        return _EagerRotary()
    if impl_name == "liger_kernel":
        return _LigerRotary()
    if impl_name == "npu":
        if not IS_NPU_AVAILABLE:
            raise RuntimeError("rotary_pos_emb implementation 'npu' requires an NPU device.")
        return _NpuRotary()
    raise KeyError(f"No async RoPE wrapper for impl={impl_name!r}")


_BUILDERS: dict[str, Callable[[str, str], OpWrapper]] = {
    "rms_norm": _build_rms_norm,
    "rotary_pos_emb": _build_rotary,
}

_SUPPORTED_IMPLEMENTATIONS: dict[str, frozenset[str]] = {
    "rms_norm": frozenset({"eager", "liger_kernel", "npu"}),
    "rotary_pos_emb": frozenset({"eager", "liger_kernel", "npu"}),
}

_DEFAULT_VARIANTS: dict[str, str] = {
    "rms_norm": "standard",
    "rotary_pos_emb": "full",
}

_CONFIG_FIELDS: dict[str, str] = {
    "rms_norm": "rms_norm_implementation",
    "rotary_pos_emb": "rotary_pos_emb_implementation",
}

_WRAPPERS: dict[tuple[str, str, str], OpWrapper] = {}


def _normalize_spec(op: str | tuple[str, str]) -> tuple[str, str]:
    if isinstance(op, tuple):
        op_name, variant = op
        return op_name, variant
    if op not in _DEFAULT_VARIANTS:
        raise KeyError(f"Unsupported async Ulysses op {op!r}. Known: {sorted(_DEFAULT_VARIANTS)}")
    return op, _DEFAULT_VARIANTS[op]


def _impl_name(op_name: str) -> str:
    config = get_ops_config()
    if config is None:
        return "eager"
    return getattr(config, _CONFIG_FIELDS[op_name])


def get_op_wrapper(op_name: str, variant: str | None = None) -> OpWrapper:
    spec = (op_name, variant) if variant is not None else op_name
    op_name, variant = _normalize_spec(spec)
    impl_name = _impl_name(op_name)
    supported = _SUPPORTED_IMPLEMENTATIONS[op_name]
    if impl_name not in supported:
        raise KeyError(
            f"Async Ulysses has no {op_name} wrapper for implementation {impl_name!r}. Supported: {sorted(supported)}."
        )
    key = (op_name, variant, impl_name)
    wrapper = _WRAPPERS.get(key)
    if wrapper is None:
        wrapper = _BUILDERS[op_name](variant, impl_name)
        _WRAPPERS[key] = wrapper
    return wrapper

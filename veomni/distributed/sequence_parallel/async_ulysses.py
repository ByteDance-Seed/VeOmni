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

import importlib
import numbers
from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributed import ProcessGroup

from veomni.utils.device import IS_CUDA_AVAILABLE, IS_NPU_AVAILABLE
from veomni.utils.logging import get_logger

from .comm import get_ulysses_sequence_parallel_group, get_ulysses_sequence_parallel_world_size
from .ulysses import all_to_all_tensor
from .utils import padding_tensor_for_seqeunce_parallel, unpadding_tensor_for_seqeunce_parallel


if IS_NPU_AVAILABLE:
    import torch_npu  # noqa: F401  (used in the IS_CUDA_AVAILABLE-else branches)


if TYPE_CHECKING:
    from veomni.ops.kernels.fused_ulysses_projection.ws_push import WsPushDispatch

logger = get_logger(__name__)
fused_layer_norm_cuda = None


def divide_qkv_linear_weight(weight: Tensor, dim: int):
    return weight.chunk(3, dim=dim)


def divide_qkv_linear_bias(bias: Tensor, dim: int):
    if bias is not None:
        return bias.chunk(3, dim=dim)
    else:
        return None, None, None


def _split_fused_qkv_views(
    qkv_weight: Tensor,
    qkv_bias: Optional[Tensor],
    n_q: int,
    n_kv: int,
    head_dim: int,
) -> tuple[
    Tensor,
    Tensor,
    Tensor,
    Optional[Tensor],
    Optional[Tensor],
    Optional[Tensor],
    int,
    int,
]:
    """Slice a packed ``[n_q*hd | n_kv*hd | n_kv*hd, in_dim]`` ``qkv_weight``
    (and matching ``qkv_bias``) into the three q/k/v views the eager-fallback
    path and the backward GEMMs expect.
    """
    q_out = n_q * head_dim
    kv_out = n_kv * head_dim
    expected_out = q_out + 2 * kv_out
    if qkv_weight.shape[0] != expected_out:
        raise ValueError(
            f"qkv_weight.shape[0]={qkv_weight.shape[0]} != (n_q + 2*n_kv) * head_dim "
            f"= ({n_q} + 2*{n_kv}) * {head_dim} = {expected_out}"
        )
    if qkv_bias is not None and qkv_bias.shape[0] != expected_out:
        raise ValueError(
            f"qkv_bias.shape[0]={qkv_bias.shape[0]} != (n_q + 2*n_kv) * head_dim = {expected_out}"
        )
    q_w = qkv_weight[:q_out]
    k_w = qkv_weight[q_out : q_out + kv_out]
    v_w = qkv_weight[q_out + kv_out :]
    q_b = k_b = v_b = None
    if qkv_bias is not None:
        q_b = qkv_bias[:q_out]
        k_b = qkv_bias[q_out : q_out + kv_out]
        v_b = qkv_bias[q_out + kv_out :]
    return q_w, k_w, v_w, q_b, k_b, v_b, q_out, kv_out


def _qkv_via_ws_push(
    hidden_states: Tensor,
    dispatch: "WsPushDispatch",
    seq_dimension: int,
    unpadded_dim_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the WS-PUSH fused GEMM + Ulysses a2a and return ``(q, k, v)`` in the
    unified (already unpad'd + contiguous) shape the rest of forward expects.

    The kernel itself owns the clone (``ws_push_forward_impl`` returns
    detached copies by default), so this helper only does the
    unpad + ``.contiguous()`` finalization to mirror the eager branch.
    """
    from veomni.ops.kernels.fused_ulysses_projection.ws_push import ws_push_forward_impl

    q, k, v = ws_push_forward_impl(
        hidden_states,
        None,  # W_qkv unused when W_qkv_B is provided
        dispatch.state,
        W_qkv_B=dispatch.W_qkv_B,
        bias_B=dispatch.bias_B,
    )
    q = unpadding_tensor_for_seqeunce_parallel(q, seq_dimension, unpadded_dim_size)
    k = unpadding_tensor_for_seqeunce_parallel(k, seq_dimension, unpadded_dim_size)
    v = unpadding_tensor_for_seqeunce_parallel(v, seq_dimension, unpadded_dim_size)
    q = q.contiguous()
    k = k.contiguous()
    # v left as-is to mirror the eager branch.
    return q, k, v


def _qkv_via_eager(
    hidden_states: Tensor,
    q_weight: Tensor,
    q_bias: Optional[Tensor],
    k_weight: Tensor,
    k_bias: Optional[Tensor],
    v_weight: Tensor,
    v_bias: Optional[Tensor],
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_dimension: int,
    head_dimension: int,
    unpadded_dim_size: int,
    sp_group: Optional[ProcessGroup],
    need_repeat_kv: bool,
    n_repeat: int,
    batch_size: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Run the eager 3xF.linear + 3x async-a2a path. Mirrors the original
    behaviour: q & k a2a + unpad + ``.contiguous()`` before return; v a2a
    launches here but is collected inside this helper (no caller-side guard)."""
    # q projection + launch
    q = F.linear(hidden_states, q_weight, q_bias)
    q = q.view(batch_size, -1, num_q_heads, head_dim)
    q_res = all_to_all_tensor(q, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    # k projection + launch
    k = F.linear(hidden_states, k_weight, k_bias)
    k = k.view(batch_size, -1, num_kv_heads, head_dim)
    if need_repeat_kv:
        k = torch.repeat_interleave(k, dim=2, repeats=n_repeat)
    k_res = all_to_all_tensor(k, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    # v projection + launch
    v = F.linear(hidden_states, v_weight, v_bias)
    v = v.view(batch_size, -1, num_kv_heads, head_dim)
    if need_repeat_kv:
        v = torch.repeat_interleave(v, dim=2, repeats=n_repeat)
    v_res = all_to_all_tensor(v, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True)

    # collect q, k
    q = q_res()
    q = unpadding_tensor_for_seqeunce_parallel(q, seq_dimension, unpadded_dim_size)
    k = k_res()
    k = unpadding_tensor_for_seqeunce_parallel(k, seq_dimension, unpadded_dim_size)
    q = q.contiguous()
    k = k.contiguous()

    # collect v
    v = v_res()
    v = unpadding_tensor_for_seqeunce_parallel(v, seq_dimension, unpadded_dim_size)
    return q, k, v


class AsyncUlyssesQKVProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Tensor,
        seq_dimension: int,
        head_dimension: int,
        q_weight: Tensor,
        q_bias: Tensor,
        k_weight: Tensor,
        k_bias: Tensor,
        v_weight: Tensor,
        v_bias: Tensor,
        norm_type: str,
        norm_q_weight: Tensor,
        norm_q_bias: Tensor,
        norm_k_weight: Tensor,
        norm_k_bias: Tensor,
        normalized_shape: int,
        eps: float,
        unpadded_dim_size: int,
        head_dim: int,
        group: ProcessGroup,
        dispatch: Optional["WsPushDispatch"] = None,
        # --- Fused signature: when ``qkv_weight`` is provided, the legacy
        # q/k/v_weight/bias args must all be None. Views into qkv_weight are
        # derived locally for the eager fallback and backward GEMMs; the
        # backward returns ``grad_qkv_weight`` (not three separate grads).
        qkv_weight: Optional[Tensor] = None,
        qkv_bias: Optional[Tensor] = None,
        n_q: Optional[int] = None,
        n_kv: Optional[int] = None,
    ):
        # ---------- Signature dispatch ----------
        fused_signature = qkv_weight is not None
        if fused_signature:
            if any(t is not None for t in (q_weight, k_weight, v_weight, q_bias, k_bias, v_bias)):
                raise ValueError(
                    "AsyncUlyssesQKVProjection: pass either ``qkv_weight`` OR the "
                    "separate q/k/v_weight kwargs, not both."
                )
            if n_q is None or n_kv is None or head_dim is None:
                raise ValueError("AsyncUlyssesQKVProjection fused signature requires n_q, n_kv, head_dim.")
            (
                q_weight,
                k_weight,
                v_weight,
                q_bias,
                k_bias,
                v_bias,
                _q_out,
                _kv_out,
            ) = _split_fused_qkv_views(qkv_weight, qkv_bias, n_q, n_kv, head_dim)
            ctx._fused_signature = True
            ctx._fused_q_out = _q_out
            ctx._fused_kv_out = _kv_out
        else:
            ctx._fused_signature = False

        sp_group = get_ulysses_sequence_parallel_group() if group is None else group
        ulysses_size = get_ulysses_sequence_parallel_world_size()

        num_q_heads = q_weight.shape[0] // head_dim
        num_kv_heads = k_weight.shape[0] // head_dim
        batch_size = hidden_states.shape[0]

        assert num_q_heads % ulysses_size == 0, (
            f"num_query_heads ({num_q_heads}) must be divisible by ulysses_size ({ulysses_size})"
        )

        if ulysses_size > num_kv_heads:
            assert ulysses_size % num_kv_heads == 0, (
                f"ulysses_size ({ulysses_size}) must be divisible by num_key_value_heads ({num_kv_heads})"
            )
            ctx.need_repeat_kv = True
            ctx.n_repeat = ulysses_size // num_kv_heads
            ctx.original_num_kv_heads = num_kv_heads
        else:
            ctx.need_repeat_kv = False

        if dispatch is not None:
            from veomni.ops.kernels.fused_ulysses_projection.ws_push import (
                validate_ws_push_dispatch,
            )

            validate_ws_push_dispatch(
                dispatch,
                hidden_states,
                num_q_heads,
                num_kv_heads,
                seq_dimension,
                head_dimension,
                unpadded_dim_size,
                q_bias,
                k_bias,
                v_bias,
                group,
            )
            q, k, v = _qkv_via_ws_push(hidden_states, dispatch, seq_dimension, unpadded_dim_size)
        else:
            q, k, v = _qkv_via_eager(
                hidden_states,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                num_q_heads,
                num_kv_heads,
                head_dim,
                seq_dimension,
                head_dimension,
                unpadded_dim_size,
                sp_group,
                ctx.need_repeat_kv,
                ctx.n_repeat if ctx.need_repeat_kv else 1,
                batch_size,
            )

        # qk normalization (if needed)
        if norm_type is not None:
            if isinstance(normalized_shape, numbers.Integral):
                normalized_shape = (normalized_shape,)
            normalized_shape = torch.Size(normalized_shape)
            norm_q_weight = norm_q_weight.contiguous()
            norm_k_weight = norm_k_weight.contiguous()
            output_q, mean_q, invvar_q = None, None, None
            output_k, mean_k, invvar_k = None, None, None
            if IS_CUDA_AVAILABLE:
                global fused_layer_norm_cuda
                if fused_layer_norm_cuda is None:
                    fused_layer_norm_cuda = importlib.import_module("fused_layer_norm_cuda")
            if norm_type == "rmsnorm":
                if IS_CUDA_AVAILABLE:
                    output_q, invvar_q = fused_layer_norm_cuda.rms_forward_affine(
                        q, normalized_shape, norm_q_weight, eps
                    )
                    output_k, invvar_k = fused_layer_norm_cuda.rms_forward_affine(
                        k, normalized_shape, norm_k_weight, eps
                    )
                else:
                    output_q, invvar_q = torch_npu.npu_rms_norm(q, norm_q_weight, eps)
                    output_k, invvar_k = torch_npu.npu_rms_norm(k, norm_k_weight, eps)
            elif norm_type == "layernorm":
                output_q, mean_q, invvar_q = fused_layer_norm_cuda.forward_affine(
                    q, normalized_shape, norm_q_weight, norm_q_bias, eps
                )
                output_k, mean_k, invvar_k = fused_layer_norm_cuda.forward_affine(
                    k, normalized_shape, norm_k_weight, norm_k_bias, eps
                )
            else:
                raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
        else:
            output_q = q
            output_k = k
            mean_q = None
            mean_k = None
            invvar_q = None
            invvar_k = None

        # save ctx for backward
        ctx.sp_group = sp_group
        ctx.head_dimension = head_dimension
        ctx.seq_dimension = seq_dimension
        ctx.norm_type = norm_type
        ctx.normalized_shape = normalized_shape
        ctx.eps = eps
        ctx.save_for_backward(
            hidden_states,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            q,
            norm_q_weight,
            norm_q_bias,
            mean_q,
            invvar_q,
            k,
            norm_k_weight,
            norm_k_bias,
            mean_k,
            invvar_k,
        )

        return output_q, output_k, v

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        # get ctx for backward
        sp_group = ctx.sp_group
        seq_dimension = ctx.seq_dimension
        head_dimension = ctx.head_dimension
        norm_type = ctx.norm_type
        normalized_shape = ctx.normalized_shape
        need_repeat_kv = ctx.need_repeat_kv
        if need_repeat_kv:
            n_repeat = ctx.n_repeat
            original_num_kv_heads = ctx.original_num_kv_heads
        eps = ctx.eps
        (
            hidden_states,
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            q,
            norm_q_weight,
            norm_q_bias,
            mean_q,
            invvar_q,
            k,
            norm_k_weight,
            norm_k_bias,
            mean_k,
            invvar_k,
        ) = ctx.saved_tensors

        # initialize grads
        grad_hidden_states = None
        grad_q_weight = None
        grad_q_bias = None
        grad_k_weight = None
        grad_k_bias = None
        grad_v_weight = None
        grad_v_bias = None
        grad_norm_q_weight = None
        grad_norm_q_bias = None
        grad_norm_k_weight = None
        grad_norm_k_bias = None

        # v grad communication launch
        grad_v = grad_output[2].contiguous()
        grad_v = padding_tensor_for_seqeunce_parallel(grad_v, dim=seq_dimension)
        grad_v_res = all_to_all_tensor(
            grad_v,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # qk normalization backward (if needed)
        if norm_type is not None:
            if norm_type == "rmsnorm":
                if IS_CUDA_AVAILABLE:
                    grad_k, grad_norm_k_weight = fused_layer_norm_cuda.rms_backward_affine(
                        grad_output[1].contiguous(),
                        invvar_k,
                        k,
                        normalized_shape,
                        norm_k_weight,
                        eps,
                        False,
                    )
                    grad_q, grad_norm_q_weight = fused_layer_norm_cuda.rms_backward_affine(
                        grad_output[0].contiguous(),
                        invvar_q,
                        q,
                        normalized_shape,
                        norm_q_weight,
                        eps,
                        False,
                    )
                else:
                    grad_k, grad_norm_k_weight = torch_npu.npu_rms_norm_backward(
                        grad_output[1].contiguous(),
                        k,
                        norm_k_weight,
                        invvar_k,
                    )

                    grad_q, grad_norm_q_weight = torch_npu.npu_rms_norm_backward(
                        grad_output[0].contiguous(),
                        q,
                        norm_q_weight,
                        invvar_q,
                    )
            elif norm_type == "layernorm":
                grad_k, grad_norm_k_weight, grad_norm_k_bias = fused_layer_norm_cuda.backward_affine(
                    grad_output[1].contiguous(),
                    mean_k,
                    invvar_k,
                    k,
                    normalized_shape,
                    norm_k_weight,
                    norm_k_bias,
                    eps,
                    False,
                )
                grad_q, grad_norm_q_weight, grad_norm_q_bias = fused_layer_norm_cuda.backward_affine(
                    grad_output[0].contiguous(),
                    mean_q,
                    invvar_q,
                    q,
                    normalized_shape,
                    norm_q_weight,
                    norm_q_bias,
                    eps,
                    False,
                )
            else:
                raise NotImplementedError(f"{norm_type} is not supported in async-ulysses now!")
        else:
            grad_k = grad_output[1].contiguous()
            grad_q = grad_output[0].contiguous()
            grad_norm_k_weight = None
            grad_norm_q_weight = None

        # v grad communication collect
        grad_v = grad_v_res()
        if need_repeat_kv:
            grad_v = grad_v.reshape(
                grad_v.shape[0], grad_v.shape[1], original_num_kv_heads, n_repeat, grad_v.shape[-1]
            ).sum(dim=3)

        # k grad communication launch
        grad_k = padding_tensor_for_seqeunce_parallel(grad_k, dim=seq_dimension)
        grad_k_res = all_to_all_tensor(
            grad_k,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # v projection grad
        grad_v = grad_v.reshape(grad_v.shape[0], grad_v.shape[1], -1)
        grad_v_input = grad_v @ v_weight
        grad_v_weight = grad_v.transpose(-1, -2) @ hidden_states
        if v_bias is not None and ctx.needs_input_grad[7]:
            grad_v_bias = grad_v.sum(0)

        # k grad communication collect
        grad_k = grad_k_res()
        if need_repeat_kv:
            grad_k = grad_k.reshape(
                grad_k.shape[0], grad_k.shape[1], original_num_kv_heads, n_repeat, grad_k.shape[-1]
            ).sum(dim=3)

        # q grad communication launch
        grad_q = padding_tensor_for_seqeunce_parallel(grad_q, dim=seq_dimension)
        grad_q_res = all_to_all_tensor(
            grad_q,
            scatter_dim=seq_dimension,
            gather_dim=head_dimension,
            group=sp_group,
            async_op=True,
        )

        # k projection grad
        grad_k = grad_k.reshape(grad_k.shape[0], grad_k.shape[1], -1)
        grad_k_input = grad_k @ k_weight
        grad_k_weight = grad_k.transpose(-1, -2) @ hidden_states
        if k_bias is not None and ctx.needs_input_grad[5]:
            grad_k_bias = grad_k.sum(0)

        # q grad communication collect
        grad_q = grad_q_res()

        # q projection grad
        grad_q = grad_q.reshape(grad_q.shape[0], grad_q.shape[1], -1)
        grad_q_input = grad_q @ q_weight
        grad_q_weight = grad_q.transpose(-1, -2) @ hidden_states
        if q_bias is not None and ctx.needs_input_grad[3]:
            grad_q_bias = grad_q.sum(0)

        # grad
        grad_hidden_states = grad_q_input + grad_k_input + grad_v_input

        # ---------- Fused-signature grad assembly ----------
        # When the caller passed a single ``qkv_weight`` Parameter (see
        # FusedQKVLinear), grads must land on it as one tensor rather than
        # three separate q/k/v_weight grads.
        grad_qkv_weight = None
        grad_qkv_bias = None
        if getattr(ctx, "_fused_signature", False):
            _q_out = ctx._fused_q_out
            _kv_out = ctx._fused_kv_out
            total_out = _q_out + 2 * _kv_out
            # backward emitted: per-batch outer products kept separate (the
            # autograd engine reduces across leading dims during accumulate).
            ref = grad_q_weight  # legacy partial grad to crib shape & dtype from
            in_dim = ref.shape[-1]
            grad_qkv_weight = torch.empty(
                *ref.shape[:-2],
                total_out,
                in_dim,
                dtype=ref.dtype,
                device=ref.device,
            )
            grad_qkv_weight[..., :_q_out, :].copy_(grad_q_weight)
            grad_qkv_weight[..., _q_out : _q_out + _kv_out, :].copy_(grad_k_weight)
            grad_qkv_weight[..., _q_out + _kv_out :, :].copy_(grad_v_weight)
            if grad_q_bias is not None and grad_k_bias is not None and grad_v_bias is not None:
                ref_b = grad_q_bias
                grad_qkv_bias = torch.empty(
                    *ref_b.shape[:-1],
                    total_out,
                    dtype=ref_b.dtype,
                    device=ref_b.device,
                )
                grad_qkv_bias[..., :_q_out].copy_(grad_q_bias)
                grad_qkv_bias[..., _q_out : _q_out + _kv_out].copy_(grad_k_bias)
                grad_qkv_bias[..., _q_out + _kv_out :].copy_(grad_v_bias)
            # In fused mode the original q/k/v_weight inputs are None, so
            # their grad slots must also be None to avoid PyTorch crashing
            # on a non-None grad for a None input.
            grad_q_weight = grad_q_bias = None
            grad_k_weight = grad_k_bias = None
            grad_v_weight = grad_v_bias = None

        return (
            grad_hidden_states,
            None,
            None,
            grad_q_weight,
            grad_q_bias,
            grad_k_weight,
            grad_k_bias,
            grad_v_weight,
            grad_v_bias,
            None,
            grad_norm_q_weight,
            grad_norm_q_bias,
            grad_norm_k_weight,
            grad_norm_k_bias,
            None,
            None,
            None,
            None,
            None,
            None,  # dispatch
            grad_qkv_weight,
            grad_qkv_bias,
            None,  # n_q
            None,  # n_kv
        )


class AsyncUlyssesOutputProjection(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        hidden_states: Tensor,
        seq_dimension: int,
        head_dimension: int,
        proj_weight: Tensor,
        proj_bias: Tensor,
        unpadded_dim_size: int,
        group: ProcessGroup,
    ):
        sp_group = get_ulysses_sequence_parallel_group() if group is None else group

        # out projection
        hidden_states = padding_tensor_for_seqeunce_parallel(hidden_states, seq_dimension)
        hidden_states = all_to_all_tensor(
            hidden_states, scatter_dim=seq_dimension, gather_dim=head_dimension, group=sp_group
        )
        ctx.num_heads = hidden_states.shape[head_dimension]
        ctx.head_dim = hidden_states.shape[-1]

        hidden_states = hidden_states.view(hidden_states.shape[0], hidden_states.shape[1], -1)
        o = F.linear(hidden_states, proj_weight, proj_bias)

        # save ctx for backward
        ctx.sp_group = sp_group
        ctx.head_dimension = head_dimension
        ctx.seq_dimension = seq_dimension
        ctx.unpadded_dim_size = unpadded_dim_size

        ctx.save_for_backward(
            hidden_states,
            proj_weight,
            proj_bias,
        )

        return o

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor):
        # get ctx for backward
        sp_group = ctx.sp_group
        head_dimension = ctx.head_dimension
        seq_dimension = ctx.seq_dimension
        unpadded_dim_size = ctx.unpadded_dim_size
        (
            hidden_states,
            proj_weight,
            proj_bias,
        ) = ctx.saved_tensors
        num_heads = ctx.num_heads
        head_dim = ctx.head_dim

        # initialize grads
        grad_o = None
        grad_proj_weight = None
        grad_proj_bias = None

        # output grad
        grad_o = grad_output[0] @ (proj_weight)
        grad_o = grad_o.reshape(grad_o.shape[0], -1, num_heads, head_dim)

        # output grad communication launch
        grad_out_res = all_to_all_tensor(
            grad_o, scatter_dim=head_dimension, gather_dim=seq_dimension, group=sp_group, async_op=True
        )

        grad_proj_weight = grad_output[0].transpose(-1, -2) @ (hidden_states)
        if proj_bias is not None and ctx.needs_input_grad[3]:
            grad_proj_bias = grad_output[0].sum(0)

        # output grad communication collect
        grad_o = grad_out_res()
        grad_o = unpadding_tensor_for_seqeunce_parallel(grad_o, seq_dimension, unpadded_dim_size)

        return (
            grad_o,
            None,
            None,
            grad_proj_weight,
            grad_proj_bias,
            None,
            None,
        )


def async_ulysses_qkv_projection(
    hidden_states: Tensor = None,
    seq_dimension: int = None,
    head_dimension: int = None,
    q_weight: Tensor = None,
    q_bias: Optional[Tensor] = None,
    k_weight: Tensor = None,
    k_bias: Optional[Tensor] = None,
    v_weight: Tensor = None,
    v_bias: Optional[Tensor] = None,
    norm_type: str = None,
    norm_q_weight: Optional[Tensor] = None,
    norm_q_bias: Optional[Tensor] = None,
    norm_k_weight: Optional[Tensor] = None,
    norm_k_bias: Optional[Tensor] = None,
    normalized_shape: Optional[int] = None,
    eps: Optional[float] = None,
    unpadded_dim_size: int = None,
    head_dim: int = None,
    group: Optional[ProcessGroup] = None,
    dispatch: Optional["WsPushDispatch"] = None,
    # ---------- Fused signature (FusedQKVLinear callers) ----------
    qkv_weight: Optional[Tensor] = None,
    qkv_bias: Optional[Tensor] = None,
    n_q: Optional[int] = None,
    n_kv: Optional[int] = None,
):
    # Transparent dispatch: when the caller exposes a fused ``qkv_weight``
    # (the FusedQKVLinear path used by qwen3_vl / qwen3_vl_moe) and did not
    # explicitly hand us a ``WsPushDispatch``, consult the process-wide
    # active manager via ``WsPushDispatch.try_resolve_auto``. Returns None
    # when no manager is published / shape mismatch / non-symm-mem host /
    # the caller is on the legacy three-weight signature, in which case
    # forward falls through to the eager path.
    if dispatch is None and hidden_states is not None and qkv_weight is not None:
        from veomni.ops.kernels.fused_ulysses_projection.ws_push import WsPushDispatch

        dispatch = WsPushDispatch.try_resolve_auto(
            hidden_states,
            qkv_weight=qkv_weight,
            qkv_bias=qkv_bias,
        )

    return AsyncUlyssesQKVProjection.apply(
        hidden_states,
        seq_dimension,
        head_dimension,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        norm_type,
        norm_q_weight,
        norm_q_bias,
        norm_k_weight,
        norm_k_bias,
        normalized_shape,
        eps,
        unpadded_dim_size,
        head_dim,
        group,
        dispatch,
        qkv_weight,
        qkv_bias,
        n_q,
        n_kv,
    )


def async_ulysses_output_projection(
    hidden_states: Optional[Tensor] = None,
    seq_dimension: int = None,
    head_dimension: int = None,
    proj_weight: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    unpadded_dim_size: Optional[int] = None,
    group: Optional[ProcessGroup] = None,
):
    return AsyncUlyssesOutputProjection.apply(
        hidden_states,
        seq_dimension,
        head_dimension,
        proj_weight,
        proj_bias,
        unpadded_dim_size,
        group,
    )

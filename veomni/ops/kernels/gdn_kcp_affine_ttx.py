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

"""Ascend TTX fused local-affine pre-scan for KCP.

Scope lock: only the LTR ``{he, M}`` recurrence. Does **not** touch AG /
zigzag / prefix-merge / ``S_init``.

INV-7: kernel may mix bf16/fp32 internally; returned ``he``/``M`` are **fp32**.

Grid (default)::

    program = (segment, head, output_column_tile)
    X = [he | M]   # last-dim concat, width V+K
    BC = 16|32     # columns per program (UB-resident K×BC fp32 slab)

Packed samples launch as one grid. The public dispatcher is locked to the
BC8/M1 backward contract and never falls back to a torch implementation.
"""

from __future__ import annotations

import weakref
from dataclasses import dataclass
from operator import index
from typing import Optional, Sequence, Tuple

import torch
from torch import Tensor

from veomni.ops.kernels.gated_delta_rule.normalization import producer_dtype_l2norm


try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover
    triton = None  # type: ignore
    tl = None  # type: ignore
    _TRITON_OK = False


def _npu_ready() -> bool:
    return bool(getattr(torch, "npu", None)) and torch.npu.is_available()


@dataclass(frozen=True)
class TtxBc8M1Config:
    """Typed, immutable production kernel configuration."""

    forward_column_tile: int = 32
    backward_time_tile: int = 128
    backward_replay_column_tile: int = 8
    state_dtype: torch.dtype = torch.float32


TTX_BC8_M1_CONFIG = TtxBc8M1Config()


# The same packed CU tensor is shared by every GDN layer in one model forward.
# Cache its immutable host representation by Tensor identity and version so the
# custom autograd Function never introduces a per-layer device-to-host sync.
_CU_HOST_POINTS_CACHE: dict[int, tuple[weakref.ReferenceType, int, Tuple[int, ...]]] = {}


def _copy_cu_points_to_host(cu_seqlens: Tensor) -> Tuple[int, ...]:
    """Materialize packed boundaries once for one live metadata Tensor."""
    return tuple(int(point) for point in cu_seqlens.detach().cpu().tolist())


def _cached_host_cu_points(cu_seqlens: Tensor) -> Tuple[int, ...]:
    """Return versioned immutable host boundaries without per-layer D2H sync."""
    try:
        version = int(cu_seqlens._version)
    except RuntimeError:
        # Inference tensors do not expose a version counter. They are not the
        # production training path, so fail safe by avoiding a stale cache.
        return _copy_cu_points_to_host(cu_seqlens)

    identity = id(cu_seqlens)
    cached = _CU_HOST_POINTS_CACHE.get(identity)
    if cached is not None and cached[0]() is cu_seqlens and cached[1] == version:
        return cached[2]

    points = _copy_cu_points_to_host(cu_seqlens)

    def remove_dead_tensor(reference, *, tensor_id=identity):
        current = _CU_HOST_POINTS_CACHE.get(tensor_id)
        if current is not None and current[0] is reference:
            _CU_HOST_POINTS_CACHE.pop(tensor_id, None)

    reference = weakref.ref(cu_seqlens, remove_dead_tensor)
    _CU_HOST_POINTS_CACHE[identity] = (reference, version, points)
    return points


def _normalize_host_cu_points(cu_seqlens_list: Sequence[int]) -> Tuple[int, ...]:
    """Copy canonical host boundaries without accepting lossy coercions."""
    if isinstance(cu_seqlens_list, Tensor):
        raise TypeError("canonical host cu_seqlens_list must be a Python integer sequence, not a Tensor")
    points = []
    for point in cu_seqlens_list:
        if isinstance(point, bool) or isinstance(point, Tensor):
            raise TypeError("canonical host cu_seqlens_list entries must be exact integers")
        try:
            points.append(index(point))
        except TypeError as exc:
            raise TypeError("canonical host cu_seqlens_list entries must be exact integers") from exc
    return tuple(points)


def validate_ttx_bc8_m1_contract() -> TtxBc8M1Config:
    """Return the fixed production contract without mutating process globals."""
    config = TTX_BC8_M1_CONFIG
    if config.forward_column_tile != 32:
        raise RuntimeError("ttx_bc8_m1 forward column tile must remain 32")
    if config.backward_time_tile != 128 or config.backward_replay_column_tile != 8:
        raise RuntimeError("ttx_bc8_m1 backward tiles must remain BT=128 and BC=8")
    if config.state_dtype is not torch.float32:
        raise RuntimeError("ttx_bc8_m1 recurrent-state boundary must remain float32")
    return config


def validate_ttx_bc8_m1_shape(*, k_dim: int, v_dim: int) -> None:
    """Validate the intersection of the frozen forward/backward shape domains."""

    config = validate_ttx_bc8_m1_contract()
    if (
        k_dim <= 0
        or v_dim <= 0
        or k_dim > 128
        or v_dim > 128
        or (k_dim & (k_dim - 1)) != 0
        or v_dim % config.forward_column_tile != 0
        or k_dim % config.forward_column_tile != 0
    ):
        raise RuntimeError(
            "ttx_bc8_m1 forward/backward common domain requires 0<K,V<=128, "
            f"power-of-two K, and K/V divisible by 32; got K={k_dim}, V={v_dim}"
        )


def validate_ttx_bc8_m1_cu_seqlens(
    cu_seqlens: Tensor | Sequence[int],
    *,
    batch_size: int,
    token_count: int,
    cu_seqlens_list: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Validate packed CU boundaries before any TTX kernel can consume them."""
    if batch_size != 1:
        raise ValueError("ttx_bc8_m1 varlen affine summary expects batch=1 packed layout")
    if isinstance(cu_seqlens, Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int32:
            raise ValueError("ttx_bc8_m1 cu_seqlens must be a 1D int32 tensor with at least two boundaries")
        if cu_seqlens.device.type not in {"cpu", "npu"}:
            raise ValueError("ttx_bc8_m1 cu_seqlens must be a host or same-runtime NPU tensor")
        if cu_seqlens_list is None:
            # Public callers without canonical host metadata retain a safe,
            # versioned fallback.  The model path always supplies the host
            # list and therefore never synchronizes this device tensor.
            points = _cached_host_cu_points(cu_seqlens)
        else:
            points = _normalize_host_cu_points(cu_seqlens_list)
            if cu_seqlens.numel() != len(points):
                raise ValueError(
                    "ttx_bc8_m1 device/host cu_seqlens boundary count mismatch: "
                    f"device={cu_seqlens.numel()} host={len(points)}"
                )
    else:
        if cu_seqlens_list is not None:
            raise ValueError("cu_seqlens_list is only valid with a cu_seqlens Tensor")
        points = _normalize_host_cu_points(cu_seqlens)
    if not points or points[0] != 0:
        raise ValueError(f"ttx_bc8_m1 cu_seqlens must start at 0, got {points[:3]}")
    if any(right < left for left, right in zip(points, points[1:])):
        raise ValueError(f"ttx_bc8_m1 cu_seqlens must be nondecreasing, got {points}")
    if points[-1] != token_count:
        raise ValueError(f"ttx_bc8_m1 cu_seqlens must end at T={token_count}, got {points[-3:]}")
    return points


def validate_ttx_bc8_m1_inputs(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
) -> None:
    """Validate every rank before any rank enters the affine all-gather."""

    validate_ttx_bc8_m1_contract()
    expected_prefix = tuple(key.shape[:3])
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")
    if tuple(value.shape[:3]) != expected_prefix:
        raise ValueError(f"key/value [B,T,H] mismatch: key={tuple(key.shape)} value={tuple(value.shape)}")
    if tuple(g.shape) != expected_prefix or tuple(beta.shape) != expected_prefix:
        raise ValueError(
            f"g/beta must match key [B,T,H]={expected_prefix}, got g={tuple(g.shape)} beta={tuple(beta.shape)}"
        )
    if any(operand.device != key.device for operand in (value, g, beta)) or key.device.type != "npu":
        raise RuntimeError("KCP ttx_bc8_m1 requires key/value/g/beta on one Ascend NPU device")
    if not all(operand.is_floating_point() for operand in (key, value, g, beta)):
        raise TypeError("key/value/g/beta must be floating-point tensors")
    k_dim = int(key.shape[-1])
    v_dim = int(value.shape[-1])
    validate_ttx_bc8_m1_shape(k_dim=k_dim, v_dim=v_dim)
    if cu_seqlens is not None:
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2 or cu_seqlens.dtype != torch.int32:
            raise ValueError("ttx_bc8_m1 cu_seqlens must be a 1D int32 tensor with at least two boundaries")
        if cu_seqlens.device.type not in {"cpu", "npu"}:
            raise ValueError("ttx_bc8_m1 cu_seqlens must be a host or same-runtime NPU tensor")
        if cu_seqlens.device.type == "npu" and cu_seqlens.device != key.device:
            raise ValueError("NPU ttx_bc8_m1 cu_seqlens must be on the same device as key")
    if not _TRITON_OK or not _npu_ready():
        raise RuntimeError("KCP ttx_bc8_m1 requires Triton on an available Ascend NPU")


def _prepare_ttx_forward_operands(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    use_qk_l2norm: bool,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Prepare the exact operands consumed by the TTX forward kernels.

    The local GDN core keeps ``key``/``value``/``beta`` in their producer
    dtype while ``g`` is commonly fp32.  Triton converts every load to fp32
    for the recurrence, so changing the storage dtype here is both unnecessary
    and mathematically observable (most importantly for ``exp(g)``).  Preserve
    each operand dtype and only make the post-normalization views contiguous.
    """
    expected_prefix = tuple(key.shape[:3])
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")
    if tuple(value.shape[:3]) != expected_prefix:
        raise ValueError(f"key/value [B,T,H] mismatch: key={tuple(key.shape)} value={tuple(value.shape)}")
    if tuple(g.shape) != expected_prefix or tuple(beta.shape) != expected_prefix:
        raise ValueError(
            f"g/beta must match key [B,T,H]={expected_prefix}, got g={tuple(g.shape)} beta={tuple(beta.shape)}"
        )
    if any(operand.device != key.device for operand in (value, g, beta)):
        raise ValueError("key/value/g/beta must be on the same device")
    if not all(operand.is_floating_point() for operand in (key, value, g, beta)):
        raise TypeError("key/value/g/beta must be floating-point tensors")

    if use_qk_l2norm:
        raise ValueError(
            "TTX operands must be pre-normalized with producer_dtype_l2norm; "
            "the internal fp32 normalization path is forbidden"
        )
    return (
        key.contiguous(),
        value.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        key.new_empty(0, dtype=torch.float32),
        key.new_empty(0, dtype=torch.float32),
    )


def ttx_bc8_m1_torch_reference(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Torch oracle for the production TTX operand and recurrence contract."""
    from veomni.distributed.context_parallel.gdn_kcp import local_affine_summary_fused_torch

    if use_qk_l2norm:
        key = producer_dtype_l2norm(key, eps=eps)
    operands = _prepare_ttx_forward_operands(
        key,
        value,
        g,
        beta,
        use_qk_l2norm=False,
        eps=eps,
    )
    return local_affine_summary_fused_torch(
        *operands[:4],
        cu_seqlens=cu_seqlens,
        use_qk_l2norm=False,
        eps=eps,
    )


if _TRITON_OK:

    @triton.jit(do_not_specialize=["T"])
    def _local_affine_summary_head_grid_reference_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        M_ptr,
        T,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BK: tl.constexpr,
        BV: tl.constexpr,
    ):
        """Reference head-grid kernel retained for numerical development tests."""
        i_h = tl.program_id(0)

        o_k0 = tl.arange(0, BK)
        o_v0 = tl.arange(0, BV)
        NK = (K + BK - 1) // BK
        NV = (V + BV - 1) // BV
        for i_k in range(0, NK):
            for i_v in range(0, NV):
                offs_k = i_k * BK + o_k0
                offs_v = i_v * BV + o_v0
                mask = (offs_k[:, None] < K) & (offs_v[None, :] < V)
                tl.store(
                    he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                    tl.zeros([BK, BV], dtype=tl.float32),
                    mask=mask,
                )
        for i_r in range(0, NK):
            for i_c in range(0, NK):
                offs_r = i_r * BK + o_k0
                offs_c = i_c * BK + o_k0
                mask = (offs_r[:, None] < K) & (offs_c[None, :] < K)
                eye = tl.where(
                    (offs_r[:, None] == offs_c[None, :]) & mask,
                    1.0,
                    0.0,
                ).to(tl.float32)
                tl.store(
                    M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                    eye,
                    mask=mask,
                )

        p_k = k_ptr + i_h * K
        p_v = v_ptr + i_h * V
        p_g = g_ptr + i_h
        p_beta = beta_ptr + i_h
        stride_t_k = H * K
        stride_t_v = H * V
        stride_t_h = H

        for _t in range(0, T):
            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eg = tl.exp(b_g)

            for i_v in range(0, NV):
                offs_v = i_v * BV + tl.arange(0, BV)
                mask_v = offs_v < V
                b_khe = tl.zeros([BV], dtype=tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_khe += tl.sum(b_k[:, None] * b_he, 0)
                b_v = tl.load(p_v + offs_v, mask=mask_v, other=0).to(tl.float32)
                for i_k in range(0, NK):
                    offs_k = i_k * BK + tl.arange(0, BK)
                    mask_k = offs_k < K
                    b_k = tl.load(p_k + offs_k, mask=mask_k, other=0).to(tl.float32)
                    b_he = tl.load(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        mask=mask_k[:, None] & mask_v[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_he = b_eg * (b_he - b_beta * b_k[:, None] * b_khe[None, :]) + (
                        b_beta * b_k[:, None] * b_v[None, :]
                    )
                    tl.store(
                        he_ptr + (i_h * K * V) + offs_k[:, None] * V + offs_v[None, :],
                        b_he,
                        mask=mask_k[:, None] & mask_v[None, :],
                    )

            for i_c in range(0, NK):
                offs_c = i_c * BK + tl.arange(0, BK)
                mask_c = offs_c < K
                b_kM = tl.zeros([BK], dtype=tl.float32)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_kM += tl.sum(b_k[:, None] * b_M, 0)
                for i_r in range(0, NK):
                    offs_r = i_r * BK + tl.arange(0, BK)
                    mask_r = offs_r < K
                    b_k = tl.load(p_k + offs_r, mask=mask_r, other=0).to(tl.float32)
                    b_M = tl.load(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        mask=mask_r[:, None] & mask_c[None, :],
                        other=0.0,
                    ).to(tl.float32)
                    b_M = b_eg * (b_M - b_beta * b_k[:, None] * b_kM[None, :])
                    tl.store(
                        M_ptr + (i_h * K * K) + offs_r[:, None] * K + offs_c[None, :],
                        b_M,
                        mask=mask_r[:, None] & mask_c[None, :],
                    )

            p_k += stride_t_k
            p_v += stride_t_v
            p_g += stride_t_h
            p_beta += stride_t_h

    @triton.jit
    def _local_affine_he_coltile_kernel(
        k_ptr,
        v_ptr,
        g_ptr,
        beta_ptr,
        he_ptr,
        starts_ptr,
        ends_ptr,
        H: tl.constexpr,
        K: tl.constexpr,
        V: tl.constexpr,
        BC: tl.constexpr,
    ):
        """(segment, head, he_column_tile): UB-resident he[K,BC], one store."""
        i_s = tl.program_id(0)
        i_h = tl.program_id(1)
        i_c = tl.program_id(2)

        start = tl.load(starts_ptr + i_s)
        end = tl.load(ends_ptr + i_s)
        T = end - start

        col0 = i_c * BC
        offs_c = col0 + tl.arange(0, BC)
        offs_k = tl.arange(0, K)
        mask_c = offs_c < V

        stride_t_k = H * K
        stride_t_v = H * V
        stride_t_h = H
        p_k = k_ptr + start * stride_t_k + i_h * K
        p_v = v_ptr + start * stride_t_v + i_h * V
        p_g = g_ptr + start * stride_t_h + i_h
        p_beta = beta_ptr + start * stride_t_h + i_h

        he_ub = tl.zeros([K, BC], dtype=tl.float32)
        for _t in range(0, T):
            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eg = tl.exp(b_g)
            b_k = tl.load(p_k + offs_k).to(tl.float32)
            b_v = tl.load(p_v + offs_c, mask=mask_c, other=0).to(tl.float32)
            b_khe = tl.sum(b_k[:, None] * he_ub, 0)
            he_ub = b_eg * (he_ub - b_beta * b_k[:, None] * b_khe[None, :]) + (b_beta * b_k[:, None] * b_v[None, :])
            p_k += stride_t_k
            p_v += stride_t_v
            p_g += stride_t_h
            p_beta += stride_t_h

        out_base = he_ptr + (((i_s * H + i_h) * K) * V) + offs_k[:, None] * V + offs_c[None, :]
        tl.store(out_base, he_ub, mask=mask_c[None, :])

    @triton.jit
    def _local_affine_M_coltile_kernel(
        k_ptr,
        g_ptr,
        beta_ptr,
        M_ptr,
        starts_ptr,
        ends_ptr,
        H: tl.constexpr,
        K: tl.constexpr,
        BC: tl.constexpr,
    ):
        """(segment, head, M_column_tile): UB-resident M[K,BC], one store."""
        i_s = tl.program_id(0)
        i_h = tl.program_id(1)
        i_c = tl.program_id(2)

        start = tl.load(starts_ptr + i_s)
        end = tl.load(ends_ptr + i_s)
        T = end - start

        col0 = i_c * BC
        offs_c = col0 + tl.arange(0, BC)
        offs_k = tl.arange(0, K)
        mask_c = offs_c < K

        stride_t_k = H * K
        stride_t_h = H
        p_k = k_ptr + start * stride_t_k + i_h * K
        p_g = g_ptr + start * stride_t_h + i_h
        p_beta = beta_ptr + start * stride_t_h + i_h

        M_ub = tl.where(
            (offs_k[:, None] == offs_c[None, :]) & mask_c[None, :],
            1.0,
            0.0,
        ).to(tl.float32)

        for _t in range(0, T):
            b_g = tl.load(p_g).to(tl.float32)
            b_beta = tl.load(p_beta).to(tl.float32)
            b_eg = tl.exp(b_g)
            b_k = tl.load(p_k + offs_k).to(tl.float32)
            b_kM = tl.sum(b_k[:, None] * M_ub, 0)
            M_ub = b_eg * (M_ub - b_beta * b_k[:, None] * b_kM[None, :])
            p_k += stride_t_k
            p_g += stride_t_h
            p_beta += stride_t_h

        out_base = M_ptr + (((i_s * H + i_h) * K) * K) + offs_k[:, None] * K + offs_c[None, :]
        tl.store(out_base, M_ub, mask=mask_c[None, :])


def ttx_local_affine_he_m(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Fused LTR affine on one contiguous segment → fp32 ``he``, ``M``."""
    if not _TRITON_OK:
        raise RuntimeError("triton unavailable for TTX local-affine kernel")
    if not _npu_ready():
        raise RuntimeError("Ascend NPU unavailable for TTX local-affine kernel")
    if key.ndim != 3 or value.ndim != 3:
        raise ValueError(f"expected [T,H,D], got key={tuple(key.shape)} value={tuple(value.shape)}")

    t_len, num_heads, k_dim = int(key.shape[0]), int(key.shape[1]), int(key.shape[2])
    v_dim = int(value.shape[-1])
    if t_len == 0:
        he = torch.zeros(num_heads, k_dim, v_dim, device=key.device, dtype=torch.float32)
        M = torch.eye(k_dim, device=key.device, dtype=torch.float32).expand(num_heads, -1, -1).contiguous()
        return he, M

    # Route through batched col-tile path (S=1).
    hm = ttx_local_affine_summary(
        key.unsqueeze(0),
        value.unsqueeze(0),
        g.unsqueeze(0),
        beta.unsqueeze(0),
        cu_seqlens=None,
        use_qk_l2norm=False,
    )
    # hm: [1,H,K,V+K]
    from veomni.distributed.context_parallel.gdn_kcp import unpack_affine_hm

    he, M = unpack_affine_hm(hm[0], v_dim=v_dim)
    return he, M


def _launch_coltile(
    k_c: Tensor,
    v_c: Tensor,
    g_c: Tensor,
    b_c: Tensor,
    starts: Tensor,
    ends: Tensor,
    *,
    num_segs: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
    bc: int,
) -> Tuple[Tensor, Tensor]:
    """Packed [T,H,D] inputs + device starts/ends → he/M [S,H,K,*]."""
    assert _TRITON_OK
    if k_dim % bc != 0 or v_dim % bc != 0:
        raise RuntimeError(f"K={k_dim} V={v_dim} must be divisible by BC={bc}")
    if k_dim > 128:
        raise RuntimeError(f"col-tile affine requires K<=128 for UB slab, got K={k_dim}")

    he = torch.empty(num_segs, num_heads, k_dim, v_dim, device=k_c.device, dtype=torch.float32)
    M = torch.empty(num_segs, num_heads, k_dim, k_dim, device=k_c.device, dtype=torch.float32)

    nc_he = v_dim // bc
    nc_m = k_dim // bc
    _local_affine_he_coltile_kernel[(num_segs, num_heads, nc_he)](
        k_c,
        v_c,
        g_c,
        b_c,
        he,
        starts,
        ends,
        H=num_heads,
        K=k_dim,
        V=v_dim,
        BC=bc,
    )
    _local_affine_M_coltile_kernel[(num_segs, num_heads, nc_m)](
        k_c,
        g_c,
        b_c,
        M,
        starts,
        ends,
        H=num_heads,
        K=k_dim,
        BC=bc,
    )
    return he, M


def _launch_head_grid_reference(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
) -> Tuple[Tensor, Tensor]:
    """Per-segment head-only reference grid (not used by production dispatch)."""
    t_len, num_heads, k_dim = int(key.shape[0]), int(key.shape[1]), int(key.shape[2])
    v_dim = int(value.shape[-1])
    k_c = key.contiguous()
    v_c = value.contiguous()
    g_c = g.contiguous()
    b_c = beta.contiguous()
    he = torch.empty(num_heads, k_dim, v_dim, device=k_c.device, dtype=torch.float32)
    M = torch.empty(num_heads, k_dim, k_dim, device=k_c.device, dtype=torch.float32)

    def _pow2_ge(n: int) -> int:
        p = 16
        while p < n:
            p <<= 1
        return p

    bk = _pow2_ge(min(64, k_dim) if k_dim <= 64 else 64)
    bv = _pow2_ge(min(64, v_dim) if v_dim <= 64 else 64)
    _local_affine_summary_head_grid_reference_kernel[(num_heads,)](
        k_c,
        v_c,
        g_c,
        b_c,
        he,
        M,
        t_len,
        H=num_heads,
        K=k_dim,
        V=v_dim,
        BK=bk,
        BV=bv,
    )
    return he, M


def _ttx_local_affine_summary_fwd(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Non-differentiable TTX body (writes ``hm`` via ``torch.empty`` + kernel)."""
    from veomni.distributed.context_parallel.gdn_kcp import pack_affine_hm

    if not _TRITON_OK:
        raise RuntimeError("triton unavailable for TTX local-affine kernel")
    if not _npu_ready():
        raise RuntimeError("Ascend NPU unavailable for TTX local-affine kernel")
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")

    device = key.device
    if device.type != "npu":
        # Host inject paths: move once, compute, move hm back.
        device = torch.device("npu")
        key = key.to(device)
        value = value.to(device)
        g = g.to(device)
        beta = beta.to(device)
        if cu_seqlens is not None:
            cu_seqlens = cu_seqlens.to(device)
        return_to = "cpu"
    else:
        return_to = None

    k_c, v_c, gg_c, bb_c, _key_rstd, _normalized_key_fp32 = _prepare_ttx_forward_operands(
        key,
        value,
        g,
        beta,
        use_qk_l2norm=use_qk_l2norm,
        eps=eps,
    )

    bsz = int(k_c.shape[0])
    t_total = int(k_c.shape[1])
    num_heads = int(k_c.shape[2])
    k_dim = int(k_c.shape[3])
    v_dim = int(v_c.shape[-1])

    if cu_seqlens is None:
        # One segment per batch row.
        starts = torch.arange(0, bsz * t_total, t_total, device=device, dtype=torch.int32)
        ends = starts + t_total
        num_segs = bsz
        # Merge batch into token axis for kernel: [B*T,H,D]
        k_flat = k_c.reshape(bsz * t_total, num_heads, k_dim)
        v_flat = v_c.reshape(bsz * t_total, num_heads, v_dim)
        g_flat = gg_c.reshape(bsz * t_total, num_heads)
        b_flat = bb_c.reshape(bsz * t_total, num_heads)
    else:
        if bsz != 1:
            raise ValueError("varlen affine summary expects batch=1 packed layout")
        # Device-resident boundaries — no .cpu().tolist().
        if cu_seqlens.device.type != device.type:
            cu_seqlens = cu_seqlens.to(device)
        cu_i = cu_seqlens.to(dtype=torch.int32).reshape(-1)
        num_segs = int(cu_i.numel()) - 1
        if num_segs <= 0:
            raise ValueError("cu_seqlens must contain at least one segment")
        starts = cu_i[:-1].contiguous()
        ends = cu_i[1:].contiguous()
        k_flat = k_c[0]
        v_flat = v_c[0]
        g_flat = gg_c[0]
        b_flat = bb_c[0]

    config = validate_ttx_bc8_m1_contract()
    bc = config.forward_column_tile
    if k_dim > 128 or v_dim % bc != 0 or k_dim % bc != 0:
        raise RuntimeError(f"ttx_bc8_m1 supports K<=128 with K and V divisible by 32; got K={k_dim}, V={v_dim}")
    he, M = _launch_coltile(
        k_flat,
        v_flat,
        g_flat,
        b_flat,
        starts,
        ends,
        num_segs=num_segs,
        num_heads=num_heads,
        k_dim=k_dim,
        v_dim=v_dim,
        bc=bc,
    )
    out = [pack_affine_hm(he[s], M[s]) for s in range(num_segs)]
    hm = torch.stack(out, dim=0)

    if hm.dtype != torch.float32:
        raise RuntimeError(f"INV-7: TTX hm must be float32, got {hm.dtype}")
    if return_to == "cpu":
        hm = hm.cpu()
    return hm


class _TtxLocalAffineSummaryFn(torch.autograd.Function):
    """TTX forward with the frozen BC8/M1 analytical backward."""

    @staticmethod
    def forward(
        ctx,
        key: Tensor,
        value: Tensor,
        g: Tensor,
        beta: Tensor,
        cu_marker: Optional[Tensor],
        cu_pts: Optional[Tuple[int, ...]],
        eps: float,
    ) -> Tensor:
        ctx.use_qk_l2norm = False
        ctx.eps = float(eps)
        ctx.has_cu = cu_marker is not None
        cu_seqlens = cu_marker if ctx.has_cu else None
        if ctx.has_cu != (cu_pts is not None):
            raise ValueError("TTX cu_seqlens tensor and immutable host points must be provided together")
        if cu_pts is not None:
            ctx.cu_pts = validate_ttx_bc8_m1_cu_seqlens(
                cu_marker,
                batch_size=int(key.shape[0]),
                token_count=int(key.shape[1]),
                cu_seqlens_list=cu_pts,
            )
        else:
            ctx.cu_pts = None
        key_operand, value_operand, g_operand, beta_operand, _key_rstd, _normalized_key_fp32 = (
            _prepare_ttx_forward_operands(
                key,
                value,
                g,
                beta,
                use_qk_l2norm=ctx.use_qk_l2norm,
                eps=ctx.eps,
            )
        )
        # Backward must replay the exact storage operands used by forward.
        # Normalization, when requested by the public entrypoint, lives outside
        # this custom Function and therefore owns its VJP in the normal graph.
        ctx.save_for_backward(
            key_operand,
            value_operand,
            g_operand,
            beta_operand,
            cu_marker,
        )
        with torch.no_grad():
            hm = _ttx_local_affine_summary_fwd(
                key_operand,
                value_operand,
                g_operand,
                beta_operand,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm=False,
                eps=ctx.eps,
            )
        return hm

    @staticmethod
    def backward(ctx, grad_hm: Tensor):
        (
            key_operand,
            value_operand,
            g_operand,
            beta_operand,
            cu_marker,
        ) = ctx.saved_tensors
        cu_seqlens = cu_marker if ctx.has_cu else None
        cu_pts = getattr(ctx, "cu_pts", None)
        from veomni.ops.kernels.gdn_kcp_affine_ttx_bwd import ttx_local_affine_analytical_bwd

        gk, gv, gg, gb = ttx_local_affine_analytical_bwd(
            key_operand,
            value_operand,
            g_operand,
            beta_operand,
            grad_hm.contiguous(),
            cu_seqlens=cu_seqlens,
            cu_pts=cu_pts,
            use_qk_l2norm=False,
            eps=ctx.eps,
        )
        return gk, gv, gg, gb, None, None, None


def ttx_local_affine_summary(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    cu_seqlens_list: Optional[Sequence[int]] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Run the fp32-boundary TTX BC8/M1 affine summary."""
    operands = (key, value, g, beta)
    if any(operand.device != key.device for operand in operands[1:]) or key.device.type != "npu":
        raise RuntimeError("KCP ttx_bc8_m1 autograd requires key/value/g/beta on one Ascend NPU device")
    validate_ttx_bc8_m1_inputs(key, value, g, beta, cu_seqlens=cu_seqlens)
    if cu_seqlens is None:
        if cu_seqlens_list is not None:
            raise ValueError("cu_seqlens_list requires a cu_seqlens Tensor")
        cu_pts = None
    else:
        cu_pts = validate_ttx_bc8_m1_cu_seqlens(
            cu_seqlens,
            batch_size=int(key.shape[0]),
            token_count=int(key.shape[1]),
            cu_seqlens_list=cu_seqlens_list,
        )
    if use_qk_l2norm:
        # Normalize outside the custom Function so KCP and the local GDR core
        # can share the producer-dtype expression and its exact autograd graph.
        key = producer_dtype_l2norm(key, eps=eps)
    return _TtxLocalAffineSummaryFn.apply(key, value, g, beta, cu_seqlens, cu_pts, float(eps))


_TTX_FORWARD_BACKWARD_WARMUP_CACHE: set[tuple] = set()


def warmup_ttx_bc8_m1_forward_backward(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
) -> None:
    """Compile and execute the production TTX forward+VJP once per signature.

    KCP calls this on every CP rank before its first readiness collective.
    That includes terminal/inactive ranks which may only become scan owners in
    a later dynamic batch.  Warming the analytical backward here also moves its
    lazy module import and first Triton launches before any AG/RS or ownership
    A2A can be entered, so a catchable failure is coordinated fail-closed.
    """

    validate_ttx_bc8_m1_inputs(key, value, g, beta, cu_seqlens=None)
    signature = (
        key.device.type,
        key.device.index,
        int(key.shape[2]),
        int(key.shape[3]),
        int(value.shape[3]),
        key.dtype,
        value.dtype,
        g.dtype,
        beta.dtype,
    )
    if signature in _TTX_FORWARD_BACKWARD_WARMUP_CACHE:
        return

    # Reentrant activation checkpointing runs its first forward under
    # ``torch.no_grad()``.  The readiness warmup must nevertheless exercise the
    # real custom backward before any rank can enter KCP collectives.
    with torch.inference_mode(False), torch.enable_grad():
        token_count = 128
        num_heads = int(key.shape[2])
        key_dim = int(key.shape[3])
        value_dim = int(value.shape[3])
        warm_key = torch.zeros(
            1, token_count, num_heads, key_dim, device=key.device, dtype=key.dtype, requires_grad=True
        )
        warm_value = torch.zeros(
            1, token_count, num_heads, value_dim, device=value.device, dtype=value.dtype, requires_grad=True
        )
        warm_g = torch.zeros(1, token_count, num_heads, device=g.device, dtype=g.dtype, requires_grad=True)
        warm_beta = torch.zeros(1, token_count, num_heads, device=beta.device, dtype=beta.dtype, requires_grad=True)
        # Repeated empty boundaries cover packed dynamic batches without changing
        # the fixed non-empty signature used to compile every production kernel.
        warm_cu = torch.tensor([0, 0, token_count], device=key.device, dtype=torch.int32)
        warm_hm = ttx_local_affine_summary(
            warm_key,
            warm_value,
            warm_g,
            warm_beta,
            cu_seqlens=warm_cu,
            use_qk_l2norm=False,
        )
        torch.autograd.grad(
            warm_hm,
            (warm_key, warm_value, warm_g, warm_beta),
            grad_outputs=torch.zeros_like(warm_hm),
        )
    torch.npu.synchronize()
    _TTX_FORWARD_BACKWARD_WARMUP_CACHE.add(signature)


def warmup_ttx_bc8_m1_forward_backward_for_shapes(
    *,
    device: torch.device,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    key_dtype: torch.dtype,
    value_dtype: torch.dtype,
    g_dtype: torch.dtype,
    beta_dtype: torch.dtype,
) -> None:
    """Warm the production TTX forward+VJP from a model-level shape contract.

    Decoder layers are wrapped by non-reentrant activation checkpointing, so
    the first TTX launch must happen before entering a checkpoint.  This
    helper intentionally constructs the same small synthetic tensors as the
    tensor-based warmup, but keeps model code independent from the private
    cache and custom autograd implementation.
    """

    device = torch.device(device)
    if device.type != "npu":
        raise RuntimeError(f"TTX KCP warmup requires an Ascend NPU device, got {device}")
    for name, value in (
        ("num_heads", num_heads),
        ("key_dim", key_dim),
        ("value_dim", value_dim),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer, got {value!r}")
    token_count = 128
    key = torch.zeros(1, token_count, num_heads, key_dim, device=device, dtype=key_dtype, requires_grad=True)
    value = torch.zeros(1, token_count, num_heads, value_dim, device=device, dtype=value_dtype, requires_grad=True)
    g = torch.zeros(1, token_count, num_heads, device=device, dtype=g_dtype, requires_grad=True)
    beta = torch.zeros(1, token_count, num_heads, device=device, dtype=beta_dtype, requires_grad=True)
    warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)


__all__ = [
    "TTX_BC8_M1_CONFIG",
    "TtxBc8M1Config",
    "ttx_bc8_m1_torch_reference",
    "ttx_local_affine_he_m",
    "ttx_local_affine_summary",
    "validate_ttx_bc8_m1_cu_seqlens",
    "validate_ttx_bc8_m1_contract",
    "validate_ttx_bc8_m1_inputs",
    "validate_ttx_bc8_m1_shape",
    "warmup_ttx_bc8_m1_forward_backward",
    "warmup_ttx_bc8_m1_forward_backward_for_shapes",
]

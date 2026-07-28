# Copyright (c) 2025 VeOmni Authors.
"""GDN KCP (P1): local affine summary + all-gather{he,M} + prefix-merge.

Contract: [[GDN State-Passing CP Implementation Contract 20260727]]

Replaces serial P2P state passing at runtime (serial code retained as golden).
Communication volume ∝ ``CP × H × K × (K+V)`` — independent of sequence length S.

Algorithm (FLA ``fla/ops/cp`` semantics, eager portable implementation):
1. Non-last ranks: local gated-delta scan from zero → affine ``S_final = M @ S_init + he``
   packed as ``hm = [he | M]`` in **float32** (INV-7).
2. ``all_gather`` ``hm`` on ``cp_group`` — buffer **must** stay fp32.
3. Non-first ranks: prefix-merge preceding ranks' ``(he, M)`` → ``S_init`` (fp32).
4. Run local ``chunk_gated_delta_rule(..., initial_state=S_init)`` on ``S/cp`` tokens.
   Do **not** chain serial send/recv.

INV-7 (dtype lock): all-gather / ``hm`` / ``S_init`` buffers are explicit **float32**,
aligned with Mojo GDR ``final_state`` and serial P0. Do **not** change to bf16 —
serial v6 hit bf16 template × fp32 state byte-misalignment → NaN (same class of bug).

Lossy vs serial: accepted; keep ``state_passing_serial`` as the lossless golden and
validate P1 vs P0 with an independent tol (CP-C). Default ``gdn_cp_impl`` stays
``gather_full_replicated`` — kcp is opt-in (lossy + eager pre-scan cost).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup


def pack_affine_hm(he: Tensor, M: Tensor) -> Tensor:
    """Pack ``hm[..., :V]=he``, ``hm[..., V:]=M``. Shapes ``[*, H, K, V]`` / ``[*, H, K, K]``."""
    if he.shape[:-1] != M.shape[:-1] or he.shape[-2] != M.shape[-2]:
        raise ValueError(f"he/M shape mismatch: he={tuple(he.shape)} M={tuple(M.shape)}")
    if he.dtype != torch.float32 or M.dtype != torch.float32:
        raise ValueError("INV-7: he/M must be float32 before pack")
    return torch.cat([he, M], dim=-1)


def unpack_affine_hm(hm: Tensor, *, v_dim: int) -> Tuple[Tensor, Tensor]:
    if hm.dtype != torch.float32:
        raise ValueError("INV-7: hm must be float32")
    he = hm[..., :v_dim]
    M = hm[..., v_dim:]
    k_dim = int(he.shape[-2])
    if int(M.shape[-1]) != k_dim:
        raise ValueError(f"M last dim {M.shape[-1]} != K {k_dim}")
    return he, M


def _eye_m(num_heads: int, k_dim: int, *, device: torch.device) -> Tensor:
    eye = torch.eye(k_dim, device=device, dtype=torch.float32)
    return eye.view(1, k_dim, k_dim).expand(num_heads, k_dim, k_dim).contiguous()


def local_affine_summary_recurrent(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Eager local ``hm=[he|M]`` for one CP rank (zero initial state).

    Layouts match modeling GDN core (pre-transpose):
      key/value: ``[1, T, H, K/V]`` (varlen packed) or ``[B, T, H, K/V]``
      g/beta: ``[1, T, H]`` / ``[B, T, H]``
      cu_seqlens: optional varlen boundaries on the T axis of batch 0.

    Returns ``hm`` with shape ``[N, H, K, V+K]`` float32.
    """
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")
    if use_qk_l2norm:
        key = key * torch.rsqrt(key.pow(2).sum(dim=-1, keepdim=True) + eps)
    k = key.float()
    v = value.float()
    gg = g.float()
    bb = beta.float()
    bsz, _t, num_heads, k_dim = k.shape
    v_dim = int(v.shape[-1])

    if cu_seqlens is None:
        ranges = [(b, 0, int(k.shape[1])) for b in range(bsz)]
    else:
        if bsz != 1:
            raise ValueError("varlen affine summary expects batch=1 packed layout")
        pts = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if not pts or pts[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {pts[:3]}")
        ranges = [(0, pts[i], pts[i + 1]) for i in range(len(pts) - 1)]

    eye = _eye_m(num_heads, k_dim, device=k.device)
    out = []
    for _b, start, end in ranges:
        he = torch.zeros(num_heads, k_dim, v_dim, device=k.device, dtype=torch.float32)
        M = eye.clone()
        for t in range(start, end):
            eg = gg[_b, t].exp()  # [H]
            kt = k[_b, t]  # [H, K]
            vt = v[_b, t]  # [H, V]
            bt = bb[_b, t]  # [H]
            # M_step = eg * (I - beta * k⊗k); he_step = (beta*k)⊗v
            outer = kt.unsqueeze(-1) * kt.unsqueeze(-2)  # [H,K,K]
            M_step = eg[:, None, None] * (eye - bt[:, None, None] * outer)
            he_step = (bt[:, None] * kt).unsqueeze(-1) * vt.unsqueeze(-2)  # [H,K,V]
            he = torch.einsum("hki,hiv->hkv", M_step, he) + he_step
            M = torch.einsum("hki,hij->hkj", M_step, M)
        out.append(pack_affine_hm(he, M))
    return torch.stack(out, dim=0)


def prefix_merge_initial_state(
    ag_hm: Tensor,
    *,
    cp_rank: int,
    v_dim: int,
) -> Tensor:
    """Prefix-merge ``ag_hm[0..cp_rank)`` → ``S_init`` ``[N,H,K,V]`` float32.

    ``ag_hm``: ``[CP, N, H, K, V+K]``.
    """
    if ag_hm.dtype != torch.float32:
        raise ValueError("INV-7: ag_hm must be float32")
    if ag_hm.ndim != 5:
        raise ValueError(f"ag_hm must be [CP,N,H,K,V+K], got {tuple(ag_hm.shape)}")
    cp_size, num_seqs, num_heads, k_dim, kvk = ag_hm.shape
    if int(cp_rank) < 0 or int(cp_rank) >= cp_size:
        raise ValueError(f"cp_rank {cp_rank} out of range for cp_size {cp_size}")
    if kvk != k_dim + v_dim:
        raise ValueError(f"hm last dim {kvk} != K+V ({k_dim}+{v_dim})")
    s = torch.zeros(num_seqs, num_heads, k_dim, v_dim, device=ag_hm.device, dtype=torch.float32)
    if int(cp_rank) == 0:
        return s
    for r in range(int(cp_rank)):
        he, M = unpack_affine_hm(ag_hm[r], v_dim=v_dim)
        s = torch.einsum("nhki,nhiv->nhkv", M, s) + he
    return s


class _KcpAllGatherHm(torch.autograd.Function):
    """Differentiable all-gather for fp32 ``hm`` on ``cp_group``."""

    @staticmethod
    def forward(ctx: Any, local_hm: Tensor, group: ProcessGroup, cp_size: int, cp_rank: int) -> Tensor:
        # INV-7: AG buffer fp32 — matches final_state; bf16 causes byte-misalign NaN
        # (same pitfall as serial state_passing v6). Do not widen/narrow here.
        if local_hm.dtype != torch.float32:
            raise RuntimeError(
                "INV-7: KCP all-gather buffer must be float32 "
                "(align final_state; bf16 byte-misalign → NaN, serial v6 pitfall)"
            )
        if not torch.isfinite(local_hm).all():
            raise RuntimeError(
                f"kcp: refusing to all-gather non-finite hm (cp_rank={cp_rank})"
            )
        ctx.group = group
        ctx.cp_size = int(cp_size)
        ctx.cp_rank = int(cp_rank)
        if int(cp_size) <= 1:
            return local_hm.unsqueeze(0)
        gathered = [torch.empty_like(local_hm) for _ in range(int(cp_size))]
        dist.all_gather(gathered, local_hm.contiguous(), group=group)
        return torch.stack(gathered, dim=0)

    @staticmethod
    def backward(ctx: Any, grad_ag: Tensor) -> Tuple[Tensor, None, None, None]:
        if ctx.cp_size <= 1:
            return grad_ag.squeeze(0), None, None, None
        return grad_ag[ctx.cp_rank].contiguous(), None, None, None


def all_gather_affine_hm(
    local_hm: Tensor,
    *,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
) -> Tensor:
    """All-gather local ``hm[N,H,K,V+K]`` → ``[CP,N,H,K,V+K]`` (fp32, INV-7)."""
    return _KcpAllGatherHm.apply(local_hm, cp_group, int(cp_size), int(cp_rank))


def resolve_kcp_initial_state(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    sample_bos_on_rank: Optional[Sequence[bool]] = None,
) -> Tensor:
    """End-to-end KCP ``S_init`` for this rank (autograd-safe through AG+merge).

    Last rank skips local summary (contributes zeros); first rank returns zeros
    without merge. Sample-BOS ranks force that sample's ``S_init`` to zero.
    """
    v_dim = int(value.shape[-1])
    if int(cp_size) <= 1:
        n = 1 if cu_seqlens is None else int(cu_seqlens.numel() - 1)
        if cu_seqlens is None:
            n = int(key.shape[0])
        return torch.zeros(
            n,
            int(key.shape[2]),
            int(key.shape[3]),
            v_dim,
            device=key.device,
            dtype=torch.float32,
        )

    if int(cp_rank) < int(cp_size) - 1:
        local_hm = local_affine_summary_recurrent(
            key,
            value,
            g,
            beta,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm=use_qk_l2norm,
        )
    else:
        # Shape probe without scan work.
        if cu_seqlens is None:
            n = int(key.shape[0])
        else:
            n = int(cu_seqlens.numel() - 1)
        local_hm = torch.zeros(
            n,
            int(key.shape[2]),
            int(key.shape[3]),
            v_dim + int(key.shape[3]),
            device=key.device,
            dtype=torch.float32,
        )

    ag_hm = all_gather_affine_hm(
        local_hm, cp_group=cp_group, cp_size=cp_size, cp_rank=cp_rank
    )
    s_init = prefix_merge_initial_state(ag_hm, cp_rank=cp_rank, v_dim=v_dim)

    if sample_bos_on_rank is not None:
        from .gdn_scan_cp import apply_sample_bos_zero_s_init

        s_init = apply_sample_bos_zero_s_init(s_init, sample_bos_on_rank=sample_bos_on_rank)
    return s_init


def assert_kcp_comm_bytes_independent_of_seq(
    *,
    cp_size: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
    num_seqs: int = 1,
) -> int:
    """Return AG payload bytes; caller checks equality across seq lengths."""
    # each rank contributes hm[N,H,K,V+K] fp32; AG receives cp copies
    elems = int(num_seqs) * int(num_heads) * int(k_dim) * (int(v_dim) + int(k_dim))
    return elems * 4 * int(cp_size)


__all__ = [
    "all_gather_affine_hm",
    "assert_kcp_comm_bytes_independent_of_seq",
    "local_affine_summary_recurrent",
    "pack_affine_hm",
    "prefix_merge_initial_state",
    "resolve_kcp_initial_state",
    "unpack_affine_hm",
]

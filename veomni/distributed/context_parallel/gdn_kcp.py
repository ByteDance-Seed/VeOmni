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

"""Affine-prefix KCP for Gated DeltaNet context parallelism.

KCP reuses the validated lossless ownership route for physical-to-owned token
movement.  It replaces only the recurrent-state P2P chain with a fixed-size
affine summary collective.  Communication volume is proportional to
``CP * H * K * (K + V)`` and independent of sequence length.

Algorithm:
1. Non-last ranks: local gated-delta scan from zero → affine ``S_final = M @ S_init + he``
   packed as ``hm = [he | M]`` in **float32** (INV-7).
2. ``all_gather`` ``hm`` on ``cp_group`` — buffer **must** stay fp32.
3. Non-first ranks: prefix-merge preceding ranks' ``(he, M)`` → ``S_init`` (fp32).
4. Run local ``chunk_gated_delta_rule(..., initial_state=S_init)`` on the
   ownership rank's tokens without a recurrent-state P2P chain.

INV-7 (dtype lock): all-gather / ``hm`` / ``S_init`` buffers are explicit **float32**,
matching the GDN kernel's recurrent-state boundary. Do **not** change this
collective boundary to bf16.

The Ascend production backend is the fail-closed TTX BC8/M1 implementation.
Portable torch implementations are retained only as explicit numerical
references; production dispatch never silently falls back to them.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed import ProcessGroup

from veomni.ops.kernels.gated_delta_rule.normalization import producer_dtype_l2norm

from .gdn_lossless import GdnLosslessRuntimePlan
from .gdn_runtime import GdnCpOperation, GdnCpPhase, GdnCpRuntimeIdentity, GdnCpRuntimeObserver


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


def _prepare_affine_inputs(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
    use_qk_l2norm: bool,
    eps: float,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, List[Tuple[int, int, int]]]:
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError(f"key/value must be 4D [B,T,H,D], got {tuple(key.shape)} / {tuple(value.shape)}")
    if use_qk_l2norm:
        key = producer_dtype_l2norm(key, dim=-1, eps=eps)
    k = key.float()
    v = value.float()
    gg = g.float()
    bb = beta.float()
    bsz = int(k.shape[0])
    if cu_seqlens is None:
        ranges = [(b, 0, int(k.shape[1])) for b in range(bsz)]
    else:
        if bsz != 1:
            raise ValueError("varlen affine summary expects batch=1 packed layout")
        pts = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
        if not pts or pts[0] != 0:
            raise ValueError(f"cu_seqlens must start at 0, got {pts[:3]}")
        ranges = [(0, pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
    return k, v, gg, bb, ranges


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

    This is the portable numerical reference. Optimized implementations must
    match it within their documented floating-point tolerance.
    """
    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    num_heads = int(k.shape[2])
    k_dim = int(k.shape[3])
    v_dim = int(v.shape[-1])
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


def local_affine_summary_fused_torch(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
) -> Tensor:
    """Same LTR fp32 recurrence as eager in a head-batched ``bmm`` form.

    Still O(T) sequential (associativity not reordered — required for bit-close
    vs eager). It is a portable reference, not a production backend.
    INV-7: all state stays float32.
    """
    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    num_heads = int(k.shape[2])
    k_dim = int(k.shape[3])
    v_dim = int(v.shape[-1])
    eye = _eye_m(num_heads, k_dim, device=k.device)
    out = []
    for _b, start, end in ranges:
        # Contiguous time slices for fewer indexing ops.
        kt_all = k[_b, start:end].contiguous()  # [T,H,K]
        vt_all = v[_b, start:end].contiguous()
        eg_all = gg[_b, start:end].exp().contiguous()
        bt_all = bb[_b, start:end].contiguous()
        he = torch.zeros(num_heads, k_dim, v_dim, device=k.device, dtype=torch.float32)
        M = eye.clone()
        for t in range(int(kt_all.shape[0])):
            eg = eg_all[t]
            kt = kt_all[t]
            vt = vt_all[t]
            bt = bt_all[t]
            outer = kt.unsqueeze(-1) * kt.unsqueeze(-2)
            M_step = eg[:, None, None] * (eye - bt[:, None, None] * outer)
            he_step = (bt[:, None] * kt).unsqueeze(-1) * vt.unsqueeze(-2)
            # bmm over heads: [H,K,K] @ [H,K,V] / [H,K,K]
            he = torch.bmm(M_step, he) + he_step
            M = torch.bmm(M_step, M)
        out.append(pack_affine_hm(he, M))
    return torch.stack(out, dim=0)


def _l2norm_vjp(x: Tensor, grad_y: Tensor, *, eps: float) -> Tensor:
    """VJP of ``y = x * rsqrt(sum(x^2)+eps)`` on the last dim."""
    sumsq = x.pow(2).sum(dim=-1, keepdim=True).add(eps)
    inv = sumsq.rsqrt()
    # d/dx (x * inv) = inv * I - x x^T * inv^3
    return grad_y * inv - x * (grad_y * x).sum(dim=-1, keepdim=True) * (inv * inv * inv)


def _affine_bwd_chunk() -> int:
    return 128


def local_affine_summary_analytical_bwd(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    grad_hm: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
    chunk_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Analytical reverse-scan VJP for local affine summary (fp32).

    Checkpointed reverse: save ``(he,M)`` every ``chunk_size`` tokens in a
    single forward sweep, then recompute each chunk and apply the closed-form
    VJP of ``A=e^g(I-βkkᵀ), b=βk vᵀ``. No autograd graph of T steps, no
    Sherman–Morrison (avoids ``1-β‖k‖²`` drift).

    Numerical reference implementation for the analytical VJP. Production
    dispatch uses the registered TTX analytical backward path instead.
    """
    key_in = key
    k, v, gg, bb, ranges = _prepare_affine_inputs(
        key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
    )
    num_heads = int(k.shape[2])
    k_dim = int(k.shape[3])
    v_dim = int(v.shape[-1])
    eye = _eye_m(num_heads, k_dim, device=k.device)
    bt_chunk = int(chunk_size) if chunk_size is not None else _affine_bwd_chunk()

    grad_k = torch.zeros_like(k)
    grad_v = torch.zeros_like(v)
    grad_g = torch.zeros_like(gg)
    grad_b = torch.zeros_like(bb)

    if grad_hm.ndim != 4:
        raise ValueError(f"grad_hm must be [N,H,K,V+K], got {tuple(grad_hm.shape)}")
    if int(grad_hm.shape[0]) != len(ranges):
        raise ValueError(f"grad_hm N={grad_hm.shape[0]} != num ranges {len(ranges)}")

    for seg_i, (_b, start, end) in enumerate(ranges):
        kt_all = k[_b, start:end].contiguous()
        vt_all = v[_b, start:end].contiguous()
        gg_all = gg[_b, start:end].contiguous()
        bb_all = bb[_b, start:end].contiguous()
        t_len = int(kt_all.shape[0])
        dhe, dM = unpack_affine_hm(grad_hm[seg_i].float(), v_dim=v_dim)

        he = torch.zeros(num_heads, k_dim, v_dim, device=k.device, dtype=torch.float32)
        M = eye.clone()
        ckpt_he: List[Tensor] = [he.clone()]
        ckpt_M: List[Tensor] = [M.clone()]
        for t in range(t_len):
            eg = gg_all[t].exp()
            kt = kt_all[t]
            vt = vt_all[t]
            bt = bb_all[t]
            outer = kt.unsqueeze(-1) * kt.unsqueeze(-2)
            A = eg[:, None, None] * (eye - bt[:, None, None] * outer)
            b_mat = (bt[:, None] * kt).unsqueeze(-1) * vt.unsqueeze(-2)
            he = torch.bmm(A, he) + b_mat
            M = torch.bmm(A, M)
            if (t + 1) % bt_chunk == 0 or (t + 1) == t_len:
                ckpt_he.append(he.clone())
                ckpt_M.append(M.clone())

        n_ck = len(ckpt_he) - 1
        for c in range(n_ck - 1, -1, -1):
            t0 = c * bt_chunk
            t1 = min(t_len, (c + 1) * bt_chunk)
            he_s = ckpt_he[c].clone()
            M_s = ckpt_M[c].clone()
            he_list: List[Tensor] = []
            M_list: List[Tensor] = []
            for t in range(t0, t1):
                he_list.append(he_s.clone())
                M_list.append(M_s.clone())
                eg = gg_all[t].exp()
                kt = kt_all[t]
                vt = vt_all[t]
                bt = bb_all[t]
                outer = kt.unsqueeze(-1) * kt.unsqueeze(-2)
                A = eg[:, None, None] * (eye - bt[:, None, None] * outer)
                b_mat = (bt[:, None] * kt).unsqueeze(-1) * vt.unsqueeze(-2)
                he_s = torch.bmm(A, he_s) + b_mat
                M_s = torch.bmm(A, M_s)

            for t in range(t1 - 1, t0 - 1, -1):
                store_i = t - t0
                he_t = he_list[store_i]
                M_t = M_list[store_i]
                eg = gg_all[t].exp()
                kt = kt_all[t]
                vt = vt_all[t]
                bt = bb_all[t]
                tr_dA = (dhe * he_t).sum(dim=(-1, -2)) + (dM * M_t).sum(dim=(-1, -2))
                k_he = torch.einsum("hk,hkv->hv", kt, he_t)
                k_dhe = torch.einsum("hk,hkv->hv", kt, dhe)
                k_Mt = torch.einsum("hk,hkj->hj", kt, M_t)
                k_dM = torch.einsum("hk,hkj->hj", kt, dM)
                k_dA_k = (k_he * k_dhe).sum(dim=-1) + (k_Mt * k_dM).sum(dim=-1)
                deg = tr_dA - bt * k_dA_k
                grad_g[_b, start + t] += deg * eg
                grad_b[_b, start + t] += -eg * k_dA_k + torch.einsum("hk,hkv,hv->h", kt, dhe, vt)
                dk = (
                    -eg[:, None]
                    * bt[:, None]
                    * (
                        torch.einsum("hkv,hv->hk", dhe, k_he)
                        + torch.einsum("hkv,hv->hk", he_t, k_dhe)
                        + torch.einsum("hkj,hj->hk", dM, k_Mt)
                        + torch.einsum("hkj,hj->hk", M_t, k_dM)
                    )
                )
                dk = dk + bt[:, None] * torch.einsum("hkv,hv->hk", dhe, vt)
                dv = bt[:, None] * torch.einsum("hkv,hk->hv", dhe, kt)
                grad_k[_b, start + t] += dk
                grad_v[_b, start + t] += dv
                dhe = eg[:, None, None] * (dhe - bt[:, None, None] * kt.unsqueeze(-1) * k_dhe.unsqueeze(-2))
                dM = eg[:, None, None] * (dM - bt[:, None, None] * kt.unsqueeze(-1) * k_dM.unsqueeze(-2))

    if use_qk_l2norm:
        grad_k = _l2norm_vjp(key_in.float(), grad_k, eps=eps)

    return (
        grad_k.to(dtype=key_in.dtype),
        grad_v.to(dtype=value.dtype),
        grad_g.to(dtype=g.dtype),
        grad_b.to(dtype=beta.dtype),
    )


def ensure_affine_hm_fp32(hm: Tensor, *, where: str = "kcp") -> Tensor:
    """Cast the affine-summary boundary to fp32 without a host sync."""
    if hm.dtype != torch.float32:
        hm = hm.float().contiguous()
    if hm.dtype != torch.float32:
        raise RuntimeError(f"{where}: hm must be float32 before all-gather, got {hm.dtype}")
    return hm


def resolve_local_affine_impl(name: Optional[str] = None) -> str:
    """Resolve an explicit affine backend without a silent fallback."""
    key = ("ttx_bc8_m1" if name is None else name).strip().lower()
    if key in ("eager", "recurrent", "golden"):
        return "eager"
    if key in ("fused_torch", "reference", "torch_reference"):
        return "fused_torch"
    if key in ("ttx", "npu", "ttx_bc8_m1"):
        return "ttx"
    raise ValueError(
        f"Unknown KCP affine implementation {name!r}. Expected 'ttx_bc8_m1', 'torch_reference', or 'eager'."
    )


def local_affine_summary(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    eps: float = 1e-6,
    impl: Optional[str] = None,
) -> Tensor:
    """Dispatch the KCP local affine summary."""
    kind = resolve_local_affine_impl(impl)
    if kind == "eager":
        return local_affine_summary_recurrent(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )
    if kind == "fused_torch":
        return local_affine_summary_fused_torch(
            key, value, g, beta, cu_seqlens=cu_seqlens, use_qk_l2norm=use_qk_l2norm, eps=eps
        )
    if kind == "ttx":
        # Fail closed: the optimized training path never falls back to torch.
        try:
            from veomni.ops.kernels.gdn_kcp_affine_ttx import (
                ttx_local_affine_summary,
                validate_ttx_bc8_m1_contract,
            )
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                f"KCP ttx_bc8_m1 requested but its kernel module is unavailable. import_error={exc}"
            ) from exc
        validate_ttx_bc8_m1_contract()
        try:
            hm = ttx_local_affine_summary(
                key,
                value,
                g,
                beta,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm=use_qk_l2norm,
                eps=eps,
            )
        except Exception as exc:
            raise RuntimeError(
                f"KCP ttx_bc8_m1 failed; refusing a silent torch fallback. {type(exc).__name__}: {exc}"
            ) from exc
        return ensure_affine_hm_fp32(hm, where="ttx_local_affine_summary")
    raise AssertionError(f"unreachable KCP affine implementation {kind!r}")


def _validate_local_affine_preflight(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cu_seqlens: Optional[Tensor],
    affine_impl: str,
) -> None:
    """Run rank-local, side-effect-free validation before the readiness collective."""

    kind = resolve_local_affine_impl(affine_impl)
    if kind != "ttx":
        return
    from veomni.ops.kernels.gdn_kcp_affine_ttx import (
        validate_ttx_bc8_m1_inputs,
        warmup_ttx_bc8_m1_forward_backward,
    )

    validate_ttx_bc8_m1_inputs(key, value, g, beta, cu_seqlens=cu_seqlens)
    warmup_ttx_bc8_m1_forward_backward(key, value, g, beta)


def prepare_kcp_ttx_warmup(
    *,
    device: torch.device,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    key_dtype: torch.dtype,
    value_dtype: torch.dtype,
    g_dtype: torch.dtype,
    beta_dtype: torch.dtype,
    cp_group: ProcessGroup,
    reference: Tensor,
) -> None:
    """Prepare TTX forward+VJP before decoder-layer checkpointing.

    The first TTX launch is local and may import/compile/execute a custom
    autograd function.  Running it inside a non-reentrant checkpoint is
    invalid because checkpoint recomputation can replay the launch while the
    KCP readiness graph is already being rebuilt.  All ranks therefore do the
    local warmup first, then enter one scalar readiness collective so a single
    rank's deterministic compile/shape failure is reported uniformly instead
    of leaving peers in an unmatched checkpoint path.
    """

    local_error: Exception | None = None
    try:
        from veomni.ops.kernels.gdn_kcp_affine_ttx import warmup_ttx_bc8_m1_forward_backward_for_shapes

        warmup_ttx_bc8_m1_forward_backward_for_shapes(
            device=device,
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            key_dtype=key_dtype,
            value_dtype=value_dtype,
            g_dtype=g_dtype,
            beta_dtype=beta_dtype,
        )
    except Exception as exc:
        local_error = exc

    status = torch.tensor(
        [int(local_error is not None)],
        device=reference.device,
        dtype=torch.int32,
    )
    if int(dist.get_world_size(group=cp_group)) > 1:
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=cp_group)
    if int(status.item()) != 0:
        detail = f" local_error={type(local_error).__name__}: {local_error}" if local_error else ""
        raise RuntimeError(f"coordinated KCP TTX warmup failed on at least one CP rank.{detail}")


def _coordinate_local_affine_readiness(
    *,
    local_error: Exception | None,
    local_launched: bool,
    expect_affine_scan: bool,
    cp_group: ProcessGroup,
    reference: Tensor,
    observer: GdnCpRuntimeObserver | None,
) -> bool:
    """Coordinate one layer's first local-affine setup before affine AG.

    Terminal/empty ranks do not execute the TTX scan, so they must wait on this
    scalar readiness collective instead of entering the affine all-gather while
    another rank is unwinding a validation/compile/first-launch exception.
    The owning model layer requests this handshake once, then records readiness
    on that layer instance. Steady-state forwards never call this function.

    Unexpected device faults after a signature has passed are reported locally
    like any other accelerator-kernel failure. This handshake coordinates
    catchable deterministic import/device/shape/compile/first-launch failures;
    it cannot recover an accelerator kernel that never returns.
    """

    if observer is not None:
        observer.observe_cp_ranks(range(observer.identity.cp_size))
        observer.enter(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
    status = torch.tensor(
        [int(local_error is not None), int(local_launched)],
        device=reference.device,
        dtype=torch.int32,
    )
    try:
        dist.all_reduce(status, op=dist.ReduceOp.MAX, group=cp_group)
    except Exception:
        if observer is not None:
            observer.error(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
        raise
    failed = int(status[0].item())
    any_launched = bool(status[1].item())
    if failed != 0:
        if observer is not None:
            observer.error(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
        if local_error is not None:
            raise RuntimeError(
                "KCP coordinated local-affine preflight failed on this CP rank; affine all-gather was not entered"
            ) from local_error
        raise RuntimeError(
            "KCP coordinated local-affine preflight failed on another CP rank; affine all-gather was not entered"
        )
    if expect_affine_scan and not any_launched:
        if observer is not None:
            observer.error(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
        raise RuntimeError("KCP readiness expected an affine scan, but no CP rank launched one")
    if observer is not None:
        observer.exit(GdnCpOperation.KCP_AFFINE_READY, GdnCpPhase.FORWARD)
    return any_launched


def kcp_plan_requires_affine_scan(plan: GdnLosslessRuntimePlan) -> bool:
    """Return the global, rank-consistent answer to whether any owner has a successor."""

    return any(
        sample.is_active and sample.successor_rank is not None
        for rank_plan in plan.global_plan.ranks
        for sample in rank_plan.samples
    )


def prefix_merge_initial_state(
    ag_hm: Tensor,
    *,
    cp_rank: int,
    v_dim: int,
) -> Tensor:
    """Prefix-merge ``ag_hm[0..cp_rank)`` → ``S_init`` ``[N,H,K,V]`` float32.

    ``ag_hm``: ``[CP, N, H, K, V+K]``.

    Rank 0 returns numeric zeros with **no** ``ag_hm`` dependency — callers that
    need AG/RS collective liveness must re-attach via
    :func:`attach_zero_valued_dep` (see :func:`resolve_kcp_initial_state`).
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


def kcp_zero_participation_token(*tensors: Tensor) -> Tensor:
    """Scalar ``0`` that still depends on every input (autograd participation).

    Isolated ``requires_grad_(True)`` zeros are **not** enough: last-rank plain
    zeros never connect ``key/value/g/beta``, so ``_KcpAllGatherHm`` drops out of
    the graph and that rank skips reduce-scatter while peers still enter it.

    Non-empty tensors use one element. Empty owners use their empty reduction,
    which still keeps an autograd edge without materializing data.
    """
    acc: Optional[Tensor] = None
    for tensor in tensors:
        if tensor is None:
            continue
        if tensor.numel() == 0:
            piece = tensor.sum() * 0
        else:
            piece = torch.nan_to_num(tensor[(0,) * tensor.ndim], nan=0.0, posinf=0.0, neginf=0.0) * 0
        acc = piece if acc is None else (acc + piece)
    if acc is None:
        raise ValueError("kcp_zero_participation_token requires at least one tensor")
    return acc


def attach_zero_valued_dep(dst: Tensor, *sources: Tensor) -> Tensor:
    """Return ``dst`` unchanged numerically, but backward still visits ``sources``."""
    token = kcp_zero_participation_token(*sources)
    return dst + token.to(device=dst.device, dtype=dst.dtype)


def assert_kcp_cp_group_identity(
    cp_group: ProcessGroup,
    *,
    cp_size: int,
    cp_rank: int,
) -> None:
    """Fail-closed: ``cp_group`` world-size/rank must match caller geometry."""
    world = int(dist.get_world_size(group=cp_group))
    rank = int(dist.get_rank(group=cp_group))
    if world != int(cp_size):
        raise RuntimeError(f"KCP cp_group world_size={world} != cp_size={int(cp_size)}")
    if rank != int(cp_rank):
        raise RuntimeError(f"KCP cp_group rank={rank} != cp_rank={int(cp_rank)}")


def _deterministic_reduce_scatter_sum(grad_ag: Tensor, *, group: ProcessGroup) -> Tensor:
    """Route then sum an all-gather VJP in a fixed source-rank order.

    HCCL ``reduce_scatter(SUM)`` is numerically correct but its internal
    floating-point reduction order is not guaranteed.  Starting at CP4 that
    made otherwise identical KCP backward passes differ by a few fp32 ULPs.
    ``all_to_all_single`` performs only data movement: source rank ``r`` sends
    ``grad_ag[dst]`` to destination ``dst``.  The receiver then accumulates
    source chunks explicitly in rank order, making the result repeatable while
    retaining the reduce-scatter wire-volume class.
    """

    cp_size = int(dist.get_world_size(group=group))
    if grad_ag.dtype != torch.float32:
        raise RuntimeError("KCP all-gather backward buffer must be float32")
    if grad_ag.ndim < 2 or int(grad_ag.shape[0]) != cp_size:
        raise RuntimeError(f"KCP all-gather backward leading dim {tuple(grad_ag.shape)} != CP size {cp_size}")
    # Flatten the payload behind the CP axis.  Gloo and HCCL both define the
    # equal-split contract on dim 0, while some HCCL versions only accept the
    # variable payload reliably as a contiguous 2-D buffer.
    routed = grad_ag.contiguous().view(cp_size, -1)
    received_by_source = torch.empty_like(routed)
    dist.all_to_all_single(received_by_source, routed, group=group)
    out = received_by_source[0].view_as(grad_ag[0]).clone()
    for source_rank in range(1, cp_size):
        out.add_(received_by_source[source_rank].view_as(out))
    return out


class _KcpAllGatherHm(torch.autograd.Function):
    """Differentiable all-gather for fp32 ``hm`` on ``cp_group``.

    ``participate`` is an explicit zero-valued token input so last-rank (plain
    zero ``local_hm``) and any other non-grad ``hm`` still keep this Function in
    the autograd graph; every CP rank must enter :meth:`backward` (reduce-scatter)
    or zigzag A2A collectives desynchronize.
    """

    @staticmethod
    def forward(
        ctx: Any,
        local_hm: Tensor,
        participate: Tensor,
        group: ProcessGroup,
        cp_size: int,
        cp_rank: int,
        observer: GdnCpRuntimeObserver | None,
    ) -> Tensor:
        if local_hm.dtype != torch.float32:
            raise RuntimeError("KCP all-gather buffer must be float32")
        ctx.group = group
        ctx.cp_size = int(cp_size)
        ctx.cp_rank = int(cp_rank)
        ctx.observer = observer
        ctx.participate_shape = tuple(participate.shape)
        ctx.participate_dtype = participate.dtype
        ctx.participate_device = participate.device
        # Zero-valued: keeps participate in the graph without changing AG payload.
        send = local_hm + participate.to(device=local_hm.device, dtype=local_hm.dtype) * 0
        if int(cp_size) <= 1:
            return send.unsqueeze(0)
        gathered = [torch.empty_like(send) for _ in range(int(cp_size))]
        if observer is not None:
            observer.observe_cp_ranks(range(int(cp_size)))
            observer.enter(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.FORWARD)
        try:
            dist.all_gather(gathered, send.contiguous(), group=group)
        except Exception:
            if observer is not None:
                observer.error(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.FORWARD)
            raise
        if observer is not None:
            observer.exit(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.FORWARD)
        return torch.stack(gathered, dim=0)

    @staticmethod
    def backward(ctx: Any, grad_ag: Tensor) -> Tuple[Tensor, Optional[Tensor], None, None, None, None]:
        if ctx.cp_size <= 1:
            out_single = grad_ag.squeeze(0)
        else:
            # The inverse of all-gather is SUM reduce-scatter. Every rank must
            # execute the collective, including BOS and terminal owners. Route
            # without reduction, then add in a fixed source-rank order so CP4+
            # does not inherit HCCL's non-deterministic floating-point order.
            grad_ag = grad_ag.contiguous()
            if ctx.observer is not None:
                ctx.observer.enter(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.BACKWARD)
            try:
                out_single = _deterministic_reduce_scatter_sum(grad_ag, group=ctx.group)
            except Exception:
                if ctx.observer is not None:
                    ctx.observer.error(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.BACKWARD)
                raise
            if ctx.observer is not None:
                ctx.observer.exit(GdnCpOperation.KCP_AFFINE_AG, GdnCpPhase.BACKWARD)
        participate_grad = torch.zeros(
            ctx.participate_shape,
            dtype=ctx.participate_dtype,
            device=ctx.participate_device,
        )
        return out_single, participate_grad, None, None, None, None


def all_gather_affine_hm(
    local_hm: Tensor,
    *,
    cp_group: ProcessGroup,
    cp_size: int,
    cp_rank: int,
    participate: Optional[Tensor] = None,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """All-gather local ``hm[N,H,K,V+K]`` → ``[CP,N,H,K,V+K]`` (fp32, INV-7).

    ``participate``: optional zero-valued token (see
    :func:`kcp_zero_participation_token`). When omitted, a detached zero is used
    — callers with non-grad ``local_hm`` **must** pass a live token or AG/RS will
    be skipped on that rank.
    """
    if participate is None:
        participate = local_hm.new_zeros(())
    return _KcpAllGatherHm.apply(local_hm, participate, cp_group, int(cp_size), int(cp_rank), observer)


def _identity_affine_hm(
    *,
    num_seqs: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
    reference: Tensor,
) -> Tensor:
    """Return ``he=0, M=I`` for a terminal owner that skips its pre-scan."""
    he = torch.zeros(num_seqs, num_heads, k_dim, v_dim, device=reference.device, dtype=torch.float32)
    eye = torch.eye(k_dim, device=reference.device, dtype=torch.float32)
    matrix = eye.view(1, 1, k_dim, k_dim).expand(num_seqs, num_heads, k_dim, k_dim).contiguous()
    return pack_affine_hm(he, matrix)


def resolve_kcp_initial_state(
    key: Tensor,
    value: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    cp_group: ProcessGroup,
    plan: GdnLosslessRuntimePlan,
    cu_seqlens: Optional[Tensor] = None,
    use_qk_l2norm: bool = True,
    affine_impl: str = "ttx_bc8_m1",
    extra_participation: Tensor | None = None,
    coordinate_readiness: bool = True,
    observer: GdnCpRuntimeObserver | None = None,
) -> Tensor:
    """Build this ownership rank's KCP initial state with AG/RS autograd.

    The ownership plan is the single layout authority. Empty sample segments
    contribute identity transforms, BOS/inactive outputs are masked to zero,
    and every rank retains a graph edge to the all-gather so backward collective
    ordinals stay live.
    """
    cp_size = plan.cp_size
    cp_rank = plan.cp_rank
    if cp_size <= 1:
        v_dim = int(value.shape[-1])
        n = int(key.shape[0]) if cu_seqlens is None else int(cu_seqlens.numel() - 1)
        return torch.zeros(
            n,
            int(key.shape[2]),
            int(key.shape[3]),
            v_dim,
            device=key.device,
            dtype=torch.float32,
        )

    assert_kcp_cp_group_identity(cp_group, cp_size=cp_size, cp_rank=cp_rank)
    expect_affine_scan = kcp_plan_requires_affine_scan(plan)

    def build_local_summary() -> tuple[Tensor, Tensor, int, int, bool]:
        affine_kind = resolve_local_affine_impl(affine_impl)
        if observer is not None and affine_kind != "ttx":
            raise ValueError(
                "A production KCP runtime observer may only attest the ttx_bc8_m1 backend; "
                f"got affine_impl={affine_impl!r}"
            )
        if observer is not None:
            expected_identity = GdnCpRuntimeIdentity(
                implementation="kcp",
                ownership_plan_hash=plan.plan_hash,
                cp_size=cp_size,
                cp_rank=cp_rank,
                affine_backend="ttx_bc8_m1",
            )
            if observer.identity != expected_identity:
                raise ValueError("KCP runtime observer identity does not match the live ownership plan")
        v_dim = int(value.shape[-1])
        n = int(key.shape[0]) if cu_seqlens is None else int(cu_seqlens.numel() - 1)
        if n != len(plan.local.samples):
            raise ValueError(f"KCP segment count {n} does not match ownership samples {len(plan.local.samples)}")
        participate = kcp_zero_participation_token(key, value, g, beta)
        terminal_owner = all(not sample.is_active or sample.successor_rank is None for sample in plan.local.samples)
        # On the first plan that actually needs an affine scan, every rank
        # validates and warms the production forward+VJP before the readiness
        # collective.  This includes terminal/inactive owners, so later dynamic
        # batches cannot expose an unwarmed rank after the layer is marked
        # ready.  All-terminal plans skip TTX entirely; steady state relies on
        # the public TTX entry's local validation and has no readiness overhead.
        if coordinate_readiness and expect_affine_scan:
            _validate_local_affine_preflight(
                key,
                value,
                g,
                beta,
                cu_seqlens=cu_seqlens,
                affine_impl=affine_impl,
            )
        if terminal_owner:
            local_hm = _identity_affine_hm(
                num_seqs=n,
                num_heads=int(key.shape[2]),
                k_dim=int(key.shape[3]),
                v_dim=v_dim,
                reference=key,
            )
            local_hm = attach_zero_valued_dep(local_hm, participate)
        else:
            local_hm = local_affine_summary(
                key,
                value,
                g,
                beta,
                cu_seqlens=cu_seqlens,
                use_qk_l2norm=use_qk_l2norm,
                impl=affine_impl,
            )
        return local_hm, participate, n, v_dim, not terminal_owner

    if coordinate_readiness:
        local_result = None
        local_error = None
        try:
            local_result = build_local_summary()
        except Exception as exc:
            local_error = exc
        _coordinate_local_affine_readiness(
            local_error=local_error,
            local_launched=bool(local_result is not None and local_result[4]),
            expect_affine_scan=expect_affine_scan,
            cp_group=cp_group,
            reference=key,
            observer=observer,
        )
        if local_result is None:
            raise AssertionError("coordinated KCP readiness succeeded without a local affine summary")
    else:
        # Per-layer readiness was already coordinated. This is the hot path:
        # no readiness tensor, collective, host sync, or observer READY event.
        local_result = build_local_summary()

    local_hm, participate, n, v_dim, _local_launched = local_result
    if extra_participation is not None:
        # Route auxiliary ownership inputs (notably query) through local_hm so
        # their A2A backward is topologically downstream of the KCP AG/RS.
        local_hm = attach_zero_valued_dep(local_hm, extra_participation)

    local_hm = ensure_affine_hm_fp32(local_hm, where="resolve_kcp_initial_state")

    ag_hm = all_gather_affine_hm(
        local_hm,
        cp_group=cp_group,
        cp_size=cp_size,
        cp_rank=cp_rank,
        participate=participate,
        observer=observer,
    )
    s_init = prefix_merge_initial_state(ag_hm, cp_rank=cp_rank, v_dim=v_dim)
    active_non_bos = [sample.is_active and not sample.is_bos_owner for sample in plan.local.samples]
    mask_shape = (n,) + (1,) * (s_init.ndim - 1)
    mask = s_init.new_tensor(active_non_bos, dtype=s_init.dtype).reshape(mask_shape)
    s_init = s_init * mask

    # BOS and inactive masks would otherwise drop the AG edge on some ranks.
    s_init = attach_zero_valued_dep(s_init, ag_hm)
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
    "assert_kcp_cp_group_identity",
    "attach_zero_valued_dep",
    "ensure_affine_hm_fp32",
    "kcp_plan_requires_affine_scan",
    "kcp_zero_participation_token",
    "local_affine_summary",
    "local_affine_summary_analytical_bwd",
    "local_affine_summary_fused_torch",
    "local_affine_summary_recurrent",
    "pack_affine_hm",
    "prepare_kcp_ttx_warmup",
    "prefix_merge_initial_state",
    "resolve_kcp_initial_state",
    "resolve_local_affine_impl",
    "unpack_affine_hm",
]

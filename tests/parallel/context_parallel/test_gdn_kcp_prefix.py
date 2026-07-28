# Copyright (c) 2025 VeOmni Authors.
"""Host tests for GDN KCP local-affine + prefix-merge (P1)."""
from __future__ import annotations

import torch

from veomni.distributed.context_parallel.gdn_kcp import (
    assert_kcp_comm_bytes_independent_of_seq,
    local_affine_summary_recurrent,
    pack_affine_hm,
    prefix_merge_initial_state,
    unpack_affine_hm,
)
from veomni.distributed.context_parallel.gdn_scan_cp import (
    gdn_replication_factor_for_impl,
    normalize_gdn_cp_impl,
)


def _naive_final_state(key, value, g, beta, *, initial_state=None, use_qk_l2norm=True):
    """Token-loop GDN state (matches local_affine_summary_recurrent)."""
    if use_qk_l2norm:
        key = key * torch.rsqrt(key.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    k, v, gg, bb = key.float(), value.float(), g.float(), beta.float()
    bsz, t, h, kd = k.shape
    vd = v.shape[-1]
    assert bsz == 1
    s = (
        initial_state.float()
        if initial_state is not None
        else torch.zeros(1, h, kd, vd, dtype=torch.float32)
    )
    s = s[0]
    for i in range(t):
        eg = gg[0, i].exp()
        kt, vt, bt = k[0, i], v[0, i], bb[0, i]
        s = s * eg[:, None, None]
        bv = bt[:, None] * (vt - (s * kt[..., None]).sum(-2))
        s = s + kt.unsqueeze(-1) * bv.unsqueeze(-2)
    return s.unsqueeze(0)


def test_kcp_impl_identity():
    assert normalize_gdn_cp_impl("kcp") == "kcp"
    assert gdn_replication_factor_for_impl("kcp", cp_size=8) == 1


def test_affine_summary_matches_zero_init_final_state():
    torch.manual_seed(0)
    b, t, h, kd, vd = 1, 48, 2, 8, 4
    key = torch.randn(b, t, h, kd)
    value = torch.randn(b, t, h, vd)
    g = torch.randn(b, t, h) * 0.1
    beta = torch.sigmoid(torch.randn(b, t, h))
    hm = local_affine_summary_recurrent(key, value, g, beta, use_qk_l2norm=True)
    he, M = unpack_affine_hm(hm[0], v_dim=vd)
    # From zero init: S_final = he
    s_ref = _naive_final_state(key, value, g, beta, use_qk_l2norm=True)[0]
    torch.testing.assert_close(he, s_ref, atol=1e-4, rtol=1e-4)
    # M @ 0 + he == he; also check M applied to random S0
    s0 = torch.randn_like(s_ref)
    s_lin = torch.einsum("hki,hiv->hkv", M, s0) + he
    s_ref2 = _naive_final_state(
        key, value, g, beta, initial_state=s0.unsqueeze(0), use_qk_l2norm=True
    )[0]
    torch.testing.assert_close(s_lin, s_ref2, atol=1e-4, rtol=1e-4)


def test_prefix_merge_equals_serial_chain():
    """CP2: merge(hm0) then apply hm1 ≡ full-sequence final state."""
    torch.manual_seed(1)
    b, t, h, kd, vd = 1, 64, 2, 8, 4
    key = torch.randn(b, t, h, kd)
    value = torch.randn(b, t, h, vd)
    g = torch.randn(b, t, h) * 0.05
    beta = torch.sigmoid(torch.randn(b, t, h))
    mid = t // 2
    hm0 = local_affine_summary_recurrent(
        key[:, :mid], value[:, :mid], g[:, :mid], beta[:, :mid], use_qk_l2norm=True
    )
    hm1 = local_affine_summary_recurrent(
        key[:, mid:], value[:, mid:], g[:, mid:], beta[:, mid:], use_qk_l2norm=True
    )
    ag = torch.stack([hm0[0], hm1[0]], dim=0).unsqueeze(1)  # [CP=2,N=1,...]
    # rank1 S_init = prefix merge of rank0 only
    s1 = prefix_merge_initial_state(ag, cp_rank=1, v_dim=vd)[0]
    he0, _ = unpack_affine_hm(hm0[0], v_dim=vd)
    torch.testing.assert_close(s1, he0, atol=1e-5, rtol=1e-5)
    # final via affine on rank1 block
    he1, M1 = unpack_affine_hm(hm1[0], v_dim=vd)
    s_final = torch.einsum("hki,hiv->hkv", M1, s1) + he1
    s_ref = _naive_final_state(key, value, g, beta, use_qk_l2norm=True)[0]
    torch.testing.assert_close(s_final, s_ref, atol=1e-4, rtol=1e-4)


def test_comm_bytes_independent_of_seq():
    b64 = assert_kcp_comm_bytes_independent_of_seq(
        cp_size=4, num_heads=8, k_dim=128, v_dim=128, num_seqs=1
    )
    b256 = assert_kcp_comm_bytes_independent_of_seq(
        cp_size=4, num_heads=8, k_dim=128, v_dim=128, num_seqs=1
    )
    assert b64 == b256
    # formula: CP * N * H * K * (K+V) * 4
    assert b64 == 4 * 1 * 8 * 128 * (128 + 128) * 4


def test_pack_roundtrip_fp32_only():
    he = torch.randn(2, 4, 3, dtype=torch.float32)
    M = torch.randn(2, 4, 4, dtype=torch.float32)
    hm = pack_affine_hm(he, M)
    he2, M2 = unpack_affine_hm(hm, v_dim=3)
    torch.testing.assert_close(he2, he)
    torch.testing.assert_close(M2, M)

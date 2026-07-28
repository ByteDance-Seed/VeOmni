#!/usr/bin/env python3
"""Host smoke for KCP affine + prefix-merge (avoids full veomni import)."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / "veomni" / "distributed" / "context_parallel"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


for name, path in [
    ("veomni", ROOT / "veomni"),
    ("veomni.distributed", ROOT / "veomni" / "distributed"),
    ("veomni.distributed.context_parallel", CP),
]:
    m = types.ModuleType(name)
    m.__path__ = [str(path)]
    sys.modules[name] = m

_load(
    "veomni.distributed.context_parallel.sharding",
    CP / "sharding.py",
    "veomni.distributed.context_parallel",
)
scan = _load(
    "veomni.distributed.context_parallel.gdn_scan_cp",
    CP / "gdn_scan_cp.py",
    "veomni.distributed.context_parallel",
)
kcp = _load(
    "veomni.distributed.context_parallel.gdn_kcp",
    CP / "gdn_kcp.py",
    "veomni.distributed.context_parallel",
)


def _naive_final_state(key, value, g, beta, *, initial_state=None, use_qk_l2norm=True):
    if use_qk_l2norm:
        key = key * torch.rsqrt(key.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
    k, v, gg, bb = key.float(), value.float(), g.float(), beta.float()
    _b, t, h, kd = k.shape
    vd = v.shape[-1]
    s = (
        initial_state.float()[0]
        if initial_state is not None
        else torch.zeros(h, kd, vd, dtype=torch.float32)
    )
    for i in range(t):
        eg = gg[0, i].exp()
        kt, vt, bt = k[0, i], v[0, i], bb[0, i]
        s = s * eg[:, None, None]
        bv = bt[:, None] * (vt - (s * kt[..., None]).sum(-2))
        s = s + kt.unsqueeze(-1) * bv.unsqueeze(-2)
    return s


def main() -> None:
    assert scan.normalize_gdn_cp_impl("kcp") == "kcp"
    assert scan.gdn_replication_factor_for_impl("kcp", cp_size=8) == 1

    torch.manual_seed(0)
    b, t, h, kd, vd = 1, 48, 2, 8, 4
    key = torch.randn(b, t, h, kd)
    value = torch.randn(b, t, h, vd)
    g = torch.randn(b, t, h) * 0.1
    beta = torch.sigmoid(torch.randn(b, t, h))
    hm = kcp.local_affine_summary_recurrent(key, value, g, beta, use_qk_l2norm=True)
    he, M = kcp.unpack_affine_hm(hm[0], v_dim=vd)
    s_ref = _naive_final_state(key, value, g, beta, use_qk_l2norm=True)
    torch.testing.assert_close(he, s_ref, atol=1e-4, rtol=1e-4)
    s0 = torch.randn_like(s_ref)
    s_lin = torch.einsum("hki,hiv->hkv", M, s0) + he
    s_ref2 = _naive_final_state(
        key, value, g, beta, initial_state=s0.unsqueeze(0), use_qk_l2norm=True
    )
    torch.testing.assert_close(s_lin, s_ref2, atol=1e-4, rtol=1e-4)
    print("PASS affine_identity")

    torch.manual_seed(1)
    t = 64
    key = torch.randn(b, t, h, kd)
    value = torch.randn(b, t, h, vd)
    g = torch.randn(b, t, h) * 0.05
    beta = torch.sigmoid(torch.randn(b, t, h))
    mid = t // 2
    hm0 = kcp.local_affine_summary_recurrent(
        key[:, :mid], value[:, :mid], g[:, :mid], beta[:, :mid], use_qk_l2norm=True
    )
    hm1 = kcp.local_affine_summary_recurrent(
        key[:, mid:], value[:, mid:], g[:, mid:], beta[:, mid:], use_qk_l2norm=True
    )
    ag = torch.stack([hm0[0], hm1[0]], dim=0).unsqueeze(1)
    s1 = kcp.prefix_merge_initial_state(ag, cp_rank=1, v_dim=vd)[0]
    he0, _ = kcp.unpack_affine_hm(hm0[0], v_dim=vd)
    torch.testing.assert_close(s1, he0, atol=1e-5, rtol=1e-5)
    he1, M1 = kcp.unpack_affine_hm(hm1[0], v_dim=vd)
    s_final = torch.einsum("hki,hiv->hkv", M1, s1) + he1
    s_ref = _naive_final_state(key, value, g, beta, use_qk_l2norm=True)
    torch.testing.assert_close(s_final, s_ref, atol=1e-4, rtol=1e-4)
    print("PASS prefix_merge_vs_serial")

    b64 = kcp.assert_kcp_comm_bytes_independent_of_seq(
        cp_size=4, num_heads=8, k_dim=128, v_dim=128, num_seqs=1
    )
    assert b64 == 4 * 1 * 8 * 128 * (128 + 128) * 4
    print("PASS comm_bytes")
    print("ALL_KCP_HOST_SMOKE_OK")


if __name__ == "__main__":
    main()

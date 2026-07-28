#!/usr/bin/env python3
"""Host CP-B smoke: serial state-passing vs dense full scan (no NPU/FLA)."""
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
CP = ROOT / "veomni" / "distributed" / "context_parallel"


def _load(name: str, path: Path, package: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    mod.__package__ = package
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + eps)


def torch_chunk_gated_delta_rule(
    query,
    key,
    value,
    g,
    beta,
    chunk_size=64,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    **kwargs,
):
    """Local copy of the modeling reference (dense, no cu_seqlens)."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query, dim=-1, eps=1e-6)
        key = _l2norm(key, dim=-1, eps=1e-6)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32) for x in (query, key, value, beta, g)
    ]

    batch_size, num_heads, sequence_length, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    pad_size = (chunk_size - sequence_length % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad_size))
    key = F.pad(key, (0, 0, 0, pad_size))
    value = F.pad(value, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))
    g = F.pad(g, (0, pad_size))
    total_sequence_length = sequence_length + pad_size
    scale = 1 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = ((g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ key.transpose(-1, -2)) * decay_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = attn @ v_beta
    k_cumdecay = attn @ (k_beta * g.exp().unsqueeze(-1))
    last_recurrent_state = (
        torch.zeros(batch_size, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device)
        if initial_state is None
        else initial_state.to(value)
    )
    core_attn_out = torch.zeros_like(value)
    for i in range(0, total_sequence_length // chunk_size):
        q_i, k_i, v_i = query[:, :, i], key[:, :, i], value[:, :, i]
        attn_i = q_i @ k_i.transpose(-1, -2) * decay_mask[:, :, i]
        v_prime = (k_cumdecay[:, :, i]) @ last_recurrent_state
        v_new = v_i - v_prime
        attn_inter = (q_i * g[:, :, i, :, None].exp()) @ last_recurrent_state
        core_attn_out[:, :, i] = attn_inter + attn_i @ v_new
        last_recurrent_state = (
            last_recurrent_state * g[:, :, i, -1, None, None].exp()
            + (k_i * (g[:, :, i, -1, None] - g[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_new
        )

    if not output_final_state:
        last_recurrent_state = None
    core_attn_out = core_attn_out.reshape(
        core_attn_out.shape[0], core_attn_out.shape[1], -1, core_attn_out.shape[-1]
    )
    core_attn_out = core_attn_out[:, :, :sequence_length]
    core_attn_out = core_attn_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_attn_out, last_recurrent_state


def _run_serial_vs_dense(cp_size: int, seq_len: int, *, atol: float = 2e-3) -> float:
    torch.manual_seed(0)
    b, h, dk, dv = 1, 2, 8, 16
    assert seq_len % cp_size == 0
    local = seq_len // cp_size
    q = torch.randn(b, seq_len, h, dk)
    k = torch.randn(b, seq_len, h, dk)
    v = torch.randn(b, seq_len, h, dv)
    beta = torch.rand(b, seq_len, h)
    g = -torch.rand(b, seq_len, h)

    dense_out, _ = torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False
    )

    parts = []
    s = None
    for r in range(cp_size):
        sl = slice(r * local, (r + 1) * local)
        if s is None:
            s = q.new_zeros(b, h, dk, dv)
        out_r, s = torch_chunk_gated_delta_rule(
            q[:, sl],
            k[:, sl],
            v[:, sl],
            g=g[:, sl],
            beta=beta[:, sl],
            initial_state=s,
            output_final_state=True,
        )
        parts.append(out_r)
    serial_out = torch.cat(parts, dim=1)
    err = (serial_out - dense_out).abs().max().item()
    if err > atol:
        raise AssertionError(f"cp={cp_size} seq={seq_len} max_abs_err={err} > {atol}")
    return err


def _run_repartition_plus_serial_layout(scan, sharding, cp_size: int, seq_len: int) -> None:
    """Zigzag shards → contiguous blocks → serial scan layout equals dense order."""
    full = torch.arange(seq_len, dtype=torch.float32).view(1, seq_len, 1)
    zigzag = [
        sharding.balanced_cp_slice(full, cp_size=cp_size, cp_rank=r, dim=1) for r in range(cp_size)
    ]
    blocks = scan.simulate_zigzag_to_block(zigzag, cp_size=cp_size, seq_dim=1)
    local = seq_len // cp_size
    for r, block in enumerate(blocks):
        expected = full[:, r * local : (r + 1) * local, :]
        assert torch.equal(block, expected)


def main() -> None:
    for name, path in [
        ("veomni", ROOT / "veomni"),
        ("veomni.distributed", ROOT / "veomni" / "distributed"),
        ("veomni.distributed.context_parallel", CP),
    ]:
        m = types.ModuleType(name)
        m.__path__ = [str(path)]
        sys.modules[name] = m

    sharding = _load(
        "veomni.distributed.context_parallel.sharding",
        CP / "sharding.py",
        "veomni.distributed.context_parallel",
    )
    scan = _load(
        "veomni.distributed.context_parallel.gdn_scan_cp",
        CP / "gdn_scan_cp.py",
        "veomni.distributed.context_parallel",
    )

    assert scan.gdn_replication_factor_for_impl("state_passing_serial", cp_size=4) == 1
    assert scan.normalize_gdn_cp_impl("serial") == scan.GDN_CP_IMPL_STATE_PASSING_SERIAL

    errs = []
    n_ok = 0
    for cp_size in (2, 4):
        for seq_len in (128, 256):
            _run_repartition_plus_serial_layout(scan, sharding, cp_size, seq_len)
            errs.append(_run_serial_vs_dense(cp_size, seq_len))
            n_ok += 1

    # Grad: serial state kept in-graph matches dense cumsum.
    torch.manual_seed(1)
    x = torch.randn(1, 64, 4, requires_grad=True)
    x.cumsum(dim=1).sum().backward()
    g_dense = x.grad.detach().clone()
    cp = 4
    local = 64 // cp
    x2 = x.detach().clone().requires_grad_(True)
    s = torch.zeros(1, 1, 4)
    outs = []
    for r in range(cp):
        chunk = x2[:, r * local : (r + 1) * local]
        y_r = s + chunk.cumsum(dim=1)
        outs.append(y_r)
        s = y_r[:, -1:, :]
    torch.cat(outs, dim=1).sum().backward()
    if not torch.allclose(x2.grad, g_dense, atol=1e-5):
        raise AssertionError("serial cumsum autograd mismatch vs dense")

    print(
        f"CP_B_SERIAL_SMOKE_OK cases={n_ok} "
        f"max_abs_err={max(errs):.3e} mean_abs_err={sum(errs)/len(errs):.3e}"
    )


if __name__ == "__main__":
    main()

# coding=utf-8
# Copyright 2025-2026 The Moonshot AI Team. All rights reserved.
"""Eager fallbacks for the `fla` kernels used by Kimi-K3 linear attention.

Imported by ``.modeling_kimi_k3`` only when `fla` is unavailable (e.g. on
Ascend NPUs). Numerics follow the fla reference semantics; the chunked KDA
implementation is vendored from MindSpeed-MM's ``chunk_kda_naive`` (pure
PyTorch, autograd-friendly, functionally equivalent to
``fla.ops.kda.chunk_kda``).
"""

import functools
import os

import torch
import torch.nn as nn
import torch.nn.functional as F


def tensor_cache(fn):
    """Identity stand-in for ``fla.utils.tensor_cache``.

    The real decorator memoizes results keyed by CPU tensor content. The eager
    fallback simply recomputes; correctness is unaffected.
    """
    functools.wraps(fn)
    return fn


def prepare_lens_from_mask(mask: torch.Tensor) -> torch.Tensor:
    return (mask != 0).sum(-1)


def prepare_cu_seqlens_from_mask(mask: torch.Tensor, dtype=torch.int32) -> torch.Tensor:
    lens = prepare_lens_from_mask(mask)
    cu = torch.cat([torch.zeros(1, device=mask.device, dtype=lens.dtype), lens.cumsum(0)])
    return cu.to(dtype)


class ShortConvolution(nn.Conv1d):
    """Eager depthwise causal short convolution with the fla call interface.

    ``forward(x, cache=None, output_final_state=False, cu_seqlens=None)`` where
    ``x`` is ``[B, T, H]`` (or packed ``[1, total_T, H]`` together with
    ``cu_seqlens``). Output is ``silu(conv(x))``; the cache stores the last
    ``kernel_size - 1`` *inputs*. The varlen+cache combination (packed decode)
    is not exercised by training and is not supported here.
    """

    def __init__(self, hidden_size: int, kernel_size: int, activation: str | None = "silu", **kwargs):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=False)
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.activation = activation

    def _activation_fn(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation in ("silu", "swish"):
            return F.silu(x)
        if self.activation is None:
            return x
        raise NotImplementedError(f"Unsupported activation for ShortConvolution: {self.activation!r}")

    def forward(self, x, cache=None, output_final_state=False, cu_seqlens=None, channel_slice=None, **kwargs):
        """``channel_slice=(start, end)`` runs the depthwise conv on a contiguous
        channel range only (KDA SP head-sharding: the input then carries just
        this rank's head slice, so the weight is sliced to match)."""
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if cu_seqlens is not None:
            cu = cu_seqlens.tolist()
            segments, states = [], []
            for s, e in zip(cu[:-1], cu[1:]):
                seg, st = self._conv_seq(x[:, s:e, :], cache, output_final_state, channel_slice)
                segments.append(seg)
                if st is not None:
                    states.append(st)
            y = torch.cat(segments, dim=1)
            final_state = torch.stack(states, dim=0) if states else None
            return y, final_state
        return self._conv_seq(x, cache, output_final_state, channel_slice)

    def _conv_seq(self, x, cache, output_final_state, channel_slice=None):
        x_t = x.transpose(1, 2)
        if cache is not None:
            x_t = torch.cat([cache.to(x_t.dtype), x_t], dim=-1)
        else:
            x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        weight = self.weight if channel_slice is None else self.weight[channel_slice[0] : channel_slice[1]]
        y = F.conv1d(x_t, weight, groups=weight.shape[0])
        final_state = x_t[..., -(self.kernel_size - 1) :] if output_final_state else None
        return self._activation_fn(y).transpose(1, 2), final_state


class FusedRMSNormGated(nn.Module):
    """Eager gated RMSNorm: ``rms_norm(x) * sigmoid(gate)`` (fla-compatible interface)."""

    def __init__(self, dim: int, eps: float = 1e-5, activation: str = "sigmoid", **kwargs):
        super().__init__()
        self.eps = eps
        self.activation = activation

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x_f = x.float()
        x_f = x_f * torch.rsqrt(x_f.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.activation == "sigmoid":
            g = torch.sigmoid(g.float())
        elif self.activation in ("silu", "swish"):
            g = F.silu(g.float())
        else:
            raise NotImplementedError(f"Unsupported gated activation: {self.activation!r}")
        return (x_f * g).to(dtype)


def fused_recurrent_kda(*args, **kwargs):
    raise NotImplementedError(
        "fused_recurrent_kda eager fallback is not implemented; it is only used on the "
        "single-token decode path. Use chunk mode (training) or install fla."
    )


# Vendored from MindSpeed-MM mindspeed_mm/fsdp/ops/kda/chunk_kda_naive.py


def l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Aligned with triton l2norm_fwd: normalize in fp32, cast back to the input dtype."""
    original_dtype = x.dtype
    x = x.float()
    inv_norm = torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)
    return (x * inv_norm).to(original_dtype)


def kda_gate(
    g: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
) -> torch.Tensor:
    """
    Gate transform aligned with the gate part of triton kda_gate_chunk_cumsum (fp32).

    Args:
        g: Raw gate input of shape [..., H, K].
        A_log: Shape [H].
        dt_bias: Shape [H * K], optional.
        lower_bound: Forget gate lower bound, optional.

    Returns:
        Log-space decay of shape [..., H, K], fp32.
    """
    H, K = g.shape[-2:]
    g = g.float()
    if dt_bias is not None:
        g = g + dt_bias.float().view(H, K)
    A = A_log.float().view(H, 1).exp()
    if lower_bound is not None:
        g = lower_bound * torch.sigmoid(A * g)
    else:
        g = -A * F.softplus(g)
    return g


def _chunk_kda_core(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    chunk_size: int,
    return_h: bool = False,
):
    """
    Chunk KDA forward for equal-length sequences (small-op version,
    WY representation + inter-chunk recurrence).

    Args:
        q, k, g: [B, T, H, K], fp32 (q/k already l2-normalized, g already
            gate-transformed).
        v: [B, T, H, V], fp32.
        beta: [B, T, H], fp32, used as-is (no sigmoid).
        scale: Attention scale.
        initial_state: [B, H, K, V] or None.
        output_final_state: Whether to return the final state.
        chunk_size: Chunk size (64, same as the triton version).
        return_h: Whether to return the intermediate state h at each chunk entry.

    Returns:
        o: [B, T, H, V], fp32.
        final_state: [B, H, K, V] fp32 or None.
        h: [B, NT, H, K, V] fp32 or None.
    """
    B, T, H, K = k.shape
    V = v.shape[-1]

    # switch to head-first layout [B, H, T, D]
    q, k, v, g, beta = [x.transpose(1, 2).contiguous() for x in (q, k, v, g, beta)]

    # pad to a multiple of chunk_size
    pad_size = (chunk_size - T % chunk_size) % chunk_size
    total_length = T + pad_size
    q = F.pad(q, (0, 0, 0, pad_size)) * scale
    k = F.pad(k, (0, 0, 0, pad_size))
    v = F.pad(v, (0, 0, 0, pad_size))
    g = F.pad(g, (0, 0, 0, pad_size))
    beta = F.pad(beta, (0, pad_size))

    v_beta = v * beta.unsqueeze(-1)
    k_beta = k * beta.unsqueeze(-1)

    # reshape to [B, H, NT, C, D]
    q, k, v, g, k_beta, v_beta = [x.reshape(B, H, -1, chunk_size, x.shape[-1]) for x in (q, k, v, g, k_beta, v_beta)]
    NT = q.shape[2]

    # chunk-local cumsum, same as triton chunk_local_cumsum
    g = g.cumsum(dim=-2)

    # ---------- intra chunk: WY representation ----------
    # decay_mask[i, j] = exp(g_i - g_j)
    # Clamp the masked upper-triangular region to avoid exp overflow to inf
    # (masked to zero in the forward, but inf would produce NaN gradients in
    # the backward); 80 is below the fp32 exp overflow threshold (~88.7).
    mask_lower = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=0)
    g_diff = (g.unsqueeze(-2) - g.unsqueeze(-3)).clamp(max=80)
    decay_mask = g_diff.exp()

    # Akk[i, j] = -beta_i * <k_i, k_j> * exp(g_i - g_j), strictly lower triangular
    attn = -(k_beta.unsqueeze(-2) * k.unsqueeze(-3) * decay_mask).sum(dim=-1)
    attn = attn.masked_fill(mask_lower, 0)
    # forward substitution for (I + tril(Akk))^{-1} (row by row, same
    # recurrence as the triton kernel)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)

    u = attn @ v_beta  # [B, H, NT, C, V]
    w = attn @ (k_beta * g.exp())  # [B, H, NT, C, K]

    # ---------- inter-chunk recurrence ----------
    if initial_state is None:
        S = torch.zeros(B, H, K, V, dtype=torch.float32, device=q.device)
    else:
        S = initial_state.to(torch.float32)
    o = torch.zeros_like(v)
    h = torch.zeros(B, NT, H, K, V, dtype=torch.float32, device=q.device) if return_h else None

    mask_intra = torch.triu(torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        if return_h:
            h[:, :, i] = S
        # inter: (q * exp(g)) @ S (q already carries the scale)
        attn_inter = (q_i * g_i.exp()) @ S
        # intra: Aqk[i, j] = <q_i, k_j> * exp(g_i - g_j), lower triangular incl. diagonal
        attn_intra = (q_i.unsqueeze(-2) * k_i.unsqueeze(-3) * decay_mask[:, :, i]).sum(dim=-1)
        attn_intra = attn_intra.masked_fill(mask_intra, 0)
        # delta rule: remove the old value already stored in the state
        v_new = u_i - w_i @ S
        o[:, :, i] = attn_inter + attn_intra @ v_new
        # state update: decay the whole state by exp(g_last), then write this chunk's delta
        S = S * g_i[:, :, -1].exp().unsqueeze(-1) + (k_i * (g_i[:, :, -1:] - g_i).exp()).transpose(-1, -2) @ v_new

    final_state = S if output_final_state else None

    o = o.reshape(B, H, total_length, V)[:, :, :T]
    o = o.transpose(1, 2).contiguous()
    return o, final_state, h


def chunk_kda_naive(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    allow_neg_eigval: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    cu_seqlens_cpu: torch.LongTensor | None = None,
    safe_gate: bool = False,
    lower_bound: float | None = None,
    disable_recompute: bool = False,
    return_intermediate_states: bool = False,
    transpose_state_layout: bool = False,
    chunk_size: int = 64,
    **kwargs,
):
    r"""
    A small-operator (pure PyTorch) implementation of chunk_kda, with the
    same interface and semantics as ``fla.ops.kda.chunk_kda``
    (flash-linear-attention v0.5.2).

    Args:
        q (torch.Tensor): Queries of shape `[B, T, H, K]`.
        k (torch.Tensor): Keys of shape `[B, T, H, K]`.
        v (torch.Tensor): Values of shape `[B, T, H, V]`.
        g (torch.Tensor): Gate input of shape `[B, T, H, K]`.
            With `use_gate_in_kernel=True` this is the raw input and the gate
            transform is applied internally; otherwise it is expected to be
            the pre-computed log-space decay.
        beta (torch.Tensor): Betas of shape `[B, T, H]`.
            Raw values by default; with `use_beta_sigmoid_in_kernel=True`,
            `beta = scale * sigmoid(beta)` is applied internally in fp32
            (scale = 2.0 if `allow_neg_eigval` else 1.0).
        scale (Optional[float]): Attention scale, defaults to `K ** -0.5`.
        initial_state (Optional[torch.Tensor]): Initial state of shape
            `[N, H, K, V]` (`[N, H, V, K]` with `transpose_state_layout=True`),
            fp32. N equals B for equal-length inputs, or the number of
            sequences for varlen inputs.
        output_final_state (bool): Whether to return the final state.
        use_qk_l2norm_in_kernel (bool): Whether to apply l2norm to q/k internally.
        use_gate_in_kernel (bool): Whether to apply the KDA gate transform
            internally. When True, `A_log` ([H]) and the optional `dt_bias`
            ([H * K]) must be passed via kwargs.
        use_beta_sigmoid_in_kernel (bool): Whether to apply
            `scale * sigmoid(beta)` internally. Default: `False`.
        allow_neg_eigval (bool): Only takes effect together with
            `use_beta_sigmoid_in_kernel=True`, scaling sigmoid(beta) by 2.0.
            Default: `False`.
        cu_seqlens (Optional[torch.LongTensor]): Cumulative sequence lengths
            `[N+1]` for varlen mode; the input must be a flattened batch with
            B == 1.
        safe_gate (bool): Kernel performance hint, no semantic effect;
            accepted and ignored.
        lower_bound (Optional[float]): Forget gate lower bound, used together
            with `use_gate_in_kernel=True`.
        disable_recompute (bool): Kernel performance hint, no semantic
            effect; accepted and ignored.
        return_intermediate_states (bool): Whether to additionally return the
            intermediate state h at each chunk entry.
        transpose_state_layout (bool): Use the transposed state layout
            `[N, H, V, K]`.
        chunk_size (int): Chunk size, default 64 (same as the triton version).

    Returns:
        - `return_intermediate_states=False`: (o, final_state)
            o: `[B, T, H, V]`, same dtype as the input q.
            final_state: `[N, H, K, V]` (`[N, H, V, K]` with
            `transpose_state_layout=True`), fp32; None when
            `output_final_state=False`.
        - `return_intermediate_states=True`: (o, final_state, h)
            h: `[B, NT, H, K, V]` (`[B, NT, H, V, K]` with
            `transpose_state_layout=True`), same dtype as the input q.
            In varlen mode B == 1 and NT is the total number of chunks across
            all sequences.
    """
    A_log = kwargs.get("A_log")
    dt_bias = kwargs.get("dt_bias")
    if use_gate_in_kernel:
        assert A_log is not None, "A_log must be provided when use_gate_in_kernel=True."
    if allow_neg_eigval and not use_beta_sigmoid_in_kernel:
        raise ValueError("`allow_neg_eigval=True` requires `use_beta_sigmoid_in_kernel=True`.")
    assert q.shape == k.shape == g.shape, "q, k, g must have the same shape."
    assert beta.shape == q.shape[:3], "beta must be of shape (batch size, seq len, num of head)."
    assert v.shape == (*q.shape[:3], v.shape[-1]), "v must be of shape (batch size, seq len, num of head, head dim)."
    if cu_seqlens is not None:
        assert q.shape[0] == 1, f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."

    if scale is None:
        scale = k.shape[-1] ** -0.5
    input_dtype = q.dtype

    # 1. l2norm (normalize in fp32, then cast back to the input dtype, same as triton l2norm_fwd)
    if use_qk_l2norm_in_kernel:
        q = l2norm(q)
        k = l2norm(k)

    # 2. gate transform (fp32)
    if use_gate_in_kernel:
        g = kda_gate(g, A_log, dt_bias, lower_bound)
    else:
        g = g.float()
    q, k, v, beta = q.float(), k.float(), v.float(), beta.float()

    # 3. beta sigmoid in fp32 (same as fla's fused_beta_sigmoid)
    if use_beta_sigmoid_in_kernel:
        beta = torch.sigmoid(beta) * (2.0 if allow_neg_eigval else 1.0)

    # 4. state layout: internally always [N, H, K, V]
    if transpose_state_layout and initial_state is not None:
        initial_state = initial_state.transpose(-1, -2)

    # 5. per-sequence computation (each sequence is independent in varlen mode)
    if cu_seqlens is None:
        o, final_state, h = _chunk_kda_core(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            output_final_state,
            chunk_size,
            return_h=return_intermediate_states,
        )
    else:
        cu = cu_seqlens.tolist()
        outs, states, hs = [], [], []
        for n in range(len(cu) - 1):
            s, e = cu[n], cu[n + 1]
            h0 = initial_state[n : n + 1] if initial_state is not None else None
            o_n, state_n, h_n = _chunk_kda_core(
                q[:, s:e],
                k[:, s:e],
                v[:, s:e],
                g[:, s:e],
                beta[:, s:e],
                scale,
                h0,
                output_final_state,
                chunk_size,
                return_h=return_intermediate_states,
            )
            outs.append(o_n)
            if output_final_state:
                states.append(state_n)
            if return_intermediate_states:
                hs.append(h_n)
        o = torch.cat(outs, dim=1)
        final_state = torch.cat(states, dim=0) if output_final_state else None
        h = torch.cat(hs, dim=1) if return_intermediate_states else None

    # 6. restore layout and dtype
    o = o.to(input_dtype)
    if transpose_state_layout:
        if final_state is not None:
            final_state = final_state.transpose(-1, -2)
        if h is not None:
            h = h.transpose(-1, -2)
    if h is not None:
        h = h.to(input_dtype)

    if return_intermediate_states:
        return o, final_state, h
    return o, final_state


chunk_kda = chunk_kda_naive


# ==============================================================================
# Optional timing instrumentation for the head-sharded KDA forward.
# Enabled via K3_KDA_SP_TIMING=1; event pairs are resolved at interpreter exit
# (no device sync while training) and dumped per rank. Imported by
# the generated modeling files (gpu + npu) alongside the kernels above.
# ==============================================================================
_KDA_TIMING_EVENTS: list = []
_KDA_TIMING_REGISTERED = False


def _kda_timeit(phase: str):
    """Return a STARTED phase timer; call ``__exit__()`` on it to stop."""

    class _PhaseTimer:
        def __init__(self):
            _ev = torch.npu if hasattr(torch, "npu") else torch.cuda
            self.e0 = _ev.Event(enable_timing=True)
            self.e1 = _ev.Event(enable_timing=True)
            self.e0.record()

        def __exit__(self, *exc):
            self.e1.record()
            _KDA_TIMING_EVENTS.append((phase, self.e0, self.e1))

    return _PhaseTimer()


def _kda_timing_dump():
    rank = os.environ.get("RANK", "?")
    totals = {}
    for phase, e0, e1 in _KDA_TIMING_EVENTS:
        try:
            us = e0.elapsed_time(e1) * 1000.0
        except Exception:
            continue
        t, c = totals.get(phase, (0.0, 0))
        totals[phase] = (t + us, c + 1)
    with open(f"/tmp/kda_timing_rank{rank}.txt", "w") as f:
        f.write("phase\ttotal_ms\tcalls\tus_per_call\n")
        for phase, (t, c) in sorted(totals.items(), key=lambda kv: -kv[1][0]):
            f.write(f"{phase}\t{t / 1000.0:.1f}\t{c}\t{t / max(c, 1):.0f}\n")


def _kda_timing_maybe_register():
    global _KDA_TIMING_REGISTERED
    if not _KDA_TIMING_REGISTERED:
        import atexit

        atexit.register(_kda_timing_dump)
        _KDA_TIMING_REGISTERED = True

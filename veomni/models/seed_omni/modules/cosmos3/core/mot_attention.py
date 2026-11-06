"""Joint two-pass attention for the Cosmos3 mixture-of-towers block.

The two towers share one softmax but not one projection. That is two kernel calls:
``und`` tokens attend causally among themselves, ``gen`` tokens attend to every
token of their own sample (und K/V followed by gen K/V).

A packed (TND) micro-batch runs each pass as one varlen call: the reasoner pass
uses ``und -> und`` segments, the generation pass ``gen -> und+gen`` segments.
Each sample's K/V block is contiguous (:func:`.packed_layout.interleave_und_gen`),
so samples are isolated by the layout rather than by a mask.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from veomni.ops import VeomniOp

from .packed_layout import PackedLayout, interleave_und_gen


# Attention rows that accept a packed call with independent ``cu_seq_lens_q`` /
# ``cu_seq_lens_k``. ``eager`` is served by :func:`packed_visibility_mask` instead.
PACKED_VARLEN_IMPLS = frozenset(
    {
        "sdpa",
        "veomni_sdpa",
        "flash_attention_2",
        "flash_attention_2_hub",
        "flash_attention_3",
        "flash_attention_3_hub",
        "flash_attention_4",
        "veomni_flash_attention_2",
        "veomni_flash_attention_2_hub",
        "veomni_flash_attention_3",
        "veomni_flash_attention_3_hub",
        "veomni_flash_attention_4",
    }
)


class MotRMSNorm(nn.Module):
    """RMSNorm through the ``rms_norm/standard`` op."""

    def __init__(self, size: int, *, eps: float, config=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))
        impl = config.ops_config.rms_norm_implementation if config is not None else "eager"
        self.veomni_rms_norm = VeomniOp("rms_norm", "standard", impl)

    def forward(self, hidden: Tensor) -> Tensor:
        return self.veomni_rms_norm(hidden, self.weight, eps=self.eps)


def bind_mot_attention(module: nn.Module, *, impl: str | None = None, config=None) -> None:
    """Attach the ``attention/standard`` handle and the attributes HF attention interfaces read."""
    if impl is None:
        impl = config.ops_config.attn_implementation if config is not None else "veomni_sdpa"
    module.veomni_attn = VeomniOp("attention", "standard", impl)
    module.is_causal = False
    module.layer_idx = getattr(module, "layer_idx", None)
    module.num_key_value_groups = module.num_attention_heads // module.num_key_value_heads
    module.config = SimpleNamespace(_attn_implementation=impl)


def eager_attention_forward(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None = None,
    dropout: float = 0.0,
    scaling: float | None = None,
    is_causal: bool | None = None,
    **kwargs,
) -> tuple[Tensor, None]:
    """Model-local ``eager`` row: plain SDPA math."""
    key = _repeat_kv(key, module.num_key_value_groups)
    value = _repeat_kv(value, module.num_key_value_groups)
    causal = bool(module.is_causal) if is_causal is None else bool(is_causal)
    output = F.scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=attention_mask,
        dropout_p=dropout,
        scale=scaling,
        is_causal=causal and attention_mask is None,
    )
    return output.transpose(1, 2).contiguous(), None


def _repeat_kv(hidden: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden
    batch, heads, length, head_dim = hidden.shape
    hidden = hidden[:, :, None, :, :].expand(batch, heads, groups, length, head_dim)
    return hidden.reshape(batch, heads * groups, length, head_dim)


def _attend(
    module: nn.Module, query: Tensor, key: Tensor, value: Tensor, *, is_causal: bool, scaling, dropout
) -> Tensor:
    """One dense attention call on ``(T, heads, head_dim)``; returns the same layout."""
    output, _ = module.veomni_attn(
        module,
        query.unsqueeze(0).transpose(1, 2),
        key.unsqueeze(0).transpose(1, 2),
        value.unsqueeze(0).transpose(1, 2),
        None,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        skip_ulysses=True,
    )
    return output.squeeze(0)


def packed_visibility_mask(
    query_lengths: Sequence[int],
    key_lengths: Sequence[int],
    *,
    is_causal: bool,
    device: torch.device | str | None = None,
) -> Tensor:
    """Block-diagonal ``(1, 1, sum(q), sum(k))`` visibility of a packed call."""
    query_lengths = tuple(int(length) for length in query_lengths)
    key_lengths = tuple(int(length) for length in key_lengths)
    if is_causal and query_lengths != key_lengths:
        raise ValueError(f"a causal packed call needs equal segment lengths, got {query_lengths} and {key_lengths}.")
    query_total, key_total = sum(query_lengths), sum(key_lengths)
    mask = torch.zeros((query_total, key_total), dtype=torch.bool, device=device)
    query_start = key_start = 0
    for query_length, key_length in zip(query_lengths, key_lengths):
        block = torch.ones((query_length, key_length), dtype=torch.bool, device=device)
        if is_causal:
            block = block.tril()
        mask[query_start : query_start + query_length, key_start : key_start + key_length] = block
        query_start += query_length
        key_start += key_length
    return mask.view(1, 1, query_total, key_total)


def packed_attention_impl(module: nn.Module) -> str:
    """The resolved attention row of ``module`` (``eager`` / ``veomni_sdpa`` / ...)."""
    return str(module.veomni_attn.impl)


def _attend_varlen(
    module: nn.Module,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    cu_query: Tensor,
    cu_key: Tensor,
    query_lengths: Sequence[int],
    key_lengths: Sequence[int],
    is_causal: bool,
    scaling: float | None,
    dropout: float,
) -> Tensor:
    """One packed (TND) attention call on ``(T, heads, head_dim)`` tensors."""
    impl = packed_attention_impl(module)
    if impl == "eager":
        mask = packed_visibility_mask(query_lengths, key_lengths, is_causal=is_causal, device=query.device)
        output = F.scaled_dot_product_attention(
            query.unsqueeze(0).transpose(1, 2),
            _repeat_kv(key.unsqueeze(0).transpose(1, 2), module.num_key_value_groups),
            _repeat_kv(value.unsqueeze(0).transpose(1, 2), module.num_key_value_groups),
            attn_mask=mask,
            dropout_p=dropout,
            scale=scaling,
        )
        return output.squeeze(0).transpose(0, 1).contiguous()
    if impl not in PACKED_VARLEN_IMPLS:
        # A dense call over the whole pack would let samples attend to each other.
        raise NotImplementedError(f"Cosmos3 packed attention does not support the {impl!r} attention row.")
    output, _ = module.veomni_attn(
        module,
        query.unsqueeze(0).transpose(1, 2),
        key.unsqueeze(0).transpose(1, 2),
        value.unsqueeze(0).transpose(1, 2),
        None,
        dropout=dropout,
        scaling=scaling,
        is_causal=is_causal,
        skip_ulysses=True,
        cu_seq_lens_q=cu_query,
        cu_seq_lens_k=cu_key,
        max_length_q=max(int(length) for length in query_lengths),
        max_length_k=max(int(length) for length in key_lengths),
    )
    return output.squeeze(0)


def joint_attention(
    module: nn.Module,
    q_und: Tensor,
    k_und: Tensor,
    v_und: Tensor,
    q_gen: Tensor,
    k_gen: Tensor,
    v_gen: Tensor,
    *,
    k_und_for_gen: Tensor | None = None,
    scaling: float | None = None,
    dropout: float = 0.0,
    layout: PackedLayout | None = None,
) -> tuple[Tensor, Tensor]:
    """Causal reasoner pass + full generation pass; ``layout=None`` is one dense sample.

    ``k_und_for_gen`` optionally replaces the und keys in the generation pass only.
    """
    und_keys = k_und if k_und_for_gen is None else k_und_for_gen
    if layout is None:
        out_und = _attend(module, q_und, k_und, v_und, is_causal=True, scaling=scaling, dropout=dropout)
        full_key = torch.cat((und_keys, k_gen), dim=0)
        full_value = torch.cat((v_und, v_gen), dim=0)
        out_gen = _attend(module, q_gen, full_key, full_value, is_causal=False, scaling=scaling, dropout=dropout)
        return out_und, out_gen

    out_und = _attend_varlen(
        module,
        q_und,
        k_und,
        v_und,
        cu_query=layout.und_cu,
        cu_key=layout.und_cu,
        query_lengths=layout.und_lengths,
        key_lengths=layout.und_lengths,
        is_causal=True,
        scaling=scaling,
        dropout=dropout,
    )
    out_gen = _attend_varlen(
        module,
        q_gen,
        interleave_und_gen(und_keys, k_gen, layout),
        interleave_und_gen(v_und, v_gen, layout),
        cu_query=layout.gen_cu,
        cu_key=layout.joint_cu,
        query_lengths=layout.gen_lengths,
        key_lengths=layout.joint_lengths,
        is_causal=False,
        scaling=scaling,
        dropout=dropout,
    )
    return out_und, out_gen


class MotAttention(nn.Module):
    """Two QKVO projection sets (one per tower) feeding one joint attention.

    Attribute names are the released checkpoint's: the reasoner tower uses
    ``to_q/to_k/to_v/to_out`` and ``norm_q/norm_k``, the generation tower the
    diffusers "added" spellings ``add_{q,k,v}_proj/to_add_out`` and
    ``norm_added_{q,k}``.
    """

    def __init__(self, config, *, layer_idx: int = 0) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        head_dim = config.head_dim
        bias = config.attention_bias
        eps = config.rms_norm_eps
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.scaling = head_dim**-0.5
        self.attention_dropout = config.attention_dropout

        query_out = self.num_attention_heads * head_dim
        key_value_out = self.num_key_value_heads * head_dim
        self.to_q = nn.Linear(hidden_size, query_out, bias=bias)
        self.to_k = nn.Linear(hidden_size, key_value_out, bias=bias)
        self.to_v = nn.Linear(hidden_size, key_value_out, bias=bias)
        self.to_out = nn.Linear(query_out, hidden_size, bias=bias)

        def qk_norm(enabled: bool) -> nn.Module:
            return MotRMSNorm(head_dim, eps=eps, config=config) if enabled else nn.Identity()

        # Registration order is load-bearing and must not be regrouped: it fixes the order of
        # ``named_parameters()``, which FSDP2 uses to lay out each block's flat gradient buffer
        # and its manual-prefetch list. Reordering these submodules leaves the forward bitwise
        # equal but shifts the last bit of some gradients, which bf16 weights and Adam amplify
        # into a visibly different grad_norm within two steps at dp=8 (dp=4 is unaffected).
        self.norm_q = qk_norm(config.qk_norm_for_text)
        self.norm_k = qk_norm(config.qk_norm_for_text)
        # Only when the generation pass normalises K but the reasoner does not.
        self.k_norm_und_for_gen = (
            MotRMSNorm(head_dim, eps=eps, config=config)
            if config.use_und_k_norm_for_gen and config.qk_norm_for_diffusion and not config.qk_norm_for_text
            else None
        )
        self.norm_added_q = qk_norm(config.qk_norm_for_diffusion)
        self.norm_added_k = qk_norm(config.qk_norm_for_diffusion)
        self.add_q_proj = nn.Linear(hidden_size, query_out, bias=bias)
        self.add_k_proj = nn.Linear(hidden_size, key_value_out, bias=bias)
        self.add_v_proj = nn.Linear(hidden_size, key_value_out, bias=bias)
        self.to_add_out = nn.Linear(query_out, hidden_size, bias=bias)

        rope_impl = config.ops_config.rotary_pos_emb_implementation if config is not None else "eager"
        self.veomni_rope = VeomniOp("rope", "full", rope_impl)
        bind_mot_attention(self, config=config)

    def _project(self, proj_q: nn.Linear, proj_k: nn.Linear, proj_v: nn.Linear, hidden: Tensor):
        query = proj_q(hidden).view(-1, self.num_attention_heads, self.head_dim)
        key = proj_k(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        value = proj_v(hidden).view(-1, self.num_key_value_heads, self.head_dim)
        return query, key, value

    def forward(
        self,
        hidden_und: Tensor,
        hidden_gen: Tensor,
        *,
        und_rope: tuple[Tensor, Tensor],
        gen_rope: tuple[Tensor, Tensor],
        layout: PackedLayout | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``(L_und, hidden)``, ``(L_gen, hidden)`` in, the same shapes out."""
        q_und, k_und, v_und = self._project(self.to_q, self.to_k, self.to_v, hidden_und)
        q_und = self.norm_q(q_und)
        k_und = self.norm_k(k_und)
        k_und_for_gen = None
        if self.k_norm_und_for_gen is not None:
            k_und_for_gen = self.veomni_rope(k_und, self.k_norm_und_for_gen(k_und), und_rope[0], und_rope[1])[1]
        q_und, k_und = self.veomni_rope(q_und, k_und, und_rope[0], und_rope[1])

        q_gen, k_gen, v_gen = self._project(self.add_q_proj, self.add_k_proj, self.add_v_proj, hidden_gen)
        q_gen = self.norm_added_q(q_gen)
        k_gen = self.norm_added_k(k_gen)
        q_gen, k_gen = self.veomni_rope(q_gen, k_gen, gen_rope[0], gen_rope[1])

        out_und, out_gen = joint_attention(
            self,
            q_und,
            k_und,
            v_und,
            q_gen,
            k_gen,
            v_gen,
            k_und_for_gen=k_und_for_gen,
            scaling=self.scaling,
            dropout=self.attention_dropout,
            layout=layout,
        )
        out_und = self.to_out(out_und.reshape(out_und.shape[0], -1))
        out_gen = self.to_add_out(out_gen.reshape(out_gen.shape[0], -1))
        return out_und, out_gen

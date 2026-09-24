"""Hardware-agnostic eager fallback for the Qwen4-Exp QSA indexer."""

from __future__ import annotations

import torch

from .common import project_qsa_indexer_inputs
from .reference import qsa_reference_mask


@torch.no_grad()
def qsa_indexer_forward_eager(
    indexer: torch.nn.Module,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: torch.Tensor,
    past_key_values=None,
    cu_seq_lens_q: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the exact eager QSA reference for any four-dimensional mask.

    ``cu_seq_lens_q`` is accepted for call-site compatibility. The packed
    block-diagonal mask already encodes the same boundaries, so no extra split is
    needed on the eager path.
    """
    del cu_seq_lens_q
    query, raw_keys, full_cos, full_sin = project_qsa_indexer_inputs(
        indexer,
        hidden_states,
        position_embeddings,
        past_key_values,
    )
    return qsa_reference_mask(
        query,
        raw_keys,
        full_cos,
        full_sin,
        attention_mask,
        indexer.k_layernorm.weight,
        rms_norm_eps=indexer.k_layernorm.eps,
        token_budget=indexer.token_budget,
        compress_ratio=indexer.compress_ratio,
    )

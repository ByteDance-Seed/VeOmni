"""
JanusTextEmbed — :class:`TextEmbed` + Janus image-boundary token emitters.

Why a Janus-specific subclass?
------------------------------
Boundary tokens — :code:`<begin_of_image>` / :code:`<end_of_image>` —
delimit a VQ image span in the AR stream.  Emitting them is a *model*
concern, not a *framework* concern: the FSM should not know that one
particular vocabulary uses ``100016`` / ``100593`` as image boundaries.

So instead of having the FSM "append a token" between states, we add
two explicit call-site methods on the model::

    text_embed.emit_image_start()  # → <boi> token + its inputs_embeds
    text_embed.emit_image_end()    # → <eoi> token + its inputs_embeds

The YAML wires these up as nodes (``emit_image_start`` /
``emit_image_end``) and the FSM body lists the corresponding edges.
The bridge states (``image_vq_start`` / ``image_vq_end``) run those
nodes the same way ``text_ar`` runs ``tok_encode`` — same call-site
convention, no special kwargs, no on_exit / on_enter hooks.

Output protocol
---------------
Both emit methods return the same key set as :meth:`TextEmbed.encode`
plus a synthesised ``last_token_id`` so an FSM ``token_match`` transition
on the boundary id fires immediately after the bridge state runs::

    {"input_ids":      LongTensor(B, 1) — the boundary id,
     "inputs_embeds":  FloatTensor(B, 1, hidden) — wte lookup of that id,
     "last_token_id":  LongTensor(B)   — the boundary id (transition trigger)}

Batch size is inferred from any tensor present in ``ctx`` so the emit
fits naturally into the FSM step body without needing an explicit
``batch_size`` argument.
"""

from typing import Any, Dict

import torch

from ...base.text_embed import TextEmbed
from .configuration import JanusTextEmbedConfig


class JanusTextEmbed(TextEmbed):
    """:class:`TextEmbed` + ``emit_image_start`` / ``emit_image_end``."""

    config_class = JanusTextEmbedConfig

    # ── Janus boundary-token emitters ─────────────────────────────────────────

    def emit_image_start(self, **ctx: Any) -> Dict[str, Any]:
        """Emit a single ``<begin_of_image>`` step (id from config)."""
        return self._emit(self.config.begin_of_image_token_id, ctx)

    def emit_image_end(self, **ctx: Any) -> Dict[str, Any]:
        """Emit a single ``<end_of_image>`` step (id from config)."""
        return self._emit(self.config.end_of_image_token_id, ctx)

    def _emit(self, token_id: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        batch_size = _infer_batch_size(ctx)

        ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
        inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        return {
            "input_ids": ids,
            "inputs_embeds": inputs_embeds,
            "last_token_id": ids.squeeze(-1),
        }


def _infer_batch_size(ctx: Dict[str, Any]) -> int:
    """Best-effort batch-size inference from tensors already in ``ctx``."""
    for key in ("input_ids", "inputs_embeds", "hidden_states", "attention_mask"):
        v = ctx.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.size(0))
    return 1

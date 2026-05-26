"""
JanusTextEncoder — :class:`TextEncoder` + Janus image-boundary token emitters.

Why a Janus-specific subclass?
------------------------------
Boundary tokens — :code:`<begin_of_image>` / :code:`<end_of_image>` —
delimit a VQ image span in the AR stream.  Emitting them is a *model*
concern, not a *framework* concern: the FSM should not know that one
particular vocabulary uses specific numeric ids as image boundaries.

So instead of having the FSM "append a token" between states, we add
two explicit call-site methods on the model::

    text_encoder.emit_image_start()  # → <boi> token + its inputs_embeds
    text_encoder.emit_image_end()    # → <eoi> token + its inputs_embeds

The YAML wires these up as nodes (``emit_image_start`` /
``emit_image_end``) and the FSM body lists the corresponding edges.
The bridge states (``image_vq_start`` / ``image_vq_end``) run those
nodes the same way ``text_ar`` runs ``tok_encode`` — same call-site
convention, no special kwargs, no on_exit / on_enter hooks.

FSM transition signals (inference)
----------------------------------
After :meth:`decode` samples a token, this module — not the outer FSM —
decides whether the token means "start image generation" or "text done"
and writes a one-shot string into ``ctx["module_signal"]``:

* ``"start_image_gen"`` — sampled token is ``<begin_of_image>``
* ``"text_done"``       — sampled token is ``</s>`` (eos)

Token ids are resolved from the wired tokenizer via :meth:`set_tokenizer`.

Output protocol (emit methods)
------------------------------
Both emit methods return the same key set as :meth:`TextEncoder.encode`
plus ``last_token_id``::

    {"input_ids":      LongTensor(B, 1) — the boundary id,
     "inputs_embeds":  FloatTensor(B, 1, hidden) — wte lookup of that id,
     "last_token_id":  LongTensor(B)   — the boundary id}

Batch size is inferred from any tensor present in ``ctx`` so the emit
fits naturally into the FSM step body without needing an explicit
``batch_size`` argument.
"""

from typing import Any, Dict, Optional

import torch

from veomni.models.seed_omni.generation_graph import FSM_SIGNAL_KEY

from ...base.text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig


# Signal *values* written to ``ctx[FSM_SIGNAL_KEY]`` by :meth:`decode`.
SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"


class JanusTextEncoder(TextEncoder):
    """:class:`TextEncoder` + ``emit_image_start`` / ``emit_image_end``."""

    config_class = JanusTextEncoderConfig

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)
        self._tokenizer: Any = None
        self._boi_token_id: Optional[int] = None
        self._eoi_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Resolve Janus special-token ids from the global tokenizer."""
        self._tokenizer = tokenizer
        self._boi_token_id = _resolve_token_id(
            tokenizer,
            ("<begin_of_image>", getattr(tokenizer, "boi_token", None)),
        )
        self._eoi_token_id = _resolve_token_id(
            tokenizer,
            ("<end_of_image>", getattr(tokenizer, "eoi_token", None)),
        )
        eos = getattr(tokenizer, "eos_token_id", None)
        self._eos_token_id = int(eos) if eos is not None else None
        if self._boi_token_id is not None:
            self.config.begin_of_image_token_id = self._boi_token_id
        if self._eoi_token_id is not None:
            self.config.end_of_image_token_id = self._eoi_token_id

    # ── Inference: decode + FSM signals ───────────────────────────────────────

    def decode(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        out = super().decode(
            hidden_states=hidden_states,
            labels=labels,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )
        if labels is not None or "last_token_id" not in out:
            return out

        token_id = _scalar_token_id(out["last_token_id"])
        if token_id is not None:
            if self._boi_token_id is not None and token_id == self._boi_token_id:
                out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            elif self._eos_token_id is not None and token_id == self._eos_token_id:
                out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
        return out

    # ── Janus boundary-token emitters ─────────────────────────────────────────

    def emit_image_start(self, **ctx: Any) -> Dict[str, Any]:
        """Emit a single ``<begin_of_image>`` step."""
        token_id = self._boi_token_id if self._boi_token_id is not None else self.config.begin_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_start requires begin_of_image_token_id — "
                "call set_tokenizer() before inference."
            )
        return self._emit(token_id, ctx)

    def emit_image_end(self, **ctx: Any) -> Dict[str, Any]:
        """Emit a single ``<end_of_image>`` step."""
        token_id = self._eoi_token_id if self._eoi_token_id is not None else self.config.end_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_end requires end_of_image_token_id — "
                "call set_tokenizer() before inference."
            )
        return self._emit(token_id, ctx)

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


def _resolve_token_id(tokenizer: Any, candidates: tuple) -> Optional[int]:
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in candidates:
        if not cand:
            continue
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != unk:
            return int(tid)
    return None


def _scalar_token_id(value: Any) -> Optional[int]:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return None
        return int(value.reshape(-1)[0].item())
    if value is None:
        return None
    return int(value)


def _infer_batch_size(ctx: Dict[str, Any]) -> int:
    """Best-effort batch-size inference from tensors already in ``ctx``."""
    for key in ("input_ids", "inputs_embeds", "hidden_states", "attention_mask"):
        v = ctx.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.size(0))
    return 1

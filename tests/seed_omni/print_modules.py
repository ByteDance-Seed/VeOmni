"""Reusable print-only OmniModule stand-ins for graph-flow tests.

These modules implement the SeedOmni V2 :class:`OmniModule` mixin contract
without any real ML — every call appends a structured event to a shared
``log`` list, and tensors are replaced by string sentinels.  They let the
tests assert *graph behaviour* (topo order, FSM transitions, edge routing)
in isolation from torch / FSDP / weights.

Conventions
-----------
* Every module records ``"<name>.<method>(<sorted-kwarg-keys>)"`` on call.
* Real-shaped outputs are stand-in strings (e.g. ``"<embed:input_ids=10>"``)
  so downstream nodes can carry them through edge routing without relying
  on tensor maths.
* Modules that take part in the *training* graph emit a ``_loss`` key whose
  value is a zero-dim ``torch.Tensor`` — :class:`OmniModel.forward` sums
  them into the total scalar.  Modules that only run in inference omit
  ``_loss`` from their training-side outputs.

The design here mirrors a Janus-style joint training / generation setup
(text + vision encoders → AR backbone → text decode + VQ image decode)
without committing to specific module classes.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from veomni.models.seed_omni import OmniModule


# ── Base ──────────────────────────────────────────────────────────────────────


class _PrintBase(OmniModule, nn.Module):
    """Mixin-first base: ``OmniModule`` hooks + ``nn.Module`` for ``ModuleDict``.

    Subclasses override :meth:`forward` (and optionally other methods) to
    return whatever stand-in dict the test needs.  The shared ``log`` list
    is the canonical record of what happened — assert against it.
    """

    def __init__(self, name: str, log: List[str]):
        super().__init__()
        self._mod_name = name
        self._log = log

    def _record(self, method: str, **kwargs: Any) -> None:
        keys = sorted(kwargs.keys())
        self._log.append(f"{self._mod_name}.{method}({','.join(keys)})")


def _scalar_loss(value: float = 0.5) -> torch.Tensor:
    """Zero-dim tensor — matches the ``_loss`` protocol."""
    return torch.tensor(value, dtype=torch.float32)


# ── Vision encoder (training-only here) ───────────────────────────────────────


class PrintVisionEncoder(_PrintBase):
    """Single-method module: ``forward(pixel_values)`` → ``image_embeds``."""

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("forward", **kwargs)
        return {"image_embeds": "<vis_embeds>"}


# ── VQVAE: encode (training) + decode (training & inference) ─────────────────


class PrintVQVAE(_PrintBase):
    """Two call-sites on one module — the cross-modality codec.

    * ``encode(pixel_values)``       → ``gen_embeds`` + ``vq_token_ids``
      (training only; teacher-forces the AR backbone).
    * ``decode(hidden_states, gt_token_ids?)``
      → training: ``_loss`` (CE over VQ tokens at gen-image positions).
      → inference: ``embed`` (next-step input for the AR backbone).
    """

    def encode(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("encode", **kwargs)
        return {
            "gen_embeds": "<vq_gen_embeds>",
            "vq_token_ids": "<vq_token_ids>",
        }

    def decode(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("decode", **kwargs)
        if kwargs.get("gt_token_ids") is not None:
            return {"_loss": _scalar_loss(0.7)}
        return {"embed": "<vq_decode_embed>"}


# ── Text embedding head: encode (wte) + decode (LM head) ─────────────────────


class PrintTextEmbed(_PrintBase):
    """Word-token embedding + LM head packaged as one module.

    ``encode(input_ids)`` mirrors the wte lookup; ``decode(...)`` is the LM
    head.  In training the decode receives ``labels`` and emits ``_loss``;
    in inference it samples a token from a deterministic plan.

    Parameters
    ----------
    token_script:
        Sequence of ``last_token_id`` values to emit, one per
        ``decode`` call in inference.  Once exhausted the module emits
        ``2`` (``</s>``) forever.  This makes FSM transitions
        reproducible without any sampling logic.
    """

    def __init__(
        self,
        name: str,
        log: List[str],
        token_script: Optional[Sequence[int]] = None,
    ):
        super().__init__(name, log)
        self._token_script: List[int] = list(token_script or [])
        self._cursor: int = 0

    def encode(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("encode", **kwargs)
        return {"inputs_embeds": f"<wte:{kwargs.get('input_ids', '?')}>"}

    def decode(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("decode", **kwargs)
        if kwargs.get("labels") is not None:
            return {"_loss": _scalar_loss(0.3)}
        # Inference: sample from the canned script.
        if self._cursor < len(self._token_script):
            tok = self._token_script[self._cursor]
            self._cursor += 1
        else:
            tok = 2
        return {"input_ids": tok, "last_token_id": tok}

    def reset_cursor(self) -> None:
        """Reset the token-script cursor.  Call between independent generate runs."""
        self._cursor = 0


# ── AR backbone — has both forward (training) and generate_step (inference) ──


class PrintARBackbone(_PrintBase):
    """The shared AR backbone (e.g. Llama).

    * Training: ``forward(inputs_embeds, ...)`` → ``hidden_states`` + a tiny
      ``_loss`` to test multi-loss aggregation.
    * Inference: ``generate_step(...)`` → ``hidden_states`` only (no loss).
    """

    def forward(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("forward", **kwargs)
        return {
            "hidden_states": "<ar_hidden>",
            "_loss": _scalar_loss(0.2),
        }

    def generate_step(self, **kwargs: Any) -> Dict[str, Any]:
        self._record("generate_step", **kwargs)
        return {"hidden_states": "<ar_hidden_gen>"}

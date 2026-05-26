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

from typing import Any, Sequence

import torch
import torch.nn as nn

from veomni.models.seed_omni import OmniModule
from veomni.models.seed_omni.generation_graph import FSM_SIGNAL_KEY


# Signal *values* for ``ctx[FSM_SIGNAL_KEY]`` (mirrors JanusTextEncoder).
SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"
SIGNAL_IMAGE_COMPLETE = "image_complete"
TOK_BOI = 100016
TOK_EOI = 100593
TOK_EOS = 2


def _infer_batch_size(kwargs: dict[str, Any]) -> int:
    v = kwargs.get("input_ids")
    if isinstance(v, torch.Tensor) and v.dim() >= 1:
        return int(v.size(0))
    return 1


# ── Base ──────────────────────────────────────────────────────────────────────


class _PrintBase(OmniModule, nn.Module):
    """Mixin-first base: ``OmniModule`` hooks + ``nn.Module`` for ``ModuleDict``.

    Subclasses override :meth:`forward` (and optionally other methods) to
    return whatever stand-in dict the test needs.  The shared ``log`` list
    is the canonical record of what happened — assert against it.
    """

    def __init__(self, name: str, log: list[str]):
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
    """Single-method module: ``forward(pixel_values)`` → ``image_embeds``.

    Mirrors the inference fast-skip contract: when ``pixel_values`` is
    ``None`` the module returns ``{}`` and the FSM's permissive routing
    silently drops the outgoing edge.  Training-side dummy forward (the
    other path) is the trainer's responsibility — Step 2 territory.
    """

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        self._record("forward", **kwargs)
        if kwargs.get("pixel_values") is None:
            return {}
        return {"image_embeds": "<vis_embeds>"}

    def generate_step(self, **kwargs: Any) -> dict[str, Any]:
        return self.forward(**kwargs)


# ── VQVAE: encode (training) + decode (training & inference) ─────────────────


class PrintVQVAE(_PrintBase):
    """Two call-sites on one module — the cross-modality codec.

    * ``encode(pixel_values)``       → ``gen_embeds`` + ``vq_token_ids``
      (training only; teacher-forces the AR backbone).
    * ``decode(hidden_states, gt_token_ids?)``
      → training: ``_loss`` (CE over VQ tokens at gen-image positions).
      → inference: ``embed`` (next-step input for the AR backbone).

    Inference termination signal
    ----------------------------
    A real Janus VQ decoder loops over a fixed grid (24×24 = 576 patches)
    and raises an internal "image complete" event on the final patch; the
    FSM's ``image_vq`` state runs ``token_length: variable`` and listens
    for that event via a ``module_signal(image_complete)`` transition.

    Tests configure the simulated grid size via ``image_steps`` — after
    ``image_steps`` consecutive inference ``decode()`` calls the module
    appends ``module_signal=<image_complete>`` to its output dict and resets the
    counter (so the next image span starts fresh).  ``image_steps=None``
    (the default) disables the signal entirely.
    """

    def __init__(
        self,
        name: str,
        log: list[str],
        image_steps: int | None = None,
    ):
        super().__init__(name, log)
        self._image_steps: int | None = image_steps
        self._decode_calls: int = 0

    def encode(self, **kwargs: Any) -> dict[str, Any]:
        self._record("encode", **kwargs)
        return {
            "gen_embeds": "<vq_gen_embeds>",
            "vq_token_ids": "<vq_token_ids>",
        }

    def decode(self, **kwargs: Any) -> dict[str, Any]:
        self._record("decode", **kwargs)
        if kwargs.get("gt_token_ids") is not None:
            return {"_loss": _scalar_loss(0.7)}
        # Inference path — emit `image_complete` on the last patch of the
        # simulated grid, then reset the counter for the next image span.
        self._decode_calls += 1
        out: dict[str, Any] = {"embed": "<vq_decode_embed>"}
        if self._image_steps is not None and self._decode_calls >= self._image_steps:
            out[FSM_SIGNAL_KEY] = SIGNAL_IMAGE_COMPLETE
            self._decode_calls = 0
        return out


# ── Text embedding head: encode (wte) + decode (LM head) ─────────────────────


class PrintTextEmbed(_PrintBase):
    """Word-token embedding + LM head packaged as one module.

    ``encode(input_ids)`` mirrors the wte lookup; ``decode(...)`` is the LM
    head.  In training the decode receives ``labels`` and emits ``_loss``;
    in inference it samples a token from a deterministic plan.

    Parameters
    ----------
    token_script:
        Sequence of token ids to emit, one per ``decode`` call in
        inference.  Once exhausted the module emits ``2`` (``</s>``)
        forever.  This makes FSM transitions reproducible without any
        sampling logic.
    """

    def __init__(
        self,
        name: str,
        log: list[str],
        token_script: Sequence[int] | None = None,
    ):
        super().__init__(name, log)
        self._token_script: list[int] = list(token_script or [])
        self._cursor: int = 0

    def encode(self, **kwargs: Any) -> dict[str, Any]:
        self._record("encode", **kwargs)
        return {"inputs_embeds": f"<wte:{kwargs.get('input_ids', '?')}>"}

    def decode(self, **kwargs: Any) -> dict[str, Any]:
        self._record("decode", **kwargs)
        if kwargs.get("labels") is not None:
            return {"_loss": _scalar_loss(0.3)}
        # Inference: sample from the canned script, then emit module_signal flags.
        if self._cursor < len(self._token_script):
            tok = self._token_script[self._cursor]
            self._cursor += 1
        else:
            tok = TOK_EOS
        batch_size = _infer_batch_size(kwargs)
        out: dict[str, Any] = {
            "input_ids": torch.full((batch_size, 1), tok, dtype=torch.long),
        }
        if tok == TOK_BOI:
            out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
        elif tok == TOK_EOS:
            out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
        return out

    def reset_cursor(self) -> None:
        """Reset the token-script cursor.  Call between independent generate runs."""
        self._cursor = 0

    # ── Janus boundary-token emitters (mirrors JanusTextEncoder) ───────────────
    def emit_image_start(self, **kwargs: Any) -> dict[str, Any]:
        return self._emit("boi", 100016, **kwargs)

    def emit_image_end(self, **kwargs: Any) -> dict[str, Any]:
        return self._emit("eoi", TOK_EOI, **kwargs)

    def _emit(self, label: str, token_id: int, **kwargs: Any) -> dict[str, Any]:
        self._record(f"emit_{label}", **kwargs)
        batch_size = _infer_batch_size(kwargs)
        return {
            "input_ids": torch.full((batch_size, 1), token_id, dtype=torch.long),
            "inputs_embeds": f"<wte:{label}>",
        }


# ── AR backbone — has both forward (training) and generate_step (inference) ──


class PrintARBackbone(_PrintBase):
    """The shared AR backbone (e.g. Llama).

    * Training: ``forward(inputs_embeds, ...)`` → ``hidden_states`` + a tiny
      ``_loss`` to test multi-loss aggregation.
    * Inference: ``generate_step(...)`` → ``hidden_states`` only (no loss).
    """

    def forward(self, **kwargs: Any) -> dict[str, Any]:
        self._record("forward", **kwargs)
        return {
            "hidden_states": "<ar_hidden>",
            "_loss": _scalar_loss(0.2),
        }

    def generate_step(self, **kwargs: Any) -> dict[str, Any]:
        self._record("generate_step", **kwargs)
        return {"hidden_states": "<ar_hidden_gen>"}

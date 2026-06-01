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
plus the boundary ``input_ids`` step::

    {"input_ids":      LongTensor(B, 1) — the boundary id,
     "inputs_embeds":  FloatTensor(B, 1, hidden) — wte lookup of that id}

Batch size is inferred from any tensor present in ``ctx`` so the emit
fits naturally into the FSM step body without needing an explicit
``batch_size`` argument.
"""

import logging
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ....conversation import ConversationPart, latest_assistant_text_token_ids
from ....generation_graph import FSM_SIGNAL_KEY
from ....graph import scalar_token_id
from ...base.text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig


logger = logging.getLogger(__name__)


# Signal *values* written to ``ctx[FSM_SIGNAL_KEY]`` by :meth:`decode`.
SIGNAL_START_IMAGE_GEN = "start_image_gen"
SIGNAL_TEXT_DONE = "text_done"

# Per-decode-call kwargs the FSM might pass through.  Anything outside this
# set is ignored when filtering ``generation_kwargs`` for the LM head.
_SAMPLING_KWARGS = ("temperature", "top_p", "do_sample")

# Janus DeepSeek prompt-format markers — byte-for-byte mirror of the
# upstream Jinja chat template at
# ``transformers.models.janus``'s ``chat_template.jinja``.  Owned by
# this module because per the V2 design the text encoder fully dictates
# the on-the-wire layout — no upstream ``apply_chat_template`` call.
#
# Whitespace is baked into the marker strings themselves so the BPE
# tokenizer sees the same byte stream HF does (each ``\n`` becomes a
# real token id ``185`` — they are NOT zero-cost separators):
#   * Trailing ``\n\n`` on the system prompt → turn separator.
#   * Trailing ``" "`` on ``<|User|>:``       → matches ``"<|User|>: "``.
#   * Leading ``\n\n`` on ``<|Assistant|>:`` → turn separator.
# Plus :meth:`_inject_chat_template` lookahead-inserts a ``"\n"`` between
# consecutive same-role parts (image → user_text) to match the per-
# content ``\n`` the Jinja template emits inside a single message.
#
# The default system prompt mirrors
# :data:`transformers.models.janus.processing_janus.DEFAULT_SYSTEM_PROMPT`.
# Janus checkpoints were aligned with this prompt prepended; omitting it
# produces incoherent generations (verified empirically against HF).
_JANUS_SYSTEM_PROMPT = (
    "You are a helpful language and vision assistant. "
    "You are able to understand the visual content that the user provides, "
    "and assist the user with a variety of tasks using natural language."
    "\n\n"
)
_JANUS_USER_PREFIX = "<|User|>: "
_JANUS_ASSISTANT_PREFIX = "\n\n<|Assistant|>:"


class JanusTextEncoder(TextEncoder):
    """:class:`TextEncoder` + ``emit_image_start`` / ``emit_image_end``.

    Inference contract (``generate`` / ``decode`` / ``emit_image_*``)
    ----------------------------------------------------------------
    All inference call-sites consume / produce a ``conversation_list``
    (see :mod:`veomni.models.seed_omni.conversation`).  Training paths
    (``forward`` / ``encode``) still use the legacy ``input_ids`` +
    ``masked_scatter`` contract and are not affected by this subclass.
    """

    config_class = JanusTextEncoderConfig

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)
        self._tokenizer: Any = None
        self._bos_token_id: Optional[int] = None
        self._boi_token_id: Optional[int] = None
        self._eoi_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None
        # Understanding-image placeholder id (``<image_placeholder>``).  Each
        # ``image`` turn expands to ``num_image_tokens`` copies of this id in
        # the training ``input_ids`` so the backbone's ``masked_scatter``
        # injects the SigLIP patch embeddings at exactly those positions.
        self._image_token_id: Optional[int] = None

    def set_tokenizer(self, tokenizer: Any) -> None:
        """Resolve Janus special-token ids from the global tokenizer."""
        self._tokenizer = tokenizer
        bos = getattr(tokenizer, "bos_token_id", None)
        self._bos_token_id = int(bos) if bos is not None else None
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
        pad = getattr(tokenizer, "pad_token_id", None)
        # Pad falls back to eos when the tokenizer doesn't ship one — same
        # convention HF uses when building uncond inputs in
        # `JanusForConditionalGeneration.generate`
        # (`modeling_janus.py:1282-1285`).  We need a real id here, otherwise
        # the CFG uncond branch can't replace non-BOS positions.
        self._pad_token_id = int(pad) if pad is not None else self._eos_token_id
        self._image_token_id = _resolve_token_id(
            tokenizer,
            ("<image_placeholder>", getattr(tokenizer, "image_token", None)),
        )
        if self._boi_token_id is not None:
            self.config.begin_of_image_token_id = self._boi_token_id
        if self._eoi_token_id is not None:
            self.config.end_of_image_token_id = self._eoi_token_id

    # ── Training: conversation-list → tokenised batch (D4) ────────────────────

    def pre_forward(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Tokenise the raw training ``conversation_list`` in place.

        The SeedOmni V2 data pipeline hands every node the raw
        ``conversation_list`` (``list[list[dict]]`` of
        ``{type, value, role, loss_mask}`` — see
        :mod:`veomni.data.multimodal.seedomni_transform`).  The text
        encoder owns chat-templating + tokenisation, so here we convert
        that raw structure into the flat ``input_ids`` / ``labels`` /
        ``attention_mask`` / ``position_ids`` the backbone + LM head
        consume, expanding each ``image`` turn to ``num_image_tokens``
        copies of ``<image_placeholder>`` (the backbone scatters SigLIP
        patches onto those positions).

        Inference (``conversation_list`` of :class:`ConversationPart`, or a
        batch that already carries ``input_ids``) is passed through
        untouched — :meth:`generate` / :meth:`encode`'s AR path handle it.
        """
        conversation = kwargs.get("conversation_list")
        if kwargs.get("input_ids") is not None or not _is_training_conversation(conversation):
            return kwargs
        batch = self._tokenize_training_batch(conversation)
        kwargs.pop("conversation_list", None)
        kwargs.update(batch)
        return kwargs

    def encode(  # type: ignore[override]
        self,
        input_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """wte lookup + echo the tokenised fields produced by :meth:`pre_forward`.

        Delegates the embedding lookup to :class:`TextEncoder.encode`, then
        re-emits ``input_ids`` / ``labels`` / ``attention_mask`` /
        ``position_ids`` in the return dict so the training edges can route
        them to ``janus_llama`` (scatter mask + masks) and ``tok_decode``
        (CE labels).  ``OmniModel`` only propagates a node's *return* dict.
        """
        base = super().encode(input_ids=input_ids, past_key_values=past_key_values, **kwargs)
        if not base:
            return base
        out: Dict[str, Any] = dict(base)
        out["input_ids"] = input_ids
        if labels is not None:
            out["labels"] = labels
        if attention_mask is not None:
            out["attention_mask"] = attention_mask
        if position_ids is not None:
            out["position_ids"] = position_ids
        return out

    def _num_image_tokens(self) -> int:
        """Understanding-image placeholder count per image (Janus: 24×24=576)."""
        return int(getattr(self.config, "num_image_tokens", 576))

    def _tokenize_training_batch(self, conversation_list: List[List[Dict[str, Any]]]) -> Dict[str, torch.Tensor]:
        """Tokenise N raw conversations into a right-padded training batch.

        Mirrors the inference chat template (:meth:`_inject_chat_template`):
        ``<bos><system>\\n\\n<|User|>: [<boi>img<eoi>\\n]<user_text>\\n\\n
        <|Assistant|>:<assistant_text><eos>``.  Loss is taken only on
        assistant-role text turns flagged ``loss_mask=1`` (plus the closing
        ``<eos>``); every prompt / image / boundary position is ``-100``.
        """
        if self._tokenizer is None:
            raise RuntimeError(
                "JanusTextEncoder training tokenisation needs a tokenizer — "
                "call set_tokenizer(global_tokenizer) on the module first."
            )
        if self._bos_token_id is None or self._eos_token_id is None:
            raise RuntimeError("JanusTextEncoder.set_tokenizer did not resolve bos/eos ids.")

        seqs: List[List[int]] = []
        label_seqs: List[List[int]] = []
        for sample in conversation_list:
            ids, labels = self._tokenize_one(sample)
            seqs.append(ids)
            label_seqs.append(labels)

        max_len = max(len(s) for s in seqs)
        pad_id = self._pad_token_id if self._pad_token_id is not None else self._eos_token_id
        device = self.embed_tokens.weight.device

        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
        labels = torch.full((len(seqs), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(seqs, label_seqs)):
            n = len(ids)
            input_ids[i, :n] = torch.tensor(ids, dtype=torch.long)
            labels[i, :n] = torch.tensor(lab, dtype=torch.long)
            attention_mask[i, :n] = 1
        position_ids = torch.arange(max_len, dtype=torch.long).unsqueeze(0).expand(len(seqs), -1)

        return {
            "input_ids": input_ids.to(device),
            "labels": labels.to(device),
            "attention_mask": attention_mask.to(device),
            "position_ids": position_ids.to(device),
        }

    def _tokenize_one(self, sample: List[Dict[str, Any]]) -> tuple[List[int], List[int]]:
        """Tokenise one raw conversation → (input_ids, labels) python lists."""
        n_img = self._num_image_tokens()
        ids: List[int] = [int(self._bos_token_id)]
        labels: List[int] = [-100]

        def _emit(token_ids: List[int], supervised: bool) -> None:
            ids.extend(token_ids)
            labels.extend(token_ids if supervised else [-100] * len(token_ids))

        _emit(self._encode_text(_JANUS_SYSTEM_PROMPT), supervised=False)

        prev_role: Optional[str] = None
        last_supervised = False
        for idx, item in enumerate(sample):
            role = item.get("role")
            typ = item.get("type")
            value = item.get("value")
            supervised = role == "assistant" and bool(item.get("loss_mask"))

            if role != prev_role:
                if role == "user":
                    _emit(self._encode_text(_JANUS_USER_PREFIX), supervised=False)
                elif role == "assistant":
                    _emit(self._encode_text(_JANUS_ASSISTANT_PREFIX), supervised=False)
                prev_role = role

            if typ == "text":
                _emit(self._encode_text(value or ""), supervised=supervised)
                last_supervised = supervised
            elif typ in ("image", "vq_image"):
                placeholder = self._image_token_id if typ == "image" else self._image_token_id
                if placeholder is None or self._boi_token_id is None or self._eoi_token_id is None:
                    raise RuntimeError(
                        "JanusTextEncoder training tokenisation needs boi/eoi/image placeholder ids — "
                        "call set_tokenizer() first."
                    )
                _emit(
                    [int(self._boi_token_id)] + [int(placeholder)] * n_img + [int(self._eoi_token_id)],
                    supervised=False,
                )
                # Per-content "\n" between an image and a following same-role
                # part (mirrors the inference chat template lookahead).
                nxt = sample[idx + 1] if idx + 1 < len(sample) else None
                if nxt is not None and nxt.get("role") == role:
                    _emit(self._encode_text("\n"), supervised=False)
                last_supervised = False
            else:
                raise NotImplementedError(f"JanusTextEncoder training tokenisation: unsupported turn type {typ!r}.")

        # Closing eos — supervised iff the final content turn carried loss.
        ids.append(int(self._eos_token_id))
        labels.append(int(self._eos_token_id) if last_supervised else -100)
        return ids, labels

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self._tokenizer(text, add_special_tokens=False)["input_ids"]

    # ── Inference: conversation-list aware encode ─────────────────────────────

    def generate(
        self,
        *,
        conversation_list: Optional[List[ConversationPart]] = None,
        past_key_values: Optional[Any] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Embed every text / token part that lacks ``inputs_embeds``.

        First call (no KV cache): tokenises each user / assistant ``text``
        part with the global tokenizer, prepends a ``bos`` token part at
        the very front of the list, and fills ``inputs_embeds`` for every
        part that doesn't already have one (boundary-token parts emitted
        by :meth:`emit_image_start` / :meth:`emit_image_end` already
        carry their embed; sampled-token parts appended by
        :meth:`decode` / :meth:`JanusVqvae.generate` likewise).

        Subsequent calls (KV cache present): walks the tail of the list
        embedding any newly-appended ``token`` part whose ``inputs_embeds``
        is still ``None``.  This is the AR fast path used by ``text_ar``.

        Classifier-free guidance (CFG) prep
        -----------------------------------
        When ``generation_kwargs['guidance_scale'] > 1.0`` and we're on
        the prompt pass (no KV cache yet), we also build the matching
        unconditional ``inputs_embeds`` and stash it under
        ``cfg_uncond_inputs_embeds`` in the return dict so
        :meth:`JanusLlama.generate` can build a bs=2 KV cache lazily on
        the first ``image_vq`` AR step.  Construction follows the HF
        reference (``modeling_janus.py:1282-1285``): same length as the
        cond prompt, every non-``<bos>`` position replaced by
        ``pad_token_id`` before the embedding lookup.  ``<begin_of_image>``
        is also kept when present, but at this point in the FSM (prompt
        pass) the boi has not yet been appended to the conversation, so
        the rule reduces to "keep BOS, pad everything else".
        """
        if conversation_list is None:
            return {}

        out: Dict[str, Any] = {"conversation_list": conversation_list}

        if past_key_values is None:
            self._inject_bos(conversation_list)
            self._inject_chat_template(conversation_list)
            for part in conversation_list:
                if part.inputs_embeds is not None:
                    continue
                self._embed_part(part)
            uncond = self._maybe_build_cfg_uncond_embeds(conversation_list, generation_kwargs)
            if uncond is not None:
                out["cfg_uncond_inputs_embeds"] = uncond
        else:
            # AR fast path: every prior part is already embedded; walk
            # backward from the tail until we hit a populated part so
            # the cost is O(new_parts), not O(total_parts).  Without
            # this guard the cost is quadratic in the number of
            # generated tokens (esp. after a 576-step image_vq span).
            tail_to_embed: List[ConversationPart] = []
            for part in reversed(conversation_list):
                if part.inputs_embeds is not None:
                    break
                tail_to_embed.append(part)
            for part in reversed(tail_to_embed):
                self._embed_part(part)
        return out

    # ── Inference: decode + FSM signals (conversation-list aware) ────────────

    def decode(  # type: ignore[override]
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        conversation_list: Optional[List[ConversationPart]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample the next text token and append a conversation part.

        Training pass-through: if ``labels`` is present we delegate to the
        base :class:`TextEncoder.decode` (CE-only path) — training code
        never sees ``conversation_list``.

        Inference (normal sampling): temperature / top-p (read from kwargs
        or from ``generation_kwargs``), append a sampled-token part with
        its embed, and emit ``"text_done"`` when the eos / pad token comes
        out or ``"start_image_gen"`` when the model naturally emits
        ``<boi>``.  Deterministic T2I image generation is driven by the
        scenario graph (``infer_gen.yaml``'s ``image_vq_start`` state runs
        ``emit_image_start``), not by this sampler.
        """
        if labels is not None:
            return super().decode(
                hidden_states=hidden_states,
                labels=labels,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )

        if conversation_list is None:
            # No conversation context — run the base sampler and add the
            # FSM signal post-hoc so legacy callers (unit tests + the
            # original FSM bodies that route ``input_ids`` instead of
            # ``conversation_list``) keep working.
            base_out = super().decode(
                hidden_states=hidden_states,
                temperature=temperature,
                top_p=top_p,
                **kwargs,
            )
            tok = scalar_token_id(base_out.get("input_ids"))
            if tok is not None:
                if self._boi_token_id is not None and tok == self._boi_token_id:
                    base_out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
                elif self._eos_token_id is not None and tok == self._eos_token_id:
                    base_out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE
            return base_out

        out: Dict[str, Any] = {"conversation_list": conversation_list}

        if hidden_states is None:
            return out

        sampling = self._extract_sampling_kwargs(generation_kwargs, temperature, top_p, kwargs)
        tok = self._sample_token(hidden_states, **sampling)
        self._append_token_part(conversation_list, tok)
        out["input_ids"] = self._token_id_tensor(tok)

        if self._boi_token_id is not None and tok == self._boi_token_id:
            out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
        elif self._eos_token_id is not None and tok == self._eos_token_id:
            out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE

        return out

    # ── Janus boundary-token emitters (conversation-list aware) ──────────────

    def emit_image_start(
        self,
        conversation_list: Optional[List[ConversationPart]] = None,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Emit a single ``<begin_of_image>`` step + append a part."""
        token_id = self._boi_token_id if self._boi_token_id is not None else self.config.begin_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_start requires begin_of_image_token_id — "
                "call set_tokenizer() before inference."
            )
        return self._emit(int(token_id), conversation_list, ctx)

    def emit_image_end(
        self,
        conversation_list: Optional[List[ConversationPart]] = None,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Emit a single ``<end_of_image>`` step + append a part."""
        token_id = self._eoi_token_id if self._eoi_token_id is not None else self.config.end_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_end requires end_of_image_token_id — "
                "call set_tokenizer() before inference."
            )
        return self._emit(int(token_id), conversation_list, ctx)

    def _emit(
        self,
        token_id: int,
        conversation_list: Optional[List[ConversationPart]],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        batch_size = _infer_batch_size(ctx)
        ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
        inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        out: Dict[str, Any] = {"input_ids": ids, "inputs_embeds": inputs_embeds}
        if conversation_list is not None:
            # Reuse the just-built tensors so the conversation part and
            # the FSM step-token share the same backing storage (and
            # batch shape) — avoids the shape drift that would happen
            # if ``_append_token_part`` rebuilt a hard-coded ``(1, 1)``
            # tensor under a B > 1 ctx.  Single-process inference still
            # forces B == 1, but the contract should not silently
            # diverge if that ever changes.
            conversation_list.append(
                ConversationPart(
                    kind="token",
                    role="assistant",
                    token_id=token_id,
                    input_ids=ids,
                    inputs_embeds=inputs_embeds,
                )
            )
            out["conversation_list"] = conversation_list
        return out

    # ── CFG: build the unconditional inputs_embeds ───────────────────────────

    def _maybe_build_cfg_uncond_embeds(
        self,
        conversation_list: List[ConversationPart],
        generation_kwargs: Optional[Dict[str, Any]],
    ) -> Optional[torch.Tensor]:
        """Build ``(1, T_prompt, hidden)`` uncond inputs_embeds, or ``None``.

        Returns ``None`` (skip CFG) when:

        * ``guidance_scale`` is missing / <= 1.0;
        * the tokenizer wasn't wired in (``set_tokenizer`` not called) — we
          can't resolve ``pad_token_id`` then;
        * any conversation part already carries non-text content (image_und
          parts have ``input_ids = None``); Janus's CFG protocol is only
          defined for the T2I path so we conservatively bail out.
        """
        if not generation_kwargs:
            return None
        cfg_w = generation_kwargs.get("guidance_scale")
        if cfg_w is None or float(cfg_w) <= 1.0:
            return None
        if self._tokenizer is None or self._pad_token_id is None or self._bos_token_id is None:
            return None

        bos_id = int(self._bos_token_id)
        pad_id = int(self._pad_token_id)
        boi_id = self._boi_token_id  # may be None if not registered
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        uncond_chunks: List[torch.Tensor] = []
        for part in conversation_list:
            if part.inputs_embeds is None:
                continue
            ids = part.input_ids
            if ids is None or ids.numel() == 0:
                # image_und / image_gen part — CFG masking rule undefined for
                # multi-modal prompts in Janus.  Abort the whole uncond build
                # and warn so users don't silently get noise-quality samples
                # (the failure mode CFG is meant to fix).
                logger.warning(
                    "JanusTextEncoder: guidance_scale > 1 was requested but the prompt "
                    "contains a non-text part (kind=%r) with no input_ids — CFG only "
                    "supports T2I prompts (text-only). Skipping uncond branch; the LLM "
                    "will run with cond logits only.",
                    part.kind,
                )
                return None
            keep_mask = ids == bos_id
            if boi_id is not None:
                keep_mask = keep_mask | (ids == int(boi_id))
            masked = torch.where(keep_mask, ids, torch.full_like(ids, pad_id))
            uncond_chunks.append(self.embed_tokens(masked).to(dtype=dtype, device=device))
        if not uncond_chunks:
            return None
        return torch.cat(uncond_chunks, dim=1)

    # ── Finalize: decode the accumulated assistant text ──────────────────────

    def finalize(self, *, ctx: Dict[str, Any], request: Dict[str, Any]) -> Dict[str, Any]:
        """Detokenize every assistant-role text token sampled in this run.

        The framework calls this when the FSM enters ``done`` (see
        :meth:`OmniModel.generate`); the result lands under
        ``ctx['finalize'][<module_name>]`` so the caller can read the
        full reply.  When no tokenizer was wired in (e.g. minimal tests),
        we return the raw token ids instead of decoded text so the hook
        is still useful.
        """
        del request
        conversation = ctx.get("conversation_list")
        if not isinstance(conversation, list):
            return {}
        token_ids = latest_assistant_text_token_ids(conversation)
        if not token_ids:
            return {}
        if self._tokenizer is None:
            return {"token_ids": token_ids}
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"text": text, "token_ids": token_ids}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _inject_bos(self, conversation_list: List[ConversationPart]) -> None:
        if self._bos_token_id is None:
            return
        already_bos = (
            conversation_list
            and conversation_list[0].kind == "token"
            and conversation_list[0].token_id == self._bos_token_id
        )
        if already_bos:
            return
        bos = ConversationPart(kind="token", role="system", token_id=int(self._bos_token_id))
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        ids = torch.tensor([[self._bos_token_id]], dtype=torch.long, device=device)
        bos.input_ids = ids
        bos.inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        conversation_list.insert(0, bos)

    def _inject_chat_template(self, conversation_list: List[ConversationPart]) -> None:
        """Insert Janus role markers + image boundaries in-place on the prompt pass.

        The text encoder is the sole authority on the on-the-wire prompt
        layout (per the V2 design — no upstream ``apply_chat_template``).
        We walk the conversation once and produce the canonical Janus
        prompt format::

            <bos><system_prompt>\\n\\n<|User|>: <boi>[image]<eoi>\\n<user_text>\\n\\n<|Assistant|>:[<assistant_text>]

        The ``\\n`` separators are baked into the role markers (see the
        ``_JANUS_*`` constants above) except the per-content ``\\n``
        between an image and a same-role text part, which is inserted
        here via a one-step lookahead — same rule as the upstream Jinja
        ``{%if not loop.last%}\\n{%endif%}``.

        Idempotent: runs once on the prompt pass (``past_key_values is
        None`` branch only) and tags each inserted marker part with
        ``meta['source'] = 'chat_template'`` so a re-run wouldn't double-
        insert (defensive; the AR fast path doesn't call us again).
        """
        if not conversation_list:
            return
        # Skip if we've already injected markers in this conversation.
        if any(p.meta.get("source") == "chat_template" for p in conversation_list):
            return

        def _text_marker(role: str, text: str) -> ConversationPart:
            return ConversationPart(kind="text", role=role, text=text, meta={"source": "chat_template"})

        def _token_marker(role: str, token_id: int) -> ConversationPart:
            return ConversationPart(
                kind="token",
                role=role,
                token_id=int(token_id),
                meta={"source": "chat_template"},
            )

        new_list: List[ConversationPart] = []
        prev_role: Optional[str] = None
        for idx, part in enumerate(conversation_list):
            # Bos sits at the very front; system prompt rides right after.
            if part.kind == "token" and part.role == "system" and part.token_id == self._bos_token_id:
                new_list.append(part)
                new_list.append(_text_marker("system", _JANUS_SYSTEM_PROMPT))
                continue

            if part.role != prev_role:
                if part.role == "user":
                    new_list.append(_text_marker("user", _JANUS_USER_PREFIX))
                elif part.role == "assistant":
                    new_list.append(_text_marker("assistant", _JANUS_ASSISTANT_PREFIX))
                prev_role = part.role

            # Wrap each image_und with <begin_of_image> / <end_of_image>
            # boundary tokens, then insert a "\n" separator iff the next
            # part is in the SAME role (i.e. image followed by user
            # text) — mirrors the per-content "\n" the upstream Jinja
            # template emits inside a single message.
            if part.kind == "image_und":
                if self._boi_token_id is not None:
                    new_list.append(_token_marker("user", self._boi_token_id))
                new_list.append(part)
                if self._eoi_token_id is not None:
                    new_list.append(_token_marker("user", self._eoi_token_id))
                next_part = conversation_list[idx + 1] if idx + 1 < len(conversation_list) else None
                if next_part is not None and next_part.role == part.role:
                    new_list.append(_text_marker(part.role, "\n"))
                continue

            new_list.append(part)

        conversation_list.clear()
        conversation_list.extend(new_list)

    def _embed_part(self, part: ConversationPart) -> None:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        if part.kind == "text":
            if not part.text:
                # Empty assistant marker: nothing to embed yet.
                return
            if self._tokenizer is None:
                raise RuntimeError(
                    "JanusTextEncoder.generate needs a tokenizer for text parts — "
                    "call set_tokenizer(global_tokenizer) on the module first."
                )
            ids = self._tokenizer(part.text, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = ids.to(device=device)
            part.input_ids = ids
            part.inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
            return

        if part.kind == "token":
            if part.token_id is None:
                return
            ids = torch.tensor([[int(part.token_id)]], dtype=torch.long, device=device)
            part.input_ids = ids
            part.inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)

        # image_und / image_gen parts are not the text encoder's responsibility.

    def _append_token_part(
        self,
        conversation_list: List[ConversationPart],
        token_id: int,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        conversation_list.append(
            ConversationPart(
                kind="token",
                role="assistant",
                token_id=int(token_id),
                input_ids=ids,
                inputs_embeds=inputs_embeds,
            )
        )

    def _token_id_tensor(self, token_id: int) -> torch.Tensor:
        device = self.embed_tokens.weight.device
        return torch.tensor([[int(token_id)]], dtype=torch.long, device=device)

    def _sample_token(
        self,
        hidden_states: torch.Tensor,
        temperature: float = 1.0,
        top_p: float = 1.0,
        do_sample: bool = True,
    ) -> int:
        hidden_states = hidden_states.to(self.device)
        last = hidden_states[:, -1, :]
        logits = self._project(last) if last.dim() == 2 else self._project(last.squeeze(0))
        if logits.dim() == 3:
            logits = logits.squeeze(1)
        if not do_sample:
            return int(logits.argmax(dim=-1).item())
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-6)
        if top_p < 1.0:
            logits = self._top_p_filter(logits, top_p)
        probs = F.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        return int(token.item())

    @staticmethod
    def _extract_sampling_kwargs(
        generation_kwargs: Optional[Dict[str, Any]],
        temperature: float,
        top_p: float,
        kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {"temperature": temperature, "top_p": top_p, "do_sample": True}
        if generation_kwargs:
            for k in _SAMPLING_KWARGS:
                if k in generation_kwargs:
                    merged[k] = generation_kwargs[k]
        for k in _SAMPLING_KWARGS:
            if k in kwargs:
                merged[k] = kwargs[k]
        return merged


def _is_training_conversation(conversation: Any) -> bool:
    """True iff ``conversation`` is the raw training shape ``list[list[dict]]``.

    The training data pipeline emits per-sample lists of plain
    ``{type, value, role, loss_mask}`` dicts; inference uses
    :class:`ConversationPart` objects.  We dispatch the training
    tokenisation only on the former.
    """
    return (
        isinstance(conversation, list)
        and len(conversation) > 0
        and isinstance(conversation[0], list)
        and len(conversation[0]) > 0
        and isinstance(conversation[0][0], dict)
    )


def _resolve_token_id(tokenizer: Any, candidates: tuple) -> Optional[int]:
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in candidates:
        if not cand:
            continue
        tid = tokenizer.convert_tokens_to_ids(cand)
        if tid is not None and tid != unk:
            return int(tid)
    return None


def _infer_batch_size(ctx: Dict[str, Any]) -> int:
    """Best-effort batch-size inference from tensors already in ``ctx``."""
    for key in ("input_ids", "inputs_embeds", "hidden_states", "attention_mask"):
        v = ctx.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.size(0))
    return 1

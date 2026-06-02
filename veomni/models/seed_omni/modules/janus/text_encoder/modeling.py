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

Token ids are resolved from the wired conversation tokenizer via
:meth:`set_conversation_tokenizer`.

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

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from ....conversation import (
    ConversationItem,
    TrainConversation,
    TrainSegment,
    assemble_labels,
    get_llm_embed,
    get_token_id,
    is_embedded,
    is_train_conversation,
    item_role,
    latest_assistant_text_token_ids,
    maybe_merge_outputs,
    needs_embedding,
    seal_phase_outputs,
    set_llm_embed,
)
from ....generation_graph import FSM_SIGNAL_KEY
from ....graph import scalar_token_id
from ...base.text_encoder.modeling import TextEncoder
from .configuration import JanusTextEncoderConfig


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
        self._conversation_tokenizer: Any = None
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
        # Base id of the 576 positional generation tokens ``<image_0>``..
        # ``<image_575>`` (Janus: 100017..100592).  A ``vq_image`` turn
        # expands to ``[gen_base + k for k in range(num_image_tokens)]`` so
        # the gen-image placeholder ids are *distinct* from the understanding
        # placeholder (no ``masked_scatter`` collision in mixed batches) and
        # self-describe their grid position.  ``gen_image_mask`` (returned by
        # :meth:`encode`) marks exactly these positions for the backbone
        # scatter + the VQ gen-loss.
        self._gen_image_base: Optional[int] = None
        # Sampled text token ids for the current assistant turn (not stored
        # on the conversation list — only embeds live there).
        self._text_token_cache: list[int] = []

    def reset_inference_state(self) -> None:
        """Clear per-request text token cache."""
        self._text_token_cache.clear()

    def set_conversation_tokenizer(self, conversation_tokenizer: Any) -> None:
        """Resolve Janus special-token ids from the global conversation tokenizer."""
        self._conversation_tokenizer = conversation_tokenizer
        bos = getattr(conversation_tokenizer, "bos_token_id", None)
        self._bos_token_id = int(bos) if bos is not None else None
        self._boi_token_id = _resolve_token_id(
            conversation_tokenizer,
            ("<begin_of_image>", getattr(conversation_tokenizer, "boi_token", None)),
        )
        self._eoi_token_id = _resolve_token_id(
            conversation_tokenizer,
            ("<end_of_image>", getattr(conversation_tokenizer, "eoi_token", None)),
        )
        eos = getattr(conversation_tokenizer, "eos_token_id", None)
        self._eos_token_id = int(eos) if eos is not None else None
        pad = getattr(conversation_tokenizer, "pad_token_id", None)
        # Pad falls back to eos when the tokenizer doesn't ship one — same
        # convention HF uses when building uncond inputs in
        # `JanusForConditionalGeneration.generate`
        # (`modeling_janus.py:1282-1285`).  We need a real id here, otherwise
        # the CFG uncond branch can't replace non-BOS positions.
        self._pad_token_id = int(pad) if pad is not None else self._eos_token_id
        self._image_token_id = _resolve_token_id(
            conversation_tokenizer,
            ("<image_placeholder>", getattr(conversation_tokenizer, "image_token", None)),
        )
        self._gen_image_base = _resolve_token_id(conversation_tokenizer, ("<image_0>",))
        if self._boi_token_id is not None:
            self.config.begin_of_image_token_id = self._boi_token_id
        if self._eoi_token_id is not None:
            self.config.end_of_image_token_id = self._eoi_token_id

    # ── Training: conversation-list → tokenised batch (D4) ────────────────────

    def pre_forward(self, **kwargs: Any) -> Dict[str, Any]:  # type: ignore[override]
        """Pass the training carrier through to :meth:`encode`.

        In the V2 segment-driven contract the text encoder no longer emits flat
        ``input_ids`` / ``attention_mask`` / ``position_ids`` / ``gen_image_mask``
        tensors.  All tokenisation + wte embedding happens in :meth:`encode`,
        which fills the :class:`TrainConversation` carrier's per-sample segment
        list.  Inference (``ConversationPart`` list / ``input_ids`` batch) is
        passed through untouched — :meth:`generate` handles it.
        """
        return kwargs

    def encode(  # type: ignore[override]
        self,
        input_ids: Optional[torch.LongTensor] = None,
        conversation_list: Optional[Any] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the per-sample embedding segments on the training carrier.

        Training (``conversation_list`` is a :class:`TrainConversation`): walk
        each sample's raw conversation, apply the Janus chat template, tokenise
        + wte-embed the text runs, splice in the SigLIP / VQ patch embeds for
        ``image`` / ``vq_image`` turns (wrapped in ``<boi>``/``<eoi>``), and
        write the ordered :class:`TrainSegment` list back onto the carrier.
        No flat ``input_ids`` / masks are emitted — the backbone derives
        everything from the segments.  Returns ``{"conversation_list": carrier}``.

        Legacy / inference path (``input_ids`` only): delegate to
        :class:`TextEncoder.encode` (wte lookup → ``inputs_embeds``).
        """
        if is_train_conversation(conversation_list):
            self._build_segments(conversation_list)
            return {"conversation_list": conversation_list}
        return super().encode(input_ids=input_ids, past_key_values=past_key_values, **kwargs)

    def _num_image_tokens(self) -> int:
        """Understanding-image placeholder count per image (Janus: 24×24=576)."""
        return int(getattr(self.config, "num_image_tokens", 576))

    def _build_segments(self, conv: TrainConversation) -> None:
        """Fill ``conv.segments`` — one ordered ``TrainSegment`` list per sample.

        Mirrors the inference chat template (:meth:`_inject_chat_template`):
        ``<bos><system>\\n\\n<|User|>: [<boi>img<eoi>\\n]<user_text>\\n\\n
        <|Assistant|>:<assistant_text><eos>``.  Text runs become wte-embedded
        segments; ``image`` / ``vq_image`` turns become patch-embed segments
        sourced from ``conv.und_embeds`` / ``conv.gen_embeds`` (one row per
        sample — at most one image of each modality per sample, see
        ``image_inputs.build_pixel_values_batch``).  Loss is taken only on
        assistant-role text turns flagged ``loss_mask=1`` (plus the closing
        ``<eos>``); image / boundary / prompt positions get ``label_ids=-100``,
        and ``vq_image`` positions carry the teacher VQ ids in ``gen_ids``.
        """
        if self._conversation_tokenizer is None:
            raise RuntimeError(
                "JanusTextEncoder training tokenisation needs a tokenizer — "
                "call set_conversation_tokenizer(conversation_tokenizer) on the module first."
            )
        if self._bos_token_id is None or self._eos_token_id is None:
            raise RuntimeError("JanusTextEncoder.set_conversation_tokenizer did not resolve bos/eos ids.")
        conv.segments = [self._segment_one(sample, conv, i) for i, sample in enumerate(conv.raw)]

    def _segment_one(
        self, sample: List[Dict[str, Any]], conv: TrainConversation, sample_idx: int
    ) -> List[TrainSegment]:
        """Build the ordered ``TrainSegment`` list for one raw conversation."""
        device = self.embed_tokens.weight.device
        segs: List[TrainSegment] = []

        def _add_text(token_ids: List[int], supervised: bool) -> None:
            if not token_ids:
                return
            ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
            embeds = self.embed_tokens(ids_t)  # (L, D)
            label_ids = ids_t.clone() if supervised else torch.full_like(ids_t, -100)
            gen_ids = torch.full_like(ids_t, -100)
            segs.append(TrainSegment(embeds=embeds, label_ids=label_ids, gen_ids=gen_ids))

        def _add_image(typ: str) -> None:
            if typ == "image":
                if conv.und_embeds is None:
                    raise RuntimeError(
                        "JanusTextEncoder: `image` turn but carrier has no und_embeds (run JanusSiglip)."
                    )
                embeds = conv.und_embeds[sample_idx]  # (P, D)
                p = embeds.size(0)
                gen_ids = torch.full((p,), -100, dtype=torch.long, device=device)
            else:  # vq_image
                if conv.gen_embeds is None or conv.gen_token_ids is None:
                    raise RuntimeError(
                        "JanusTextEncoder: `vq_image` turn but carrier has no gen_embeds (run JanusVqvae)."
                    )
                embeds = conv.gen_embeds[sample_idx]  # (P, D)
                p = embeds.size(0)
                gen_ids = conv.gen_token_ids[sample_idx].to(device=device, dtype=torch.long)  # (P,)
            label_ids = torch.full((p,), -100, dtype=torch.long, device=device)
            segs.append(TrainSegment(embeds=embeds, label_ids=label_ids, gen_ids=gen_ids))

        _add_text([int(self._bos_token_id)], supervised=False)
        _add_text(self._encode_text(_JANUS_SYSTEM_PROMPT), supervised=False)

        prev_role: Optional[str] = None
        last_supervised = False
        for idx, item in enumerate(sample):
            role = item.get("role")
            typ = item.get("type")
            value = item.get("value")
            supervised = role == "assistant" and bool(item.get("loss_mask"))

            if role != prev_role:
                if role == "user":
                    _add_text(self._encode_text(_JANUS_USER_PREFIX), supervised=False)
                elif role == "assistant":
                    _add_text(self._encode_text(_JANUS_ASSISTANT_PREFIX), supervised=False)
                prev_role = role

            if typ == "text":
                _add_text(self._encode_text(value or ""), supervised=supervised)
                last_supervised = supervised
            elif typ in ("image", "vq_image"):
                if self._boi_token_id is None or self._eoi_token_id is None:
                    raise RuntimeError(
                        "JanusTextEncoder training tokenisation needs boi/eoi ids — call set_conversation_tokenizer() first."
                    )
                _add_text([int(self._boi_token_id)], supervised=False)
                _add_image(typ)
                _add_text([int(self._eoi_token_id)], supervised=False)
                # Per-content "\n" between an image and a following same-role
                # part (mirrors the inference chat template lookahead).
                nxt = sample[idx + 1] if idx + 1 < len(sample) else None
                if nxt is not None and nxt.get("role") == role:
                    _add_text(self._encode_text("\n"), supervised=False)
                last_supervised = False
            else:
                raise NotImplementedError(f"JanusTextEncoder training tokenisation: unsupported turn type {typ!r}.")

        # Closing eos — supervised iff the final content turn carried loss.
        _add_text([int(self._eos_token_id)], supervised=last_supervised)
        return segs

    def _encode_text(self, text: str) -> List[int]:
        if not text:
            return []
        return self._conversation_tokenizer(text, add_special_tokens=False)["input_ids"]

    # ── Inference: conversation-list aware encode ─────────────────────────────

    def generate(
        self,
        *,
        conversation_list: Optional[List[ConversationItem]] = None,
        past_key_values: Optional[Any] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        """Embed every text / token part that is not yet embedded.

        First call (no KV cache): tokenises each user / assistant ``text``
        part, prepends a ``bos`` token at the front, and embeds every part
        that still needs embedding (boundary-token parts from
        :meth:`emit_image_start` / :meth:`emit_image_end` already carry
        embeds in ``value``).

        Subsequent calls (KV cache present): embeds newly-appended ``token``
        parts whose ``value`` is still a scalar id.
        """
        if conversation_list is None:
            return {}

        out: Dict[str, Any] = {"conversation_list": conversation_list}

        if past_key_values is None:
            self._inject_bos(conversation_list)
            self._inject_chat_template(conversation_list)
            for part in conversation_list:
                if not needs_embedding(part):
                    continue
                self._embed_part(part)
        return out

    def ar_step(
        self,
        hidden_states: Optional[torch.Tensor] = None,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Monolithic text AR step: decode hidden → id → embed → merge.

        Reads backbone ``hidden_states``, samples one text token, stores the
        id in the module-private cache, replaces the trailing ``output``
        item's ``value`` with the wte embed, merges adjacent text-phase
        outputs, and raises FSM signals on boundary tokens.
        """
        if conversation_list is None:
            return {}
        out: Dict[str, Any] = {"conversation_list": conversation_list}
        if hidden_states is None or not conversation_list:
            return out

        sampling = self._extract_sampling_kwargs(generation_kwargs, temperature, top_p, kwargs)
        tok = self._sample_token(hidden_states, **sampling)
        self._text_token_cache.append(int(tok))

        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        ids = torch.tensor([[int(tok)]], dtype=torch.long, device=device)
        embed = self.embed_tokens(ids).to(dtype=dtype)

        tail = conversation_list[-1]
        if tail.type != "output":
            raise ValueError(
                f"JanusTextEncoder.ar_step expects the conversation tail to be type='output', got {tail.type!r}."
            )
        set_llm_embed(tail, embed, token_id=int(tok))
        tail.meta["input_ids"] = ids
        maybe_merge_outputs(conversation_list, phase="text")

        if self._boi_token_id is not None and tok == self._boi_token_id:
            self._maybe_arm_cfg_for_image_gen(conversation_list, generation_kwargs, out)
            seal_phase_outputs(conversation_list, phase="text", new_type="text")
            tail.type = "soi"
            tail.meta.pop("phase", None)
            out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            out["ar_phase"] = "vq"
        elif self._eos_token_id is not None and tok == self._eos_token_id:
            seal_phase_outputs(conversation_list, phase="text", new_type="text")
            out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE

        return out

    # ── Inference: decode + FSM signals (conversation-list aware) ────────────

    def decode(  # type: ignore[override]
        self,
        hidden_states: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
        top_p: float = 1.0,
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Sample the next text token and append a conversation part.

        Training (V2 segment-driven): when ``conversation_list`` is a
        :class:`TrainConversation`, build the per-position text labels from its
        segments (``label_ids``) and delegate to the base
        :class:`TextEncoder.decode` shifted-CE path.  The backbone assembled
        ``hidden_states`` from the same segment order + right-pad, so labels
        and logits line up position-for-position.  Legacy ``labels``-tensor
        callers (unit tests) still work via the branch below.

        Inference (normal sampling): temperature / top-p (read from kwargs
        or from ``generation_kwargs``), append a sampled-token part with
        its embed, and emit ``"text_done"`` when the eos / pad token comes
        out or ``"start_image_gen"`` when the model naturally emits
        ``<boi>``.  Deterministic T2I image generation is driven by the
        scenario graph (``infer_gen.yaml``'s ``image_vq_start`` state runs
        ``emit_image_start``), not by this sampler.
        """
        if hidden_states is None and is_train_conversation(conversation_list):
            hidden_states = conversation_list.hidden_states
        if hidden_states is not None and is_train_conversation(conversation_list):
            if conversation_list.segments is None:
                raise ValueError("JanusTextEncoder.decode: TrainConversation has no segments (encode must run first).")
            labels = assemble_labels(conversation_list.segments, key="label_ids").to(hidden_states.device)
            # Trim any SP right-pad so logits/labels align (no-op when SP off).
            hidden_states = hidden_states[:, : labels.size(1)]
            return super().decode(hidden_states=hidden_states, labels=labels, **kwargs)

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
        conversation_list: Optional[List[ConversationItem]] = None,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Emit ``<begin_of_image>`` as a standalone ``soi`` conversation item."""
        token_id = self._boi_token_id if self._boi_token_id is not None else self.config.begin_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_start requires begin_of_image_token_id — "
                "call set_conversation_tokenizer() before inference."
            )
        if conversation_list is not None:
            seal_phase_outputs(conversation_list, phase="text", new_type="text")
        out: Dict[str, Any] = {}
        self._maybe_arm_cfg_for_image_gen(
            conversation_list,
            ctx.get("generation_kwargs"),
            out,
        )
        out.update(self._emit_boundary(int(token_id), "soi", conversation_list, ctx))
        out["ar_phase"] = "vq"
        return out

    def emit_image_end(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Emit ``<end_of_image>`` as a standalone ``eoi`` conversation item."""
        token_id = self._eoi_token_id if self._eoi_token_id is not None else self.config.end_of_image_token_id
        if token_id is None:
            raise RuntimeError(
                "JanusTextEncoder.emit_image_end requires end_of_image_token_id — "
                "call set_conversation_tokenizer() before inference."
            )
        if conversation_list is not None:
            seal_phase_outputs(conversation_list, phase="vq", new_type="image")
        out = self._emit_boundary(int(token_id), "eoi", conversation_list, ctx)
        out["ar_phase"] = "text"
        out["collapse_cfg"] = True
        return out

    def _emit_boundary(
        self,
        token_id: int,
        item_type: str,
        conversation_list: Optional[List[ConversationItem]],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        batch_size = _infer_batch_size(ctx)
        ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
        inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        out: Dict[str, Any] = {"input_ids": ids, "inputs_embeds": inputs_embeds}
        if conversation_list is not None:
            conversation_list.append(
                ConversationItem(
                    type=item_type,
                    value=inputs_embeds,
                    meta={"role": "assistant", "token_id": int(token_id), "input_ids": ids},
                )
            )
            out["conversation_list"] = conversation_list
        return out

    def _emit(
        self,
        token_id: int,
        conversation_list: Optional[List[ConversationItem]],
        ctx: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Legacy emit helper — prefer :meth:`_emit_boundary`."""
        return self._emit_boundary(token_id, "token", conversation_list, ctx)

    # ── CFG: arm at SOI from ``generation_kwargs`` ────────────────────────────

    def _maybe_arm_cfg_for_image_gen(
        self,
        conversation_list: Optional[List[ConversationItem]],
        generation_kwargs: Optional[Dict[str, Any]],
        out: Dict[str, Any],
    ) -> None:
        """Opt into CFG when ``<soi>`` opens the image span.

        Reads ``guidance_scale`` from ``generation_kwargs`` (opaque to the
        framework).  When ``> 1``, writes a one-shot ``ctx["cfg"]`` arm signal
        plus ``cfg_uncond_inputs_embeds`` for the backbone to expand KV on its
        next forward.  Llama clears ``cfg["enabled"]`` after the cache is built.
        """
        if conversation_list is None or not generation_kwargs:
            return
        cfg_w = generation_kwargs.get("guidance_scale")
        if cfg_w is None or float(cfg_w) <= 1.0:
            return
        uncond = self._build_cfg_uncond_embeds(conversation_list)
        if uncond is None:
            return
        out["cfg"] = {"enabled": True, "guidance_scale": float(cfg_w)}
        out["cfg_uncond_inputs_embeds"] = uncond

    def _build_cfg_uncond_embeds(
        self,
        conversation_list: List[ConversationItem],
    ) -> Optional[torch.Tensor]:
        """Build ``(1, T_prompt, hidden)`` uncond inputs_embeds, or ``None``.

        Returns ``None`` when the tokenizer is unavailable or the prompt is
        not text-only (e.g. I2T with an ``image`` part).
        """
        if self._conversation_tokenizer is None or self._pad_token_id is None or self._bos_token_id is None:
            return None

        if any(p.type == "image" for p in conversation_list):
            return None

        bos_id = int(self._bos_token_id)
        pad_id = int(self._pad_token_id)
        boi_id = self._boi_token_id  # may be None if not registered
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        uncond_chunks: List[torch.Tensor] = []
        for part in conversation_list:
            if part.type == "output":
                continue
            embed = get_llm_embed(part)
            if embed is None:
                continue
            ids = part.meta.get("input_ids")
            if ids is None:
                if part.type == "image":
                    return None
                continue
            keep_mask = ids == bos_id
            if boi_id is not None:
                keep_mask = keep_mask | (ids == int(boi_id))
            masked = torch.where(keep_mask, ids, torch.full_like(ids, pad_id))
            uncond_chunks.append(self.embed_tokens(masked).to(dtype=dtype, device=device))
        if not uncond_chunks:
            return None
        return torch.cat(uncond_chunks, dim=1)

    # ── Finalize: decode the accumulated assistant text ──────────────────────

    def finalize(self, *, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Detokenize and flush the module-private text token cache.

        Flushes when this module raised ``text_done`` / ``start_image_gen``,
        or when ``module_signal`` is absent (``max_new_tokens`` forced cleanup)
        and tokens remain buffered.
        """
        if not self._text_token_cache:
            return {}
        signal = ctx.get(FSM_SIGNAL_KEY)
        if signal is not None and signal not in (SIGNAL_TEXT_DONE, SIGNAL_START_IMAGE_GEN):
            return {}
        return self._flush_text_generated(ctx)

    def _flush_text_generated(self, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Decode cached token ids, clear the cache, return a ``generated`` payload."""
        conversation = ctx.get("conversation_list")
        token_ids = list(self._text_token_cache)
        self._text_token_cache.clear()
        boundary = {t for t in (self._eos_token_id, self._boi_token_id) if t is not None}
        while token_ids and token_ids[-1] in boundary:
            token_ids.pop()
        if not token_ids and isinstance(conversation, list):
            token_ids = latest_assistant_text_token_ids(conversation)
        if not token_ids:
            return {}
        meta = {"token_ids": token_ids}
        if self._conversation_tokenizer is None:
            return {"generated": {"type": "text", "value": "", "meta": meta}}
        text = self._conversation_tokenizer.decode(token_ids, skip_special_tokens=True)
        return {"generated": {"type": "text", "value": text, "meta": meta}}

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _inject_bos(self, conversation_list: List[ConversationItem]) -> None:
        if self._bos_token_id is None:
            return
        already_bos = (
            conversation_list
            and conversation_list[0].type == "token"
            and item_role(conversation_list[0]) == "system"
            and get_token_id(conversation_list[0]) == self._bos_token_id
        )
        if already_bos:
            return
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        ids = torch.tensor([[self._bos_token_id]], dtype=torch.long, device=device)
        embed = self.embed_tokens(ids).to(dtype=dtype)
        conversation_list.insert(
            0,
            ConversationItem(
                type="token",
                value=embed,
                meta={"role": "system", "token_id": int(self._bos_token_id), "input_ids": ids},
            ),
        )

    def _inject_chat_template(self, conversation_list: List[ConversationItem]) -> None:
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

        def _text_marker(role: str, text: str) -> ConversationItem:
            return ConversationItem(type="text", value=text, meta={"role": role, "source": "chat_template"})

        def _token_marker(role: str, token_id: int) -> ConversationItem:
            return ConversationItem(
                type="token",
                value=int(token_id),
                meta={"role": role, "source": "chat_template", "token_id": int(token_id)},
            )

        new_list: List[ConversationItem] = []
        prev_role: Optional[str] = None
        for idx, part in enumerate(conversation_list):
            role = item_role(part)
            if part.type == "token" and role == "system" and get_token_id(part) == self._bos_token_id:
                new_list.append(part)
                new_list.append(_text_marker("system", _JANUS_SYSTEM_PROMPT))
                continue

            if role != prev_role:
                if role == "user":
                    new_list.append(_text_marker("user", _JANUS_USER_PREFIX))
                elif role == "assistant":
                    new_list.append(_text_marker("assistant", _JANUS_ASSISTANT_PREFIX))
                prev_role = role

            if part.type == "image" and role == "user":
                if self._boi_token_id is not None:
                    new_list.append(_token_marker("user", self._boi_token_id))
                new_list.append(part)
                if self._eoi_token_id is not None:
                    new_list.append(_token_marker("user", self._eoi_token_id))
                next_part = conversation_list[idx + 1] if idx + 1 < len(conversation_list) else None
                if next_part is not None and item_role(next_part) == role:
                    new_list.append(_text_marker(role, "\n"))
                continue

            new_list.append(part)

        conversation_list.clear()
        conversation_list.extend(new_list)

    def _embed_part(self, part: ConversationItem) -> None:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype

        if part.type == "text":
            if not part.value:
                return
            if self._conversation_tokenizer is None:
                raise RuntimeError(
                    "JanusTextEncoder.generate needs a tokenizer for text parts — "
                    "call set_conversation_tokenizer(conversation_tokenizer) on the module first."
                )
            ids = self._conversation_tokenizer(part.value, return_tensors="pt", add_special_tokens=False)["input_ids"]
            ids = ids.to(device=device)
            embed = self.embed_tokens(ids).to(dtype=dtype)
            part.meta["input_ids"] = ids
            part.value = embed
            return

        if part.type == "token":
            tid = get_token_id(part)
            if tid is None:
                return
            if is_embedded(part):
                return
            ids = torch.tensor([[int(tid)]], dtype=torch.long, device=device)
            embed = self.embed_tokens(ids).to(dtype=dtype)
            set_llm_embed(part, embed, token_id=int(tid))
            part.meta["input_ids"] = ids

    def _append_token_part(
        self,
        conversation_list: List[ConversationItem],
        token_id: int,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> None:
        device = self.embed_tokens.weight.device
        dtype = self.embed_tokens.weight.dtype
        if inputs_embeds is None:
            ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
            conversation_list.append(
                ConversationItem(
                    type="token",
                    value=inputs_embeds,
                    meta={"role": "assistant", "token_id": int(token_id), "input_ids": ids},
                )
            )
        else:
            conversation_list.append(
                ConversationItem(
                    type="token",
                    value=inputs_embeds,
                    meta={"role": "assistant", "token_id": int(token_id)},
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


def _resolve_token_id(tokenizer: Any, candidates: tuple) -> Optional[int]:
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in candidates:
        if not cand:
            continue
        try:
            tid = tokenizer.convert_tokens_to_ids(cand)
        except (KeyError, ValueError):
            # Some (mock / minimal) tokenizers raise on unknown tokens rather
            # than returning unk — treat as unresolved and try the next.
            continue
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

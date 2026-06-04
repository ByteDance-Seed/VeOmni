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
After :meth:`ar_step` samples a token, this module — not the outer FSM —
decides whether the token means "start image generation" or "text done"
and writes a one-shot string into ``ctx["module_signal"]``:

* ``"start_image_gen"`` — sampled token is ``<begin_of_image>``
* ``"text_done"``       — sampled token is ``</s>`` (eos)

Token ids are resolved from this module's own tokenizer asset
(``tokenizer.json`` next to ``config.json``, same layout as SigLIP's processor).

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

from veomni.utils.model_outputs import CausalLMOutputWithLogProbs
from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import (
    ConversationItem,
    get_llm_embed,
    get_token_id,
    is_dummy,
    is_embedded,
    item_role,
    latest_assistant_text_token_ids,
    maybe_merge_outputs,
    needs_embedding,
    seal_phase_outputs,
    set_llm_embed,
)
from ....generation_graph import FSM_SIGNAL_KEY
from ...base.text_encoder.modeling import TextEncoder
from .chat_template import (
    JanusChatMarkers,
    apply_janus_chat_template,
    pack_text_input_ids,
)
from .configuration import JanusTextEncoderConfig


# Signal *values* written to ``ctx[FSM_SIGNAL_KEY]`` by :meth:`ar_step`.
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

# Each non-text turn is exactly one ``MODALITY_SLOT_ID`` in the flat stream (wte id 0).
# Encoder modules expand the real patch sequence elsewhere on the conversation item.
MODALITY_SLOT_ID = 0


class JanusTextEncoder(TextEncoder):
    """:class:`TextEncoder` + ``emit_image_start`` / ``emit_image_end``.

    Inference contract (``generate`` / ``decode`` / ``emit_image_*``)
    ----------------------------------------------------------------
    All inference call-sites consume / produce a ``conversation_list``
    (see :mod:`veomni.models.seed_omni.conversation`).  Training uses
    :meth:`pre_forward` → :meth:`encode` → :meth:`post_forward` on the same
    carrier; inference still uses :meth:`generate` until it shares
    :meth:`_prepare_sample_training`.
    """

    config_class = JanusTextEncoderConfig

    def __init__(self, config: JanusTextEncoderConfig):
        super().__init__(config)
        self._tokenizer: Optional[Any] = None
        self._chat_markers: Optional[JanusChatMarkers] = None
        self._bos_token_id: Optional[int] = None
        self._boi_token_id: Optional[int] = None
        self._eoi_token_id: Optional[int] = None
        self._eos_token_id: Optional[int] = None
        self._pad_token_id: Optional[int] = None
        # Sampled text token ids for the current assistant turn (not stored
        # on the conversation list — only embeds live there).
        self._text_token_cache: list[int] = []

        # training cache
        self._conversation_carrier: Any = None
        self._encode_batch_shape: torch.LongTensor | None = None

    def reset_inference_state(self) -> None:
        """Clear per-request text token cache."""
        self._text_token_cache.clear()

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._bos_token_id = int(tokenizer.bos_token_id)
        self._eos_token_id = int(tokenizer.eos_token_id)
        self._pad_token_id = int(tokenizer.pad_token_id)
        self._boi_token_id = int(tokenizer.boi_token_id)
        self._eoi_token_id = int(tokenizer.eoi_token_id)
        self._chat_markers = JanusChatMarkers(
            bos_token=str(tokenizer.bos_token),
            eos_token=str(tokenizer.eos_token),
            boi_token=str(tokenizer.boi_token),
            eoi_token=str(tokenizer.eoi_token),
            system_prompt=_JANUS_SYSTEM_PROMPT,
            user_prefix=_JANUS_USER_PREFIX,
            assistant_prefix=_JANUS_ASSISTANT_PREFIX,
        )

    # ── Janus Text Encoder Main Function ───────────────────────────────────────
    def encode(  # type: ignore[override]
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Any = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Wte lookup only: ``input_ids`` → ``inputs_embeds``."""
        input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        embeds = self.embed_tokens(input_ids)
        return {
            "inputs_embeds": embeds.squeeze(0) if embeds.size(0) == 1 else embeds,
        }

    # ── Inference: decode + FSM signals (conversation-list aware) ────────────

    def decode(  # type: ignore[override]
        self,
        hidden_states: Optional[torch.Tensor] = None,
        shift_labels: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithLogProbs:
        """LM-head projection for training; inference sampling uses :meth:`ar_step`."""
        return super().decode(
            hidden_states=hidden_states,
            shift_labels=shift_labels,
            **kwargs,
        )

    # ── Training: conversation-list carrier (pre → encode → post) ─────────────

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Stash ``conversation_list``; ``encode`` builds flat ``input_ids`` + per-item meta."""
        assert method in ("encode", "decode")

        if method == "encode":
            self._conversation_carrier = conversation_list
            input_ids = self._prepare_encode_inputs(self._conversation_carrier)
            return {"input_ids": input_ids}
        else:  # decode
            self._conversation_carrier = conversation_list
            hidden_states, shift_labels = self._prepare_decode_inputs(self._conversation_carrier)
            return {"hidden_states": hidden_states, "shift_labels": shift_labels}

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        """Scatter flat wte output back onto text items; rebuild sample order for the backbone."""
        assert method in ("encode", "decode")
        if method == "encode":
            conversation = self._conversation_carrier
            batch_shape = self._encode_batch_shape
            self._conversation_carrier = None
            self._encode_batch_shape = None
            inputs_embeds = outputs.get("inputs_embeds")
            self._scatter_text_embeds(conversation, unflatten(inputs_embeds, batch_shape))
            return {"conversation_list": conversation}

        if method == "decode":
            outputs = super().post_forward(method=method, **outputs)
            conversation = self._conversation_carrier
            self._conversation_carrier = None
            outputs["conversation_list"] = conversation
            return outputs

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, bool]:
        """Apply template in-place per sample; ``extend`` text token tensors → ``naflatten``."""
        input_ids: list[torch.Tensor] = []
        self._encode_batch_shape = None
        for sample in conversation_list or []:
            self._prepare_sample_training(sample)
            input_ids.extend(pack_text_input_ids(sample))
        input_ids, self._encode_batch_shape = naflatten(input_ids)
        return input_ids

    def _prepare_sample_training(self, sample: list[ConversationItem]) -> None:
        """Apply chat template to ``sample`` in place: tokenize text rows, merge adjacent text."""
        parts = apply_janus_chat_template(sample, self._chat_markers)
        self._tokenize_template_parts(parts)
        parts = self._merge_consecutive_text_parts(parts)
        sample.clear()
        sample.extend(parts)

    def _merge_consecutive_text_parts(self, parts: list[ConversationItem]) -> list[ConversationItem]:
        """Merge adjacent ``type='text'`` rows (concat token ids, labels, attention_mask)."""
        merged: list[ConversationItem] = []
        for part in parts:
            if merged and merged[-1].type == "text" and part.type == "text":
                prev = merged[-1]
                prev.value = torch.cat([prev.value, part.value])
                prev.meta["labels"] = torch.cat([prev.meta["labels"], part.meta["labels"]])
                prev.meta["attention_mask"] = torch.cat([prev.meta["attention_mask"], part.meta["attention_mask"]])
                continue
            merged.append(part)
        return merged

    def _tokenize_template_parts(self, parts: list[ConversationItem]) -> None:
        """Token ids in ``value``; ``labels`` / ``attention_mask`` in ``meta``."""
        device = self.device
        for part in parts:
            if part.type == "text":
                text = part.value
                loss_mask = int(part.meta.pop("loss_mask"))
                input_ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
                labels = input_ids if loss_mask else [-100] * len(input_ids)
                part.value = torch.tensor(input_ids, device=device, dtype=torch.long)
                part.meta["labels"] = torch.tensor(labels, device=device, dtype=torch.long)
                part.meta["attention_mask"] = torch.ones(len(input_ids), dtype=torch.long, device=device)

    def _scatter_text_embeds(
        self,
        conversation_list: list[list[ConversationItem]],
        segment_embeds: list[torch.Tensor],
    ) -> None:
        """Write WTE segments back onto ``type='text'`` items in pack order."""
        dtype = self.dtype
        segment_embeds_iterator = iter(segment_embeds)
        for sample in conversation_list:
            for part in sample:
                if part.type != "text":
                    continue
                part.value = next(segment_embeds_iterator).to(device=self.device, dtype=dtype)
        if next(segment_embeds_iterator, None) is not None:
            raise RuntimeError(
                f"text segment count mismatch: scattered {len(segment_embeds)}, expected {len(conversation_list)}"
            )

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[list[list[ConversationItem]]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Flat concat of all non-dummy hidden states + labels, then shift once.

        Samples are not padded separately — the whole micro-batch is one
        sequence so downstream CE can token-mean over all supervised positions.
        Image items without ``meta['labels']`` get ``-100`` filler rows matching
        their hidden length.
        """
        hidden_states_chunks: list[torch.Tensor] = []
        label_chunks: list[torch.Tensor] = []

        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                hidden_states = part.value
                if hidden_states.dim() == 3:
                    hidden_states = hidden_states.squeeze(0)
                if part.type == "text":
                    labels = part.meta["labels"]
                    assert labels.shape[0] == hidden_states.shape[0]
                    hidden_states_chunks.append(hidden_states)
                    label_chunks.append(labels)
                elif part.type == "image":
                    hidden_states_chunks.append(hidden_states[-1:])
                    label_chunks.append(torch.full((1,), -100, dtype=torch.long, device=hidden_states.device))

        hidden_states = torch.cat(hidden_states_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)

        labels = labels[..., 1:].contiguous()
        shift_labels = F.pad(labels, (0, 1), "constant", -100)
        return hidden_states, shift_labels

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

        device = self.device
        dtype = self.dtype
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

        if tok == self._boi_token_id:
            self._maybe_arm_cfg_for_image_gen(conversation_list, generation_kwargs, out)
            seal_phase_outputs(conversation_list, phase="text", new_type="text")
            tail.type = "soi"
            tail.meta.pop("phase", None)
            out[FSM_SIGNAL_KEY] = SIGNAL_START_IMAGE_GEN
            out["ar_phase"] = "vq"
        elif tok == self._eos_token_id:
            seal_phase_outputs(conversation_list, phase="text", new_type="text")
            out[FSM_SIGNAL_KEY] = SIGNAL_TEXT_DONE

        return out

    # ── Janus boundary-token emitters (conversation-list aware) ──────────────

    def emit_image_start(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **ctx: Any,
    ) -> Dict[str, Any]:
        """Emit ``<begin_of_image>`` as a standalone ``soi`` conversation item."""
        token_id = self._boi_token_id
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
        token_id = self._eoi_token_id
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
        device = self.device
        dtype = self.dtype
        batch_size = _infer_batch_size(ctx)
        ids = torch.full((batch_size, 1), token_id, dtype=torch.long, device=device)
        inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
        out: Dict[str, Any] = {"input_ids": ids, "inputs_embeds": inputs_embeds}
        if conversation_list is not None:
            conversation_list.append(
                ConversationItem(
                    type=item_type,
                    value=inputs_embeds,
                    role="assistant",
                    meta={"token_id": int(token_id), "input_ids": ids},
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

        Returns ``None`` when the prompt is not text-only (e.g. I2T with an ``image`` part).
        """
        if any(p.type == "image" for p in conversation_list):
            return None

        bos_id = int(self._bos_token_id)
        pad_id = int(self._pad_token_id)
        boi_id = self._boi_token_id  # may be None if not registered
        device = self.device
        dtype = self.dtype

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
        text = self._tokenizer.decode(token_ids, skip_special_tokens=True)
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
        device = self.device
        dtype = self.dtype
        ids = torch.tensor([[self._bos_token_id]], dtype=torch.long, device=device)
        embed = self.embed_tokens(ids).to(dtype=dtype)
        conversation_list.insert(
            0,
            ConversationItem(
                type="token",
                value=embed,
                role="system",
                meta={"token_id": int(self._bos_token_id), "input_ids": ids},
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
            return ConversationItem(type="text", value=text, role=role, meta={"source": "chat_template"})

        def _token_marker(role: str, token_id: int) -> ConversationItem:
            return ConversationItem(
                type="token",
                value=int(token_id),
                role=role,
                meta={"source": "chat_template", "token_id": int(token_id)},
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
                new_list.append(_token_marker("user", self._boi_token_id))
                new_list.append(part)
                new_list.append(_token_marker("user", self._eoi_token_id))
                next_part = conversation_list[idx + 1] if idx + 1 < len(conversation_list) else None
                if next_part is not None and item_role(next_part) == role:
                    new_list.append(_text_marker(role, "\n"))
                continue

            new_list.append(part)

        conversation_list.clear()
        conversation_list.extend(new_list)

    def _embed_part(self, part: ConversationItem) -> None:
        device = self.device
        dtype = self.dtype

        if part.type == "text":
            if not part.value:
                return
            ids = self._tokenizer(part.value, return_tensors="pt", add_special_tokens=False)["input_ids"]
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
        device = self.device
        dtype = self.dtype
        if inputs_embeds is None:
            ids = torch.tensor([[int(token_id)]], dtype=torch.long, device=device)
            inputs_embeds = self.embed_tokens(ids).to(dtype=dtype)
            conversation_list.append(
                ConversationItem(
                    type="token",
                    value=inputs_embeds,
                    role="assistant",
                    meta={"token_id": int(token_id), "input_ids": ids},
                )
            )
        else:
            conversation_list.append(
                ConversationItem(
                    type="token",
                    value=inputs_embeds,
                    role="assistant",
                    meta={"token_id": int(token_id)},
                )
            )

    def _token_id_tensor(self, token_id: int) -> torch.Tensor:
        device = self.device
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


def _infer_batch_size(ctx: Dict[str, Any]) -> int:
    """Best-effort batch-size inference from tensors already in ``ctx``."""
    for key in ("input_ids", "inputs_embeds", "hidden_states", "attention_mask"):
        v = ctx.get(key)
        if isinstance(v, torch.Tensor) and v.dim() >= 1:
            return int(v.size(0))
    return 1

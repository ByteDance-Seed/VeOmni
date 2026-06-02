"""
JanusLlama — Janus' LLaMA backbone (no wte, no lm_head) as one OmniModule.

Mixin form: ``class JanusLlama(OmniModule, PreTrainedModel)``.

The backbone *contains* a vanilla :class:`transformers.LlamaModel` whose
``embed_tokens`` has been replaced with an :class:`nn.Identity` — the
word-token embedding lives in the sibling
:class:`~veomni.models.seed_omni.modules.base.TextEncoder` module.  This
keeps the LLaMA forward path unchanged; ``inputs_embeds`` (passed in by
the graph from the ``tok_encode`` node) is what actually flows through.

Multi-modal embedding scatter
-----------------------------
:meth:`pre_forward` is the only place that knows how to merge image
embeddings into ``inputs_embeds``: ``masked_scatter`` at
``input_ids == config.image_token_id`` (understanding) or
``config.gen_image_token_id`` (generation) positions.  This is the
"backbone owns the multimodal merge" pattern from design.md §10.

Sequence parallelism
--------------------
When the global :class:`ParallelState` has SP enabled, :meth:`pre_forward`
slices every sequence-domain tensor with
:func:`veomni.distributed.sequence_parallel.data.sp_pad_and_slice` and
:meth:`post_forward` all-gathers ``hidden_states`` back to full length so
downstream nodes (``tok_decode`` / ``vae_decode``) receive non-sliced
tensors.

Connection outputs
------------------
``hidden_states``  — final LLaMA hidden states ``(B, T, D)``.  CE / sampling
                     live in the head modules.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import LlamaConfig, PreTrainedModel
from transformers.models.llama.modeling_llama import LlamaModel

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel.data import gather_outputs, sp_pad_and_slice

from ....conversation import (
    ArPhase,
    ConversationItem,
    append_output_hidden,
    assemble_embeds,
    collect_prompt_embeds,
    get_ar_tail_embed,
    is_dummy,
    is_train_conversation,
)
from ....module import OmniModule
from .configuration import JanusLlamaConfig


class JanusLlama(OmniModule, PreTrainedModel):
    """LLaMA backbone with multi-modal embedding scatter (no wte, no lm_head).

    Image-embedding injection (``und_image_embeds`` / ``gen_image_embeds``)
    is performed via ``masked_scatter`` using a boolean mask derived from
    ``input_ids`` placeholder tokens — same strategy as the original Janus.
    ``inputs_embeds`` is required: there is no fallback ``embed_tokens``
    lookup inside this module.
    """

    config_class = JanusLlamaConfig
    base_model_prefix = "janus_llama"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["LlamaDecoderLayer"]

    def __init__(self, config: JanusLlamaConfig):
        super().__init__(config)

        text_cfg = LlamaConfig(**config.text_config) if config.text_config else LlamaConfig()
        self._text_cfg = text_cfg
        self.language_model = LlamaModel._from_config(text_cfg)
        # Drop the embed_tokens parameters — owned by sibling TextEncoder.
        self.language_model.set_input_embeddings(nn.Identity())

        # Classifier-free guidance runtime state — set to True the first
        # time :meth:`generate` consumes a ``cfg_uncond_inputs_embeds`` from
        # ctx and expands the KV cache to bs=2.  ``_cfg_seen_vqvae`` flips
        # the moment we observe a ``meta.source == "vqvae"`` tail (i.e. we
        # entered the body of ``image_vq``) — used by the AR-branch guard
        # to detect "leaving image_vq" and raise a clear error instead of
        # silently feeding a bs=2 cache into ``text_ar`` / interleave
        # states (where ``tok_decode`` would crash on bs=2 hidden states).
        # Both are reset by :meth:`reset_inference_state` between requests.
        self._cfg_active: bool = False
        self._cfg_seen_vqvae: bool = False
        self._past_key_values: Any = None
        self._ar_phase: ArPhase = "text"

        self.post_init()

    # ── OmniModule interface ───────────────────────────────────────────────────

    def set_conversation_tokenizer(self, conversation_tokenizer: Any) -> None:
        """Resolve image-placeholder ids used by the ``masked_scatter`` merge.

        ``image_token_id`` (``<image_placeholder>``) marks where SigLIP
        understanding patches are injected; ``gen_image_token_id`` marks
        VQ generation-image positions.  Stored on ``config`` so
        :meth:`pre_forward` can build the scatter mask from ``input_ids``.
        Resolved from the wired conversation tokenizer rather than
        ``config.json`` so a single source of truth (the vocabulary) drives
        both the text encoder's placeholder expansion and the backbone's
        scatter.
        """
        und = _resolve_token_id(
            conversation_tokenizer, ("<image_placeholder>", getattr(conversation_tokenizer, "image_token", None))
        )
        if und is not None:
            self.config.image_token_id = und
        gen = _resolve_token_id(conversation_tokenizer, (getattr(conversation_tokenizer, "gen_image_token", None),))
        if gen is not None:
            self.config.gen_image_token_id = gen

    def pre_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        und_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_mask: Optional[torch.Tensor] = None,
        conversation_list: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Segment-driven splice (V2) or legacy masked_scatter merge, then SP slice.

        V2 (``conversation_list`` is a :class:`TrainConversation`): concatenate
        every per-sample segment's ``embeds`` in order → ``inputs_embeds``
        (NO ``masked_scatter``, NO placeholder ids), and derive
        ``attention_mask`` / ``position_ids`` from the assembled lengths.  The
        full-batch ``und_embeds`` / ``gen_embeds`` on the carrier are folded in
        as ``sum() * 0.0`` autograd anchors so every encoder always receives a
        (zero) gradient even on image-free micro-batches (FSDP DP alignment).
        The carrier is forwarded on so ``tok_decode`` / ``vae_decode`` rebuild
        their labels from the same segments.

        Legacy (``input_ids`` + ``*_image_embeds`` + ``gen_image_mask``):
        ``masked_scatter`` at placeholder positions — kept for any
        pre-tensorised / test callers.

        SP ``pad_and_slice`` is applied last so an image span crossing an SP
        boundary keeps its embedding; :meth:`post_forward` gathers back.
        """
        if is_train_conversation(conversation_list):
            return self._pre_forward_segments(conversation_list, kwargs)

        if inputs_embeds is None:
            raise ValueError("JanusLlama.pre_forward expects `inputs_embeds` from `tok_encode`.")

        # Multi-modal merge.  The ``+ x.sum() * 0.0`` anchor below is the
        # FSDP grad-sync guard for training-side dummy forward (see module-
        # doc "Training vs. inference no input semantics" in
        # :mod:`veomni.models.seed_omni.module`): when the placeholder mask
        # is all-False (text-only micro-batch with dummy zero image
        # embeds), ``masked_scatter`` would otherwise drop the gradient
        # path back to the upstream encoder and FSDP grad-reduce would
        # mismatch across DP ranks.  The anchor adds a zero-valued
        # contribution that *is* part of the autograd graph, so the
        # upstream module always receives a (zero) gradient.
        if und_image_embeds is not None and input_ids is not None:
            und_mask = input_ids == self.config.image_token_id
            inputs_embeds = self._scatter_by_mask(inputs_embeds, und_mask, und_image_embeds)
            if self.training:
                inputs_embeds = inputs_embeds + und_image_embeds.sum() * 0.0
        if gen_image_embeds is not None and gen_image_mask is not None:
            inputs_embeds = self._scatter_by_mask(inputs_embeds, gen_image_mask.bool(), gen_image_embeds)
            if self.training:
                inputs_embeds = inputs_embeds + gen_image_embeds.sum() * 0.0

        ps = get_parallel_state()
        if ps.sp_enabled:
            if input_ids is not None:
                input_ids = sp_pad_and_slice(input_ids, dim=1, pad_value=0)
            inputs_embeds = sp_pad_and_slice(inputs_embeds, dim=1, pad_value=0)
            if labels is not None:
                labels = sp_pad_and_slice(labels, dim=1, pad_value=-100)
            if attention_mask is not None:
                attention_mask = sp_pad_and_slice(attention_mask, dim=1, pad_value=0)
            if position_ids is not None:
                position_ids = sp_pad_and_slice(position_ids, dim=1, pad_value=0)

        return dict(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            labels=labels,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )

    def _pre_forward_segments(self, conv: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Assemble inputs from the :class:`TrainConversation` segments + anchors."""
        if conv.segments is None:
            raise ValueError(
                "JanusLlama.pre_forward: TrainConversation has no segments (text encoder must run first)."
            )
        inputs_embeds, attention_mask, position_ids = assemble_embeds(conv.segments)

        # FSDP grad-sync anchors over the FULL encoder batch tensors — a
        # micro-batch with no image of a modality still drives a (zero)
        # gradient through that encoder so DP grad-reduce stays aligned.
        if self.training:
            if conv.und_embeds is not None:
                inputs_embeds = inputs_embeds + conv.und_embeds.sum() * 0.0
            if conv.gen_embeds is not None:
                inputs_embeds = inputs_embeds + conv.gen_embeds.sum() * 0.0

        ps = get_parallel_state()
        if ps.sp_enabled:
            inputs_embeds = sp_pad_and_slice(inputs_embeds, dim=1, pad_value=0)
            attention_mask = sp_pad_and_slice(attention_mask, dim=1, pad_value=0)
            position_ids = sp_pad_and_slice(position_ids, dim=1, pad_value=0)

        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            conversation_list=conv,
            **kwargs,
        )

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """All-gather ``hidden_states`` across SP ranks (when SP enabled).

        Also stash the gathered tensor on the :class:`TrainConversation` carrier
        so downstream decode heads read it from ``conversation_list.hidden_states``
        — no edge ``output``/``as`` routing needed.
        """
        ps = get_parallel_state()
        if ps.sp_enabled and "hidden_states" in outputs:
            outputs["hidden_states"] = gather_outputs(outputs["hidden_states"], gather_dim=1, scale_grad=True)
        convo = outputs.get("conversation_list")
        if is_train_conversation(convo) and "hidden_states" in outputs:
            convo.hidden_states = outputs["hidden_states"]
        return outputs

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        conversation_list: Optional[Any] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Backbone forward.  ``inputs_embeds`` is required (no internal wte).

        In V2 training the :class:`TrainConversation` carrier is echoed in the
        output so the loss heads (``tok_decode`` / ``vae_decode``) rebuild their
        labels from the same per-sample segments the backbone spliced.
        """
        if inputs_embeds is None:
            raise ValueError(
                "JanusLlama.forward expects `inputs_embeds` (produced by "
                "`text_encoder.encode`). Word-token embedding has moved to "
                "the TextEncoder module."
            )

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **{k: v for k, v in kwargs.items() if k in ("cache_position",)},
        )
        out: Dict[str, Any] = {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
        }
        if conversation_list is not None:
            out["conversation_list"] = conversation_list
        return out

    def generate_step(
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values=None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Single auto-regressive backbone step (FSM ``image_vq`` / ``text_ar``).

        Returns hidden states only; sampling lives downstream in
        ``text_encoder.decode`` / :meth:`JanusVqvae.decode`.
        """
        if inputs_embeds is None:
            raise ValueError(
                "JanusLlama.generate_step expects `inputs_embeds` (from text_encoder.encode or vqvae.decode)."
            )
        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
        }

    # ── Inference (conversation-list aware) ──────────────────────────────────

    def reset_inference_state(self) -> None:
        """Wipe per-request runtime state — CFG flag, KV cache, etc.

        Called by :class:`OmniInferencer` between requests so the next
        prompt starts fresh (no leftover ``_cfg_active`` from a previous
        T2I call carrying over into an I2T call).
        """
        self._cfg_active = False
        self._cfg_seen_vqvae = False
        self._past_key_values = None
        self._ar_phase = "text"

    def set_ar_phase(self, phase: ArPhase) -> None:
        """Switch the active AR phase (``text`` vs ``vq``) for ``output`` tagging."""
        self._ar_phase = phase

    def collapse_cfg_cache(self) -> None:
        """Drop the unconditional half of a bs=2 KV cache after image generation."""
        if not self._cfg_active or self._past_key_values is None:
            return
        self._past_key_values = _slice_kv_cache(self._past_key_values, slice(0, 1))
        self._cfg_active = False
        self._cfg_seen_vqvae = False

    def generate(
        self,
        *,
        conversation_list: Optional[List[ConversationItem]] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cfg_uncond_inputs_embeds: Optional[torch.Tensor] = None,
        ar_phase: Optional[ArPhase] = None,
        cfg: Optional[Dict[str, Any]] = None,
        collapse_cfg: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Auto-regressive backbone step driven by a conversation list.

        Prompt pass (internal KV empty): concatenate every sealed embedded
        part (``text`` / ``image`` / ``soi`` / ``eoi`` / ``token``), prime
        the KV cache, and append an ``output`` item with backbone hidden
        states.

        AR passes: read the last position of the conversation tail embed,
        run one forward step, append a fresh ``output`` hidden item.
        ``past_key_values`` is kept on the module — callers should not
        rely on routing KV through ``ctx``.
        """
        if conversation_list is None:
            raise ValueError(
                "JanusLlama.generate expects `conversation_list` — call OmniInferencer.generate "
                "(or build a conversation manually with veomni.models.seed_omni.build_conversation)."
            )

        if collapse_cfg:
            self.collapse_cfg_cache()

        if ar_phase is not None:
            self._ar_phase = ar_phase

        cache_kwargs = {k: v for k, v in kwargs.items() if k in ("cache_position",)}
        past_kv = self._past_key_values

        if past_kv is None:
            embed_chunks = [emb.to(self.device) for emb in collect_prompt_embeds(conversation_list)]
            if not embed_chunks:
                raise ValueError(
                    "JanusLlama.generate: conversation_list has no embedded parts. "
                    "Run siglip + text_encoder generate first."
                )
            inputs_embeds = torch.cat(embed_chunks, dim=1)
            for part in conversation_list:
                if is_dummy(part) and isinstance(part.value, torch.Tensor):
                    inputs_embeds = inputs_embeds + part.value.sum() * 0.0

            lm_out = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                **cache_kwargs,
            )
            self._past_key_values = lm_out.past_key_values
            hidden = lm_out.last_hidden_state
            append_output_hidden(conversation_list, hidden, phase=self._ar_phase)
            return {
                "hidden_states": hidden,
                "conversation_list": conversation_list,
            }

        tail_embed = get_ar_tail_embed(conversation_list)
        if tail_embed is None:
            raise ValueError(
                "JanusLlama.generate (AR step) expects an embedded conversation tail. "
                "Upstream module (text_encoder.ar_step / vqvae.ar_step / emit_*) did not provide one."
            )

        cfg_out: Optional[Dict[str, Any]] = None
        if (
            isinstance(cfg, dict)
            and cfg.get("enabled")
            and cfg_uncond_inputs_embeds is not None
            and not self._cfg_active
        ):
            uncond = cfg_uncond_inputs_embeds.to(self.device)
            uncond_out = self.language_model(
                inputs_embeds=uncond,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                **cache_kwargs,
            )
            past_kv = _concat_kv_caches(past_kv, uncond_out.past_key_values)
            self._past_key_values = past_kv
            self._cfg_active = True
            cfg_out = {"enabled": False, "guidance_scale": float(cfg.get("guidance_scale", 1.0) or 1.0)}

        if self._cfg_active:
            if self._ar_phase == "vq":
                self._cfg_seen_vqvae = True
            elif self._cfg_seen_vqvae:
                self.collapse_cfg_cache()
                past_kv = self._past_key_values

        inputs_embeds = tail_embed.to(self.device)
        if self._cfg_active and inputs_embeds.size(0) == 1:
            inputs_embeds = inputs_embeds.expand(2, *inputs_embeds.shape[1:]).contiguous()

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_kv,
            use_cache=True,
            **cache_kwargs,
        )
        self._past_key_values = lm_out.past_key_values
        hidden = lm_out.last_hidden_state
        append_output_hidden(conversation_list, hidden, phase=self._ar_phase)
        out: Dict[str, Any] = {
            "hidden_states": hidden,
            "conversation_list": conversation_list,
        }
        if cfg_out is not None:
            out["cfg"] = cfg_out
        return out

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _scatter_by_mask(
        inputs_embeds: torch.Tensor,
        mask: torch.Tensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Replace ``mask`` positions in ``inputs_embeds`` with ``image_embeds``.

        ``mask`` is ``(B, T)`` bool; ``image_embeds`` is ``(B, K, D)`` where
        every sample carries ``K`` patch embeddings (real for present rows,
        dummy zeros for absent rows — the training dummy-forward keeps the
        FSDP graph aligned).  We select only the **present rows** (``mask``
        has any True) before flattening so ``masked_scatter`` consumes each
        present sample's own embeddings: a naïve full flatten would feed an
        absent row's dummy embeds into a present row when the two are not in
        ascending order within the batch (mixed und/gen/text micro-batches).
        """
        present = mask.any(dim=1)
        flat = image_embeds[present].reshape(-1, inputs_embeds.size(-1)).to(inputs_embeds.dtype)
        m = mask.unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(m, flat)


def _resolve_token_id(tokenizer: Any, candidates: tuple) -> Optional[int]:
    """Return the first candidate token string that maps to a real (non-unk) id."""
    unk = getattr(tokenizer, "unk_token_id", None)
    for cand in candidates:
        if not cand:
            continue
        try:
            tid = tokenizer.convert_tokens_to_ids(cand)
        except (KeyError, ValueError):
            continue
        if tid is not None and tid != unk:
            return int(tid)
    return None


def _slice_kv_cache(cache: Any, index: slice) -> Any:
    """Slice a ``DynamicCache`` along batch dim 0."""
    if cache is None:
        return None
    sliced = type(cache)()
    layers = getattr(cache, "layers", None)
    if layers is None:
        n_layers = len(getattr(cache, "key_cache", []))
        for i in range(n_layers):
            k = cache.key_cache[i][index]
            v = cache.value_cache[i][index]
            sliced.update(k, v, i)
        return sliced
    for i, _ in enumerate(layers):
        k = cache.layers[i].keys[index]
        v = cache.layers[i].values[index]
        sliced.update(k, v, i)
    return sliced


def _concat_kv_caches(cond: Any, uncond: Any) -> Any:
    """Batch-concat two ``DynamicCache``s along dim 0.

    Scoped to ``DynamicCache`` with all-``DynamicLayer`` layers — which is
    what plain ``LlamaModel`` produces on Janus today.  Other cache classes
    (``StaticCache``, ``HybridCache``, ``SlidingWindowCache``) require
    constructor args and / or carry per-layer state (``is_sliding``,
    ``_sliding_window_tensor``, ``cumulative_length``) that this helper does
    NOT preserve — passing them in is a bug and will either raise on the
    ``type(cond)()`` call or silently lose layer metadata.  Janus has no
    sliding-window attention so we don't hit that path; revisit when
    extending CFG to backbones that do.

    Supports both the v5.9 ``.layers[i].keys`` schema and the older
    ``.key_cache[i]`` / ``.value_cache[i]`` lists via ``getattr`` probe,
    because the transformers cache module is still in flux and we want
    the helper to keep working through minor bumps.
    """
    if cond is None or uncond is None:
        return cond or uncond
    if type(cond) is not type(uncond):
        raise TypeError(
            f"_concat_kv_caches: cond ({type(cond).__name__}) and uncond ({type(uncond).__name__}) "
            "must be the same cache class."
        )
    merged = type(cond)()
    layers = getattr(cond, "layers", None)
    if layers is None:
        # Pre-5.9 DynamicCache exposed `key_cache` / `value_cache` lists.
        n_layers = len(getattr(cond, "key_cache", []))
        for i in range(n_layers):
            k = torch.cat([cond.key_cache[i], uncond.key_cache[i]], dim=0)
            v = torch.cat([cond.value_cache[i], uncond.value_cache[i]], dim=0)
            merged.update(k, v, i)
        return merged
    for i, _ in enumerate(layers):
        k = torch.cat([cond.layers[i].keys, uncond.layers[i].keys], dim=0)
        v = torch.cat([cond.layers[i].values, uncond.layers[i].values], dim=0)
        merged.update(k, v, i)
    return merged

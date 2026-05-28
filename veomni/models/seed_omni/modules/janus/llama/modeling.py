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

from ....conversation import ConversationPart
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

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if hasattr(module, "_init_weights"):
            return

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        und_image_embeds: Optional[torch.FloatTensor] = None,
        gen_image_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Multi-modal merge + SP slice (when enabled).

        Steps (in order):

          1. Scatter ``und_image_embeds`` / ``gen_image_embeds`` into
             ``inputs_embeds`` at placeholder token positions.  Done here
             rather than in :meth:`forward` so the SP slice (step 2) sees
             a fully-merged sequence — otherwise an image whose tokens
             span an SP boundary would lose its embedding.
          2. SP pad-and-slice every sequence-domain tensor.
          3. Forward pass through the LLaMA backbone and post-forward
             gather (the latter in :meth:`post_forward`).
        """
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
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, und_image_embeds, self.config.image_token_id
            )
            if self.training:
                inputs_embeds = inputs_embeds + und_image_embeds.sum() * 0.0
        if gen_image_embeds is not None and input_ids is not None:
            inputs_embeds = self._scatter_image_embeds(
                inputs_embeds, input_ids, gen_image_embeds, self.config.gen_image_token_id
            )
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

    def post_forward(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        """All-gather ``hidden_states`` across SP ranks (when SP enabled)."""
        ps = get_parallel_state()
        if ps.sp_enabled and "hidden_states" in outputs:
            outputs["hidden_states"] = gather_outputs(outputs["hidden_states"], gather_dim=1, scale_grad=True)
        return outputs

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Backbone forward.  ``inputs_embeds`` is required (no internal wte)."""
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
        return {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
        }

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
        """Wipe per-request runtime state — CFG flag, etc.

        Called by :class:`OmniInferencer` between requests so the next
        prompt starts fresh (no leftover ``_cfg_active`` from a previous
        T2I call carrying over into an I2T call).
        """
        self._cfg_active = False
        self._cfg_seen_vqvae = False

    def generate(
        self,
        *,
        conversation_list: Optional[List[ConversationPart]] = None,
        past_key_values: Optional[Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cfg_uncond_inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Auto-regressive backbone step driven by a conversation list.

        First pass (no KV cache): concatenate every part's
        ``inputs_embeds`` along the sequence dimension and run a single
        full prompt forward.  Subsequent passes (KV cache hot):
        consume **only** the trailing part's ``inputs_embeds`` — the
        same tail-only convention HuggingFace ``generate`` uses to
        avoid re-feeding the prefix.

        Any part without ``inputs_embeds`` (e.g. an empty assistant
        marker, a token whose embedding the upstream module hasn't yet
        populated) is skipped.  Routing edges may carry the same
        ``conversation_list`` object from multiple upstream encoders
        (siglip + text_encoder) — by the time this method runs the
        topological body-execution rule guarantees every relevant part
        has been filled.

        Classifier-free guidance (CFG)
        ------------------------------
        When ``cfg_uncond_inputs_embeds`` is present in ctx (built by
        :meth:`JanusTextEncoder.generate`) and ``self._cfg_active`` is
        False, the first AR call after the prompt pass runs a *separate*
        full-prompt forward on the uncond embeds, then batch-concats the
        resulting KV cache with the existing cond cache to form a bs=2
        cache.  The tail input (``boi`` from ``tok_decode`` or any
        subsequent VQ-token embed) is broadcast from bs=1 to bs=2 so the
        backbone stays bs=2 for the rest of ``image_vq``.  The cond /
        uncond logits split + CFG mix lives downstream in
        :meth:`JanusVqvae.generate`.

        Invariant 17 (SeedOmni V2): graph nodes / FSM topology stay
        unchanged; the bs=2 expansion is fully hidden inside this module.
        See ``.agents/skills/seedomni-v2/SKILL.md`` for the design
        rationale.
        """
        if conversation_list is None:
            raise ValueError(
                "JanusLlama.generate expects `conversation_list` — call OmniInferencer.generate "
                "(or build a conversation manually with veomni.models.seed_omni.build_conversation)."
            )

        cache_kwargs = {k: v for k, v in kwargs.items() if k in ("cache_position",)}

        if past_key_values is None:
            embed_chunks = [p.inputs_embeds for p in conversation_list if p.inputs_embeds is not None]
            if not embed_chunks:
                raise ValueError(
                    "JanusLlama.generate: conversation_list has no embedded parts. "
                    "Run siglip + text_encoder generate first."
                )
            inputs_embeds = torch.cat(embed_chunks, dim=1)

            # Cond pass first — keep its hidden states for downstream
            # `tok_decode` (which only ever sees bs=1 — CFG mixing is a
            # `vae_generate`-side concern in the seed_omni V2 contract).
            lm_out = self.language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=None,
                use_cache=True,
                **cache_kwargs,
            )
            past_kv_out = lm_out.past_key_values

            if cfg_uncond_inputs_embeds is not None:
                if cfg_uncond_inputs_embeds.shape[:2] != inputs_embeds.shape[:2]:
                    raise ValueError(
                        "JanusLlama.generate: `cfg_uncond_inputs_embeds` shape "
                        f"{tuple(cfg_uncond_inputs_embeds.shape)} does not match the cond prompt "
                        f"{tuple(inputs_embeds.shape)} (text_encoder mis-built the uncond branch)."
                    )
                uncond_out = self.language_model(
                    inputs_embeds=cfg_uncond_inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    use_cache=True,
                    **cache_kwargs,
                )
                past_kv_out = _concat_kv_caches(past_kv_out, uncond_out.past_key_values)
                self._cfg_active = True

            return {
                "hidden_states": lm_out.last_hidden_state,
                "past_key_values": past_kv_out,
                "conversation_list": conversation_list,
            }
            # Note: ``cfg_uncond_inputs_embeds`` lingers in ctx after this
            # branch returns (the FSM ``ctx.update(out)`` only adds keys,
            # doesn't drain).  That is intentional: it is harmless because
            # the AR branch below never reads it (it dispatches on
            # ``self._cfg_active`` instead), and every other module's
            # ``generate`` signature absorbs unknown kwargs via ``**_``.

        tail = conversation_list[-1] if conversation_list else None
        if tail is None or tail.inputs_embeds is None:
            raise ValueError(
                "JanusLlama.generate (AR step) expects the trailing conversation part to carry "
                "`inputs_embeds`. Upstream module (text_encoder.decode / vqvae.generate / emit_*) "
                "did not populate it."
            )

        # CFG state machine, AR branch
        # ----------------------------
        # ``_cfg_seen_vqvae`` flips True the moment we observe a vqvae-sourced
        # tail (i.e. we're inside ``image_vq``).  If we *later* see a tail that
        # is NOT vqvae-sourced while ``_cfg_seen_vqvae`` is True, the FSM has
        # left ``image_vq`` and routed the bs=2 KV cache into ``text_ar`` /
        # interleave — that requires collapsing the cache to the cond half
        # first, which is not yet implemented.  Fail loudly rather than
        # silently feeding bs=2 hidden states into ``tok_decode`` (which
        # would crash inside ``int(token.item())`` on a ``(2,1)`` tensor).
        if self._cfg_active:
            tail_is_vqvae = tail.meta.get("source") == "vqvae"
            if tail_is_vqvae:
                self._cfg_seen_vqvae = True
            elif self._cfg_seen_vqvae:
                raise NotImplementedError(
                    "JanusLlama.generate: CFG (guidance_scale>1) is active and the FSM has left "
                    "the `image_vq` span — routing the bs=2 KV cache into `text_ar` / interleave "
                    "states is not supported yet (need to collapse the cache to the cond half). "
                    "Run inference with `guidance_scale<=1` for interleave / I2T graphs, or "
                    "extend `JanusLlama` with a `_collapse_cfg_cache()` helper. "
                    "See seedomni-v2 invariant 17."
                )

        inputs_embeds = tail.inputs_embeds

        if self._cfg_active and inputs_embeds.size(0) == 1:
            inputs_embeds = inputs_embeds.expand(2, *inputs_embeds.shape[1:]).contiguous()

        lm_out = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            **cache_kwargs,
        )
        return {
            "hidden_states": lm_out.last_hidden_state,
            "past_key_values": lm_out.past_key_values,
            "conversation_list": conversation_list,
        }

    # ── Internal helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _scatter_image_embeds(
        inputs_embeds: torch.Tensor,
        input_ids: torch.LongTensor,
        image_embeds: torch.Tensor,
        placeholder_token_id: int,
    ) -> torch.Tensor:
        """Replace placeholder positions in ``inputs_embeds`` with ``image_embeds``."""
        mask = (input_ids == placeholder_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        flat_embeds = image_embeds.reshape(-1, inputs_embeds.size(-1))
        return inputs_embeds.masked_scatter(mask, flat_embeds)


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

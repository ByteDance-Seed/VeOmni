"""
JanusLlama — Janus' LLaMA backbone (no wte, no lm_head) as one OmniModule.

Mixin form: ``class JanusLlama(OmniModule, PreTrainedModel)``.

The backbone *contains* VeOmni's patched :class:`~veomni.models.transformers.llama.generated.patched_modeling_llama_gpu.LlamaModel` whose
``embed_tokens`` has been replaced with an :class:`nn.Identity` — the
word-token embedding lives in the sibling
:class:`~veomni.models.seed_omni.modules.base.TextEncoder` module.  This
keeps the LLaMA forward path unchanged; ``inputs_embeds`` (passed in by
the graph from the ``tok_encode`` node) is what actually flows through.

Multi-modal embedding packing
-----------------------------
:meth:`pre_forward` concatenates each sample's embedded conversation items
into one packed sequence (``bs=1``), builds per-sample ``position_ids``
(``range(0, len)`` per sample), and precomputes Flash-Attention varlen
kwargs (``cu_seq_lens_*`` / ``max_length_*``) for the patched LLaMA
backbone.  :meth:`post_forward` unpacks ``hidden_states`` back to
right-padded ``(B, T, D)`` for the decode heads.

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

from veomni.distributed.parallel_state import get_parallel_state
from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaModel
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids
from veomni.utils.tensor_utils import naflatten, unflatten

from ....conversation import (
    ArPhase,
    ConversationItem,
    is_dummy,
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
    # Inner ``language_model`` is VeOmni's patched ``LlamaModel`` (fused ops + SP).
    supports_gradient_checkpointing = True

    def __init__(self, config: JanusLlamaConfig):
        super().__init__(config)

        text_cfg = LlamaConfig(**config.text_config) if config.text_config else LlamaConfig()
        self._text_cfg = text_cfg
        self.language_model = LlamaModel._from_config(text_cfg)
        # Drop the embed_tokens parameters — owned by sibling TextEncoder.
        self.language_model.set_input_embeddings(nn.Identity())

        # Inference state
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

        # Training state
        self._conversation_carrier: Optional[list[list[ConversationItem]]] = None
        self._pack_inputs_embeds_shape: Optional[torch.Tensor] = None

        self.post_init()

    # ── JanusLlama Main Function ─────────────────────────────────────────────────────
    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Backbone forward — ``inputs_embeds`` in, ``hidden_states`` out."""
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
        }

    # ── OmniModule interface ───────────────────────────────────────────────────

    def pre_forward(
        self,
        method: str,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Pack embedded ``conversation_list`` into a single bs=1 sequence.

        Per sample: concatenate every embedded item's ``value`` in order.
        Across samples: pack into one sequence with per-sample ``position_ids``
        ``range(0, len)``.  Precompute Flash-Attention varlen kwargs
        (``cu_seq_lens_*`` / ``max_length_*``) so decoder layers skip the
        per-layer host sync.  SP ``pad_and_slice`` is applied last;
        :meth:`post_forward` gathers and unpacks back to ``(B, T, D)``.
        """
        assert method == "forward"
        inputs_embeds, attention_mask, position_ids, inputs_embeds_shape = self._pack_conversations_for_forward(
            conversation_list
        )

        if self.training and get_parallel_state().fsdp_enabled:
            inputs_embeds = _fold_fsdp_dummy_anchors(inputs_embeds, conversation_list)

        self._conversation_carrier = conversation_list
        self._pack_inputs_embeds_shape = inputs_embeds_shape

        if get_parallel_state().sp_enabled:
            raise NotImplementedError("SP is not supported yet")
        else:
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )
        return dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            **kwargs,
        )

    def post_forward(self, method: str, **outputs: Any) -> Dict[str, Any]:
        """Unflatten packed hidden states and write them back onto ``conversation_list``."""
        assert method == "forward"
        hidden_states = outputs.get("hidden_states")

        if get_parallel_state().sp_enabled:
            raise NotImplementedError("SP is not supported yet")

        conversation = self._conversation_carrier
        pack_shape = self._pack_inputs_embeds_shape
        self._conversation_carrier = None
        self._pack_inputs_embeds_shape = None

        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        self._scatter_hidden_states(conversation, unflatten(hidden_states, pack_shape))
        return {"conversation_list": conversation}

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
        conversation_list: Optional[List[ConversationItem]] = None,
        generation_kwargs: Dict[str, Any] = dict,
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
        if self._past_key_values is None:  # first AR step
            inputs_embeds, attention_mask, position_ids, _ = self._pack_conversations_for_forward([conversation_list])
            (cu_seq_lens_q, cu_seq_lens_k), (max_length_q, max_length_k) = prepare_fa_kwargs_from_position_ids(
                position_ids
            )

            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=self._past_key_values,
                cu_seq_lens_q=cu_seq_lens_q,
                cu_seq_lens_k=cu_seq_lens_k,
                max_length_q=max_length_q,
                max_length_k=max_length_k,
                use_cache=True,
            )
            self._past_key_values = outputs["past_key_values"]

            hidden_states = outputs["hidden_states"]
            conversation_list.append(
                ConversationItem(
                    type="output",
                    value=self._tail_hidden_from_forward(hidden_states),
                    role="assistant",
                )
            )
            return {
                "conversation_list": conversation_list,
            }
        tail_part = conversation_list[-1]
        past_kv = self._past_key_values
        assert tail_part.type == "output"

        # CFG
        cfg_uncond_inputs_embeds = tail_part.meta.get("cfg_uncond_inputs_embeds", None)
        if cfg_uncond_inputs_embeds is not None and not self._cfg_active:
            uncond = cfg_uncond_inputs_embeds.to(self.device)
            inputs_embeds = tail_part.value[:, -1:].to(self.device)
            if uncond.dim() == 2:
                uncond = uncond.unsqueeze(0)
            uncond_out = self.forward(
                inputs_embeds=uncond,
                attention_mask=None,
                past_key_values=None,
                use_cache=True,
            )
            past_kv = _concat_kv_caches(past_kv, uncond_out["past_key_values"])
            self._past_key_values = past_kv
            self._cfg_active = True
            tail_part.meta.pop("cfg_uncond_inputs_embeds", None)

        inputs_embeds: torch.Tensor = tail_part.value[-1:].to(self.device)  # 1, dim
        inputs_embeds = inputs_embeds.unsqueeze(0)  # 1, 1, dim
        # if self._cfg_active:
        #     if self._ar_phase == "vq":
        #         self._cfg_seen_vqvae = True
        #     elif self._cfg_seen_vqvae:
        #         self.collapse_cfg_cache()
        #         past_kv = self._past_key_values

        if self._cfg_active:
            inputs_embeds = inputs_embeds.expand(2, *inputs_embeds.shape[1:]).contiguous()  # 2, 1, dim

        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=None,
            past_key_values=past_kv,
            use_cache=True,
        )
        self._past_key_values = outputs["past_key_values"]
        conversation_list.append(
            ConversationItem(
                type="output",
                value=self._tail_hidden_from_forward(outputs["hidden_states"]),
                role="assistant",
            )
        )
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        """Last-position backbone hidden for an ``output`` row (keeps CFG batch)."""
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()

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

    def _pack_conversations_for_forward(
        self,
        conversations: list[list[ConversationItem]],
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        int,
        list[int],
    ]:
        """Pack embedded conversations → bs=1 tensors + precomputed FA2 kwargs."""
        inputs_embeds_list = []
        attention_mask = []
        position_ids = []
        for sample in conversations:
            sample_lengths = 0
            for item in sample:
                role = item.role
                if role != "dummy":  # skip dummy items
                    embeds = item.value
                    embeds_length = embeds.size(0)
                    chunk_attention_mask = item.meta.pop("attention_mask", None)
                    if chunk_attention_mask is None:
                        chunk_attention_mask = torch.ones(embeds_length, dtype=torch.long, device=self.device)
                    inputs_embeds_list.append(embeds.to(self.device))
                    attention_mask.append(chunk_attention_mask.to(self.device))
                    sample_lengths += embeds_length
            sample_position_ids = torch.arange(sample_lengths, dtype=torch.long, device=self.device)
            position_ids.append(sample_position_ids)
        inputs_embeds, inputs_embeds_shape = naflatten(inputs_embeds_list)
        position_ids = torch.cat(position_ids, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)

        # Packed varlen path: one logical batch row for Llama + FA2 kwargs.
        if inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        return inputs_embeds, attention_mask, position_ids, inputs_embeds_shape

    def _scatter_hidden_states(
        self,
        conversation_list: list[list[ConversationItem]],
        hidden_states_list: list[torch.Tensor],
    ) -> None:
        """Write backbone hidden segments back onto non-dummy items in pack order."""
        hidden_states_list_iter = iter(hidden_states_list)
        for sample in conversation_list:
            for part in sample:
                if is_dummy(part):
                    continue
                part.value = next(hidden_states_list_iter)
        if next(hidden_states_list_iter, None) is not None:
            raise RuntimeError(
                "JanusLlama._scatter_hidden_states: segment count exceeds non-dummy conversation items."
            )


def _fold_fsdp_dummy_anchors(
    inputs_embeds: torch.Tensor,
    conversations: list[list[ConversationItem]],
) -> torch.Tensor:
    """``inputs_embeds + dummy.value.mean() * 0.0`` for each dummy item."""
    for sample in conversations:
        for part in sample:
            if not is_dummy(part):
                continue
            if not isinstance(part.value, torch.Tensor):
                continue
            fake = part.value.mean().to(device=inputs_embeds.device, dtype=inputs_embeds.dtype) * 0.0
            inputs_embeds = inputs_embeds + fake
    return inputs_embeds


def _sp_pad_tensor(
    tensor: torch.Tensor,
    dim: int,
    pad_value: int,
    pad_scale: int = 1,
) -> torch.Tensor:
    """Pad a sequence tensor so its length is divisible by ``sp_size * pad_scale``."""
    sp_size = get_parallel_state().sp_size
    seq_length = tensor.size(dim)
    scale_sp_size = sp_size * pad_scale
    sp_chunk_size = (seq_length + scale_sp_size - 1) // scale_sp_size
    pad_size = sp_chunk_size * scale_sp_size - seq_length
    if pad_size == 0:
        return tensor
    pad_shape = list(tensor.shape)
    pad_shape[dim] = pad_size
    pad = torch.full(pad_shape, fill_value=pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, pad), dim=dim)


def _sp_slice_tensor(tensor: torch.Tensor, dim: int) -> torch.Tensor:
    """Extract this SP rank's slice from an already padded sequence tensor."""
    sp_size = get_parallel_state().sp_size
    sp_rank = get_parallel_state().sp_rank
    seq_length = tensor.size(dim)
    sp_chunk_size = (seq_length + sp_size - 1) // sp_size
    return tensor.narrow(dim, sp_rank * sp_chunk_size, sp_chunk_size)


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

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
    ConversationItem,
    is_dummy,
)
from ....module import InferModuleMixin, TrainModuleMixin
from .configuration import JanusLlamaConfig


class JanusLlama(TrainModuleMixin, InferModuleMixin, PreTrainedModel):
    """LLaMA backbone (no wte, no lm_head).

    Multi-modal inputs are already embedded by the sibling encoder modules
    (text wte / SigLIP / VQVAE) and live on the ``conversation_list`` items.
    :meth:`pre_forward` simply **concatenates** every non-dummy item's
    ``value`` in order into one packed bs=1 sequence — there is no
    ``masked_scatter`` and no placeholder-token mask.  ``inputs_embeds`` is
    required; this module never falls back to an ``embed_tokens`` lookup.
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
        # ctx and expands the KV cache to bs=2.
        self._cfg_active: bool = False
        self._past_key_values: Any = None
        # Separate unconditional KV cache for CFG (bs=1, run alongside the
        # conditional cache — see :meth:`generate`).
        self._uncond_past_key_values: Any = None

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

    def reset_local_inference_state(self) -> None:
        """Wipe per-request runtime state — CFG flag, KV cache, etc.

        Called by :class:`OmniInferencer` between requests so the next
        prompt starts fresh (no leftover ``_cfg_active`` from a previous
        T2I call carrying over into an I2T call).
        """
        self._cfg_active = False
        self._past_key_values = None
        self._uncond_past_key_values = None

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Auto-regressive backbone step driven by a conversation list.

        Prompt pass (internal KV empty): concatenate every embedded part
        (``text`` / ``image`` and any boundary ``output`` rows), prime the
        KV cache, and append an ``output`` item with backbone hidden states.

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
        assert tail_part.type == "output"

        # CFG
        cfg_uncond_inputs_embeds = tail_part.meta.pop("cfg_uncond_inputs_embeds", None)
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
            self._uncond_past_key_values = uncond_out["past_key_values"]
            self._cfg_active = True
        elif tail_part.meta.get("collapse_cfg", False):
            self._uncond_past_key_values = None
            self._cfg_active = False

        inputs_embeds: torch.Tensor = tail_part.value[-1:].to(self.device)  # 1, dim
        inputs_embeds = inputs_embeds.unsqueeze(0)  # 1, 1, dim

        if self._cfg_active:
            # CFG is a bs=2 (cond / uncond) decode step.  We deliberately run the
            # two branches as two **separate bs=1 forwards** rather than one bs=2
            # forward: VeOmni's fused rotary / attention kernels are only
            # validated for the bs=1 packed-sequence path and return wrong hidden
            # states for bs>1 decoding against a ``DynamicCache`` (verified — the
            # unconditional row diverges from the HF reference; running each
            # branch at bs=1 matches HF token-for-token).  ``cond`` reads the
            # conditional cache, ``uncond`` the unconditional one; both advance
            # independently, then we stack ``(cond, uncond)`` on the batch dim so
            # downstream CFG mixing sees the expected ``(2, 1, D)`` layout.
            cond_out = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._past_key_values,
                use_cache=True,
            )
            uncond_out = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._uncond_past_key_values,
                use_cache=True,
            )
            self._past_key_values = cond_out["past_key_values"]
            self._uncond_past_key_values = uncond_out["past_key_values"]
            hidden_states = torch.cat([cond_out["hidden_states"], uncond_out["hidden_states"]], dim=0)  # 2, 1, dim
        else:
            outputs = self.forward(
                inputs_embeds=inputs_embeds,
                attention_mask=None,
                past_key_values=self._past_key_values,
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
        return {"conversation_list": conversation_list}

    @staticmethod
    def _tail_hidden_from_forward(hidden_states: torch.Tensor) -> torch.Tensor:
        """Last-position backbone hidden for an ``output`` row (keeps CFG batch)."""
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
            return hidden_states[-1:].contiguous()
        return hidden_states[:, -1:, :].contiguous()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _pack_conversations_for_forward(
        self,
        conversations: list[list[ConversationItem]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Pack embedded conversations → ``(inputs_embeds, attention_mask, position_ids, pack_shape)``."""
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

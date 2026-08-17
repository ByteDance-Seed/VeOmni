"""Janus LLaMA backbone (no wte / lm_head).

``JanusLlama(InferenceMixin, OmniPreTrainedModel)`` — patched
``LlamaModel`` with ``embed_tokens = Identity``; ``inputs_embeds`` come from
the text-encoder node.  Packing / FSDP dummy anchors live in ``accelerated.py``.

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
``forward`` is SP-unaware — it computes on whatever (already-sliced) sequence
it is handed. SP is driven OUTSIDE the model (classic single-pass Ulysses): when
``sp_size > 1`` the accelerated ``TrainingMixin``'s ``forward_pre`` hook slices
the (replicated) packed sample to this rank's ``1/sp_size`` shard (rebuilding
the varlen ``cu_seqlens``), ``forward`` runs once (its attention all-to-alls
over the SP group internally), and the ``forward_post`` hook all-gathers the
shards back to the full sequence on every rank (both in ``accelerated.py``,
each gated on ``sp_size > 1``). The rest of ``forward_post`` (the unpack /
scatter) then runs SP-agnostically on the full data.

Connection outputs
------------------
``hidden_states``  — final LLaMA hidden states ``(B, T, D)``.  CE / sampling
                     live in the head modules.
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaModel
from veomni.utils.seqlen_pos_transform_utils import prepare_fa_kwargs_from_position_ids

from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem
from ...base.llm_packing import pack_llm_conversations_for_forward
from .configuration import JanusLlamaConfig


class InferenceMixin:
    """FSM ``generate`` (with classifier-free guidance) — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`JanusLlama`'s bases: ``OmniPreTrainedModel`` ships no-op
    ``reset_local_inference_state`` / ``reset_global_inference_state`` defaults
    (kept as a safety net for modules that don't need real inference state),
    and MRO resolves left-to-right — put second, those no-ops would shadow the
    real implementations below.
    """

    def reset_local_inference_state(self) -> None:
        self._cfg_active = False
        self._uncond_past_key_values = None

    def reset_global_inference_state(self) -> None:
        self.reset_local_inference_state()
        self._past_key_values = None

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if self._past_key_values is None:
            inputs_embeds, attention_mask, position_ids, _ = pack_llm_conversations_for_forward(
                [conversation_list], self.device
            )
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
            return {"conversation_list": conversation_list}
        tail_part = conversation_list[-1]
        assert tail_part.type == "output"

        cfg_uncond_inputs_embeds = tail_part.meta.pop("cfg_uncond_inputs_embeds", None)
        if cfg_uncond_inputs_embeds is not None and not self._cfg_active:
            uncond = cfg_uncond_inputs_embeds.to(self.device)
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

        inputs_embeds: torch.Tensor = tail_part.value[-1:].to(self.device)
        inputs_embeds = inputs_embeds.unsqueeze(0)

        if self._cfg_active:
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
            hidden_states = torch.cat([cond_out["hidden_states"], uncond_out["hidden_states"]], dim=0)
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
        """Return the last-token hidden state as ``[B, 1, H]`` for VQVAE sampling."""
        if hidden_states.dim() == 3:
            return hidden_states[:, -1:, :].contiguous()
        if hidden_states.dim() == 2:
            return hidden_states.unsqueeze(1).contiguous()
        raise TypeError(f"Unexpected hidden_states shape: {tuple(hidden_states.shape)}")


class JanusLlama(InferenceMixin, OmniPreTrainedModel):
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
        self.config = config
        self.language_model = LlamaModel._from_config(self.config.text_config)
        # Drop the embed_tokens parameters — owned by sibling TextEncoder.
        self.language_model.set_input_embeddings(nn.Identity())

        self._cfg_active: bool = False
        self._past_key_values: Any = None
        self._uncond_past_key_values: Any = None
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
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


__all__ = ["InferenceMixin", "JanusLlama"]

"""Janus LLaMA backbone (no wte / lm_head).

``JanusLlama(JanusLlamaModuleMixin, PreTrainedModel)`` — patched
``LlamaModel`` with ``embed_tokens = Identity``; ``inputs_embeds`` come from
the text-encoder node.  Packing / FSDP dummy anchors live in ``modulemixin.py``.

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

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from veomni.models.transformers.llama.generated.patched_modeling_llama_gpu import LlamaModel

from .configuration import JanusLlamaConfig
from .modulemixin import JanusLlamaModuleMixin, JanusLlamaTraceMixin


class JanusLlama(JanusLlamaModuleMixin, JanusLlamaTraceMixin, PreTrainedModel):
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

"""Qwen3 AR backbone (no wte / lm_head)."""

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from veomni.models.transformers.qwen3.generated.patched_modeling_qwen3_gpu import Qwen3Model

from .configuration import Qwen3LlmConfig
from .modulemixin import Qwen3LlmModuleMixin


class Qwen3Llm(Qwen3LlmModuleMixin, PreTrainedModel):
    """Qwen3 backbone (no wte, no lm_head).

    Multi-modal inputs are already embedded by the sibling text encoder and live
    on the ``conversation_list`` items.  :meth:`pre_forward` concatenates every
    non-dummy item's ``value`` in order into one packed bs=1 sequence.
    """

    config_class = Qwen3LlmConfig
    base_model_prefix = "qwen3_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3DecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3LlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3Model._from_config(self.config.text_config)
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

"""Qwen3-VL AR backbone (no wte / lm_head, with DeepStack injection)."""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from veomni.utils.device import IS_NPU_AVAILABLE

from .configuration import Qwen3VLLlmConfig
from .modulemixin import Qwen3VLLlmModuleMixin


if IS_NPU_AVAILABLE:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_npu import Qwen3VLTextModel
else:
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import Qwen3VLTextModel


class Qwen3VLLlm(Qwen3VLLlmModuleMixin, PreTrainedModel):
    """Qwen3-VL text backbone (no wte, no lm_head).

    Token / image embeds are produced by the sibling ``qwen3vl_text_encoder`` /
    ``qwen3vl_vision`` modules and live on ``conversation_list`` items.
    :meth:`pre_forward` concatenates them per sample, rebuilds M-RoPE position
    ids, and threads the per-layer DeepStack features into the text model.
    """

    config_class = Qwen3VLLlmConfig
    base_model_prefix = "qwen3vl_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["Qwen3VLTextDecoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3VLLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = Qwen3VLTextModel._from_config(self.config.text_config)
        self.language_model.set_input_embeddings(nn.Identity())
        self.post_init()

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[List[torch.Tensor]] = None,
        past_key_values=None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": outputs.past_key_values,
        }

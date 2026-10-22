# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLaDA2.0-Uni bidirectional Bailing/DeepSeek-MoE backbone (no wte / lm_head).

Directly composes the layers, norms, rotary embeddings, and fused MoE blocks of
:mod:`veomni.models.transformers.llada2_uni.modeling_llada2_uni` so the single-model
``DllmTrainer`` path and the ``SeedOmni`` ``OmniTrainer`` path share one implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from transformers import initialization as init
from transformers.modeling_outputs import MoeModelOutputWithPast

from veomni.models.transformers.llada.dllm_utils import bidirectional_attention_mask
from veomni.models.transformers.llada2_uni.configuration_llada2_uni import LLaDA2UniConfig
from veomni.models.transformers.llada2_uni.modeling_llada2_uni import (
    LLaDA2UniDecoderLayer,
    LLaDA2UniExperts,
    LLaDA2UniGate,
    LLaDA2UniPreTrainedModel,
    LLaDA2UniRMSNorm,
    LLaDA2UniRotaryEmbedding,
)
from veomni.models.transformers.llada_moe.checkpoint_tensor_converter import (
    LLaDAMoECheckpointTensorConverter,
    convert_llada_moe_fqn_to_index_mapping,
)

from ...module_modeling_base import PretrainedOmniModule
from .configuration import LladaUniLlmConfig


class LladaUniBackbone(LLaDA2UniPreTrainedModel):
    """Bidirectional Bailing/DeepSeek-MoE decoder stack without embeddings or vision tower."""

    _no_split_modules = ["LLaDA2UniDecoderLayer"]

    def __init__(self, config: LLaDA2UniConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LLaDA2UniDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LLaDA2UniRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2UniRotaryEmbedding(config)
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> MoeModelOutputWithPast:
        if inputs_embeds is None:
            raise ValueError("LladaUniBackbone requires `inputs_embeds`.")
        bsz, seq_len = inputs_embeds.shape[:2]
        device = inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=device).unsqueeze(0).expand(bsz, -1)

        if attention_mask is not None and attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                min_val = torch.finfo(inputs_embeds.dtype).min
                mask = torch.zeros(attention_mask.shape, dtype=inputs_embeds.dtype, device=device)
                mask = mask.masked_fill(~attention_mask.to(device), min_val)
            else:
                mask = attention_mask.to(dtype=inputs_embeds.dtype, device=device)
        else:
            mask = bidirectional_attention_mask(self.config, inputs_embeds, attention_mask, position_ids)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids=position_ids)
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return MoeModelOutputWithPast(last_hidden_state=self.norm(hidden_states))


class InferenceMixin:
    """FSM ``generate`` — runs one block-diffusion denoising forward with an explicit 4D mask."""

    def generate(
        self,
        llada_inputs_embeds: Optional[torch.Tensor] = None,
        llada_attention_mask: Optional[torch.Tensor] = None,
        llada_position_ids: Optional[torch.Tensor] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del generation_kwargs, kwargs
        if llada_inputs_embeds is None:
            raise RuntimeError("llada2_uni_llm.generate expects `llada_inputs_embeds` in the FSM ctx.")

        embeds = llada_inputs_embeds.to(self.device)
        inputs_embeds = embeds.unsqueeze(0) if embeds.dim() == 2 else embeds
        bsz, seq_len = inputs_embeds.shape[:2]
        if llada_position_ids is not None:
            position_ids = llada_position_ids.to(self.device)
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=self.device).unsqueeze(0).expand(bsz, -1)

        attn = llada_attention_mask
        if attn is not None:
            attn = attn.to(self.device)
            if attn.dim() == 2:
                attn = attn.unsqueeze(0).unsqueeze(0)
            elif attn.dim() == 3:
                attn = attn.unsqueeze(1)

        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attn,
            position_ids=position_ids,
            use_cache=False,
        )
        hidden_states = outputs["hidden_states"]
        if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
            hidden_states = hidden_states.squeeze(0)
        return {"llada_hidden": hidden_states}


class LladaUniLlm(InferenceMixin, PretrainedOmniModule):
    """LLaDA2.0-Uni bidirectional MoE backbone (no wte, no lm_head)."""

    config_class = LladaUniLlmConfig
    base_model_prefix = "llada2_uni_llm"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["LLaDA2UniDecoderLayer"]
    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_attention_backend = True

    def __init__(self, config: LladaUniLlmConfig):
        super().__init__(config)
        self.config = config
        self.language_model = LladaUniBackbone._from_config(
            self.config.text_config,
            attn_implementation=getattr(config, "_attn_implementation", None),
        )
        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        std = self.config.text_config.initializer_range
        if isinstance(module, LLaDA2UniExperts):
            init.normal_(module.gate_up_proj, mean=0.0, std=std)
            init.normal_(module.down_proj, mean=0.0, std=std)
        elif isinstance(module, LLaDA2UniGate):
            init.normal_(module.weight, mean=0.0, std=std)
            init.zeros_(module.expert_bias)

    def freeze_model(self) -> None:
        if getattr(self.config, "freeze", False):
            self.language_model.requires_grad_(False)

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del use_cache
        outputs = self.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        return {
            "hidden_states": outputs.last_hidden_state,
            "past_key_values": None,
        }


def _create_llada2_uni_llm_checkpoint_tensor_converter(model: LladaUniLlm):
    return LLaDAMoECheckpointTensorConverter(num_experts=model.config.text_config.num_experts)


for _cls in (LladaUniLlm, LladaUniBackbone):
    _cls._create_checkpoint_tensor_converter = staticmethod(_create_llada2_uni_llm_checkpoint_tensor_converter)
    _cls._convert_fqn_to_index_mapping = staticmethod(convert_llada_moe_fqn_to_index_mapping)


__all__ = ["InferenceMixin", "LladaUniBackbone", "LladaUniLlm"]

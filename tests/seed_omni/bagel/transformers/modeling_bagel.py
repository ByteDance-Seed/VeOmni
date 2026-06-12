"""Minimal self-contained BAGEL transformers reference model for tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from tests.seed_omni.bagel.transformers.configuration_bagel import BagelReferenceConfig


class BagelReferenceForCausalLM(PreTrainedModel):
    config_class = BagelReferenceConfig
    base_model_prefix = "bagel"
    main_input_name = "input_ids"

    def __init__(self, config: BagelReferenceConfig) -> None:
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(config.hidden_size, config.intermediate_size),
                    nn.GELU(),
                    nn.Linear(config.intermediate_size, config.hidden_size),
                    nn.LayerNorm(config.hidden_size),
                )
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        **kwargs,
    ) -> CausalLMOutputWithPast | tuple[torch.Tensor, ...]:
        del kwargs
        if input_ids is None:
            raise ValueError("BagelReferenceForCausalLM.forward requires input_ids.")

        hidden = self.embed_tokens(input_ids)
        hidden_states: list[torch.Tensor] = [hidden]
        for layer in self.layers:
            hidden = hidden + layer(hidden)
            hidden_states.append(hidden)
        logits = self.lm_head(hidden)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)

        use_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        emit_hidden = self.config.output_hidden_states if output_hidden_states is None else output_hidden_states
        if not use_return_dict:
            values: tuple[torch.Tensor, ...] = (logits,)
            if loss is not None:
                values = (loss,) + values
            if emit_hidden:
                values = values + (tuple(hidden_states),)
            return values

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            hidden_states=tuple(hidden_states) if emit_hidden else None,
        )

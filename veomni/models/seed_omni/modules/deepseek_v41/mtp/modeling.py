# Copyright 2026 Bytedance Ltd. and/or its affiliates
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

"""Standalone DeepSeek-V4.1 DSpark / MTP OmniModule (``deepseek_v41_mtp``).

Holds the three DSpark draft stages (``model.mtp.{0,1,2}``) as an independent
node in the training and generation graphs so that:

* runs that do not train or serve DSpark omit the node completely instead of
  loading and freezing ~14 B of weights inside ``deepseek_v41_llm``;
* ``deepseek_v41_mtp`` can carry its own EP size (128 routed experts vs the
  backbone's 384), optimizer / LR, freeze policy, and FSDP2 stack.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...module_modeling_base import PretrainedOmniModule
from ..llm.attention import DeepseekV41RotaryEmbedding
from ..llm.dspark import build_dspark_blocks
from ..llm.mhc import DeepseekV41HyperConnection, expand_to_streams, identity_pre_mix
from ..llm.moe import DeepseekV41TopKRouter
from ..llm.reuse import DeepseekV4Experts
from ..llm.sequence_shard import SequenceShard
from .configuration import DeepseekV41MtpConfig


class DeepseekV41MtpBackbone(nn.Module):
    """The DSpark stage stack (``model.mtp.*``) plus its rotary frequency buffer."""

    def __init__(self, config: DeepseekV41MtpConfig) -> None:
        super().__init__()
        self.config = config
        self.mtp = build_dspark_blocks(config)
        self.rotary_emb = DeepseekV41RotaryEmbedding(config)

    def project_main(self, draft_hiddens: list[torch.Tensor]) -> torch.Tensor:
        if not self.mtp:
            raise RuntimeError("project_main was called but this module has no DSpark stages")
        return self.mtp[0].project_main(draft_hiddens)

    def forward_train(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        draft_hiddens: list[torch.Tensor],
        shard: SequenceShard | None = None,
        input_ids: torch.Tensor | None = None,
        image_mask: torch.Tensor | None = None,
        factored_markov: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Teacher-forced DSpark on the packed row."""
        if not self.mtp:
            raise RuntimeError("forward_train was called but this module has no DSpark stages")
        if len(draft_hiddens) != len(self.config.dspark_target_layer_ids):
            raise ValueError(
                f"DSpark expected {len(self.config.dspark_target_layer_ids)} target-layer hiddens, "
                f"got {len(draft_hiddens)}"
            )
        hidden_streams = expand_to_streams(inputs_embeds, self.config.hc_mult)
        pre_mix = identity_pre_mix(hidden_streams, self.config.hc_mult)
        if input_ids is not None and shard is not None:
            input_ids = shard.local_rows(F.pad(input_ids, (0, shard.full_len - input_ids.shape[1])))
        main: Any = draft_hiddens
        draft = markov_bias = None
        last = len(self.mtp) - 1
        for stage_id, stage in enumerate(self.mtp):
            stage.mlp._recompute_routing_cache = None
            stage.mlp._enable_recompute_cache = bool(getattr(stage, "gradient_checkpointing", False) and self.training)
            hidden_streams, pre_mix, main, draft, markov_bias = stage(
                hidden_streams,
                pre_mix,
                position_ids,
                self.rotary_emb,
                main,
                shard,
                input_ids if stage_id == last else None,
                image_mask,
                factored_markov if stage_id == last else False,
            )
        return draft, markov_bias


class DeepseekV41Mtp(PretrainedOmniModule):
    """``deepseek_v41_mtp`` -- the DSpark draft stages as a standalone OmniModule."""

    config_class = DeepseekV41MtpConfig
    supports_sp_sharded_handoff = True
    base_model_prefix = "deepseek_v41_mtp"
    main_input_name = "inputs_embeds"
    _no_split_modules = ["DeepseekV41DSparkBlock"]
    supports_gradient_checkpointing = True

    def _check_and_adjust_attn_implementation(self, attn_implementation, **kwargs):
        return attn_implementation

    @staticmethod
    def _create_checkpoint_tensor_converter(model: DeepseekV41Mtp):
        from ..checkpoint_tensor_converter import create_deepseek_v41_mtp_checkpoint_tensor_converter

        return create_deepseek_v41_mtp_checkpoint_tensor_converter(model)

    def __init__(self, config: DeepseekV41MtpConfig) -> None:
        super().__init__(config)
        self.config = config
        self.model = DeepseekV41MtpBackbone(config)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        if getattr(module, "is_quantized", False):
            return
        super()._init_weights(module)
        std = self.config.initializer_range
        if isinstance(module, DeepseekV41HyperConnection):
            nn.init.normal_(module.fn, mean=0.0, std=std)
            nn.init.zeros_(module.base)
            nn.init.ones_(module.scale)
        elif isinstance(module, DeepseekV41TopKRouter):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, DeepseekV4Experts) and not getattr(module, "experts_are_quantized", False):
            nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
            nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def freeze_model(self) -> None:
        if self.config.freeze:
            self.model.mtp.requires_grad_(False)

    @property
    def dspark_trains(self) -> bool:
        """Whether ``forward`` runs the teacher-forced DSpark stages this step."""
        return self.training and bool(self.model.mtp) and any(p.requires_grad for p in self.model.mtp.parameters())

    def forward(  # type: ignore[override]
        self,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        draft_hiddens: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
        sequence_shard: SequenceShard | None = None,
        dspark_input_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if not self.dspark_trains or not draft_hiddens:
            return {}
        if inputs_embeds is None:
            raise ValueError("deepseek_v41_mtp requires inputs_embeds from deepseek_v41_llm.")
        if position_ids is None:
            raise ValueError("deepseek_v41_mtp requires position_ids from deepseek_v41_llm.")
        factored_markov = bool(getattr(self, "_use_factored_dspark_markov", False))
        draft, markov_bias = self.model.forward_train(
            inputs_embeds,
            position_ids,
            draft_hiddens,
            sequence_shard,
            dspark_input_ids,
            image_mask=image_mask,
            factored_markov=factored_markov,
        )
        outputs: dict[str, Any] = {"dspark_hidden_states": draft}
        if markov_bias is not None:
            outputs["dspark_markov_bias"] = markov_bias
            markov_hw = getattr(self.model.mtp[-1], "_last_markov_head_weight", None)
            if markov_hw is not None:
                outputs["dspark_markov_head_weight"] = markov_hw
        return outputs


__all__ = [
    "DeepseekV41Mtp",
    "DeepseekV41MtpBackbone",
]

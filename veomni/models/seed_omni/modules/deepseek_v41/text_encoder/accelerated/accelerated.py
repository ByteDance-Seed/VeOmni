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

"""Training-graph hooks for the DeepSeek-V4.1 text encoder.

Everything about the encode / decode plumbing -- the SP slice and gather around
``embed_tokens``, the fused linear+CE head, the vocab-parallel lookup -- is the
shared base's and is inherited unchanged. What is V4.1's here is one thing: the
chat template the tokenizer setter binds, which has to be built with this
module's ``thinking_mode`` / ``reasoning_effort`` rather than the base's
marker-only default.
"""

from typing import Any, Dict, Optional

import torch

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....base.text_encoder.accelerated import TrainingMixin as BaseTrainingMixin
from ....base.text_encoder.accelerated import VeOmniMixin as BaseVeOmniMixin
from ..chat_template import DeepseekV41ChatTemplate
from ..configuration import DeepseekV41TextEncoderConfig
from ..modeling import DeepseekV41TextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: DeepseekV41TextEncoderConfig
    device: torch.device
    _chat_template: DeepseekV41ChatTemplate

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The shared batch still holds ngram_embeds from the sibling node.
        del kwargs
        return super().encode_pre(conversation_list)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        return super().decode_pre(conversation_list)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    _chat_template: DeepseekV41ChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = DeepseekV41ChatTemplate(
            tokenizer,
            thinking_mode=self.config.thinking_mode,
            reasoning_effort=self.config.reasoning_effort,
        )


class DeepseekV41TextEncoderAccelerated(VeOmniMixin, DeepseekV41TextEncoder):
    pass


__all__ = ["DeepseekV41TextEncoderAccelerated"]

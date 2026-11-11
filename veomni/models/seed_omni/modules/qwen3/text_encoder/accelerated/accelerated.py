"""VeOmni-accelerated Qwen3TextEncoder — training graph hooks.

FSM ``generate`` lives natively on :class:`~.modeling.Qwen3TextEncoder`;
this file owns the SP-aware training pre/forward/post hooks and ChatML
template binding.
"""

from typing import Any, Dict, Optional

import torch

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem
from ....text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ....text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ..chat_template import Qwen3ChatTemplate
from ..configuration import Qwen3TextEncoderConfig
from ..modeling import Qwen3TextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3TextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _tokenizer: Any

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """Qwen3 ChatML text encoder — accelerated wrapper."""

    config: Qwen3TextEncoderConfig
    _chat_template: Qwen3ChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Qwen3ChatTemplate(tokenizer)


class Qwen3TextEncoderAccelerated(VeOmniMixin, Qwen3TextEncoder):
    pass


__all__ = ["Qwen3TextEncoderAccelerated"]

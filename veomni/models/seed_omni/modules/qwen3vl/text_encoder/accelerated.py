"""VeOmni-accelerated Qwen3VLTextEncoder — training graph hooks.

FSM ``generate`` now lives natively on :class:`~.modeling.Qwen3VLTextEncoder`;
this file only owns the SP-aware training pre/forward/post hooks.
"""

from typing import Any, Dict, Optional

import torch

from ....mixins.training_module_mixin import post_forward, pre_forward
from ....utils.conversation import ConversationItem
from ...base.text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ...base.text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from .chat_template import Qwen3VLChatTemplate
from .configuration import Qwen3VLTextEncoderConfig
from .modeling import Qwen3VLTextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3VLTextEncoderConfig
    device: torch.device
    _chat_template: Qwen3VLChatTemplate

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
    """Qwen3-VL ``TextEncoder`` accelerated wrapper — chat-template binding only.

    The encode/decode plumbing and ChatML ``generate`` FSM live on the native
    :class:`~.modeling.Qwen3VLTextEncoder`; only the chat template binding is
    accelerated-specific (tokenizer property setter).
    """

    _chat_template: Qwen3VLChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Qwen3VLChatTemplate(tokenizer)


class Qwen3VLTextEncoderAccelerated(VeOmniMixin, Qwen3VLTextEncoder):
    pass


__all__ = ["Qwen3VLTextEncoderAccelerated"]

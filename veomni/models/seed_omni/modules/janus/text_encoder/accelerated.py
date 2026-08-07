"""VeOmni-accelerated JanusTextEncoder — training graph hooks.

FSM ``generate`` (and the ``emit_image_start`` / ``emit_image_end`` / CFG
helpers) now live natively on :class:`~.modeling.JanusTextEncoder`; this file
only owns the SP-aware training pre/forward/post hooks.
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
from .chat_template import JanusChatTemplate
from .configuration import JanusTextEncoderConfig
from .modeling import JanusTextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: JanusTextEncoderConfig
    device: torch.device
    _chat_template: JanusChatTemplate

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
    """Janus ``TextEncoder`` accelerated wrapper — chat-template binding only.

    The encode/decode plumbing and the T2I-aware ``generate`` FSM (BOS
    injection, ``<boi>`` / ``<eoi>`` signals, classifier-free guidance arming)
    live on the native :class:`~.modeling.JanusTextEncoder`; only the chat
    template binding is accelerated-specific (tokenizer property setter).
    """

    _chat_template: JanusChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = JanusChatTemplate(tokenizer)


class JanusTextEncoderAccelerated(VeOmniMixin, JanusTextEncoder):
    pass


__all__ = ["JanusTextEncoderAccelerated"]

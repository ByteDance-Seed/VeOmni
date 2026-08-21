"""VeOmni-accelerated Qwen3TextEncoder — training graph hooks.

FSM ``generate`` now lives natively on :class:`~.modeling.Qwen3TextEncoder`;
this file only owns the SP-aware training pre/forward/post hooks plus the
image-mode vision-token freeze (both genuinely accelerated-only).
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
from ...qwen3vl.text_encoder.chat_template import Qwen3VLChatTemplate
from .chat_template import Qwen3ChatTemplate
from .configuration import Qwen3TextEncoderConfig
from .modeling import Qwen3TextEncoder


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3TextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _tokenizer: Any
    embed_tokens: torch.nn.Embedding
    _trainable_row_mask: Optional[torch.Tensor]

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._trainable_row_mask: Optional[torch.Tensor] = None

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

    def freeze_model(self) -> None:
        if not self._enable_image:
            return  # fully trainable (default text-only behaviour)
        # The user can't know the vision special-token ids, but the module can:
        # resolve them from its own tokenizer so only those rows stay trainable.
        ids = [int(self._tokenizer.convert_tokens_to_ids(tok)) for tok in self._VISION_SPECIAL_TOKENS]
        weight = self.embed_tokens.weight
        weight.requires_grad_(True)
        keep = torch.zeros(weight.shape[0], dtype=torch.bool)
        keep[ids] = True
        self._trainable_row_mask = keep

        def _mask_grad(grad: torch.Tensor) -> torch.Tensor:
            mask = self._trainable_row_mask.to(device=grad.device)
            return grad * mask.unsqueeze(1).to(grad.dtype)

        weight.register_hook(_mask_grad)


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """Qwen3 ChatML text encoder, optionally image-aware — accelerated wrapper.

    Only the chat-template selection and the image-mode freeze are genuinely
    accelerated-specific; the encode/decode plumbing and ChatML ``generate``
    FSM live on the native :class:`~.modeling.Qwen3TextEncoder`.
    """

    config: Qwen3TextEncoderConfig
    _chat_template: Qwen3ChatTemplate | Qwen3VLChatTemplate

    # Vision special tokens whose embedding rows bootstrap image understanding;
    # ids are resolved from the tokenizer at freeze time (see :meth:`freeze_model`).
    _VISION_SPECIAL_TOKENS = ("<|vision_start|>", "<|vision_end|>", "<|image_pad|>")

    @property
    def _enable_image(self) -> bool:
        return self.config.enable_image

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        # Only the template differs: image mode reuses the Qwen3-VL ChatML template
        # (adds the vision wrap tokens); otherwise the text-only Qwen3 ChatML.
        if self._enable_image:
            self._chat_template = Qwen3VLChatTemplate(tokenizer)
        else:
            self._chat_template = Qwen3ChatTemplate(tokenizer)


class Qwen3TextEncoderAccelerated(VeOmniMixin, Qwen3TextEncoder):
    pass


__all__ = ["Qwen3TextEncoderAccelerated"]

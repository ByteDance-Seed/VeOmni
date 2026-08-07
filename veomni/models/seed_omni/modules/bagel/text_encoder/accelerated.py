"""VeOmni-accelerated BagelTextEncoder — training graph hooks.

FSM ``generate`` (and ``encode_image_markers``) now live natively on
:class:`~.modeling.BagelTextEncoder`; this file only owns the training
pre/forward/post hooks (packed-token embedding + MoT CE loss).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedTokenizerBase

from ....mixins.training_module_mixin import post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from ...base.text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ...base.text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from .chat_template import BagelChatTemplate
from .configuration import BagelTextEncoderConfig
from .modeling import BagelTextEncoder, prepare_bagel_encode_input_ids


class TrainingMixin(BaseTrainingMixin):
    config: BagelTextEncoderConfig
    device: torch.device
    dtype: torch.dtype
    _chat_template: BagelChatTemplate | None
    _encode_batch_shape: torch.LongTensor | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._chat_template: Optional[BagelChatTemplate] = None

    def dummy_inputs(self, kind: str = "encode") -> Dict[str, torch.Tensor]:
        """Delegate to native impl — overrides ``TrainingModuleMixin.dummy_inputs``'s
        unrelated ``(*, batch_size, device, dtype)`` signature, which otherwise wins
        this name collision by sitting ahead of :class:`BagelTextEncoder` in MRO.
        """
        return BagelTextEncoder.dummy_inputs(self, kind=kind)

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().encode_post(**outputs)

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: Optional[List[List[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        return super().decode_pre(conversation_list, **kwargs)

    @post_forward("decode")
    def decode_post(self, **outputs: Any) -> Dict[str, Any]:
        return super().decode_post(**outputs)

    def _anchor_dummy_decode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
        dummy: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Tie dummy CE loss to MoT hidden states without changing its value."""
        if conversation_list is None:
            return dummy

        anchor = None
        for item in iter_desired_items(
            conversation_list, types=["text", "image", "output"], roles=["user", "assistant"]
        ):
            value = item.value
            if not torch.is_tensor(value):
                continue
            if value.dim() == 3 and value.shape[0] == 1:
                value = value.squeeze(0)
            if value.dim() == 2 and int(value.shape[-1]) == int(self.config.hidden_size):
                anchor = value.to(device=self.device, dtype=self.dtype).sum() * 0.0
                break
        if anchor is None:
            return dummy

        return {
            "hidden_states": dummy["hidden_states"] + anchor,
            "labels": dummy["labels"],
        }

    def _prepare_encode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
    ) -> torch.Tensor:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder._prepare_encode_inputs requires conversation_list.")

        input_ids, self._encode_batch_shape = prepare_bagel_encode_input_ids(
            conversation_list,
            device=self.device,
            fallback_input_ids=self.dummy_inputs(kind="encode")["input_ids"],
        )
        return input_ids

    def _prepare_decode_inputs(
        self,
        conversation_list: Optional[List[List[ConversationItem]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if conversation_list is None:
            raise ValueError("BagelTextEncoder._prepare_decode_inputs requires conversation_list.")

        hidden_parts: List[torch.Tensor] = []
        shift_label_parts: List[torch.Tensor] = []
        for item in iter_desired_items(conversation_list, types=["text"]):
            if is_dummy(item):
                continue

            hidden_states = item.value
            labels = item.meta["labels"]
            if hidden_states.dim() == 3 and hidden_states.size(0) == 1:
                hidden_states = hidden_states.squeeze(0)
            labels = labels.reshape(-1)
            if hidden_states.shape[0] != labels.shape[0]:
                raise ValueError(
                    "BAGEL text decode requires hidden-state and label lengths to match: "
                    f"got {hidden_states.shape[0]} and {labels.shape[0]}."
                )

            shift_labels = torch.full_like(labels, -100, dtype=torch.long)
            shift_labels[:-1] = labels[1:]
            hidden_parts.append(hidden_states.to(device=self.device, dtype=self.dtype))
            shift_label_parts.append(shift_labels)

        if hidden_parts:
            hidden_states = torch.cat(hidden_parts, dim=0)
            shift_labels = torch.cat(shift_label_parts, dim=0).to(device=hidden_states.device, non_blocking=True)
            if torch.any(shift_labels != -100):
                return hidden_states, shift_labels

        dummy = self._anchor_dummy_decode_inputs(conversation_list, self.dummy_inputs(kind="decode"))
        return dummy["hidden_states"], dummy["labels"]


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """BAGEL text-embedding accelerated wrapper — chat-template binding + training hooks.

    The T2I generation FSM (``generate`` / ``encode_image_markers``) lives on
    the native :class:`~.modeling.BagelTextEncoder`; only the chat template
    binding and CE-loss training hooks are accelerated-specific.
    """

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._chat_template = BagelChatTemplate(tokenizer)


class BagelTextEncoderAccelerated(VeOmniMixin, BagelTextEncoder):
    pass


__all__ = ["BagelTextEncoderAccelerated"]

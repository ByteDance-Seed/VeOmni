"""VeOmni-accelerated Qwen3OmniTextEncoder — training graph hooks.

The FSM ``generate`` lives natively on :class:`~.modeling.Qwen3OmniTextEncoder`;
this file only owns the SP-aware training pre/forward/post hooks and the
chat-template binding.
"""

from typing import Any, Dict, Optional

import torch

from .....mixins.training_module_mixin import post_forward, pre_forward
from .....utils.conversation import ConversationItem, is_backbone_input
from ....text_encoder.accelerated import (
    TrainingMixin as BaseTrainingMixin,
)
from ....text_encoder.accelerated import (
    VeOmniMixin as BaseVeOmniMixin,
)
from ..chat_template import Qwen3OmniChatTemplate
from ..configuration import Qwen3OmniTextEncoderConfig
from ..modeling import (
    TALKER_PROMPT,
    PromptSegment,
    Qwen3OmniTextEncoder,
    talker_prompt_segments,
    talker_supervised_mask,
)


class TrainingMixin(BaseTrainingMixin):
    config: Qwen3OmniTextEncoderConfig
    device: torch.device
    _chat_template: Qwen3OmniChatTemplate

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._talker_prompt_views: Optional[list[tuple[torch.Tensor, list[PromptSegment], torch.Tensor]]] = None

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # Captured here, not in ``encode_post``: the base scatters embeds over the
        # text rows, and it is the token ids underneath that the talker rebuilds
        # its 3D RoPE from.
        if self.config.publish_talker_prompt:
            self._talker_prompt_views = [
                (
                    self._chat_template.pack_talker_input_ids(sample),
                    talker_prompt_segments(sample),
                    talker_supervised_mask(sample),
                )
                for sample in conversation_list or []
            ]
        return super().encode_pre(conversation_list, **kwargs)

    @post_forward("encode")
    def encode_post(self, **outputs: Any) -> Dict[str, Any]:
        result = super().encode_post(**outputs)
        views = self._talker_prompt_views
        self._talker_prompt_views = None
        if views is not None:
            # Looked up here rather than in the pre-hook: ``embed_tokens`` is its
            # own FSDP2 unit (``_no_split_modules = ["Embedding"]``), and a nested
            # unit may not run forward before its root — which is what a pre-hook
            # lookup would do. Once per batch, not per sample, because under the
            # ``emb`` parallel group each lookup is a collective and a count that
            # varies with the batch size hangs.
            tts_embeds = self.tts_embeds()
            _stash_talker_prompts(
                result["conversation_list"],
                [
                    self.talker_prompt_payload(ids, segments, supervised, tts_embeds)
                    for ids, segments, supervised in views
                ],
            )
        return result


def _stash_talker_prompts(
    conversation_list: list[list[ConversationItem]],
    payloads: list[Dict[str, Any]],
) -> None:
    """Leave one payload per sample on the carrier, for the talker to find.

    On the sample's first packed row: the payload describes the whole sample, and
    the carrier has no sample-level slot, so a designated row stands in for one.
    The talker scans for the key rather than assuming which row, so this choice
    stays an implementation detail.
    """
    for sample, payload in zip(conversation_list, payloads, strict=True):
        host = next(
            (item for item in sample if is_backbone_input(item)),
            None,
        )
        if host is None:
            raise ValueError(
                "qwen3omni_text_encoder: a sample has no row in the thinker's sequence, so there is nowhere to "
                "leave the talker's prompt view."
            )
        host.meta[TALKER_PROMPT] = payload


class VeOmniMixin(TrainingMixin, BaseVeOmniMixin):
    """Qwen3-Omni ``TextEncoder`` accelerated wrapper — chat-template binding only."""

    _chat_template: Qwen3OmniChatTemplate

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, tokenizer: Any) -> None:
        self._tokenizer = tokenizer
        self._chat_template = Qwen3OmniChatTemplate(
            tokenizer, add_speech_system_prompt=self.config.add_speech_system_prompt
        )


class Qwen3OmniTextEncoderAccelerated(VeOmniMixin, Qwen3OmniTextEncoder):
    pass


__all__ = ["Qwen3OmniTextEncoderAccelerated", "TrainingMixin", "VeOmniMixin"]

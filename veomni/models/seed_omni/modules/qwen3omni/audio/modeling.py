"""Qwen3-Omni audio tower as a SeedOmni encoder module."""

from typing import Any, Dict, List, Optional

import torch

from veomni.models.transformers.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeAudioEncoder

from ....utils.conversation import ConversationItem
from ...module_modeling_base import PretrainedOmniModule
from .configuration import Qwen3OmniAudioEncoderConfig
from .processing import (
    _OMNI_MEL_LEN,
    UPSTREAM_CHUNK_FRAMES,
    Qwen3OmniAudioPreprocessor,
    audio_output_lengths,
)


def encode_qwen3omni_audio_items(
    module: "Qwen3OmniAudioEncoder",
    items: List[ConversationItem],
) -> None:
    """Run the tower over ``items`` and overwrite each ``value`` with its embeds.

    Items arrive holding ``(mel_len, num_mel_bins)`` features from the CPU
    preprocessor. The tower consumes one flat concatenation of every clip plus
    the per-clip lengths, so the split on the way out has to be derived rather
    than observed.
    """
    if not items:
        return
    mel_lengths = torch.tensor(
        [int(item.meta[_OMNI_MEL_LEN]) for item in items],
        dtype=torch.long,
        device=module.device,
    )
    input_features = torch.cat([item.value.to(device=module.device) for item in items], dim=0)
    embeds = module.forward(input_features=input_features, feature_lens=mel_lengths)["audio_embeds"]
    sizes = audio_output_lengths(mel_lengths, module.config.audio_config.n_window).tolist()
    for item, chunk in zip(items, torch.split(embeds, sizes, dim=0), strict=True):
        item.value = chunk


class InferenceMixin:
    """FSM ``generate`` — encodes standalone audio items in place.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3OmniAudioEncoder`'s bases so MRO resolves this concrete
    ``generate``. This module keeps no state across FSM steps.
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        items = [p for p in conversation_list if p.type == "audio" and p.role == "user"]
        encode_qwen3omni_audio_items(self, items)
        return {"conversation_list": conversation_list}


class Qwen3OmniAudioEncoder(InferenceMixin, PretrainedOmniModule):
    """Qwen3-Omni audio tower — waveform mel features to LLM-space embeds."""

    config_class = Qwen3OmniAudioEncoderConfig
    preprocessor_class = Qwen3OmniAudioPreprocessor
    base_model_prefix = "qwen3omni_audio"
    main_input_name = "input_features"
    # The tower's output rows are a plain per-clip concatenation, and each clip's
    # output length is derivable from its mel length, so ``forward_post`` can hand
    # the backbone this rank's window item by item.
    supports_sp_sharded_handoff = True
    _no_split_modules = ["Qwen3OmniMoeAudioEncoderLayer"]
    supports_gradient_checkpointing = True

    def __init__(self, config: Qwen3OmniAudioEncoderConfig):
        super().__init__(config)
        self.config = config
        chunk = config.audio_config.n_window * 2
        if chunk != UPSTREAM_CHUNK_FRAMES:
            raise ValueError(
                f"qwen3omni_audio: n_window={config.audio_config.n_window} gives a {chunk}-frame chunk, but the "
                f"upstream audio tower hardcodes {UPSTREAM_CHUNK_FRAMES} frames when building its attention "
                f"cu_seqlens. Only n_window={UPSTREAM_CHUNK_FRAMES // 2} (as shipped) is consistent; anything "
                "else crashes inside attention with a split_with_sizes mismatch."
            )
        self.audio_tower = Qwen3OmniMoeAudioEncoder._from_config(self.config.audio_config)
        self.post_init()

    def freeze_model(self) -> None:
        """Freeze the whole tower when ``config.freeze`` (talker training)."""
        if self.config.freeze:
            self.audio_tower.requires_grad_(False)

    @property
    def compute_dtype(self) -> torch.dtype:
        """The dtype the mel features must be in to meet the tower's first conv.

        Read off the weight itself, and read *here* rather than in the caller,
        because neither of the obvious answers is right under FSDP2 mixed
        precision. ``PreTrainedModel.dtype`` reports the first parameter it
        finds, which is an fp32 *shard* while the all-gather a forward body sees
        is bf16; and a pre-forward hook that read this weight would read it
        before the unshard, getting the same fp32 answer. By the time this
        property runs, ``forward`` has been entered through ``__call__``, so the
        unshard has happened and ``conv2d1`` is a real tensor.

        ``conv2d1`` speaks for the tower: only ``Qwen3OmniMoeAudioEncoderLayer``
        is in ``_no_split_modules``, so the conv stack shares this module's root
        unit and one precision policy.
        """
        return self.audio_tower.conv2d1.weight.dtype

    def forward(  # type: ignore[override]
        self,
        input_features: Optional[torch.FloatTensor] = None,
        feature_lens: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        del kwargs
        if input_features is not None:
            input_features = input_features.to(dtype=self.compute_dtype)
        outputs = self.audio_tower(input_features, feature_lens=feature_lens)
        return {"audio_embeds": outputs.last_hidden_state}


__all__ = [
    "Qwen3OmniAudioEncoder",
    "audio_output_lengths",
    "encode_qwen3omni_audio_items",
]

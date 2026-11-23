"""Training-graph hooks for the Qwen3-Omni audio tower."""

from typing import Any, Dict, List, Optional

import torch

from .....mixins.base_mixin import BaseMixin
from .....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from .....utils.conversation import ConversationItem, iter_desired_items
from ..configuration import Qwen3OmniAudioEncoderConfig
from ..modeling import Qwen3OmniAudioEncoder, audio_output_lengths
from ..processing import _OMNI_MEL_LEN


class TrainingMixin(TrainingModuleMixin):
    """Graph hooks for the audio tower.

    The tower takes one flat concatenation of every clip in the micro-batch plus
    the per-clip mel lengths, so ``forward_pre`` builds that and remembers the
    items in order while ``forward_post`` splits the flat output back. The split
    sizes are derived from :func:`audio_output_lengths`, not observed, because
    the tower returns a single unsegmented tensor.

    Dummy items exist for the same reason as in the vision tower: on a
    micro-batch with no real audio, the worker appends one zero-feature
    ``role="dummy"`` clip per sample so this FSDP unit still runs exactly one
    forward per step and produces a gradient anchor.
    """

    config: Qwen3OmniAudioEncoderConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._audio_items: Optional[List[ConversationItem]] = None
        self._conversation_carrier: Any = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        # The graph hands a node the whole batch; this hook owns only the carrier.
        # Swallowing the rest keeps framework metadata and sibling keys from
        # failing the call.
        del kwargs
        self._conversation_carrier = conversation_list
        # Dummy anchors carry ``is_dummy`` and keep the ``role="user"`` tag of the
        # items they stand in for, so one filter selects both.
        items = list(iter_desired_items(conversation_list, types=["audio"], roles=["user"]))
        self._audio_items = items
        if not items:
            return {}
        feature_lens = torch.tensor(
            [int(item.meta[_OMNI_MEL_LEN]) for item in items],
            dtype=torch.long,
            device=self.device,
        )
        # Device only: ``forward`` owns the dtype. A pre-hook runs before FSDP2
        # unshards, so ``self.dtype`` here is the fp32 master shard, not the
        # bf16 weight the conv will actually meet.
        input_features = torch.cat([item.value.to(device=self.device) for item in items], dim=0)
        return {"input_features": input_features, "feature_lens": feature_lens}

    @post_forward("forward")
    def forward_post(self, **outputs: Any) -> Dict[str, Any]:
        items = self._audio_items or []
        embeds = outputs.get("audio_embeds")
        if items and embeds is not None:
            mel_lengths = torch.tensor([int(item.meta[_OMNI_MEL_LEN]) for item in items], dtype=torch.long)
            sizes = audio_output_lengths(mel_lengths, self.config.audio_config.n_window).tolist()
            for item, chunk in zip(items, torch.split(embeds, sizes, dim=0), strict=True):
                item.value = chunk
        self._audio_items = None
        carrier = self._conversation_carrier
        self._conversation_carrier = None
        return {"conversation_list": carrier}

    def dummy_inputs(self, *, batch_size: int, device: torch.device, dtype: torch.dtype) -> Dict[str, Any]:
        """One full CNN block per sample — the shortest clip that exercises the
        conv stack at the only window size the tower supports."""
        frames = self.config.audio_config.n_window * 2
        return {
            "input_features": torch.zeros(
                batch_size * frames, self.config.audio_config.num_mel_bins, device=device, dtype=dtype
            ),
            "feature_lens": torch.full((batch_size,), frames, dtype=torch.long, device=device),
        }


class VeOmniMixin(BaseMixin, TrainingMixin):
    """``generate()`` already lives on the native :class:`Qwen3OmniAudioEncoder`;
    this composition only adds the training hooks and hook resolution."""


class Qwen3OmniAudioEncoderAccelerated(VeOmniMixin, Qwen3OmniAudioEncoder):
    pass


__all__ = ["Qwen3OmniAudioEncoderAccelerated", "TrainingMixin", "VeOmniMixin"]

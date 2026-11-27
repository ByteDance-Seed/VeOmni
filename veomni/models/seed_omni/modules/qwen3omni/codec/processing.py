"""CPU-side resampling of assistant speech to the codec's own rate."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ......data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from ....utils.conversation import ConversationItem
from ...module_processing_base import ModulePreprocessorBase
from .configuration import Qwen3OmniCodecConfig


# ``AUDIO_METADATA_KEY`` is the clip's timeline as the data layer stated it,
# moved with the payload here once the clip has been converted. One spelling for
# the whole graph: the audio tower reads it, ``encode_wav`` requires the rate,
# and ``OmniInferencer`` needs it to write a wav.


class Qwen3OmniCodecPreprocessor(ModulePreprocessorBase):
    """Bring ``role="assistant"`` audio to the codec's rate, on the worker.

    Why this exists at all: ``encode_wav`` *refuses* audio at any other rate
    rather than resampling, because a silent conversion there would stretch the
    RVQ timeline and misalign every target frame against the text. That refusal
    needs someone to satisfy it, and the data layer cannot — it loads a clip at
    whatever rate the file holds and has no business knowing that this codec
    runs at 24 kHz while the thinker's tower next door runs at 16 kHz. The rate
    is a property of this module, so the conversion belongs to this module.

    On the dataloader worker rather than in the GPU hook for the ordinary
    reason: it is per-clip CPU work with no parameters, and the workers are
    already there.

    Only the assistant's side. The same ``type="audio"`` row is an encoder input
    when the user speaks it and a *label* when the assistant does, and the
    sibling audio preprocessor claims the former (it filters ``roles=["user"]``),
    so the two never touch the same row.
    """

    def __init__(self, sampling_rate: int) -> None:
        self._sampling_rate = int(sampling_rate)

    @classmethod
    def from_pretrained(
        cls, module_path: str, *, config_overrides: dict[str, Any] | None = None, **kwargs: Any
    ) -> "Qwen3OmniCodecPreprocessor":
        """``config_overrides`` is the module's YAML ``model_config:`` block.

        Applied rather than dropped because the rate read below is derived from
        ``mimi_config``, so a block touching it would leave this worker
        resampling to a rate the live model does not run at.
        """
        del kwargs
        config = Qwen3OmniCodecConfig.from_pretrained(module_path, **(config_overrides or {}))
        return cls(config.sampling_rate)

    @property
    def sampling_rate(self) -> int:
        return self._sampling_rate

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        """Rewrite each assistant audio row's samples and declared rate in place.

        No dummy anchor, unlike the media towers: this module is frozen under
        every graph, so it has no gradient for an anchor to keep alive, and a
        step whose batch carries no speech simply gives the codec nothing to do.
        """
        del kwargs
        if inference:
            # Nothing to tokenise: the assistant's speech is what generation
            # produces, not something it is given.
            return
        for sample in conversation_list:
            for item in sample:
                if item.role == "assistant" and item.type == "audio":
                    waveform, metadata = self._to_codec_rate(item)
                    # Payload and timeline move together: ``value`` is now a
                    # 24 kHz tensor, so metadata still claiming the loaded rate
                    # would report a duration this row never had.
                    item.value, item.meta[AUDIO_METADATA_KEY] = waveform, metadata

    def _to_codec_rate(self, item: ConversationItem) -> tuple[torch.Tensor, AudioMetadata]:
        """Mono float32 samples at the codec's rate, plus the matching metadata.

        A torch tensor, not the numpy array the data layer loaded:
        ``encode_pre`` hands ``item.value`` to ``encode_wav`` verbatim, and its
        shape normaliser calls ``.dim()``. Same as the sibling tower, which
        leaves a tensor on the row too.
        """
        waveform = item.value
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            squeezed = waveform.squeeze()
            if squeezed.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__}: assistant audio has shape {tuple(waveform.shape)}; the codec takes "
                    f"mono samples. Downmix in the data layer — averaging channels here would guess at the intent."
                )
            waveform = squeezed

        metadata = item.meta.get(AUDIO_METADATA_KEY)
        rate = getattr(metadata, "sampling_rate", None)
        if rate is None:
            raise ValueError(
                f"{type(self).__name__}: an assistant audio item states no meta['{AUDIO_METADATA_KEY}'] rate. It "
                f"is the talker's target, so a wrong rate retunes every code it is scored against — the data "
                f"layer must state what it loaded (see veomni/data/seed_omni/utils/audio.py)."
            )
        rate = int(rate)
        if rate <= 0:
            raise ValueError(
                f"{type(self).__name__}: assistant audio states meta['{AUDIO_METADATA_KEY}'].sampling_rate={rate}."
            )
        if rate == self._sampling_rate:
            return torch.from_numpy(np.ascontiguousarray(waveform)), metadata

        # Imported here rather than at module scope only to keep the cost off
        # every import of this module; librosa is a hard dependency of all three
        # extras, so this cannot fail for a reachable install.
        import librosa

        resampled = librosa.resample(waveform, orig_sr=rate, target_sr=self._sampling_rate)
        # ``resampled_to`` is what refuses a conversion that retimed the clip.
        return (
            torch.from_numpy(np.ascontiguousarray(resampled)),
            metadata.resampled_to(self._sampling_rate, len(resampled)),
        )


__all__ = ["Qwen3OmniCodecPreprocessor"]

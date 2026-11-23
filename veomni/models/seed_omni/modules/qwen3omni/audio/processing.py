"""CPU-side waveform -> log-mel features for the Qwen3-Omni audio tower."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from ......data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY
from ....utils.conversation import ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from .configuration import Qwen3OmniAudioEncoderConfig


# Per-item mel length, stashed so the GPU forward can split the flat encoder
# output back per item without recomputing it from the feature tensor.
_OMNI_MEL_LEN = "audio_mel_len"


class Qwen3OmniAudioPreprocessor(ModulePreprocessorBase):
    """Worker-side mel extraction for standalone ``type="audio"`` items.

    Holds only the picklable ``WhisperFeatureExtractor`` — never the model. Runs
    on CPU and overwrites ``item.value`` with a ``(mel_len, num_mel_bins)``
    tensor, mirroring how the vision preprocessor leaves patches on ``value``.

    The encoder wants 16kHz; note that the codec *emits* 24kHz, so input and
    output sample rates differ across the graph — which is why an item may
    declare its own rate in ``meta["audio_metadata"]`` and gets resampled here.
    An item that omits the metadata is taken to be 16kHz already, the rate
    VeOmni's own loader (``audio_utils.load_audio``) resamples to.
    See :meth:`_to_encoder_rate` for why the rate decides TMRoPE alignment.
    """

    def __init__(
        self,
        feature_extractor: Any,
        dtype: torch.dtype | None = None,
        dummy_features: torch.Tensor | None = None,
    ) -> None:
        self._feature_extractor = feature_extractor
        self._dtype = dtype
        self._dummy_features = dummy_features

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> Qwen3OmniAudioPreprocessor:
        del kwargs
        return cls(WhisperFeatureExtractor.from_pretrained(module_path))

    @property
    def sampling_rate(self) -> int:
        return int(self._feature_extractor.sampling_rate)

    def bind_dummy_inputs(self, config: Qwen3OmniAudioEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP anchor. 100 mel frames is one full CNN block, the
        smallest length that exercises the length formula."""
        self._dtype = dtype
        self._dummy_features = torch.zeros(100, config.audio_config.num_mel_bins, dtype=dtype)

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs
        for sample in conversation_list:
            items = [it for it in iter_desired_items([sample], types=["audio"], roles=["user"]) if not it.is_dummy]
            if items:
                out = self._feature_extractor(
                    [self._checked_clip(item) for item in items],
                    sampling_rate=self.sampling_rate,
                    return_attention_mask=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                # WhisperFeatureExtractor pads every clip to 30s; the attention mask
                # is the only record of true length, so trim with it here rather
                # than shipping 3000 padded frames per clip to the GPU.
                lengths = out["attention_mask"].sum(dim=-1).tolist()
                for item, features, length in zip(items, out["input_features"], lengths, strict=True):
                    item.value = features[:, :length].transpose(0, 1).contiguous().to(dtype=self._dtype)
                    item.meta[_OMNI_MEL_LEN] = int(length)
                continue
            if inference:
                continue
            if self._dummy_features is None:
                raise RuntimeError(
                    f"{type(self).__name__}: dummy inputs not bound — call bind_dummy_inputs() "
                    "before training use (pure inference never reaches this branch)."
                )
            # Per sample: this module's forward still runs on a step with no real
            # audio so its FSDP collectives stay in step with the other ranks.
            sample.append(
                ConversationItem(
                    type="audio",
                    value=self._dummy_features.clone(),
                    role="user",
                    is_dummy=True,
                    meta={_OMNI_MEL_LEN: int(self._dummy_features.size(0))},
                )
            )

    def _checked_clip(self, item: ConversationItem) -> np.ndarray:
        """Encoder-rate samples, refusing a clip the extractor would cut short.

        ``WhisperFeatureExtractor`` truncates to its 30 s window and reports
        nothing — the item just comes back claiming 30 s. In training that
        pairs a clipped encoder input with a text target covering the whole
        recording, which does not fail, it only puts a ceiling on quality; at
        inference it answers about part of what it was given. Chunking a long
        clip across windows is a real feature and not one this preprocessor
        can fake, so the honest response is to name the limit.
        """
        samples = self._to_encoder_rate(item)
        window = int(self._feature_extractor.n_samples)
        if samples.shape[0] > window:
            raise ValueError(
                f"{type(self).__name__}: clip is {samples.shape[0] / self.sampling_rate:.1f}s but the encoder's "
                f"window is {window / self.sampling_rate:.0f}s, and the feature extractor would silently truncate "
                f"it. Segment the recording into window-sized turns in the data layer."
            )
        return samples

    def _to_encoder_rate(self, item: ConversationItem) -> np.ndarray:
        """Mono float32 samples at the encoder's rate, resampling if the item declares another.

        The rate is what places a clip on TMRoPE's shared clock, so it is an
        alignment concern rather than an audio-quality one. The tower emits 13
        tokens per 100 mel frames and the backbone spends exactly one position
        id per audio token, while a video's temporal row advances
        ``second_per_grid * position_id_per_seconds``. Both come to 13 positions
        per second only while a mel frame really is 10ms, which holds at 16kHz
        alone: hand the extractor 22.05kHz unresampled and a 4s clip claims
        5.54s of that clock, drifting against a video track by 38%.

        ``WhisperFeatureExtractor`` never resamples — its ``sampling_rate``
        argument exists only to raise when the caller's rate disagrees with
        16kHz — so passing it our own rate would defeat the single guard it
        offers. Resampling has to happen before it, here.

        The item's declared timeline is left alone: this is a local copy for the
        tower, and the clip is the same number of seconds long either way.
        """
        waveform = item.value
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.detach().cpu().numpy()
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.ndim > 1:
            # A (1, N) or (N, 1) clip is unambiguous; anything else (stereo,
            # say) has no single duration to place on the clock, and flattening
            # it would invent one.
            squeezed = waveform.squeeze()
            if squeezed.ndim != 1:
                raise ValueError(
                    f"{type(self).__name__}: audio item has shape {tuple(waveform.shape)}; the tower takes mono "
                    f"samples. Downmix in the data layer — averaging channels here would guess at the intent."
                )
            waveform = squeezed

        metadata = item.meta.get(AUDIO_METADATA_KEY)
        rate = int(getattr(metadata, "sampling_rate", None) or self.sampling_rate)
        if rate == self.sampling_rate:
            return waveform
        if rate <= 0:
            raise ValueError(
                f"{type(self).__name__}: audio item states meta['{AUDIO_METADATA_KEY}'].sampling_rate={rate}, "
                f"which is not a rate."
            )
        import librosa  # optional dependency, only reachable for off-rate audio

        return librosa.resample(waveform, orig_sr=rate, target_sr=self.sampling_rate)


__all__ = ["Qwen3OmniAudioPreprocessor", "_OMNI_MEL_LEN"]

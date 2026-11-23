"""CPU-side waveform -> log-mel features for the Qwen3-Omni audio tower."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from transformers import WhisperFeatureExtractor

from ......data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY
from ....utils.conversation import _OMNI_TOKEN_LEN, ConversationItem, iter_desired_items
from ...module_processing_base import ModulePreprocessorBase
from .configuration import Qwen3OmniAudioEncoderConfig


# Per-item mel length, stashed so the GPU forward can split the flat encoder
# output back per item without recomputing it from the feature tensor.
_OMNI_MEL_LEN = "audio_mel_len"

# Upstream's ``_get_feat_extract_output_lengths`` hardcodes a 100-frame chunk in
# its ``(length // 100) * 13`` term, and the tower builds its attention
# ``cu_seqlens`` from that helper while deriving the actual hidden length from
# the real chunking. The two only agree when ``n_window * 2 == 100``. At any
# other window the tower crashes inside attention with an opaque
# ``split_with_sizes`` mismatch, so the module rejects it at construction.
UPSTREAM_CHUNK_FRAMES = 100


def _conv_stack_output_length(length: int) -> int:
    """Length after the tower's three stride-2, pad-1, kernel-3 convolutions."""
    for _ in range(3):
        length = (length - 1) // 2 + 1
    return length


def audio_output_lengths(mel_lengths: torch.Tensor, n_window: int) -> torch.Tensor:
    """Encoder output length per clip, given its mel length.

    The tower does not convolve a clip whole: it cuts it into ``n_window * 2``
    frame chunks, pads each, and keeps the unpadded part of each chunk's conv
    output. So the total is a sum over chunks, and a clip of 100 frames in two
    50-frame chunks does *not* give the same length as one 100-frame chunk.

    Upstream's ``_get_feat_extract_output_lengths`` hardcodes a chunk of 100
    (its ``(length // 100) * 13`` term), which is right for the shipped
    ``n_window=50`` and silently wrong for anything else. This computes it from
    the config instead. Getting it wrong desynchronises the sibling text
    encoder's placeholder count from the embed count, and the scatter then
    misaligns without raising.

    Lives beside the preprocessor rather than in ``modeling.py`` because the
    *worker* needs it: the encoder's output length is what this row contributes
    to the sample's token count, and ``dyn_bsz`` reads that before any model
    exists. ``modeling.py`` imports it from here for the reverse direction
    (splitting the tower's flat output back per item).
    """
    chunk = n_window * 2
    out = []
    for length in mel_lengths.tolist():
        full, remainder = divmod(int(length), chunk)
        total = full * _conv_stack_output_length(chunk)
        if remainder:
            total += _conv_stack_output_length(remainder)
        out.append(total)
    return torch.tensor(out, dtype=torch.long, device=mel_lengths.device)


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
        n_window: int = 50,
    ) -> None:
        # ``_processor`` is the attribute ``bind_module_assets`` copies onto the
        # live module and ``save_pretrained`` writes beside the weights, so the
        # extractor has to live there or a split module loses its mel config:
        # the loaded module would have nothing to save, and the exported one
        # nothing to reload. ``sampling_rate`` reads it off that same attribute.
        self._processor = feature_extractor
        self._dtype = dtype
        self._dummy_features = dummy_features
        # Only used to predict the tower's output row count for ``dyn_bsz``; the
        # tower itself reads the same value from its own checkpoint.
        self._n_window = int(n_window)

    @classmethod
    def from_pretrained(cls, module_path: str, **kwargs: Any) -> Qwen3OmniAudioPreprocessor:
        del kwargs
        return cls(WhisperFeatureExtractor.from_pretrained(module_path))

    @property
    def sampling_rate(self) -> int:
        return int(self._processor.sampling_rate)

    def bind_dummy_inputs(self, config: Qwen3OmniAudioEncoderConfig, dtype: torch.dtype | None = None) -> None:
        """Training-only FSDP anchor. 100 mel frames is one full CNN block, the
        smallest length that exercises the length formula."""
        self._dtype = dtype
        self._n_window = int(config.audio_config.n_window)
        self._dummy_features = torch.zeros(100, config.audio_config.num_mel_bins, dtype=dtype)

    def preprocess_conversations(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs
        for sample in conversation_list:
            items = [it for it in iter_desired_items([sample], types=["audio"], roles=["user"]) if not it.is_dummy]
            if items:
                out = self._processor(
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
                # The encoder's *output* row count is what this item contributes
                # to the backbone's sequence, so that is what ``dyn_bsz`` has to
                # count. Stamping the mel-frame count instead would over-report
                # by ~8x; leaving the key off drops the row from the count
                # entirely, which is what made an audio-heavy sample look short.
                token_lens = audio_output_lengths(torch.tensor(lengths, dtype=torch.long), self._n_window).tolist()
                for item, features, length, token_len in zip(
                    items, out["input_features"], lengths, token_lens, strict=True
                ):
                    item.value = features[:, :length].transpose(0, 1).contiguous().to(dtype=self._dtype)
                    item.meta[_OMNI_MEL_LEN] = int(length)
                    item.meta[_OMNI_TOKEN_LEN] = int(token_len)
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
            dummy_len = int(self._dummy_features.size(0))
            sample.append(
                ConversationItem(
                    type="audio",
                    value=self._dummy_features.clone(),
                    role="user",
                    is_dummy=True,
                    meta={
                        _OMNI_MEL_LEN: dummy_len,
                        # The anchor is a real forward for this tower, so it costs
                        # real rows: 13 per 100 mel frames. Leaving the count off
                        # made `dyn_bsz` bill a no-audio sample as zero audio tokens
                        # while the hook still processed the dummy.
                        _OMNI_TOKEN_LEN: int(audio_output_lengths(torch.tensor([dummy_len]), self._n_window)[0]),
                    },
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
        window = int(self._processor.n_samples)
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


__all__ = ["Qwen3OmniAudioPreprocessor", "UPSTREAM_CHUNK_FRAMES", "_OMNI_MEL_LEN", "audio_output_lengths"]

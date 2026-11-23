"""Qwen3-Omni audio codec as one SeedOmni module — waveform <-> RVQ codes.

Both directions live here because they define a single shared thing: what a
"code" means for this model. The halves come from different checkpoints, since
Qwen3-Omni released a decoder but not its audio tokenizer:

* ``decode_codes`` (codes -> waveform) runs Qwen's own ``code2wav`` vocoder.
  This is the inference path, and it is the right decoder because it is the one
  the talker was trained against — it produces the model's actual voice.
* ``encode_wav`` (waveform -> codes) runs ``kyutai/mimi``, whose index space the
  talker's codes turn out to share. This is what makes talker training possible
  without Qwen's unreleased tokenizer: targets can be built from raw audio.

**How far that sharing was verified**, because the naive check is misleading.
``scripts/seed_omni/qwen3omni_audio_check.py`` measures both directions on real
checkpoints. Figures below are from one run; the talker samples a different
utterance each time, so treat the gaps as the result, not the absolute values.

* Audio-level agreement is good. Feeding Mimi-encoded codes to Qwen's vocoder
  reproduces the source audio at 0.1235 log-STFT L1, on par with the 0.1256 for
  the reverse cross-check (talker codes through Mimi's decoder) and well under
  the 0.2344 control of decoding time-shuffled codes.
* Code-level agreement is **not** identity, and must not be expected to be.
  Re-encoding Qwen's decoded audio recovers 58.3% of the talker's semantic codes
  (group 0) and near-chance rates on the deep residual groups. The decisive
  control is that *Mimi against itself* is barely better (85.4% on group 0,
  12.7% on groups 3+): the loss is intrinsic RVQ residual sensitivity, not an
  encoder/decoder mismatch. So a code-level equality test would fail on a
  perfectly healthy codec — validate this codec, and any talker trained on it,
  on audio, never on exact code match beyond group 0.
"""

from typing import Any, Dict, List, Optional

import torch

# The patchgen-generated modeling deliberately excludes the talker and vocoder
# classes: VeOmni's training path forwards through the thinker only, and
# instantiating them would drag their layers into the FSDP _no_split_modules
# aggregation. They are inference-only here, so take them from upstream.
from transformers import MimiModel
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import Qwen3OmniMoeCode2Wav

from veomni.utils import logging

from ......data.seed_omni.utils.media_metadata import AUDIO_METADATA_KEY, AudioMetadata
from ....graphs.generation_graph import FSM_SIGNAL_KEY
from ....utils.conversation import ConversationItem, get_tail_output_item
from ...module_modeling_base import PretrainedOmniModule
from ..carrier import TALKER_CODES_KEY
from .configuration import Qwen3OmniCodecConfig
from .processing import Qwen3OmniCodecPreprocessor


logger = logging.get_logger(__name__)

SIGNAL_AUDIO_DONE = "audio_done"
# Where ``encode`` stashes an assistant clip's RVQ codes: on the audio item that
# produced them, not as a row of their own. A separate row would sit inside the
# conversation the chat template later walks, and the talker's targets are an
# annotation of that audio rather than another turn. (Generation is the mirror
# case — there the codes precede any waveform, so they ride an ``output`` row;
# see ``carrier.TALKER_CODES_KEY``.)
TALKER_CODE_LABELS = "qwen3omni_talker_labels"


class InferenceMixin:
    """FSM ``generate`` — turns the talker's RVQ codes into a waveform.

    Listed *before* :class:`~...module_modeling_base.PretrainedOmniModule` in
    :class:`Qwen3OmniCodec`'s bases so MRO resolves this concrete ``generate``.
    This module keeps no state across FSM steps: it runs once, at the end of the
    graph, over the complete code sequence.
    """

    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        codes = _take_codes(conversation_list)
        waveform = self.decode_codes(codes)
        # Stated in full, not just the rate: this clip was synthesised rather
        # than loaded, so this is the only place its timeline is known, and the
        # inferencer needs both halves to write a wav of the right length.
        meta = {
            AUDIO_METADATA_KEY: AudioMetadata(
                sampling_rate=self.config.sampling_rate, num_samples=int(waveform.numel())
            )
        }
        conversation_list.append(
            ConversationItem(
                type="audio",
                value=waveform,
                role="assistant",
                meta=meta,
            )
        )
        return {
            "conversation_list": conversation_list,
            FSM_SIGNAL_KEY: SIGNAL_AUDIO_DONE,
            # A ``{type, value}`` dict, not the bare waveform: ``OmniModel``
            # drops anything else on the floor, so a bare tensor would leave the
            # caller with a graph that ran to completion and produced nothing.
            "generated": {"type": "audio", "value": waveform, "meta": meta},
        }


def _take_codes(conversation_list: List[ConversationItem]) -> torch.Tensor:
    """Pull the talker's RVQ codes off the carrier.

    Selected by ``source`` rather than by type: the codes ride an ordinary
    ``type="output"`` row, so the thinker's own output rows sit alongside them
    and only the source separates the two.

    Raises rather than falling back to silence: an empty waveform coming out of
    a TTS graph looks like a model quality problem, not a wiring bug, and would
    cost far more to diagnose later.
    """
    item = get_tail_output_item(conversation_list, meta_keys=[TALKER_CODES_KEY])
    if item is None:
        raise ValueError(
            f"qwen3omni_codec: no output row stamped {TALKER_CODES_KEY!r} on the conversation — the talker "
            f"must run before the vocoder in the generation graph."
        )
    return item.value


class Qwen3OmniCodec(InferenceMixin, PretrainedOmniModule):
    """Qwen3-Omni audio codec — 24 kHz waveform <-> 16-group RVQ codes at 12.5 Hz.

    See the module docstring for which half comes from which checkpoint and for
    exactly how far the two were verified to share an index space.
    """

    config_class = Qwen3OmniCodecConfig
    base_model_prefix = "qwen3omni_codec"
    main_input_name = "codes"
    # Brings assistant speech to this codec's rate on the dataloader worker.
    # ``encode_wav`` refuses any other rate rather than resampling, and the data
    # layer has no business knowing what this rate is, so it lands here.
    preprocessor_class = Qwen3OmniCodecPreprocessor
    # Both halves' layer classes. Frozen does not mean unwrapped — FSDP2
    # shards a module whatever its ``requires_grad`` — so all four are live
    # under ``fsdp_mode: fsdp2``, which this module supports.
    #
    # ``MimiResidualVectorQuantizer`` is named here rather than inherited:
    # upstream's ``MimiModel._no_split_modules`` names the enclosing
    # ``MimiSplitResidualVectorQuantizer`` instead, and HF ``post_init`` unions
    # that in, so the effective targets nest. Wrapping must therefore run
    # bottom-up, which ``parallelize_model_fsdp2`` does.
    #
    # The parent unit would hold both children's parameters, so naming the
    # children is redundant for coverage today. It is kept because upstream's
    # list already changed under this module once (5.9 named the inner class,
    # 5.16 the outer), and the entry that survives a rename is the one this
    # module declares itself.
    #
    # One of them needs help to get there. ``MimiTransformerLayer`` is entered
    # through ``__call__`` (``self.encoder_transformer(...)``) and unshards
    # normally, but ``MimiResidualVectorQuantizer`` is only ever reached by
    # *method* (``_encode_frame`` -> ``quantizer.encode`` -> the semantic and
    # acoustic instances' ``encode``), where FSDP2 installs no hook. The
    # accelerated wrapper all-gathers the units under Mimi itself; see
    # ``accelerated.TrainingMixin._unshard_mimi_units``, which also explains why
    # it is one window rather than FSDP's per-method registration.
    _no_split_modules = [
        "Qwen3OmniMoeCode2WavTransformerLayer",
        "Qwen3OmniMoeCode2WavDecoderBlock",
        "MimiTransformerLayer",
        "MimiResidualVectorQuantizer",
    ]

    def __init__(self, config: Qwen3OmniCodecConfig):
        super().__init__(config)
        if not getattr(config, "encoder_from_checkpoint", True):
            # Only reachable when the codec is actually selected, which is what
            # makes this the right place: a thinker-only run never builds it, and
            # a run that does would otherwise train the talker against codes from
            # an encoder whose weights were never loaded.
            raise NotImplementedError(
                "qwen3omni_codec: this checkpoint was loaded from the upstream root, which ships only the "
                "decoder half of the audio codec. The encoder comes from kyutai/mimi, so split the checkpoint "
                "offline first: `python scripts/seed_omni/convert_model.py --extra mimi_path=kyutai/mimi`."
            )
        self.config = config
        self.code2wav = Qwen3OmniMoeCode2Wav._from_config(self.config.code2wav_config)
        # The full MimiModel, decoder included, even though only ``encode`` is
        # used here. Surgically dropping the decoder would save ~40% of a 384 MB
        # fp32 codec — negligible beside a 30B backbone — in exchange for
        # breaking the upstream state-dict contract, and it would remove the
        # independent decoder that made the code-space check possible.
        # Mimi resolves attention through its own legacy MIMI_ATTENTION_CLASSES
        # dict, which only knows eager/sdpa/flash_attention_2 — it raises a
        # KeyError on VeOmni's sequence-parallel variants. Pinning sdpa here
        # rather than letting the launcher's attn_implementation propagate costs
        # nothing: this encoder is frozen, tiny, and runs once per clip.
        self.config.mimi_config._attn_implementation = "sdpa"
        self.mimi = MimiModel._from_config(self.config.mimi_config)
        self._warned_encoder_dtype = False
        self.post_init()

    def freeze_model(self) -> None:
        """Always frozen, and not on a flag — a codec is never a trainee.

        Both halves are fixed by definition of the task: ``code2wav`` is the
        vocoder whose output is the thing being listened to, and the Mimi half
        defines what "a code" means for this model, so training it would move
        the talker's label space out from under it mid-run. With nothing
        trainable, ``ModuleRuntime`` also skips the optimizer and the
        checkpoint for this module.
        """
        self.requires_grad_(False)

    @torch.no_grad()
    def encode_wav(self, waveform: torch.Tensor, *, sampling_rate: int) -> torch.Tensor:
        """Encode a waveform to ``(num_quantizers, T)`` RVQ codes — the talker's targets.

        Accepts ``(N,)``, ``(1, N)`` or ``(1, 1, N)`` mono samples. Truncating
        Mimi's 32 quantizers to the decoder's 16 is exact rather than
        approximate: RVQ is sequential, so the first 16 codes of a deeper encode
        are identical to a 16-deep encode (asserted in the tests).

        ``sampling_rate`` is required rather than assumed. VeOmni's own audio
        loader (``veomni/data/multimodal/audio_utils.py``) defaults to 16 kHz
        while this codec runs at 24 kHz, so the most likely caller holds the
        wrong rate — and resampling silently here would stretch the code
        timeline and misalign every target frame.
        """
        expected_rate = self.config.sampling_rate
        if int(sampling_rate) != expected_rate:
            raise ValueError(
                f"qwen3omni_codec: got {sampling_rate} Hz audio but this codec operates at {expected_rate} Hz. "
                f"Resample before encoding."
            )
        if self.mimi.dtype != torch.float32 and not self._warned_encoder_dtype:
            # Measured on the real encoder: bf16 agrees with fp32 on only 68% of
            # codes (96.8% on the semantic group, 62.7% on groups 3+). For a
            # frozen 79M-param tokenizer that is free label noise in the talker's
            # targets, so it is worth saying out loud rather than absorbing.
            logger.warning(
                f"qwen3omni_codec: encoding targets with a {self.mimi.dtype} encoder. Against fp32 this changes "
                f"~32%% of code indices. Keep the codec in fp32 when generating talker targets."
            )
            self._warned_encoder_dtype = True
        waveform = _as_mono_batch(waveform).to(device=self.device, dtype=self.mimi.dtype)
        codes = self.mimi.encode(waveform, num_quantizers=self.config.num_quantizers).audio_codes
        return codes.squeeze(0).long()

    @torch.no_grad()
    def encode(
        self,
        waveforms: Optional[List[torch.Tensor]] = None,
        sampling_rates: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Training-graph endpoint: tokenize assistant speech into talker targets.

        One clip at a time rather than one padded batch. Padding would be
        encoded too — Mimi has no attention mask on this path — so a short clip
        batched beside a long one would end with codes for its own silence and
        train the talker to emit them. A frozen 79M encoder over a handful of
        clips is not where a step's time goes, so there is nothing to buy back.
        """
        waveforms = waveforms or []
        rates = sampling_rates or []
        return {"codes": [self.encode_wav(w, sampling_rate=r) for w, r in zip(waveforms, rates, strict=True)]}

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode ``(num_quantizers, T)`` or ``(B, num_quantizers, T)`` codes to a waveform.

        Uses the vocoder's streaming path (a sliding window with left context)
        rather than one whole-sequence pass — the vocoder's own transformer has
        ``sliding_window=72``, so a long sequence decoded in one go is both
        slower and not what upstream does.
        """
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        expected = self.config.num_quantizers
        if codes.size(1) != expected:
            raise ValueError(
                f"qwen3omni_codec: expected {expected} RVQ code groups on dim 1, got {codes.size(1)}. "
                f"Codes must be (num_quantizers, T) or (B, num_quantizers, T)."
            )
        codes = codes.to(self.device)
        return self.code2wav.chunked_decode(
            codes,
            chunk_size=self.config.chunk_size,
            left_context_size=self.config.left_context_size,
        ).float()

    def forward(  # type: ignore[override]
        self,
        codes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        del kwargs
        return {"waveform": self.decode_codes(codes)}


def _as_mono_batch(waveform: torch.Tensor) -> torch.Tensor:
    """Normalize a mono waveform to Mimi's ``(1, 1, num_samples)`` input shape."""
    if waveform.dim() == 1:
        return waveform.view(1, 1, -1)
    if waveform.dim() == 2:
        if waveform.size(0) != 1:
            raise ValueError(
                f"qwen3omni_codec: expected mono audio, got {waveform.size(0)} channels/items in {tuple(waveform.shape)}."
            )
        return waveform.unsqueeze(0)
    if waveform.dim() == 3:
        if waveform.shape[:2] != (1, 1):
            raise ValueError(
                f"qwen3omni_codec: expected a single mono clip shaped (1, 1, N), got {tuple(waveform.shape)}."
            )
        return waveform
    raise ValueError(f"qwen3omni_codec: cannot read a waveform from shape {tuple(waveform.shape)}.")


__all__ = ["Qwen3OmniCodec", "SIGNAL_AUDIO_DONE", "TALKER_CODE_LABELS"]

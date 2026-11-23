"""Config for :class:`Qwen3OmniCodec`."""

from typing import Any, Dict, Optional

from transformers import MimiConfig
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import Qwen3OmniMoeCode2WavConfig

from ...module_configuration_base import OmniModuleConfig


class Qwen3OmniCodecConfig(OmniModuleConfig):
    """Wraps the two halves of Qwen3-Omni's audio codec as one OmniModule config.

    The halves come from **different checkpoints**, which is unusual enough to
    spell out: Qwen3-Omni ships a decoder only (``code2wav``, 230 tensors, no
    quantizer), because its audio tokenizer was never released. The talker's
    codes turn out to live in Mimi's index space, so ``kyutai/mimi`` supplies
    the missing encoder. Both are frozen, so a single module owning both is the
    honest shape: one place decides what "a code" means for this model.

    ``chunk_size`` / ``left_context_size`` are the streaming-decode window
    upstream hardcodes at its single call site
    (``Qwen3OmniMoeForConditionalGeneration.generate``). They are decode-time
    knobs, not weights, so an infer YAML can override them here.
    """

    model_type = "qwen3omni_codec"
    # Registered so transformers treats the nested dicts as real configs (load /
    # save / to_dict round-trip). Registering them also means the launcher's
    # ``attn_implementation`` propagates into both halves, which is why
    # ``modeling.py`` pins the Mimi half back to SDPA — Mimi dispatches through
    # its own attention table and does not know VeOmni's variants.
    sub_configs = {"code2wav_config": Qwen3OmniMoeCode2WavConfig, "mimi_config": MimiConfig}

    def __init__(
        self,
        code2wav_config: Optional[Dict[str, Any]] = None,
        mimi_config: Optional[Dict[str, Any]] = None,
        chunk_size: int = 300,
        left_context_size: int = 25,
        **kwargs,
    ):
        self.code2wav_config = (
            Qwen3OmniMoeCode2WavConfig(**code2wav_config) if code2wav_config else Qwen3OmniMoeCode2WavConfig()
        )
        self.mimi_config = MimiConfig(**mimi_config) if mimi_config else MimiConfig()
        self.chunk_size = chunk_size
        self.left_context_size = left_context_size
        self._check_halves_agree()
        self._absorb_derived_kwargs(kwargs)
        # A codec has no wte / lm_head, so there is nothing to tie. Left True,
        # the shared post-load tie would look for an input embedding this module
        # does not have.
        kwargs["tie_word_embeddings"] = False
        super().__init__(**kwargs)

    def _check_halves_agree(self) -> None:
        """Reject an encoder/decoder pair that cannot exchange code indices or timing.

        Three invariants tie the halves together, and each is fatal rather than
        cosmetic: a code index produced against one codebook size is meaningless
        against another, an encoder shallower than the decoder cannot fill every
        code group, and the two must agree on how much audio one code frame
        spans or the output plays at the wrong speed.
        """
        encoder_codebook = self.mimi_config.codebook_size
        decoder_codebook = self.code2wav_config.codebook_size
        if encoder_codebook != decoder_codebook:
            raise ValueError(
                f"qwen3omni_codec: encoder and decoder disagree on codebook size "
                f"({encoder_codebook} vs {decoder_codebook}), so their code indices are not interchangeable."
            )
        if self.mimi_config.num_quantizers < self.code2wav_config.num_quantizers:
            raise ValueError(
                f"qwen3omni_codec: the encoder has only {self.mimi_config.num_quantizers} quantizers but the "
                f"decoder expects {self.code2wav_config.num_quantizers} code groups."
            )
        # The decoder never names a sampling rate, but it does imply one: its
        # upsampling stack fixes how many samples one code frame becomes. On the
        # real pair both sides say 1920 (24000 / 12.5 == 8*5*4*3 * 2*2). A
        # mismatch here is the nastiest failure mode available, because it does
        # not raise anywhere — it emits audio at the wrong speed, tagged with a
        # sampling rate that says otherwise.
        if self.decoder_samples_per_frame != self.encoder_samples_per_frame:
            raise ValueError(
                f"qwen3omni_codec: the halves disagree on how much audio one code frame spans — the decoder "
                f"upsamples a frame to {self.decoder_samples_per_frame} samples, but the encoder produces one "
                f"frame per {self.encoder_samples_per_frame} samples "
                f"({self.mimi_config.sampling_rate} Hz / {self.mimi_config.frame_rate} Hz). Decoded audio would "
                f"play at the wrong speed."
            )

    def _absorb_derived_kwargs(self, kwargs: Dict[str, Any]) -> None:
        """Validate and drop config keys that are now derived rather than stored.

        ``sampling_rate`` used to be a stored field on this config. It is a
        read-only property now, so leaving it in ``kwargs`` would make
        ``PretrainedConfig.__init__``'s blanket ``setattr`` raise an
        ``AttributeError`` naming the property — an error that describes the
        symptom and hides the cause. An older config that carries the key is
        accepted if it agrees with the derived value and rejected loudly if not.
        """
        for key in ("sampling_rate", "frame_rate", "num_quantizers"):
            if key not in kwargs:
                continue
            supplied = kwargs.pop(key)
            derived = getattr(self, key)
            if supplied != derived:
                raise ValueError(
                    f"qwen3omni_codec: config sets {key}={supplied}, but the encoder/decoder pair implies "
                    f"{derived}. This value is derived from the two halves — drop it from the config rather "
                    f"than overriding it."
                )

    @property
    def decoder_samples_per_frame(self) -> int:
        """Samples the vocoder emits per code frame, from its upsampling stack."""
        total = 1
        for rate in list(self.code2wav_config.upsample_rates) + list(self.code2wav_config.upsampling_ratios):
            total *= rate
        return total

    @property
    def encoder_samples_per_frame(self) -> float:
        """Samples the encoder consumes per code frame."""
        return self.mimi_config.sampling_rate / self.mimi_config.frame_rate

    @property
    def num_quantizers(self) -> int:
        """RVQ code groups, i.e. dim 1 of a code tensor.

        The decoder's value is authoritative: it is what the talker was trained
        to emit. Mimi is a 32-quantizer codec truncated to this depth, which is
        exact because RVQ is sequential — the first ``n`` codes of a deeper
        encode are the first ``n`` codes of a shallower one.
        """
        return self.code2wav_config.num_quantizers

    @property
    def sampling_rate(self) -> int:
        """Waveform rate on both sides of the codec (24 kHz).

        Read off Mimi rather than stored: the upstream ``code2wav_config`` in the
        Qwen checkpoint declares no ``sampling_rate`` at all, so a literal here
        would be an unchecked assumption sitting on the one path that turns
        codes into a ``.wav``. The encoder does declare it, so it is the source.
        """
        return self.mimi_config.sampling_rate

    @property
    def frame_rate(self) -> float:
        """Code frames per second (12.5 Hz), i.e. what one column of codes spans."""
        return self.mimi_config.frame_rate

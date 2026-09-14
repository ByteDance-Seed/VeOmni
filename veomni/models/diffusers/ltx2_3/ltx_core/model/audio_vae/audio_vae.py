from typing import Set, Tuple

import torch
import torch.nn.functional as F
from ltx_core.components.patchifiers import AudioPatchifier
from ltx_core.model.audio_vae.attention import AttentionType, make_attn
from ltx_core.model.audio_vae.causal_conv_2d import make_conv2d
from ltx_core.model.audio_vae.causality_axis import CausalityAxis
from ltx_core.model.audio_vae.downsample import build_downsampling_path
from ltx_core.model.audio_vae.ops import AudioProcessor, PerChannelStatistics
from ltx_core.model.audio_vae.resnet import ResnetBlock
from ltx_core.model.audio_vae.upsample import build_upsampling_path
from ltx_core.model.audio_vae.vocoder import Vocoder
from ltx_core.model.common.normalization import NormType, build_normalization_layer
from ltx_core.types import Audio, AudioLatentShape


LATENT_DOWNSAMPLE_FACTOR = 4


def build_mid_block(
    channels: int,
    temb_channels: int,
    dropout: float,
    norm_type: NormType,
    causality_axis: CausalityAxis,
    attn_type: AttentionType,
    add_attention: bool,
) -> torch.nn.Module:
    """Build the middle block with two ResNet blocks and optional attention."""
    mid = torch.nn.Module()
    mid.block_1 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    mid.attn_1 = (
        make_attn(channels, attn_type=attn_type, norm_type=norm_type) if add_attention else torch.nn.Identity()
    )
    mid.block_2 = ResnetBlock(
        in_channels=channels,
        out_channels=channels,
        temb_channels=temb_channels,
        dropout=dropout,
        norm_type=norm_type,
        causality_axis=causality_axis,
    )
    return mid


def run_mid_block(mid: torch.nn.Module, features: torch.Tensor) -> torch.Tensor:
    """Run features through the middle block."""
    features = mid.block_1(features, temb=None)
    features = mid.attn_1(features)
    return mid.block_2(features, temb=None)


class AudioEncoder(torch.nn.Module):
    """
    Encoder that compresses audio spectrograms into latent representations.
    The encoder uses a series of downsampling blocks with residual connections,
    attention mechanisms, and configurable causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        dropout: float = 0.0,
        resamp_with_conv: bool = True,
        in_channels: int,
        resolution: int,
        z_channels: int,
        double_z: bool = True,
        attn_type: AttentionType = AttentionType.VANILLA,
        mid_block_add_attention: bool = True,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        n_fft: int = 1024,
        is_causal: bool = True,
        mel_bins: int = 64,
        **_ignore_kwargs,
    ) -> None:
        super().__init__()

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.n_fft = n_fft
        self.is_causal = is_causal
        self.mel_bins = mel_bins

        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.z_channels = z_channels
        self.double_z = double_z
        self.norm_type = norm_type
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        self.conv_in = make_conv2d(
            in_channels,
            self.ch,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

        self.non_linearity = torch.nn.SiLU()

        self.down, block_in = build_downsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
        )

        self.mid = build_mid_block(
            channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )

        self.norm_out = build_normalization_layer(block_in, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            causality_axis=self.causality_axis,
        )

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Encode audio spectrogram into latent representations.
        Args:
            spectrogram: Input spectrogram of shape (batch, channels, time, frequency)
        Returns:
            Encoded latent representation of shape (batch, channels, frames, mel_bins)
        """
        h = self.conv_in(spectrogram)
        h = self._run_downsampling_path(h)
        h = run_mid_block(self.mid, h)
        h = self._finalize_output(h)

        return self._normalize_latents(h)

    def _run_downsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in range(self.num_resolutions):
            stage = self.down[level]
            for block_idx in range(self.num_res_blocks):
                h = stage.block[block_idx](h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != self.num_resolutions - 1:
                h = stage.downsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        h = self.norm_out(h)
        h = self.non_linearity(h)
        return self.conv_out(h)

    def _normalize_latents(self, latent_output: torch.Tensor) -> torch.Tensor:
        means = torch.chunk(latent_output, 2, dim=1)[0]
        latent_shape = AudioLatentShape(
            batch=means.shape[0],
            channels=means.shape[1],
            frames=means.shape[2],
            mel_bins=means.shape[3],
        )
        latent_patched = self.patchifier.patchify(means)
        latent_normalized = self.per_channel_statistics.normalize(latent_patched)
        return self.patchifier.unpatchify(latent_normalized, latent_shape)


def encode_audio(
    audio: Audio,
    audio_encoder: AudioEncoder,
    audio_processor: AudioProcessor | None = None,
) -> torch.Tensor:
    """Encode audio waveform into latent representation.
    Args:
        audio: Audio container with waveform tensor of shape (batch, channels, samples) and sampling rate.
        audio_encoder: Audio encoder model
        audio_processor: Audio processor model (optional, if not provided, it will be created from the audio encoder)
    """
    dtype = next(audio_encoder.parameters()).dtype
    device = next(audio_encoder.parameters()).device

    if audio_processor is None:
        audio_processor = AudioProcessor(
            target_sample_rate=audio_encoder.sample_rate,
            mel_bins=audio_encoder.mel_bins,
            mel_hop_length=audio_encoder.mel_hop_length,
            n_fft=audio_encoder.n_fft,
        ).to(device=device)

    mel_spectrogram = audio_processor.waveform_to_mel(audio.to(device=device))

    latent = audio_encoder(mel_spectrogram.to(dtype=dtype))
    return latent


class AudioDecoder(torch.nn.Module):
    """
    Symmetric decoder that reconstructs audio spectrograms from latent features.
    The decoder mirrors the encoder structure with configurable channel multipliers,
    attention resolutions, and causal convolutions.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        ch: int,
        out_ch: int,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int,
        attn_resolutions: Set[int],
        resolution: int,
        z_channels: int,
        norm_type: NormType = NormType.GROUP,
        causality_axis: CausalityAxis = CausalityAxis.WIDTH,
        dropout: float = 0.0,
        mid_block_add_attention: bool = True,
        sample_rate: int = 16000,
        mel_hop_length: int = 160,
        is_causal: bool = True,
        mel_bins: int | None = None,
    ) -> None:
        super().__init__()

        resamp_with_conv = True
        attn_type = AttentionType.VANILLA

        self.per_channel_statistics = PerChannelStatistics(latent_channels=ch)
        self.sample_rate = sample_rate
        self.mel_hop_length = mel_hop_length
        self.is_causal = is_causal
        self.mel_bins = mel_bins
        self.patchifier = AudioPatchifier(
            patch_size=1,
            audio_latent_downsample_factor=LATENT_DOWNSAMPLE_FACTOR,
            sample_rate=sample_rate,
            hop_length=mel_hop_length,
            is_causal=is_causal,
        )

        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.out_ch = out_ch
        self.give_pre_end = False
        self.tanh_out = False
        self.norm_type = norm_type
        self.z_channels = z_channels
        self.channel_multipliers = ch_mult
        self.attn_resolutions = attn_resolutions
        self.causality_axis = causality_axis
        self.attn_type = attn_type

        base_block_channels = ch * self.channel_multipliers[-1]
        base_resolution = resolution // (2 ** (self.num_resolutions - 1))
        self.z_shape = (1, z_channels, base_resolution, base_resolution)

        self.conv_in = make_conv2d(
            z_channels, base_block_channels, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )
        self.non_linearity = torch.nn.SiLU()
        self.mid = build_mid_block(
            channels=base_block_channels,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            add_attention=mid_block_add_attention,
        )
        self.up, final_block_channels = build_upsampling_path(
            ch=ch,
            ch_mult=ch_mult,
            num_resolutions=self.num_resolutions,
            num_res_blocks=num_res_blocks,
            resolution=resolution,
            temb_channels=self.temb_ch,
            dropout=dropout,
            norm_type=self.norm_type,
            causality_axis=self.causality_axis,
            attn_type=self.attn_type,
            attn_resolutions=attn_resolutions,
            resamp_with_conv=resamp_with_conv,
            initial_block_channels=base_block_channels,
        )

        self.norm_out = build_normalization_layer(final_block_channels, normtype=self.norm_type)
        self.conv_out = make_conv2d(
            final_block_channels, out_ch, kernel_size=3, stride=1, causality_axis=self.causality_axis
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """
        Decode latent features back to audio spectrograms.
        Args:
            sample: Encoded latent representation of shape (batch, channels, frames, mel_bins)
        Returns:
            Reconstructed audio spectrogram of shape (batch, channels, time, frequency)
        """
        sample, target_shape = self._denormalize_latents(sample)

        h = self.conv_in(sample)
        h = run_mid_block(self.mid, h)
        h = self._run_upsampling_path(h)
        h = self._finalize_output(h)

        return self._adjust_output_shape(h, target_shape)

    def _denormalize_latents(self, sample: torch.Tensor) -> tuple[torch.Tensor, AudioLatentShape]:
        latent_shape = AudioLatentShape(
            batch=sample.shape[0],
            channels=sample.shape[1],
            frames=sample.shape[2],
            mel_bins=sample.shape[3],
        )

        sample_patched = self.patchifier.patchify(sample)
        sample_denormalized = self.per_channel_statistics.un_normalize(sample_patched)
        sample = self.patchifier.unpatchify(sample_denormalized, latent_shape)

        target_frames = latent_shape.frames * LATENT_DOWNSAMPLE_FACTOR
        if self.causality_axis != CausalityAxis.NONE:
            target_frames = max(target_frames - (LATENT_DOWNSAMPLE_FACTOR - 1), 1)

        target_shape = AudioLatentShape(
            batch=latent_shape.batch,
            channels=self.out_ch,
            frames=target_frames,
            mel_bins=self.mel_bins if self.mel_bins is not None else latent_shape.mel_bins,
        )

        return sample, target_shape

    def _adjust_output_shape(
        self,
        decoded_output: torch.Tensor,
        target_shape: AudioLatentShape,
    ) -> torch.Tensor:
        _, _, current_time, current_freq = decoded_output.shape
        target_channels = target_shape.channels
        target_time = target_shape.frames
        target_freq = target_shape.mel_bins

        decoded_output = decoded_output[
            :, :target_channels, : min(current_time, target_time), : min(current_freq, target_freq)
        ]

        time_padding_needed = target_time - decoded_output.shape[2]
        freq_padding_needed = target_freq - decoded_output.shape[3]

        if time_padding_needed > 0 or freq_padding_needed > 0:
            padding = (
                0,
                max(freq_padding_needed, 0),
                0,
                max(time_padding_needed, 0),
            )
            decoded_output = F.pad(decoded_output, padding)

        decoded_output = decoded_output[:, :target_channels, :target_time, :target_freq]

        return decoded_output

    def _run_upsampling_path(self, h: torch.Tensor) -> torch.Tensor:
        for level in reversed(range(self.num_resolutions)):
            stage = self.up[level]
            for block_idx, block in enumerate(stage.block):
                h = block(h, temb=None)
                if stage.attn:
                    h = stage.attn[block_idx](h)

            if level != 0 and hasattr(stage, "upsample"):
                h = stage.upsample(h)

        return h

    def _finalize_output(self, h: torch.Tensor) -> torch.Tensor:
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.non_linearity(h)
        h = self.conv_out(h)
        return torch.tanh(h) if self.tanh_out else h


def decode_audio(latent: torch.Tensor, audio_decoder: "AudioDecoder", vocoder: "Vocoder") -> Audio:
    """
    Decode an audio latent representation using the provided audio decoder and vocoder.
    Args:
        latent: Input audio latent tensor.
        audio_decoder: Model to decode the latent to waveform features.
        vocoder: Model to convert decoded features to audio waveform.
    Returns:
        Decoded audio with waveform and sampling rate.
    """
    decoded_audio = audio_decoder(latent)
    waveform = vocoder(decoded_audio).squeeze(0).float()
    return Audio(waveform=waveform, sampling_rate=vocoder.output_sampling_rate)

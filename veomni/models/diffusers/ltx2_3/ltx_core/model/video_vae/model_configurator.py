import json
from pathlib import Path

import safetensors
import torch

from ltx_core.model.video_vae.enums import LogVarianceType, NormLayerType, PaddingModeType
from ltx_core.model.video_vae.video_vae import VideoDecoder, VideoEncoder


class VideoEncoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> VideoEncoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        in_channels = config.get("in_channels", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "zeros"))
        encoder_blocks = config.get("encoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        latent_log_var_str = config.get("latent_log_var", "uniform")

        return VideoEncoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=in_channels,
            out_channels=latent_channels,
            encoder_blocks=[tuple(b) for b in encoder_blocks],
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            latent_log_var=LogVarianceType(latent_log_var_str),
            encoder_spatial_padding_mode=spatial_padding_mode,
        )


class VideoDecoderConfigurator:
    @classmethod
    def from_config(cls, config: dict) -> VideoDecoder:
        config = config.get("vae", {})
        convolution_dimensions = config.get("dims", 3)
        latent_channels = config.get("latent_channels", 128)
        spatial_padding_mode = PaddingModeType(config.get("spatial_padding_mode", "reflect"))
        out_channels = config.get("out_channels", 3)
        decoder_blocks = config.get("decoder_blocks", [])
        patch_size = config.get("patch_size", 4)
        norm_layer_str = config.get("norm_layer", "pixel_norm")
        causal = config.get("causal_decoder", False)
        timestep_conditioning = config.get("timestep_conditioning", True)
        base_channels = config.get("decoder_base_channels", 128)

        return VideoDecoder(
            convolution_dimensions=convolution_dimensions,
            in_channels=latent_channels,
            out_channels=out_channels,
            decoder_blocks=[tuple(b) for b in decoder_blocks],
            patch_size=patch_size,
            norm_layer=NormLayerType(norm_layer_str),
            causal=causal,
            timestep_conditioning=timestep_conditioning,
            decoder_spatial_padding_mode=spatial_padding_mode,
            base_channels=base_channels,
        )


VAE_ENCODER_KEY_PREFIXES = ("vae.encoder.", "vae.per_channel_statistics.")
VAE_DECODER_KEY_PREFIXES = ("vae.decoder.", "vae.per_channel_statistics.")


def _load_safetensors_config(path: str) -> dict:
    path_obj = Path(path)
    if path_obj.is_dir():
        safetensor_files = list(path_obj.rglob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {path}")
        path = str(safetensor_files[0])

    with safetensors.safe_open(path, framework="pt") as f:
        meta = f.metadata()
        if meta is None or "config" not in meta:
            return {}
        return json.loads(meta["config"])


def _load_filtered_state_dict(
    path: str,
    key_prefixes: tuple[str, ...],
    device: torch.device | None = None,
) -> dict:
    path_obj = Path(path)
    if path_obj.is_dir():
        safetensor_files = list(path_obj.rglob("*.safetensors"))
    else:
        safetensor_files = [path_obj]

    sd = {}
    device = device or torch.device("cpu")
    for shard_path in safetensor_files:
        with safetensors.safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            for name in f.keys():
                if any(name.startswith(prefix) for prefix in key_prefixes):
                    new_name = name
                    for prefix in key_prefixes:
                        if name.startswith(prefix):
                            new_name = name[len(prefix) :]
                            break
                    sd[new_name] = f.get_tensor(name).to(device=device)
    return sd


def load_video_encoder(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    meta_init: bool = False,
) -> VideoEncoder:
    if isinstance(device, str):
        device = torch.device(device)

    config = _load_safetensors_config(checkpoint_path)
    encoder = VideoEncoderConfigurator.from_config(config)

    if not meta_init:
        sd = _load_filtered_state_dict(checkpoint_path, VAE_ENCODER_KEY_PREFIXES, device)
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        encoder.load_state_dict(sd, strict=False, assign=True)
        encoder = encoder.to(device=device)

    return encoder


def load_video_decoder(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    meta_init: bool = False,
) -> VideoDecoder:
    if isinstance(device, str):
        device = torch.device(device)

    config = _load_safetensors_config(checkpoint_path)
    decoder = VideoDecoderConfigurator.from_config(config)

    if not meta_init:
        sd = _load_filtered_state_dict(checkpoint_path, VAE_DECODER_KEY_PREFIXES, device)
        if dtype is not None:
            sd = {k: v.to(dtype=dtype) for k, v in sd.items()}
        decoder.load_state_dict(sd, strict=False, assign=True)
        decoder = decoder.to(device=device)

    return decoder

"""Video VAE package."""

import torch
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder
from ltx_core.model.video_vae.model_configurator import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoderConfigurator,
    VideoEncoderConfigurator,
)
from ltx_core.model.video_vae.tiling import SpatialTilingConfig, TemporalTilingConfig, TilingConfig
from ltx_core.model.video_vae.video_vae import VideoDecoder, VideoEncoder


def load_video_encoder(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    meta_init: bool = False,
) -> VideoEncoder:
    if isinstance(device, str):
        device = torch.device(device)
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoEncoderConfigurator,
        model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)


def load_video_decoder(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    meta_init: bool = False,
) -> VideoDecoder:
    if isinstance(device, str):
        device = torch.device(device)
    return SingleGPUModelBuilder(
        model_path=str(checkpoint_path),
        model_class_configurator=VideoDecoderConfigurator,
        model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
    ).build(device=device, dtype=dtype)


__all__ = [
    "SpatialTilingConfig",
    "TemporalTilingConfig",
    "TilingConfig",
    "VAE_DECODER_COMFY_KEYS_FILTER",
    "VAE_ENCODER_COMFY_KEYS_FILTER",
    "VideoDecoder",
    "VideoDecoderConfigurator",
    "VideoEncoder",
    "VideoEncoderConfigurator",
    "load_video_decoder",
    "load_video_encoder",
]

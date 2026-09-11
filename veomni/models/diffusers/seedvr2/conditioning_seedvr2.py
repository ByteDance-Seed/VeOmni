"""Frozen VAE conditioning and explicit supervised one-step fine-tuning.

The supervised target below is an integration objective, not a reproduction of
the unpublished SeedVR2 adversarial post-training.
"""

from pathlib import Path

import torch
from transformers import PretrainedConfig, PreTrainedModel


class SeedVR2ConditionConfig(PretrainedConfig):
    model_type = "seedvr2_condition"

    def __init__(self, base_model_path="", scaling_factor=0.9152, **kwargs):
        super().__init__(**kwargs)
        self.base_model_path = base_model_path
        self.scaling_factor = scaling_factor


class SeedVR2ConditionModel(PreTrainedModel):
    config_class = SeedVR2ConditionConfig

    def __init__(self, config, meta_init=False, **kwargs):
        super().__init__(config)
        self.vae = None
        self.register_buffer("text_embedding", None, persistent=False)
        if not meta_init:
            from .core.vae.attn_video_vae import VideoAutoencoderKLWrapper

            self.vae = VideoAutoencoderKLWrapper(
                spatial_downsample_factor=8,
                temporal_downsample_factor=4,
                freeze_encoder=True,
                act_fn="silu",
                block_out_channels=(128, 256, 512, 512),
                down_block_types=("DownEncoderBlock3D",) * 4,
                up_block_types=("UpDecoderBlock3D",) * 4,
                in_channels=3,
                out_channels=3,
                latent_channels=16,
                layers_per_block=2,
                norm_num_groups=32,
                slicing_sample_min_size=4,
                temporal_scale_num=2,
                inflation_mode="pad",
                use_quant_conv=False,
                use_post_quant_conv=False,
            )
            weights = Path(config.base_model_path)
            self.vae.load_state_dict(
                torch.load(weights / "ema_vae.pth", map_location="cpu", weights_only=True, mmap=True), strict=True
            )
            self.vae.requires_grad_(False).eval()
            self.vae.set_causal_slicing(split_size=4, memory_device="same")
            self.vae.set_memory_limit(conv_max_mem=0.5, norm_max_mem=0.5)
            self.text_embedding = torch.load(weights / "pos_emb.pt", map_location="cpu", weights_only=True)
            if not isinstance(self.text_embedding, torch.Tensor) or self.text_embedding.ndim != 2:
                raise ValueError("Expected the official packed positive text embedding tensor.")

    @torch.no_grad()
    def encode_video(self, video):
        """Encode normalized C,T,H,W video into scaled T,H,W,C latents."""
        if self.vae is None:
            raise RuntimeError("VAE is unavailable in offline-conditioning mode.")
        if video.ndim != 4 or video.shape[0] != 3:
            raise ValueError("Expected normalized C,T,H,W RGB video.")
        original_frames = video.shape[1]
        pad = (4 - (original_frames - 1) % 4) % 4
        if pad:
            video = torch.cat((video, video[:, -1:].expand(-1, pad, -1, -1)), dim=1)
        parameter = next(self.vae.parameters())
        video = self.vae.preprocess(video.unsqueeze(0).to(parameter.device, parameter.dtype))
        latent = self.vae.encode(video).latent
        if latent.ndim == 4:
            latent = latent.unsqueeze(2)
        return latent.squeeze(0).permute(1, 2, 3, 0) * self.config.scaling_factor

    @torch.no_grad()
    def decode_latents(self, latents):
        parameter = next(self.vae.parameters())
        value = (latents / self.config.scaling_factor).permute(3, 0, 1, 2).unsqueeze(0)
        sample = self.vae.decode(value.to(parameter.device, parameter.dtype)).sample
        if sample.ndim == 4:
            sample = sample.unsqueeze(2)
        return sample.squeeze(0)

    @torch.no_grad()
    def get_condition(self, videos, target_videos=None, **kwargs):
        if target_videos is None or len(videos) != len(target_videos):
            raise ValueError("Supervised SeedVR2 training requires paired videos and target_videos.")
        return {
            "latents": [self.encode_video(video) for video in target_videos],
            "condition_latents": [self.encode_video(video) for video in videos],
            "context": [self.text_embedding for _ in videos],
        }

    def process_condition(self, latents, condition_latents, context, **kwargs):
        if not (len(latents) == len(condition_latents) == len(context)) or not latents:
            raise ValueError("Target, condition, and context batches must be nonempty and aligned.")
        packed = {"vid": [], "txt": [], "timestep": [], "training_target": []}
        for target, condition, text in zip(latents, condition_latents, context):
            if target.shape != condition.shape:
                raise ValueError("HQ and LQ latents must have identical shapes.")
            noise = torch.randn_like(target)
            packed["vid"].append(torch.cat((noise, condition, torch.ones_like(condition[..., :1])), dim=-1))
            packed["txt"].append(text)
            packed["timestep"].append(torch.tensor([1000.0], device=target.device))
            packed["training_target"].append(noise - target)
        return packed

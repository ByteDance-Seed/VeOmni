from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from torchvision.transforms import InterpolationMode, functional
from transformers import PreTrainedModel

from .....utils import logging
from .....utils.device import get_device_type
from ..ltx_core.model.video_vae import load_video_encoder
from ..ltx_core.text_encoders.gemma.embeddings_processor import (
    EmbeddingsProcessor,
    convert_to_additive_mask,
)
from ..ltx_core.text_encoders.gemma.encoders.encoder_configurator import (
    EMBEDDINGS_PROCESSOR_KEY_REMAP,
    build_embeddings_processor,
)
from .configuration_ltx2_3_condition import LTXVideoConditionModelConfig


logger = logging.get_logger(__name__)

_BASE_SHIFT_ANCHOR = 1024
_MAX_SHIFT_ANCHOR = 4096
DEFAULT_FPS = 24


class LTX2Scheduler:
    def __init__(self):
        self.timesteps = None
        self.sigmas = None

    def set_timesteps(self, num_train_timesteps: int, device=None):
        sigmas = torch.linspace(1.0, 0.0, num_train_timesteps + 1)
        x1 = _BASE_SHIFT_ANCHOR
        x2 = _MAX_SHIFT_ANCHOR
        max_shift = 2.05
        base_shift = 0.95
        mm = (max_shift - base_shift) / (x2 - x1)
        b = base_shift - mm * x1
        sigma_shift = _MAX_SHIFT_ANCHOR * mm + b
        sigmas = torch.where(
            sigmas != 0,
            math.exp(sigma_shift) / (math.exp(sigma_shift) + (1 / sigmas - 1)),
            0,
        )
        non_zero_mask = sigmas != 0
        non_zero_sigmas = sigmas[non_zero_mask]
        one_minus_z = 1.0 - non_zero_sigmas
        scale_factor = one_minus_z[-1] / (1.0 - 0.1)
        stretched = 1.0 - (one_minus_z / scale_factor)
        sigmas[non_zero_mask] = stretched
        sigmas = sigmas.to(torch.float32)
        if device is not None:
            sigmas = sigmas.to(device)
        self.sigmas = sigmas
        self.timesteps = sigmas[:-1]

    def scale_noise(self, latents: torch.Tensor, timestep: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sigma = timestep.reshape(-1, *([1] * (latents.dim() - 1)))
        return (1.0 - sigma) * latents + sigma * noise


def _load_embeddings_processor_weights(
    embeddings_processor: EmbeddingsProcessor,
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
):
    """Load EmbeddingsProcessor weights from LTX-2 checkpoint with key remapping.

    Uses safetensors safe_open to selectively read only the needed keys,
    avoiding loading the entire (potentially 20+ GB) checkpoint into memory.
    """
    from safetensors import safe_open

    ckpt_path = Path(checkpoint_path)

    target_prefixes = tuple(EMBEDDINGS_PROCESSOR_KEY_REMAP.keys())

    if ckpt_path.is_dir():
        safetensor_files = sorted(ckpt_path.glob("*.safetensors"))
        if not safetensor_files:
            raise FileNotFoundError(f"No safetensors files found in {ckpt_path}")
    elif ckpt_path.suffix == ".safetensors":
        safetensor_files = [ckpt_path]
    else:
        all_state_dict = torch.load(str(ckpt_path), map_location=device, weights_only=True)
        processor_state_dict = {}
        for key, value in all_state_dict.items():
            for old_prefix, new_prefix in EMBEDDINGS_PROCESSOR_KEY_REMAP.items():
                if key.startswith(old_prefix):
                    new_key = new_prefix + key[len(old_prefix) :]
                    processor_state_dict[new_key] = value.to(dtype)
                    break
        missing, unexpected = embeddings_processor.load_state_dict(processor_state_dict, strict=False)
        if missing:
            logger.warning_rank0(f"EmbeddingsProcessor missing keys: {missing}")
        if unexpected:
            logger.warning_rank0(f"EmbeddingsProcessor unexpected keys: {unexpected}")
        return

    processor_state_dict = {}
    for sf_path in safetensor_files:
        with safe_open(str(sf_path), framework="pt", device=str(device)) as f:
            keys = f.keys()
            for key in keys:
                if key.startswith(target_prefixes):
                    for old_prefix, new_prefix in EMBEDDINGS_PROCESSOR_KEY_REMAP.items():
                        if key.startswith(old_prefix):
                            new_key = new_prefix + key[len(old_prefix) :]
                            processor_state_dict[new_key] = f.get_tensor(key).to(dtype)
                            break

    if not processor_state_dict:
        raise RuntimeError(
            f"No EmbeddingsProcessor weights found in {checkpoint_path}. "
            f"Expected keys starting with: {target_prefixes}"
        )

    missing, unexpected = embeddings_processor.load_state_dict(processor_state_dict, strict=False)
    if missing:
        logger.warning_rank0(f"EmbeddingsProcessor missing keys: {missing}")
    if unexpected:
        logger.warning_rank0(f"EmbeddingsProcessor unexpected keys: {unexpected}")


def _load_transformer_config(checkpoint_path: str) -> dict:
    """Load transformer config JSON from checkpoint directory for connector config."""
    ckpt_path = Path(checkpoint_path)
    config_candidates = [
        ckpt_path / "transformer" / "config.json",
        ckpt_path / "config.json",
        ckpt_path / "model_index.json",
    ]
    if ckpt_path.is_file():
        config_candidates = [ckpt_path.parent / "transformer" / "config.json", ckpt_path.parent / "config.json"]

    for candidate in config_candidates:
        if candidate.exists():
            with open(candidate) as f:
                return json.load(f)

    raise FileNotFoundError(f"No config.json found near {checkpoint_path}")


class LTXVideoConditionModel(PreTrainedModel):
    config_class = LTXVideoConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: LTXVideoConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.vae = None
        self.scheduler = None
        self.embeddings_processor = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        self._load_components()

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading LTX-Video condition components from {base}.")

        device = torch.device(get_device_type())

        transformer_config = _load_transformer_config(base)

        self.embeddings_processor = build_embeddings_processor(transformer_config, with_feature_extractor=False)

        _load_embeddings_processor_weights(self.embeddings_processor, base, device=device, dtype=torch.bfloat16)
        self.embeddings_processor = self.embeddings_processor.to(device=device, dtype=torch.bfloat16)

        if not self.config.with_audio:
            self.embeddings_processor.audio_connector = None

        vae_path = base
        if self.config.vae_subfolder:
            vae_path = f"{base}/{self.config.vae_subfolder}"
        if not self.meta_init:
            self.vae = load_video_encoder(vae_path, device=get_device_type(), dtype=torch.bfloat16, meta_init=False)
        else:
            self.vae = None

        self.scheduler = LTX2Scheduler()

        if self.meta_init:
            self.embeddings_processor.feature_extractor = None

    def _encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        height, width = video.shape[-2:]
        size = min(self.config.video_max_size, min(width, height))
        video = functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)
        video = video / 127.5 - 1.0

        vae_device = next(self.vae.parameters()).device
        vae_dtype = next(self.vae.parameters()).dtype
        video = video.to(device=vae_device, dtype=vae_dtype)
        with torch.no_grad():
            normalized_means = self.vae(video)
        return normalized_means

    @torch.no_grad()
    def get_condition(self, inputs, videos, **kwargs) -> dict[str, Any]:
        """Online encoding: Gemma + feature extractor for text, VAE for video.

        Note: For offline precompute, use the precompute_gemma_features.py script instead.
        This method requires Gemma to be loaded, which is expensive.
        """
        raise NotImplementedError(
            "Online Gemma encoding is not supported during training. "
            "Please pre-compute Gemma features using the precompute script and use offline_training mode."
        )

    def _sample_shifted_logit_normal(
        self,
        batch_size: int,
        seq_length: int,
        device: torch.device,
        std: float = 1.0,
        eps: float = 1e-3,
        uniform_prob: float = 0.1,
        min_tokens: int = 1024,
        max_tokens: int = 4096,
        min_shift: float = 0.95,
        max_shift: float = 2.05,
    ) -> torch.Tensor:
        m = (max_shift - min_shift) / (max_tokens - min_tokens)
        b = min_shift - m * min_tokens
        mu = m * seq_length + b

        normal_samples = torch.randn((batch_size,), device=device) * std + mu
        logitnormal_samples = torch.sigmoid(normal_samples)

        percentile_999 = torch.sigmoid(torch.tensor(mu + 3.0902 * std, device=device))
        percentile_005 = torch.sigmoid(torch.tensor(mu + (-2.5758) * std, device=device))

        zero_terminal_raw = (logitnormal_samples - percentile_005) / (percentile_999 - percentile_005)
        stretched_logit = torch.where(
            zero_terminal_raw >= eps,
            zero_terminal_raw,
            2 * eps - zero_terminal_raw,
        )
        stretched_logit = torch.clamp(stretched_logit, 0, 1)

        uniform = (1 - eps) * torch.rand((batch_size,), device=device) + eps
        prob = torch.rand((batch_size,), device=device)

        return torch.where(prob > uniform_prob, stretched_logit, uniform)

    def process_condition(
        self,
        latents: list[torch.Tensor],
        context: list[torch.Tensor],
        context_mask: list[torch.Tensor] | None = None,
        audio_context: list[torch.Tensor] | None = None,
        audio_latents: list[torch.Tensor] | None = None,
        first_frame_conditioning_p: float = 0.1,
        timestep_sampling_mode: str = "shifted_logit_normal",
        fps: list[float] | None = None,
    ) -> dict[str, Any]:
        """Process pre-computed features: run connectors, add noise, compute targets.

        Args:
            latents: List of video latent tensors [B, C, F, H, W].
            context: List of pre-computed video feature tensors [B, S, D] (from Gemma + feature extractor).
            context_mask: List of attention mask tensors [B, S]. If None, all tokens are valid.
            audio_context: List of pre-computed audio feature tensors [B, S, D]. If None, no audio features.
            audio_latents: List of audio latent tensors [B, C, T, F]. If None, no audio latents.
            first_frame_conditioning_p: Probability of conditioning on the first frame.
            timestep_sampling_mode: "shifted_logit_normal" or "uniform".
        """
        if timestep_sampling_mode == "uniform" and not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents[0].device)
            self._timesteps_ready = True

        device = latents[0].device
        compute_dtype = torch.float32

        packed_conditions: dict[str, list] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "context_mask": [],
            "training_target": [],
            "latents": [],
            "video_loss_mask": [],
            "audio_hidden_states": [],
            "audio_timestep": [],
            "audio_encoder_hidden_states": [],
            "audio_training_target": [],
            "audio_loss_mask": [],
            "fps": [],
        }

        for i, (sample_latents, sample_features) in enumerate(zip(latents, context)):
            sample_mask = context_mask[i] if context_mask is not None else None
            sample_audio_features = audio_context[i] if audio_context is not None else None
            sample_audio_latents = audio_latents[i] if audio_latents is not None else None

            sample_features = sample_features.to(device=device)
            if sample_mask is not None:
                sample_mask = sample_mask.to(device=device)
            else:
                sample_mask = torch.ones(sample_features.shape[:-1], device=device, dtype=torch.long)

            additive_mask = convert_to_additive_mask(sample_mask, sample_features.dtype)

            if not self.config.with_audio:
                sample_audio_features = None
            elif sample_audio_features is not None:
                sample_audio_features = sample_audio_features.to(device=device)

            self.embeddings_processor = self.embeddings_processor.to(device=device, dtype=sample_features.dtype)

            with torch.no_grad():
                video_embeds, audio_embeds, binary_mask = self.embeddings_processor.create_embeddings(
                    sample_features, sample_audio_features, additive_mask
                )

            latents_on_device = sample_latents.to(device=device, dtype=compute_dtype)
            B, C, F, H, W = latents_on_device.shape

            conditioning_mask = self._create_first_frame_conditioning_mask(
                B, F, H, W, first_frame_conditioning_p, device
            )

            noise = torch.randn(
                latents_on_device.shape,
                dtype=compute_dtype,
                device=device,
            )

            if timestep_sampling_mode == "shifted_logit_normal":
                seq_length = F * H * W
                timestep = self._sample_shifted_logit_normal(B, seq_length, device).to(
                    device=device, dtype=compute_dtype
                )
            else:
                timestep_ids = torch.randint(
                    0,
                    len(self.scheduler.timesteps),
                    (B,),
                    device=device,
                )
                timestep = self.scheduler.timesteps[timestep_ids].to(device=device, dtype=compute_dtype)

            noisy_latents = self.scheduler.scale_noise(latents_on_device, timestep, noise)
            noisy_latents = noisy_latents.to(device=device)

            conditioning_mask_3d = conditioning_mask.view(B, F, H, W)
            noisy_latents = torch.where(
                conditioning_mask_3d.unsqueeze(1),
                latents_on_device.to(device=device),
                noisy_latents,
            )

            training_target = noise.to(device=device) - latents_on_device.to(device=device)
            video_loss_mask = (~conditioning_mask).float()

            packed_conditions["hidden_states"].append(noisy_latents)
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(video_embeds)
            packed_conditions["context_mask"].append(binary_mask)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(latents_on_device.to(device=device))
            packed_conditions["video_loss_mask"].append(video_loss_mask)
            sample_fps = fps[i] if fps is not None else DEFAULT_FPS
            packed_conditions["fps"].append(sample_fps)

            if self.config.with_audio and sample_audio_latents is not None:
                audio_on_device = sample_audio_latents.to(device=device, dtype=compute_dtype)
                audio_noise = torch.randn(
                    audio_on_device.shape,
                    dtype=compute_dtype,
                    device=device,
                )

                noisy_audio = self.scheduler.scale_noise(audio_on_device, timestep, audio_noise)
                audio_training_target = audio_noise - audio_on_device

                audio_seq_len = noisy_audio.shape[2]
                audio_loss_mask = torch.ones(B, audio_seq_len, dtype=torch.float32, device=device)

                packed_conditions["audio_hidden_states"].append(noisy_audio)
                packed_conditions["audio_timestep"].append(timestep)
                packed_conditions["audio_encoder_hidden_states"].append(audio_embeds)
                packed_conditions["audio_training_target"].append(audio_training_target)
                packed_conditions["audio_loss_mask"].append(audio_loss_mask)

        if not packed_conditions["audio_hidden_states"]:
            del packed_conditions["audio_hidden_states"]
            del packed_conditions["audio_timestep"]
            del packed_conditions["audio_encoder_hidden_states"]
            del packed_conditions["audio_training_target"]
            del packed_conditions["audio_loss_mask"]

        return packed_conditions

    def _create_first_frame_conditioning_mask(
        self,
        batch_size: int,
        num_frames: int,
        height: int,
        width: int,
        conditioning_prob: float,
        device: torch.device,
    ) -> torch.Tensor:
        """Create per-sample first-frame conditioning mask.

        Returns: [B, seq_len] bool tensor where True = conditioning token.
        """
        seq_len = num_frames * height * width
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)

        first_frame_size = height * width
        if first_frame_size < seq_len:
            per_sample_condition = torch.rand(batch_size, device=device) < conditioning_prob
            mask[per_sample_condition, :first_frame_size] = True

        return mask

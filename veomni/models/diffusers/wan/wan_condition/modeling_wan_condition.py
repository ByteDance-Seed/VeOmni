from __future__ import annotations

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from transformers import AutoTokenizer, PreTrainedModel, UMT5EncoderModel

from .....utils import logging
from .config_wan_condition import WanConditionConfig


logger = logging.get_logger(__name__)


class WanConditionModel(PreTrainedModel):
    config_class = WanConditionConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: WanConditionConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self._timesteps_ready = False

        if config.load_components:
            self._load_components()

    def _load_components(self):
        if not self.config.base_model_path:
            raise ValueError("`base_model_path` is required for WanConditionModel.")

        base = self.config.base_model_path
        logger.info_rank0(f"Loading Wan condition components from {base}.")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.text_encoder = UMT5EncoderModel.from_pretrained(base, subfolder=self.config.text_encoder_subfolder)
        self.vae = AutoencoderKLWan.from_pretrained(base, subfolder=self.config.vae_subfolder)
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base, subfolder=self.config.scheduler_subfolder
        )

    def _ensure_components(self):
        if self.tokenizer is None or self.text_encoder is None or self.vae is None or self.scheduler is None:
            self._load_components()

    def _move_components_to(self, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        if self.text_encoder is not None:
            self.text_encoder.to(device=device, dtype=dtype)
        if self.vae is not None:
            self.vae.to(device=device, dtype=dtype)

    @torch.no_grad()
    def get_condition(self, **kwargs) -> dict[str, torch.Tensor]:
        """Input format is fixed by dit data_transform:
        {"inputs": ..., "outputs": ..., "images": ..., "videos": ...}
        """
        self._ensure_components()

        prompt = kwargs["inputs"]
        videos = kwargs["videos"]

        if isinstance(videos, torch.Tensor):
            video = videos
        else:
            video = videos[0]
            if not isinstance(video, torch.Tensor):
                raise ValueError("`videos` must contain torch.Tensor.")

        target_device = video.device
        self._move_components_to(target_device)

        prompt_list = prompt if isinstance(prompt, list) else [prompt]
        text_inputs = self.tokenizer(
            prompt_list,
            padding="max_length",
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt",
        )
        input_ids = text_inputs.input_ids.to(target_device)
        context = self.text_encoder(input_ids=input_ids).last_hidden_state

        if video.dim() != 5:
            raise ValueError(f"Expected 5D video tensor, got shape: {tuple(video.shape)}")
        if video.shape[1] not in (3, 16):
            # likely [B, F, C, H, W]
            video = video.permute(0, 2, 1, 3, 4).contiguous()

        video = video.to(target_device, dtype=self.vae.dtype)
        posterior = self.vae.encode(video).latent_dist
        latents = posterior.sample()
        scaling_factor = getattr(self.vae.config, "scaling_factor", 1.0)
        latents = latents * scaling_factor

        return {
            "latents": latents,
            "context": context.to(latents.device),
        }

    @torch.no_grad()
    def process_condition(self, **condition_dict) -> dict[str, torch.Tensor]:
        self._ensure_components()
        latents = condition_dict["latents"]
        context = condition_dict["context"]

        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents.device)
            self._timesteps_ready = True

        noise = torch.randn_like(latents)
        timestep_ids = torch.randint(0, len(self.scheduler.timesteps), (latents.shape[0],), device=latents.device)
        timestep = self.scheduler.timesteps[timestep_ids].to(device=latents.device, dtype=latents.dtype)
        noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
        training_target = noise - latents

        return {
            "x": noisy_latents,
            "hidden_states": noisy_latents,
            "timestep": timestep,
            "context": context,
            "encoder_hidden_states": context,
            "training_target": training_target,
            "loss_weight": torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype),
        }

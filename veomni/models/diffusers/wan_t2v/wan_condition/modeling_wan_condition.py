from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.video_processor import VideoProcessor
from torchvision.transforms import InterpolationMode, functional
from transformers import AutoTokenizer, PreTrainedModel, UMT5EncoderModel

from .....utils import logging
from .configuration_wan_condition import WanTransformer3DConditionModelConfig


def vis_video(video: torch.Tensor):
    from veomni.data.multimodal.video_utils import save_video_tensors_to_file

    video = video.permute(0, 2, 3, 1).clamp(0, 255).to(torch.uint8)
    save_video_tensors_to_file(video, "video.mp4", fps=24)


logger = logging.get_logger(__name__)


# T2V only
class WanTransformer3DConditionModel(PreTrainedModel):
    config_class = WanTransformer3DConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: WanTransformer3DConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.video_processor = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        self._load_components()

    @property
    def _execution_device(self):
        return self.vae.device

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Wan condition components from {base}.")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        if self.meta_init:
            text_encoder_config = UMT5EncoderModel.config_class.from_pretrained(
                base,
                subfolder=self.config.text_encoder_subfolder,
            )
            # Build module from config to avoid loading checkpoint tensors.
            self.text_encoder = UMT5EncoderModel._from_config(text_encoder_config)
            self.text_encoder.to(dtype=torch.bfloat16)
            self.vae = AutoencoderKLWan.from_config(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        else:
            self.text_encoder = UMT5EncoderModel.from_pretrained(
                base,
                subfolder=self.config.text_encoder_subfolder,
                torch_dtype=torch.bfloat16,
            )
            self.vae = AutoencoderKLWan.from_pretrained(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
        )
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae.config.scale_factor_spatial)

    def _encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        # resize video to max size
        height, width = video.shape[-2:]

        size = min(self.config.video_max_size, min(width, height))
        video = functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)

        video = self.video_processor.preprocess_video(video)
        video = video.to(device=self.vae.device, dtype=self.vae.dtype)
        posterior = self.vae.encode(video).latent_dist

        # save mean & logvar
        latents = posterior.parameters
        return latents.mean(dim=1)

    @torch.no_grad()
    def _get_t5_prompt_embeds(self, **kwargs):
        return WanPipeline._get_t5_prompt_embeds(self, **kwargs)

    @torch.no_grad()
    def get_condition(self, inputs, videos, **kwargs) -> dict[str, Any]:
        """
        inputs: list[str], a list of samples of prompts
        videos: list[list[torch.Tensor]] a list of samples of videos
        """
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=inputs,
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )  # bs, seqlen, dim
        context_list = [u.unsqueeze(0) for u in prompt_embeds]

        latents_list: list[list[torch.Tensor]] = []
        for sample_videos in videos:
            assert len(sample_videos) == 1, "Only one video per sample is supported for T2V"
            latents_list.append(self._encode_video_to_latents(sample_videos[0]))

        sample_condition_list = [
            {"latents": latents, "context": context} for latents, context in zip(latents_list, context_list)
        ]
        return sample_condition_list

    @torch.no_grad()
    def process_condition(
        self, latents_list: list[list[torch.Tensor]], context_list: list[torch.Tensor]
    ) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents_list[0][0].device)
            self._timesteps_ready = True

        packed_conditions: list[dict[str, torch.Tensor]] = []
        for sample_idx, sample_latents in enumerate(latents_list):
            sample_context = context_list[sample_idx]
            for latents in sample_latents:
                noise = torch.randn_like(latents)
                timestep_ids = torch.randint(
                    0, len(self.scheduler.timesteps), (latents.shape[0],), device=latents.device
                )
                timestep = self.scheduler.timesteps[timestep_ids].to(device=latents.device, dtype=latents.dtype)
                noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
                training_target = noise - latents

                context = sample_context.to(latents.device)
                if context.shape[0] != latents.shape[0]:
                    context = context.expand(latents.shape[0], -1, -1)

                packed_conditions.append(
                    {
                        "x": noisy_latents,
                        "hidden_states": noisy_latents,
                        "timestep": timestep,
                        "context": context,
                        "encoder_hidden_states": context,
                        "training_target": training_target,
                        "loss_weight": torch.ones(latents.shape[0], device=latents.device, dtype=latents.dtype),
                    }
                )

        return {"packed_conditions": packed_conditions, "latents": latents_list, "context": context_list}

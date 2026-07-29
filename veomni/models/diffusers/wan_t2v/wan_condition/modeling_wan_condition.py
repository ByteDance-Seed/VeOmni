from __future__ import annotations
from typing import Any

import torch
from diffusers import AutoencoderKLWan, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.pipelines.wan.pipeline_wan import WanPipeline
from diffusers.video_processor import VideoProcessor
from torchvision.transforms import InterpolationMode, functional
from transformers import AutoTokenizer, PreTrainedModel, UMT5EncoderModel

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .configuration_wan_condition import WanTransformer3DConditionModelConfig

logger = logging.get_logger(__name__)

class WanTransformer3DConditionModel(PreTrainedModel):
    config_class = WanTransformer3DConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: WanTransformer3DConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self.video_processor = None
        self.negative_prompt_embeds = None
        self._timesteps_ready = False
        self.meta_init = meta_init
        self.seed = config.seed
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed(self.seed + get_parallel_state().dp_rank)
        self._load_components()

    @property
    def _execution_device(self):
        return self.vae.device

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Wan condition components from {base}.")
        self.tokenizer = AutoTokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.text_encoder = UMT5EncoderModel.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        if self.meta_init:
            self.vae = AutoencoderKLWan.from_config(
                base,
                subfolder=self.config.vae_subfolder,
                torch_dtype=torch.float32,
            )
        else:
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
        self._prepare_negative_prompt_embeds()
        if self.meta_init:
            del self.text_encoder

    @torch.no_grad()
    def _prepare_negative_prompt_embeds(self):
        prompt_embeds, _ = WanPipeline.encode_prompt(
            self,
            prompt=[self.config.cfg_negative_prompt],
            do_classifier_free_guidance=False,
            max_sequence_length=self.config.max_sequence_length,
        )
        self.negative_prompt_embeds = prompt_embeds[0].unsqueeze(0)

    def _encode_video_to_latents(self, video: torch.Tensor) -> torch.Tensor:
        # resize video to max size
        height, width = video.shape[-2:]

        size = min(self.config.video_max_size, min(width, height))
        video = functional.resize(video, size, interpolation=InterpolationMode.BICUBIC).float().clamp(0, 255)
        video = self.video_processor.preprocess_video(video)
        video = video.to(device=self.vae.device, dtype=self.vae.dtype)

        # save mean & logvar
        posterior: DiagonalGaussianDistribution = self.vae.encode(video).latent_dist

        return posterior.parameters

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(
            self.vae.config.latents_mean,
            device=latents.device,
            dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        latents_std = torch.tensor(
            self.vae.config.latents_std,
            device=latents.device,
            dtype=latents.dtype
        ).view(1, self.vae.config.z_dim, 1, 1, 1)

        return (latents - latents_mean) / latents_std

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

        latents_list: list[torch.Tensor] = []
        for sample_videos in videos:
            assert len(sample_videos) == 1, "Only one video per sample is supported for T2V"
            latents_list.append(self._encode_video_to_latents(sample_videos[0]))  # 1, c, f, h, w

        return {"latents": latents_list, "context": context_list}

    def process_condition(self, latents: list[torch.Tensor], context: list[torch.Tensor]) -> dict[str, Any]:
        if not self._timesteps_ready:
            self.scheduler.set_timesteps(self.config.num_train_timesteps, device=latents[0].device)
            self._timesteps_ready = True

        packed_conditions: dict[str, list[torch.Tensor]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "training_target": [],
            "latents": [],
        }
        for sample_latents, sample_context in zip(latents, context):
            norm_latents = DiagonalGaussianDistribution(sample_latents).mode()
            norm_latents = self._normalize_latents(norm_latents)
            noise = torch.randn(  # TODO: use randn_like(generator=self.generator) when updating to torch 2.10.0
                latents.shape, dtype=latents.dtype, device=self.generator.device, generator=self.generator
            ).to(self.generator.device)
            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (latents.shape[0],),
                device=self.generator.device,
                generator=self.generator,
            ).to(latents.device)
            timestep = self.scheduler.timesteps[timestep_ids].to(device=latents.device, dtype=latents.dtype)
            noisy_latents = self.scheduler.scale_noise(latents, timestep, noise)
            training_target = noise - latents

            use_negative_context = (
                torch.rand((), device=self.generator.device, generator=self.generator) < self.config.cfg_negative_prob
            )
            if use_negative_context:
                sample_context = self.negative_prompt_embeds.to(device=latents.device, dtype=sample_context.dtype)
            else:
                sample_context = sample_context.to(latents.device)

            if self.config.expand_timesteps:
                # Wan2.2 I2V: flow-matching blend with position-dependent timesteps.
                # First frame uses clean latent as condition (t=0);
                # other frames use noisy latent at the sampled timestep.
                condition = norm_latents[:, :, 0:1, :, :]
                num_frames = norm_latents.shape[2]
                latent_h = norm_latents.shape[3]
                latent_w = norm_latents.shape[4]
                first_frame_mask = torch.ones(
                    1,
                    1,
                    num_frames,
                    latent_h,
                    latent_w,
                    dtype=norm_latents.dtype,
                    device=norm_latents.device,
                )
                first_frame_mask[:, :, 0] = 0

                hidden_states = (1 - first_frame_mask) * condition + first_frame_mask * noisy_latents

                # First frame has t=0 (clean condition). The correct velocity
                # target at t=0 is zero because the flow has not started.
                # Setting target=0 at frame 0 prevents the model from learning
                # spurious outputs that corrupt other frames through attention.
                training_target[:, :, 0, :, :] = 0.0

                # Position-dependent timestep at post-patch resolution.
                # Use integer division to match Conv3d output size exactly
                # (avoids mismatch when latent spatial dims are odd).
                p_t, p_h, p_w = self.config.patch_size
                post_patch_f = num_frames // p_t
                post_patch_h = latent_h // p_h
                post_patch_w = latent_w // p_w
                first_frame_mask_post = torch.ones(
                    1,
                    1,
                    post_patch_f,
                    post_patch_h,
                    post_patch_w,
                    dtype=norm_latents.dtype,
                    device=norm_latents.device,
                )
                first_frame_mask_post[:, :, 0] = 0

                temp_ts = (first_frame_mask_post[0, 0] * timestep).flatten()
                timestep = temp_ts.unsqueeze(0).expand(norm_latents.shape[0], -1)
            else:
                hidden_states = noisy_latents

            packed_conditions["hidden_states"].append(hidden_states)
            packed_conditions["timestep"].append(timestep)
            packed_conditions["encoder_hidden_states"].append(sample_context)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(norm_latents)

        return packed_conditions
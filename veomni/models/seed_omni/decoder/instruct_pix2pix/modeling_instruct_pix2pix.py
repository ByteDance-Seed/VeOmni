from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.image_processor import PipelineImageInput
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from transformers import PreTrainedModel

from .....utils import logging
from ....seed_omni.projector import build_feature_projector
from .configuration_instruct_pix2pix import InstructPix2PixConfig


logger = logging.get_logger(__name__)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img.retrieve_latents
def retrieve_latents(
    encoder_output: torch.Tensor, generator: Optional[torch.Generator] = None, sample_mode: str = "sample"
):
    if hasattr(encoder_output, "latent_dist") and sample_mode == "sample":
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, "latent_dist") and sample_mode == "argmax":
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, "latents"):
        return encoder_output.latents
    else:
        raise AttributeError("Could not access latents of provided encoder_output")


def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`. Based on findings of [Common Diffusion Noise Schedules and
    Sample Steps are Flawed](https://arxiv.org/pdf/2305.08891.pdf). See Section 3.4
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg


class InstructionPix2Pix(PreTrainedModel):
    config_class = InstructPix2PixConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv3d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def __init__(self, config: InstructPix2PixConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if config.unet_config is not None:
            self.unet = UNet2DConditionModel.from_config(config.unet_config)
            self.vae = AutoencoderKL.from_config(config.vae_config)
            self.noise_scheduler = DDPMScheduler.from_config(config.scheduler_config)
        else:
            self.unet, self.vae, self.noise_scheduler = None, None, None
            logger.info_rank0("Make sure you are converting checkpoints instead of training/inferencing.")

        if config.hidden_size is not None:
            self.head = build_feature_projector(config.hidden_size, config.condition_dim)
        else:
            self.head = nn.Identity()

        self.force_zeros_for_empty_prompt = config.force_zeros_for_empty_prompt
        self.is_cosxl_edit = config.is_cosxl_edit
        if self.unet is not None:
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
            self.default_sample_size = self.unet.config.sample_size
        self.watermark = None

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (
            batch_size,
            num_channels_latents,
            int(height) // self.vae_scale_factor,
            int(width) // self.vae_scale_factor,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.noise_scheduler.init_noise_sigma
        return latents

    def forward(self, input_embeds: torch.Tensor, clean_images: torch.Tensor, **kwargs):
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape, device=clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device, dtype=torch.int64
        )

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Predict the noise residual
        noise_pred = self.unet(noisy_images, timesteps, return_dict=False)[0]
        loss = F.mse_loss(noise_pred, noise)

        return loss

    def encode_prompt(
        self,
        prompt: str,
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.Tensor] = None,
        text_encoders=None,
        tokenizers=None,
    ):
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
        # Define tokenizers and text encoders
        # encoder_1 768
        # encoder_2 1280
        if prompt_embeds is None:
            # textual inversion: process multi-vector tokens if necessary
            prompt_embeds_list = []
            prompts = [prompt, prompt]
            for prompt, tokenizer, text_encoder in zip(prompts, tokenizers, text_encoders):
                text_inputs = tokenizer(
                    prompt,
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )

                text_input_ids = text_inputs.input_ids

                prompt_embeds = text_encoder(
                    text_input_ids.to(device),
                    output_hidden_states=True,
                )

                # We are only ALWAYS interested in the pooled output of the final text encoder
                pooled_prompt_embeds = prompt_embeds[0]
                prompt_embeds = prompt_embeds.hidden_states[-2]

                prompt_embeds_list.append(prompt_embeds)

            prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)  # 1, 77, 2048 (768 + 1024)

        # get unconditional embeddings for classifier free guidance
        negative_prompt_embeds = torch.zeros_like(prompt_embeds)
        negative_pooled_prompt_embeds = torch.zeros_like(pooled_prompt_embeds)

        prompt_embeds_dtype = self.unet.dtype
        prompt_embeds = prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
        bs_embed, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=prompt_embeds_dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
            bs_embed * num_images_per_prompt,
            -1,  # this is only the text_encoder_2_feature
        )
        if do_classifier_free_guidance:
            negative_pooled_prompt_embeds = negative_pooled_prompt_embeds.repeat(1, num_images_per_prompt).view(
                bs_embed * num_images_per_prompt, -1
            )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_image_latents(
        self,
        image,
        image_embeds,
        batch_size,
        num_images_per_prompt,
        dtype,
        device,
        do_classifier_free_guidance,
        generator=None,
    ):
        if image_embeds is not None:
            image_latents = image_embeds
        else:
            image = image.to(device=device, dtype=dtype)

            if image.shape[1] == 4:
                image_latents = image
            else:
                # make sure the VAE is in float32 mode, as it overflows in float16
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
                if needs_upcasting:
                    image = image.float()
                    self.upcast_vae()

                image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)

            image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        if image_latents.dtype != self.vae.dtype:
            image_latents = image_latents.to(dtype=self.vae.dtype)

        if self.is_cosxl_edit:
            image_latents = image_latents * self.vae.config.scaling_factor

        return image_latents

    def _get_add_time_ids(self, original_size, crops_coords_top_left, target_size, dtype):
        add_time_ids = list(original_size + crops_coords_top_left + target_size)

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        return add_time_ids

    def set_trainable_parameters(self, mode=True):
        for name, param in self.head.named_parameters():
            param.requires_grad = mode

    @torch.no_grad()
    def generate(
        self,
        image: PipelineImageInput = None,
        image_embeds: torch.Tensor = None,
        height: Optional[int] = 768,
        width: Optional[int] = 768,
        num_inference_steps: int = 100,
        denoising_end: Optional[float] = None,
        guidance_scale: float = 5.0,
        image_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds=None,
        pooled_prompt_embeds: Optional[torch.Tensor] = None,
        negative_pooled_prompt_embeds=None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        guidance_rescale: float = 0.0,
        original_size: Tuple[int, int] = None,
        crops_coords_top_left: Tuple[int, int] = (0, 0),
        target_size: Tuple[int, int] = None,
    ):
        # 0. Default height and width to unet
        if image_embeds is not None:
            height = image_embeds.shape[2] * self.vae_scale_factor
            width = image_embeds.shape[3] * self.vae_scale_factor
        elif image is not None:
            height = image.shape[2]
            width = image.shape[3]
        else:
            height = height or self.default_sample_size * self.vae_scale_factor
            width = width or self.default_sample_size * self.vae_scale_factor

        original_size = original_size or (height, width)
        target_size = target_size or (height, width)

        batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        # 2. Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        # 3. Prepare Image latents
        if image is not None or image_embeds is not None:
            image_latents = self.prepare_image_latents(
                image,
                image_embeds,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                self.device,
                do_classifier_free_guidance,
            )
        else:
            image_latents = None

        # 4. Prepare latent variables
        num_channels_latents = self.vae.config.latent_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            self.device,
            generator,
            latents,
        )

        # 5. Check that shapes of latents and image match the UNet channels
        if image_latents is not None:
            num_channels_image = image_latents.shape[1]
            if num_channels_latents + num_channels_image != self.unet.config.in_channels:
                raise ValueError(
                    f"Incorrect configuration settings! The config of `pipeline.unet`: {self.unet.config} expects"
                    f" {self.unet.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                    f" `num_channels_image`: {num_channels_image} "
                    f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                    " `pipeline.unet` or your `image` input."
                )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = {"generator": None}

        # 7. Prepare added time ids & embeddings
        add_text_embeds = pooled_prompt_embeds

        add_time_ids = self._get_add_time_ids(
            original_size,
            crops_coords_top_left,
            target_size,
            dtype=prompt_embeds.dtype,
        )

        if do_classifier_free_guidance:
            # The extra concat similar to how it's done in SD InstructPix2Pix.
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0)
            add_text_embeds = torch.cat(
                [add_text_embeds, negative_pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0
            )
            add_time_ids = torch.cat([add_time_ids, add_time_ids, add_time_ids], dim=0)

        prompt_embeds = prompt_embeds.to(self.device)
        add_text_embeds = add_text_embeds.to(self.device)
        add_time_ids = add_time_ids.to(self.device).repeat(batch_size * num_images_per_prompt, 1)

        # 8. Denoising loop
        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance
            # is applied for both the text and the input image.
            latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents

            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)
            if image_latents is not None:
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, image_latents], dim=1)

            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                scaled_latent_model_input,
                t,
                encoder_hidden_states=prompt_embeds,
                cross_attention_kwargs=cross_attention_kwargs,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]

            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred.chunk(3)
                noise_pred = (
                    noise_pred_uncond
                    + guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
                )

            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.noise_scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)

        if not output_type == "latent":
            # make sure the VAE is in float32 mode, as it overflows in float16
            needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

            # unscale/denormalize the latents
            # denormalize with the mean and std if available and not None
            latents = latents / self.vae.config.scaling_factor
            image = self.vae.decode(latents, return_dict=False)[0]

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            return StableDiffusionXLPipelineOutput(images=image)
        else:
            return StableDiffusionXLPipelineOutput(images=latents)

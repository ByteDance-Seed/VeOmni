from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel
from diffusers.image_processor import PipelineImageInput
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from diffusers.models.downsampling import Downsample2D
from diffusers.models.upsampling import Upsample2D
from diffusers.pipelines.stable_diffusion_3.pipeline_output import StableDiffusion3PipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from torch.nn import MSELoss
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from .....utils import logging
from ....seed_omni.projector import build_feature_projector
from .configuration_ultra_edit import UltraEditConfig


logger = logging.get_logger(__name__)


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


@dataclass
class UltraEditOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None


class ConvGenerationHead(nn.Module):
    def __init__(
        self,
        feature_size: int,
        hidden_size: int,
        output_size: int,
        kernel_size1: int = 14,
        kernel_size2: int = 2,
        restore_size: int = 512,
    ) -> None:
        super().__init__()
        self.proj = build_feature_projector(feature_size, hidden_size * 4)
        self.kernel_size1 = kernel_size1
        self.kernel_size2 = kernel_size2
        self.restore_size = restore_size
        self.hidden_size = hidden_size
        self.block_out_channels = [8, 16, 32, 64]
        self.upsampler = Upsample2D(hidden_size, use_conv=True, out_channels=self.block_out_channels[0])
        blocks = []
        for index in range(1, len(self.block_out_channels)):
            in_channel = self.block_out_channels[index - 1]
            out_channel = self.block_out_channels[index]
            blocks.append(nn.GroupNorm(num_groups=in_channel // 4, num_channels=in_channel, eps=1e-6))
            blocks.append(nn.SiLU())
            blocks.append(Downsample2D(in_channel, use_conv=True, out_channels=out_channel))
        blocks += [
            nn.GroupNorm(
                num_groups=self.block_out_channels[-1] // 4, num_channels=self.block_out_channels[-1], eps=1e-6
            ),
            nn.SiLU(),
            nn.Conv2d(self.block_out_channels[-1], output_size * 2, kernel_size=3, padding=1),
        ]
        self.downsampler = nn.Sequential(*blocks)

    def forward(self, x, image_grid_thw):
        x = self.proj(x)
        x = x.view(-1, self.hidden_size)
        x = x.transpose(1, 0).reshape(x.shape[0], -1, 1, 1)
        x = self.upsampler(x, output_size=(self.kernel_size1, self.kernel_size1))
        cu_index, restored_x = 0, []
        for grid_thw in image_grid_thw:
            h_factor = grid_thw[1]
            w_factor = grid_thw[2]
            token_num = h_factor * w_factor
            latent_x = x[cu_index : cu_index + token_num].reshape(
                h_factor // 2, w_factor // 2, x.shape[1], x.shape[2] * 2, x.shape[3] * 2
            )
            cu_index += token_num
            latent_x = latent_x.permute(2, 0, 3, 1, 4).reshape(
                1, x.shape[1], h_factor * self.kernel_size1, w_factor * self.kernel_size1
            )
            pad_w = (self.restore_size - w_factor * self.kernel_size1) // 2
            pad_h = (self.restore_size - h_factor * self.kernel_size1) // 2
            latent_x = F.pad(latent_x, (pad_w, pad_w, pad_h, pad_h), mode="constant", value=0)
            restored_x.append(latent_x)

        restored_x = torch.cat(restored_x, dim=0)
        latent = self.downsampler(restored_x)
        posterior = DiagonalGaussianDistribution(latent)
        kl_loss = posterior.kl()
        latent = posterior.mode()
        return latent, kl_loss.mean()


class UltraEdit(PreTrainedModel):
    config_class = UltraEditConfig
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.ConvTranspose2d, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # init.xavier_uniform(module.weight.data)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            # init.xavier_uniform(module.weight.data)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.GroupNorm):
            if module.weight is not None:
                module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.zero_()

    def __init__(self, config: UltraEditConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        if config.transformer_config is not None:
            self.transformer = SD3Transformer2DModel.from_config(config.transformer_config)
            self.vae = AutoencoderKL.from_config(config.vae_config)
            self.noise_scheduler = FlowMatchEulerDiscreteScheduler.from_config(config.scheduler_config)
        else:
            self.transformer, self.vae, self.noise_scheduler = None, None, None
            logger.info_rank0("Make sure you are converting checkpoints instead of training/inferencing.")

        if config.output_size is not None:
            self.gen_head = build_feature_projector(config.output_size, config.output_size)
            self.proj_head = ConvGenerationHead(
                config.output_size,
                config.proj_hidden_dim,
                config.condition_dim,
                kernel_size1=config.patch_size1,
                kernel_size2=config.patch_size2,
                restore_size=512,
            )
        else:
            self.gen_head = nn.Identity()
            self.proj_head = nn.Identity()

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if self.vae is not None else 8
        self.default_sample_size = self.transformer.config.sample_size if self.transformer is not None else 128
        self.dummy_text_embeds = nn.Parameter(torch.zeros(1, 154, 4096), requires_grad=False)
        self.dummy_pooled_text_embeds = nn.Parameter(torch.zeros(1, 2048), requires_grad=False)
        # self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        if latents is not None:
            return latents.to(device=device, dtype=dtype)
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
        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        text_encoder=None,
        tokenizer=None,
    ):
        if text_input_ids is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=77,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            text_input_ids = text_inputs.input_ids
        batch_size = text_input_ids.shape[0]
        prompt_embeds = text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape

        # duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        text_input_ids: Optional[torch.Tensor] = None,
        num_images_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        clip_skip: Optional[int] = None,
        text_encoder=None,
        tokenizer=None,
    ):
        if text_input_ids is None:
            prompt = [prompt] if isinstance(prompt, str) else prompt

            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )

            text_input_ids = text_inputs.input_ids
        batch_size = text_input_ids.shape[0]
        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
        pooled_prompt_embeds = prompt_embeds[0]

        if clip_skip is None:
            prompt_embeds = prompt_embeds.hidden_states[-2]
        else:
            prompt_embeds = prompt_embeds.hidden_states[-(clip_skip + 2)]

        prompt_embeds = prompt_embeds.to(dtype=text_encoder.dtype, device=device)

        _, seq_len, _ = prompt_embeds.shape
        # duplicate text embeddings for each generation per prompt, using mps friendly method
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        pooled_prompt_embeds = pooled_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        pooled_prompt_embeds = pooled_prompt_embeds.view(batch_size * num_images_per_prompt, -1)

        return prompt_embeds, pooled_prompt_embeds

    def encode_prompt(
        self,
        prompt: Union[str, List[str]] = None,
        input_ids: Optional[List[torch.Tensor]] = [None, None, None],
        device: Optional[torch.device] = None,
        num_images_per_prompt: int = 1,
        do_classifier_free_guidance: bool = True,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        clip_skip: Optional[int] = None,
        text_encoders=[None, None, None],
        tokenizers=[None, None, None],
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt_embeds is None:
            prompt_embed, pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                text_input_ids=input_ids[0],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                text_encoder=text_encoders[0],
                tokenizer=tokenizers[0],
            )
            prompt_2_embed, pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=prompt,
                text_input_ids=input_ids[1],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=clip_skip,
                text_encoder=text_encoders[1],
                tokenizer=tokenizers[1],
            )
            clip_prompt_embeds = torch.cat([prompt_embed, prompt_2_embed], dim=-1)

            t5_prompt_embed = self._get_t5_prompt_embeds(
                prompt=prompt,
                text_input_ids=input_ids[2],
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                tokenizer=tokenizers[2],
                text_encoder=text_encoders[2],
            )

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds, (0, t5_prompt_embed.shape[-1] - clip_prompt_embeds.shape[-1])
            )

            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embed], dim=-2)
            pooled_prompt_embeds = torch.cat([pooled_prompt_embed, pooled_prompt_2_embed], dim=-1)

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt_embed, negative_pooled_prompt_embed = self._get_clip_prompt_embeds(
                prompt=[""],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                text_encoder=text_encoders[0],
                tokenizer=tokenizers[0],
            )
            negative_prompt_2_embed, negative_pooled_prompt_2_embed = self._get_clip_prompt_embeds(
                prompt=[""],
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                clip_skip=None,
                text_encoder=text_encoders[1],
                tokenizer=tokenizers[1],
            )
            negative_clip_prompt_embeds = torch.cat([negative_prompt_embed, negative_prompt_2_embed], dim=-1)

            t5_negative_prompt_embed = self._get_t5_prompt_embeds(
                prompt=[""],
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                tokenizer=tokenizers[2],
                text_encoder=text_encoders[2],
            )

            negative_clip_prompt_embeds = torch.nn.functional.pad(
                negative_clip_prompt_embeds,
                (0, t5_negative_prompt_embed.shape[-1] - negative_clip_prompt_embeds.shape[-1]),
            )

            negative_prompt_embeds = torch.cat([negative_clip_prompt_embeds, t5_negative_prompt_embed], dim=-2)
            negative_pooled_prompt_embeds = torch.cat(
                [negative_pooled_prompt_embed, negative_pooled_prompt_2_embed], dim=-1
            )

        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def prepare_image_latents(
        self, image, batch_size, num_images_per_prompt, dtype, device, do_classifier_free_guidance, generator=None
    ):
        image = image.to(device=device, dtype=dtype)
        batch_size = batch_size * num_images_per_prompt

        if image.shape[1] == self.vae.config.latent_channels:
            image_latents = image
        else:
            image_latents = retrieve_latents(self.vae.encode(image), sample_mode="argmax")
            # ? normalize image latents
            # image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor

        image_latents = torch.cat([image_latents], dim=0)

        if do_classifier_free_guidance:
            uncond_image_latents = torch.zeros_like(image_latents)
            image_latents = torch.cat([image_latents, image_latents, uncond_image_latents], dim=0)

        return image_latents

    def set_projector_trainable_only(self):
        self.requires_grad_(False)
        self.gen_head.requires_grad_(True)
        self.proj_head.requires_grad_(True)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def image_guidance_scale(self):
        return self._image_guidance_scale

    @property
    def clip_skip(self):
        return self._clip_skip

    @property
    def do_classifier_free_guidance(self):
        return self.guidance_scale > 1.0 and self.image_guidance_scale >= 1.0

    @property
    def joint_attention_kwargs(self):
        return self._joint_attention_kwargs

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def interrupt(self):
        return self._interrupt

    def get_sigmas(self, timesteps, n_dim, dtype):
        sigmas = self.noise_scheduler.sigmas.to(device=self.device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(self.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def lm_head(self, hidden_states, labels=None, features=None, image_grid_thw=None, mask=None, **kwargs):
        pred_feature = self.gen_head(hidden_states)

        loss = None
        if labels is not None:
            loss_func = MSELoss()
            mse_loss = loss_func(pred_feature, labels)

            original_image_embeds, kl_loss = self.proj_head(pred_feature, image_grid_thw)
            latents = self.vae.encode(features).latent_dist.mode()
            latents = latents * self.vae.config.scaling_factor
            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
            u = torch.normal(mean=0.0, std=0.1, size=(bsz,))
            u = torch.nn.functional.sigmoid(u)

            indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
            timesteps = self.noise_scheduler.timesteps[indices].to(self.device)

            # Add noise to the latents according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
            noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents
            concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)

            mask_embeds = self.vae.encode(mask).latent_dist.mode()
            concatenated_noisy_latents = torch.cat([concatenated_noisy_latents, mask_embeds], dim=1)

            prompt_embeds = self.dummy_text_embeds.repeat(bsz, 1, 1)
            pooled_prompt_embeds = self.dummy_pooled_text_embeds.repeat(bsz, 1)

            model_pred = self.transformer(
                hidden_states=concatenated_noisy_latents,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]

            model_pred = model_pred * (-sigmas) + noisy_model_input
            weighting = torch.ones_like(sigmas)

            target = latents
            # Get the target for loss depending on the prediction type
            diff_loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
            )
            loss = mse_loss + torch.mean(diff_loss) + kl_loss

        return UltraEditOutput(
            loss=loss,
            logits=pred_feature,
        )

    def forward(
        self,
        origin_pixel_values,
        edited_pixel_values,
        mask_pixel_values,
        prompt_embeds,
        pooled_prompt_embeds,
        **kwargs,
    ):
        latents = self.vae.encode(edited_pixel_values).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        # Sample a random timestep for each image
        # for weighting schemes where we sample timesteps non-uniformly
        # See 3.1 in the SD3 paper ($rf/lognorm(0.00,1.00)$).
        u = torch.normal(mean=0.0, std=0.1, size=(bsz,))
        u = torch.nn.functional.sigmoid(u)

        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(self.device)

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        sigmas = self.get_sigmas(timesteps, n_dim=latents.ndim, dtype=latents.dtype)
        noisy_model_input = sigmas * noise + (1.0 - sigmas) * latents

        # Get the additional image embedding for conditioning.
        # Instead of getting a diagonal Gaussian here, we simply take the mode.
        original_image_embeds = self.vae.encode(origin_pixel_values).latent_dist.mode()
        concatenated_noisy_latents = torch.cat([noisy_model_input, original_image_embeds], dim=1)

        mask_embeds = self.vae.encode(mask_pixel_values).latent_dist.mode()
        concatenated_noisy_latents = torch.cat([concatenated_noisy_latents, mask_embeds], dim=1)

        model_pred = self.transformer(
            hidden_states=concatenated_noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]

        model_pred = model_pred * (-sigmas) + noisy_model_input
        weighting = torch.ones_like(sigmas)

        target = latents
        # Get the target for loss depending on the prediction type
        loss = torch.mean(
            (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1), 1
        )

        return torch.mean(loss)

    @torch.no_grad()
    def generate(
        self,
        image: PipelineImageInput = None,
        num_inference_steps: int = 50,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        image_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_pooled_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        clip_skip: Optional[int] = None,
        **kwargs,
    ):
        self._guidance_scale = guidance_scale
        self._image_guidance_scale = image_guidance_scale
        self._clip_skip = clip_skip
        self._joint_attention_kwargs = joint_attention_kwargs
        self._interrupt = False

        batch_size = prompt_embeds.shape[0]

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 and image_guidance_scale >= 1.0

        if do_classifier_free_guidance:
            # The extra concat similar to how it's done in SD InstructPix2Pix.
            prompt_embeds = torch.cat([prompt_embeds, negative_prompt_embeds, negative_prompt_embeds], dim=0)
            pooled_prompt_embeds = torch.cat(
                [pooled_prompt_embeds, negative_pooled_prompt_embeds, negative_pooled_prompt_embeds], dim=0
            )
        # 1. Prepare timesteps
        self.noise_scheduler.set_timesteps(num_inference_steps, device=self.device)
        timesteps = self.noise_scheduler.timesteps
        # 2. Prepare Image latents
        pad_height = (512 - image.shape[2]) // 2
        pad_width = (512 - image.shape[3]) // 2
        mask_img = torch.ones_like(image)
        padding = (pad_width, pad_width, pad_height, pad_height)
        image = F.pad(image, padding, mode="constant", value=0)
        mask_img = F.pad(mask_img, padding, mode="constant", value=0)
        image_latents = self.prepare_image_latents(
            image,
            batch_size,
            num_images_per_prompt,
            prompt_embeds.dtype,
            self.device,
            do_classifier_free_guidance,
        )

        height, width = image_latents.shape[-2:]
        height = height * self.vae_scale_factor
        width = width * self.vae_scale_factor

        # 3. Prepare latent variables
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

        num_channels_image = image_latents.shape[1]
        if mask_img is not None:
            mask_img = self.image_processor.preprocess(mask_img)
            mask_image_latents = self.prepare_image_latents(
                mask_img,
                batch_size,
                num_images_per_prompt,
                prompt_embeds.dtype,
                self.device,
                self.do_classifier_free_guidance,
            )
            num_channels_image += mask_image_latents.shape[1]

        # 4. Check that shapes of latents and image match the UNet channels
        if num_channels_latents + num_channels_image != self.transformer.config.in_channels:
            raise ValueError(
                f"Incorrect configuration settings! The config of `pipeline.unet`: {self.transformer.config} expects"
                f" {self.transformer.config.in_channels} but received `num_channels_latents`: {num_channels_latents} +"
                f" `num_channels_image`: {num_channels_image} "
                f" = {num_channels_latents + num_channels_image}. Please verify the config of"
                " `pipeline.unet` or your `image` input."
            )

        # 5. Denoising loop
        for i, t in enumerate(timesteps):
            # Expand the latents if we are doing classifier free guidance.
            # The latents are expanded 3 times because for pix2pix the guidance
            # is applied for both the text and the input image.
            latent_model_input = torch.cat([latents] * 3) if do_classifier_free_guidance else latents
            timestep = t.expand(latent_model_input.shape[0])

            # concat latents, image_latents in the channel dimension
            scaled_latent_model_input = torch.cat([latent_model_input, image_latents], dim=1)
            if mask_img is not None:
                scaled_latent_model_input = torch.cat([scaled_latent_model_input, mask_image_latents], dim=1)
            # predict the noise residual
            noise_pred = self.transformer(
                scaled_latent_model_input,
                timestep=timestep,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                joint_attention_kwargs=self.joint_attention_kwargs,
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

            # compute the previous noisy sample x_t -> x_t-1
            latents_dtype = latents.dtype
            latents = self.noise_scheduler.step(noise_pred, t, latents, return_dict=False)[0]
            if latents.dtype != latents_dtype:
                if torch.backends.mps.is_available():
                    # some platforms (eg. apple mps) misbehave due to a pytorch bug: https://github.com/pytorch/pytorch/pull/99272
                    latents = latents.to(latents_dtype)
        if not output_type == "latent":
            latents = latents / self.vae.config.scaling_factor
            latent_height = pad_height // self.vae_scale_factor
            latent_width = pad_width // self.vae_scale_factor
            latents = latents[:, :, latent_height:-latent_height, latent_width:-latent_width]
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            return StableDiffusion3PipelineOutput(images=image)
        else:
            return StableDiffusion3PipelineOutput(images=latents)

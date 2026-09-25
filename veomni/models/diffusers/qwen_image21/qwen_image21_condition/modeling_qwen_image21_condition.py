from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler, QwenImage21Pipeline
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchvision.transforms import InterpolationMode, functional
from transformers import PreTrainedModel, Qwen3VLForConditionalGeneration, Qwen3VLProcessor

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .....utils.device import get_device_type
from .configuration_qwen_image21_condition import QwenImage21ConditionModelConfig


logger = logging.get_logger(__name__)


class QwenImage21ConditionModel(PreTrainedModel):
    config_class = QwenImage21ConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: QwenImage21ConditionModelConfig, meta_init: bool = False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.processor = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self._prompt_pipeline = None
        self.meta_init = meta_init
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed((config.seed or 0) + get_parallel_state().dp_rank)
        self._load_components()

    @property
    def _execution_device(self):
        if self.vae is not None:
            return self.vae.device
        if self.text_encoder is not None:
            return self.text_encoder.device
        return torch.device(get_device_type())

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Qwen-Image-2.1 condition components from {base}.")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base, subfolder=self.config.scheduler_subfolder
        )
        if self.meta_init:
            return

        self.processor = Qwen3VLProcessor.from_pretrained(base, subfolder=self.config.processor_subfolder)
        self.text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            base,
            subfolder=self.config.vae_subfolder,
            torch_dtype=torch.bfloat16,
        )

    def _get_prompt_pipeline(self):
        if self._prompt_pipeline is None:
            self._prompt_pipeline = QwenImage21Pipeline(
                scheduler=self.scheduler,
                vae=self.vae,
                text_encoder=self.text_encoder,
                processor=self.processor,
                transformer=None,
            )
        return self._prompt_pipeline

    @torch.no_grad()
    def encode_prompt(self, prompt: str | list[str], device: torch.device | None = None):
        prompt_embeds, prompt_embeds_mask, image_pad_mask = self._get_prompt_pipeline().encode_prompt(
            prompt=prompt,
            device=device or self._execution_device,
        )
        if image_pad_mask.any():
            raise ValueError("Qwen-Image-2.1 text-to-image training does not accept condition-image tokens.")
        return prompt_embeds, prompt_embeds_mask

    def _image_to_tensor(self, image) -> torch.Tensor:
        image = image.convert("RGBA")
        image = functional.resize(
            image,
            [self.config.height, self.config.width],
            interpolation=InterpolationMode.BICUBIC,
        )
        return functional.to_tensor(image).unsqueeze(0).unsqueeze(2).mul(2.0).sub(1.0)

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            return [] if length is None else [None] * length
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, channels, frames, height, width = latents.shape
        if frames != 1:
            raise ValueError(f"Qwen-Image-2.1 text-to-image training expects one latent frame, got {frames}.")
        return latents[:, :, 0].reshape(batch_size, channels, height * width).transpose(1, 2)

    def _calculate_shift(self, image_seq_len: int) -> float:
        scheduler_config = self.scheduler.config
        base_seq_len = scheduler_config.base_image_seq_len
        max_seq_len = scheduler_config.max_image_seq_len
        base_shift = scheduler_config.base_shift
        max_shift = scheduler_config.max_shift
        slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        return image_seq_len * slope + base_shift - slope * base_seq_len

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        return (latents - mean) / std

    def _encode_image_to_latents(self, image):
        image_tensor = self._image_to_tensor(image).to(device=self.vae.device, dtype=self.vae.dtype)
        posterior = self.vae.encode(image_tensor).latent_dist
        parameters = posterior.parameters
        latent_height, latent_width = parameters.shape[-2:]
        return parameters, [(1, int(latent_height), int(latent_width))]

    @torch.no_grad()
    def get_condition(self, inputs, images, **kwargs) -> dict[str, Any]:
        prompts = inputs if isinstance(inputs, list) else [inputs]
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompts)

        latents = []
        img_shapes = []
        for sample_images in images:
            sample_images = self._as_list(sample_images)
            if len(sample_images) != 1:
                raise ValueError("Qwen-Image-2.1 text-to-image training expects one target image per sample.")
            sample_latents, sample_shapes = self._encode_image_to_latents(sample_images[0])
            latents.append(sample_latents)
            img_shapes.append(sample_shapes)

        return {
            "latents": latents,
            "encoder_hidden_states": [prompt_embeds[index : index + 1] for index in range(len(prompts))],
            "encoder_hidden_states_mask": (
                [None] * len(prompts)
                if prompt_embeds_mask is None
                else [prompt_embeds_mask[index : index + 1] for index in range(len(prompts))]
            ),
            "img_shapes": img_shapes,
        }

    def rng_state_dict(self):
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state):
        self.generator.set_state(state["generator"])

    def process_condition(
        self,
        latents=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        img_shapes=None,
        **kwargs,
    ) -> dict[str, Any]:
        if "hidden_states" in kwargs and "training_target" in kwargs and "img_mask" in kwargs:
            ready_inputs = dict(kwargs)
            if latents is not None:
                ready_inputs["latents"] = latents
            if encoder_hidden_states is not None:
                ready_inputs["encoder_hidden_states"] = encoder_hidden_states
            if encoder_hidden_states_mask is not None:
                ready_inputs["encoder_hidden_states_mask"] = encoder_hidden_states_mask
            if img_shapes is not None:
                ready_inputs["img_shapes"] = img_shapes
            return ready_inputs
        if latents is None or encoder_hidden_states is None or img_shapes is None:
            raise ValueError(
                "Qwen-Image-2.1 condition processing requires latents, prompt embeddings, and img_shapes."
            )
        if self.vae is None:
            raise NotImplementedError(
                "Qwen-Image-2.1 currently supports online training only; offline training needs cached VAE metadata."
            )

        latents_list = self._as_list(latents)
        sample_count = len(latents_list)
        encoder_list = self._as_list(encoder_hidden_states, sample_count)
        encoder_mask_list = self._as_list(encoder_hidden_states_mask, sample_count)
        img_shapes_list = self._as_list(img_shapes, sample_count)
        packed = {
            "hidden_states": [],
            "encoder_hidden_states": [],
            "encoder_hidden_states_mask": [],
            "timestep": [],
            "img_shapes": [],
            "img_mask": [],
            "training_target": [],
            "latents": [],
        }

        for parameters, context, context_mask, shapes in zip(
            latents_list, encoder_list, encoder_mask_list, img_shapes_list
        ):
            latent_dist = DiagonalGaussianDistribution(parameters)
            clean_latents = self._normalize_latents(latent_dist.sample(generator=self.generator))
            clean_latents = clean_latents.to(self.generator.device)
            packed_clean = self._pack_latents(clean_latents)
            noise = torch.randn(
                clean_latents.shape,
                dtype=clean_latents.dtype,
                device=clean_latents.device,
                generator=self.generator,
            )
            num_train_timesteps = self.scheduler.config.num_train_timesteps
            self.scheduler.set_timesteps(
                num_train_timesteps,
                device=clean_latents.device,
                mu=self._calculate_shift(packed_clean.shape[1]),
            )
            timestep_index = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (clean_latents.shape[0],),
                device=clean_latents.device,
                generator=self.generator,
            )
            timestep = self.scheduler.timesteps[timestep_index.to(self.scheduler.timesteps.device)].to(
                clean_latents.device
            )
            noisy_latents = self.scheduler.scale_noise(clean_latents, timestep, noise)
            target = noise - clean_latents

            packed_noisy = self._pack_latents(noisy_latents)
            packed_target = self._pack_latents(target)
            target_slots, remainder = divmod(packed_noisy.shape[1], 4)
            if remainder:
                raise ValueError("Qwen-Image-2.1 target latent token count must be divisible by four.")
            img_mask = torch.cat(
                [
                    torch.zeros(packed_noisy.shape[0], context.shape[1], dtype=torch.bool, device=packed_noisy.device),
                    torch.ones(packed_noisy.shape[0], target_slots, dtype=torch.bool, device=packed_noisy.device),
                ],
                dim=1,
            )

            packed["hidden_states"].append(packed_noisy)
            packed["encoder_hidden_states"].append(context.to(packed_noisy.device))
            packed["encoder_hidden_states_mask"].append(
                None if context_mask is None else context_mask.to(packed_noisy.device)
            )
            packed["timestep"].append(timestep.to(dtype=packed_noisy.dtype) / 1000)
            packed["img_shapes"].append(shapes)
            packed["img_mask"].append(img_mask)
            packed["training_target"].append(packed_target)
            packed["latents"].append(packed_clean)

        return packed

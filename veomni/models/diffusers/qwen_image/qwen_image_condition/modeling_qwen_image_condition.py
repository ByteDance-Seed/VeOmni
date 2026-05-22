from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchvision.transforms import InterpolationMode, functional
from transformers import PreTrainedModel, Qwen2_5_VLForConditionalGeneration, Qwen2Tokenizer

from .....distributed.parallel_state import get_parallel_state
from .....utils import logging
from .....utils.device import get_device_type
from .configuration_qwen_image_condition import QwenImageConditionModelConfig


logger = logging.get_logger(__name__)


class QwenImageConditionModel(PreTrainedModel):
    config_class = QwenImageConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: QwenImageConditionModelConfig, meta_init=False, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.tokenizer = None
        self.text_encoder = None
        self.vae = None
        self.scheduler = None
        self._timesteps_ready = False
        self._timesteps_image_seq_len: int | None = None
        self.meta_init = meta_init
        self.seed = config.seed
        self.generator = torch.Generator(device=torch.device(get_device_type()))
        self.generator.manual_seed((self.seed or 0) + get_parallel_state().dp_rank)
        self._load_components()

    @staticmethod
    def _calculate_shift(
        image_seq_len: int,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.15,
    ) -> float:
        if max_seq_len == base_seq_len:
            raise ValueError(
                "FlowMatchEulerDiscreteScheduler config has equal base_image_seq_len and "
                f"max_image_seq_len (={base_seq_len}); cannot derive dynamic-shift `mu`."
            )
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        return image_seq_len * m + b

    @property
    def _execution_device(self):
        if self.vae is not None:
            return self.vae.device
        if self.text_encoder is not None:
            return self.text_encoder.device
        return torch.device(get_device_type())

    def _load_components(self):
        base = self.config.base_model_path
        logger.info_rank0(f"Loading Qwen-Image condition components from {base}.")
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            base,
            subfolder=self.config.scheduler_subfolder,
        )
        if self.meta_init:
            return

        self.tokenizer = Qwen2Tokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            torch_dtype=torch.bfloat16,
        )
        self.vae = AutoencoderKLQwenImage.from_pretrained(
            base,
            subfolder=self.config.vae_subfolder,
            torch_dtype=torch.float32,
        )

    @staticmethod
    def _as_list(value: Any, length: int | None = None) -> list[Any]:
        if value is None:
            if length is None:
                return []
            return [None] * length
        if isinstance(value, list):
            return value
        return [value]

    @staticmethod
    def _pack_latents(latents: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels_latents, _num_frames, height, width = latents.shape
        latents = latents[:, :, 0]
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        latents_std = torch.tensor(self.vae.config.latents_std, device=latents.device, dtype=latents.dtype).view(
            1, self.vae.config.z_dim, 1, 1, 1
        )
        return (latents - latents_mean) / latents_std

    def _image_to_tensor(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        image = functional.resize(
            image,
            [self.config.height, self.config.width],
            interpolation=InterpolationMode.BICUBIC,
        )
        image = functional.to_tensor(image).unsqueeze(0).unsqueeze(2)
        return image.mul(2.0).sub(1.0)

    def _extract_masked_hidden(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        bool_mask = mask.bool()
        valid_lengths = bool_mask.sum(dim=1)
        selected = hidden_states[bool_mask]
        return torch.split(selected, valid_lengths.tolist(), dim=0)

    @torch.no_grad()
    def _get_qwen_prompt_embeds(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt

        template = self.config.prompt_template_encode
        drop_idx = self.config.prompt_template_encode_start_idx
        txt = [template.format(item) for item in prompt]
        txt_tokens = self.tokenizer(
            txt,
            max_length=self.config.max_sequence_length + drop_idx,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        encoder_hidden_states = self.text_encoder(
            input_ids=txt_tokens.input_ids,
            attention_mask=txt_tokens.attention_mask,
            output_hidden_states=True,
        )
        hidden_states = encoder_hidden_states.hidden_states[-1]
        split_hidden_states = self._extract_masked_hidden(hidden_states, txt_tokens.attention_mask)
        split_hidden_states = [item[drop_idx:] for item in split_hidden_states]
        attn_mask_list = [
            torch.ones(item.size(0), dtype=torch.long, device=item.device) for item in split_hidden_states
        ]
        max_seq_len = max(item.size(0) for item in split_hidden_states)
        prompt_embeds = torch.stack(
            [
                torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))])
                for item in split_hidden_states
            ]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in attn_mask_list]
        )
        return prompt_embeds.to(dtype=dtype, device=device), encoder_attention_mask

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: str | list[str],
        device: torch.device | None = None,
        prompt_embeds: torch.Tensor | None = None,
        prompt_embeds_mask: torch.Tensor | None = None,
    ):
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_embeds_mask = self._get_qwen_prompt_embeds(prompt, device)

        prompt_embeds = prompt_embeds[:, : self.config.max_sequence_length]
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        if prompt_embeds_mask is not None:
            prompt_embeds_mask = prompt_embeds_mask[:, : self.config.max_sequence_length]
            prompt_embeds_mask = prompt_embeds_mask.view(batch_size, seq_len)
            if bool(prompt_embeds_mask.all()):
                prompt_embeds_mask = None

        return prompt_embeds, prompt_embeds_mask

    def _encode_image_to_latents(self, image) -> tuple[torch.Tensor, list[tuple[int, int, int]]]:
        # Return the raw posterior parameters ([1, 2*z_dim, F, H, W]) alongside the
        # packed image grid ``[(1, H // 2, W // 2)]`` so downstream callers can resample a
        # fresh latent on every training step (mode() reduces diversity) without
        # re-deriving the grid from the latent shape.
        image_tensor = self._image_to_tensor(image).to(device=self.vae.device, dtype=self.vae.dtype)
        posterior: DiagonalGaussianDistribution = self.vae.encode(image_tensor).latent_dist
        parameters = posterior.parameters
        latent_height, latent_width = int(parameters.shape[-2]), int(parameters.shape[-1])
        return parameters, [(1, latent_height // 2, latent_width // 2)]

    @torch.no_grad()
    def get_condition(self, inputs, images, **kwargs) -> dict[str, Any]:
        prompts = inputs if isinstance(inputs, list) else [inputs]
        prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt=prompts)

        latents_list = []
        img_shapes_list = []
        for sample_images in images:
            sample_images = self._as_list(sample_images)
            if len(sample_images) != 1:
                raise ValueError("Qwen-Image text-to-image training expects exactly one target image per sample.")
            sample_params, sample_img_shapes = self._encode_image_to_latents(sample_images[0])
            latents_list.append(sample_params)
            img_shapes_list.append(sample_img_shapes)

        encoder_hidden_states = [prompt_embeds[idx : idx + 1] for idx in range(len(prompts))]
        if prompt_embeds_mask is None:
            encoder_hidden_states_mask = [None] * len(prompts)
        else:
            encoder_hidden_states_mask = [prompt_embeds_mask[idx : idx + 1] for idx in range(len(prompts))]

        return {
            "latents": latents_list,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "img_shapes": img_shapes_list,
        }

    def process_condition(
        self,
        latents=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        img_shapes=None,
        **kwargs,
    ) -> dict[str, Any]:
        if "hidden_states" in kwargs and "training_target" in kwargs:
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
                "Qwen-Image condition processing requires latents (raw VAE posterior parameters "
                "from ``get_condition``), encoder hidden states, and img_shapes."
            )

        latents_list = self._as_list(latents)
        encoder_hidden_states_list = self._as_list(encoder_hidden_states, len(latents_list))
        encoder_hidden_states_mask_list = self._as_list(encoder_hidden_states_mask, len(latents_list))
        img_shapes_list = self._as_list(img_shapes, len(latents_list))

        def _seq_len(grid: list[tuple[int, int, int]]) -> int:
            return sum(f * h * w for f, h, w in grid)

        image_seq_len = _seq_len(img_shapes_list[0])
        for idx, sample_img_shapes in enumerate(img_shapes_list[1:], start=1):
            sample_seq_len = _seq_len(sample_img_shapes)
            if sample_seq_len != image_seq_len:
                # ``mu`` shifts the FlowMatchEulerDiscreteScheduler schedule based on the
                # packed image sequence length; a single ``set_timesteps`` call therefore
                # encodes one resolution. Mixing packed lengths in one micro-batch would
                # silently train later samples against the wrong schedule.
                raise ValueError(
                    "Qwen-Image condition expects all samples in a micro-batch to share the "
                    f"same packed image sequence length; got {image_seq_len} at index 0 and "
                    f"{sample_seq_len} at index {idx}."
                )
        if (not self._timesteps_ready) or (self._timesteps_image_seq_len != image_seq_len):
            scheduler_cfg = self.scheduler.config
            mu = self._calculate_shift(
                image_seq_len,
                scheduler_cfg.get("base_image_seq_len", 256),
                scheduler_cfg.get("max_image_seq_len", 4096),
                scheduler_cfg.get("base_shift", 0.5),
                scheduler_cfg.get("max_shift", 1.15),
            )
            self.scheduler.set_timesteps(
                self.config.num_train_timesteps,
                device=self.generator.device,
                mu=mu,
            )
            self._timesteps_ready = True
            self._timesteps_image_seq_len = image_seq_len

        packed_conditions: dict[str, list[Any]] = {
            "hidden_states": [],
            "timestep": [],
            "encoder_hidden_states": [],
            "encoder_hidden_states_mask": [],
            "img_shapes": [],
            "training_target": [],
            "latents": [],
        }
        if self.config.guidance_scale is not None:
            packed_conditions["guidance"] = []

        for sample_params, sample_context, sample_context_mask, sample_img_shapes in zip(
            latents_list, encoder_hidden_states_list, encoder_hidden_states_mask_list, img_shapes_list
        ):
            sample_latents_raw = DiagonalGaussianDistribution(sample_params).mode()
            sample_latents_norm = self._normalize_latents(sample_latents_raw).to(self.generator.device)
            sample_latents = self._pack_latents(sample_latents_norm)

            noise = torch.randn(
                sample_latents.shape,
                dtype=sample_latents.dtype,
                device=self.generator.device,
                generator=self.generator,
            )
            timestep_ids = torch.randint(
                0,
                len(self.scheduler.timesteps),
                (sample_latents.shape[0],),
                device=self.generator.device,
                generator=self.generator,
            ).to(sample_latents.device)
            timestep = self.scheduler.timesteps[timestep_ids].to(
                device=sample_latents.device, dtype=sample_latents.dtype
            )
            noisy_latents = self.scheduler.scale_noise(sample_latents, timestep, noise)
            training_target = noise - sample_latents

            packed_conditions["hidden_states"].append(noisy_latents)
            packed_conditions["timestep"].append(timestep / 1000)
            packed_conditions["encoder_hidden_states"].append(sample_context.to(sample_latents.device))
            if sample_context_mask is None:
                packed_conditions["encoder_hidden_states_mask"].append(None)
            else:
                packed_conditions["encoder_hidden_states_mask"].append(sample_context_mask.to(sample_latents.device))
            packed_conditions["img_shapes"].append(sample_img_shapes)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(sample_latents)
            if self.config.guidance_scale is not None:
                guidance = torch.full(
                    [sample_latents.shape[0]],
                    self.config.guidance_scale,
                    device=sample_latents.device,
                    dtype=torch.float32,
                )
                packed_conditions["guidance"].append(guidance)

        return packed_conditions

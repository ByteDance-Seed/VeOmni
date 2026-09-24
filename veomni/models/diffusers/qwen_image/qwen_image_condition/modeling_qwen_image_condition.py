from __future__ import annotations

from typing import Any

import torch
from diffusers import AutoencoderKLQwenImage, FlowMatchEulerDiscreteScheduler
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from torchvision.transforms import InterpolationMode, functional
from transformers import PreTrainedModel, Qwen2Tokenizer, Qwen2VLProcessor, Qwen2_5_VLForConditionalGeneration

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
        self.processor = None
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
        # Offline training needs normalization metadata, but no VAE weights.
        self.vae_config = AutoencoderKLQwenImage.load_config(base, subfolder=self.config.vae_subfolder)
        spatial_factor = 2 ** len(self.vae_config["temperal_downsample"])
        if self.config.height % (2 * spatial_factor) or self.config.width % (2 * spatial_factor):
            raise ValueError(f"Qwen-Image height and width must be divisible by {2 * spatial_factor}.")
        if self.config.training_recipe == "diffsynth":
            # DiffSynth's Qwen-Image SFT recipe uses a fixed exponential shift
            # and terminal sigma, independent of the inference resolution.
            self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(
                self.scheduler.config,
                num_train_timesteps=1000,
                use_dynamic_shifting=True,
                time_shift_type="exponential",
                shift_terminal=0.02,
                invert_sigmas=False,
                use_karras_sigmas=False,
                use_exponential_sigmas=False,
                use_beta_sigmas=False,
            )
        if self.meta_init:
            return

        self.tokenizer = Qwen2Tokenizer.from_pretrained(base, subfolder=self.config.tokenizer_subfolder)
        if self.config.enable_edit:
            if not self.config.processor_path:
                raise ValueError("Qwen-Image edit training requires `processor_path` for the multimodal processor.")
            logger.info_rank0(f"Loading Qwen-Image edit processor from {self.config.processor_path}.")
            self.processor = Qwen2VLProcessor.from_pretrained(self.config.processor_path)
        text_encoder_kwargs = {"torch_dtype": torch.bfloat16}
        if self.config.text_encoder_attn_implementation:
            text_encoder_kwargs["attn_implementation"] = self.config.text_encoder_attn_implementation
        self.text_encoder = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base,
            subfolder=self.config.text_encoder_subfolder,
            **text_encoder_kwargs,
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
        if latents.ndim != 5 or latents.shape[2] != 1 or any(dim % 2 for dim in latents.shape[-2:]):
            raise ValueError("Qwen-Image latents must have shape [B, C, 1, H, W] with even H and W.")
        batch_size, num_channels_latents, _num_frames, height, width = latents.shape
        latents = latents[:, :, 0]
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)

    def _normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents_mean = torch.tensor(self.vae_config["latents_mean"], device=latents.device, dtype=latents.dtype).view(
            1, self.vae_config["z_dim"], 1, 1, 1
        )
        latents_std = torch.tensor(self.vae_config["latents_std"], device=latents.device, dtype=latents.dtype).view(
            1, self.vae_config["z_dim"], 1, 1, 1
        )
        return (latents - latents_mean) / latents_std

    def _image_to_tensor(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.config.image_resize_mode == "center_crop":
            width, height = image.size
            scale = max(self.config.width / width, self.config.height / height)
            image = functional.resize(
                image, (round(height * scale), round(width * scale)), interpolation=InterpolationMode.BILINEAR
            )
            image = functional.center_crop(image, (self.config.height, self.config.width))
        else:
            image = functional.resize(
                image, [self.config.height, self.config.width], interpolation=InterpolationMode.BICUBIC
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
    def _get_qwen_edit_prompt_embeds(
        self,
        prompt: str | list[str],
        edit_images_per_sample: list[list[Any]],
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        """Encode text + source images into prompt embeddings via the multimodal processor.

        ``edit_images_per_sample`` is one list of source PIL images per prompt. Each
        prompt is encoded independently because the number of source images may vary
        across samples; the per-sample embeddings are then padded and stacked. This
        mirrors DiffSynth's ``encode_prompt_edit_multi`` (multi-image editing).
        """
        if self.processor is None:
            raise ValueError("Qwen-Image edit training requires a multimodal processor.")
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if len(prompt) != len(edit_images_per_sample):
            raise ValueError("Qwen-Image edit expects one source-image list per prompt.")

        template = self.config.edit_prompt_template
        drop_idx = self.config.edit_prompt_template_start_idx
        img_prompt_template = self.config.edit_img_prompt_template

        all_embeds: list[torch.Tensor] = []
        all_masks: list[torch.Tensor] = []
        for item, edit_images in zip(prompt, edit_images_per_sample):
            edit_images = self._as_list(edit_images)
            base_img_prompt = "".join(img_prompt_template.format(i + 1) for i in range(len(edit_images)))
            txt = template.format(base_img_prompt + item)
            model_inputs = self.processor(
                text=txt,
                images=edit_images,
                padding=True,
                return_tensors="pt",
            ).to(device)
            encoder_hidden_states = self.text_encoder(
                input_ids=model_inputs.input_ids,
                attention_mask=model_inputs.attention_mask,
                pixel_values=model_inputs.pixel_values,
                image_grid_thw=model_inputs.image_grid_thw,
                output_hidden_states=True,
            )
            hidden_states = encoder_hidden_states.hidden_states[-1]
            split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
            split_hidden_states = [item[drop_idx:] for item in split_hidden_states]
            single_embeds = split_hidden_states[0]
            all_embeds.append(single_embeds)
            all_masks.append(torch.ones(single_embeds.size(0), dtype=torch.long, device=single_embeds.device))

        max_seq_len = max(item.size(0) for item in all_embeds)
        prompt_embeds = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0), item.size(1))]) for item in all_embeds]
        )
        encoder_attention_mask = torch.stack(
            [torch.cat([item, item.new_zeros(max_seq_len - item.size(0))]) for item in all_masks]
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
        # packed image grid. Both online and offline training take posterior.mode()
        # before normalization, matching the DiffSynth Qwen-Image VAE convention.
        image_tensor = self._image_to_tensor(image).to(device=self.vae.device, dtype=self.vae.dtype)
        posterior: DiagonalGaussianDistribution = self.vae.encode(image_tensor).latent_dist
        parameters = posterior.parameters
        latent_height, latent_width = int(parameters.shape[-2]), int(parameters.shape[-1])
        return parameters, [(1, latent_height // 2, latent_width // 2)]

    @torch.no_grad()
    def get_condition(self, inputs, images, **kwargs) -> dict[str, Any]:
        prompts = inputs if isinstance(inputs, list) else [inputs]
        if not prompts or len(prompts) != len(images) or not all(isinstance(prompt, str) for prompt in prompts):
            raise ValueError("Qwen-Image expects equally sized, nonempty lists of text prompts and target images.")

        outputs_list = kwargs.get("outputs") or []
        edit_images_per_sample = None
        if self.config.enable_edit:
            edit_images_per_sample = [
                sample_outputs.get("edit_images", []) for sample_outputs in outputs_list
            ]
            if len(edit_images_per_sample) != len(prompts):
                raise ValueError("Qwen-Image edit training requires one source-image list per sample.")
            prompt_embeds, prompt_embeds_mask = self._get_qwen_edit_prompt_embeds(
                prompts, edit_images_per_sample
            )
        else:
            prompt_embeds, prompt_embeds_mask = self.encode_prompt(prompt=prompts)

        latents_list = []
        img_shapes_list = []
        for sample_images in images:
            sample_images = self._as_list(sample_images)
            if len(sample_images) != 1:
                raise ValueError("Qwen-Image training expects exactly one target image per sample.")
            sample_params, sample_img_shapes = self._encode_image_to_latents(sample_images[0])
            latents_list.append(sample_params)
            img_shapes_list.append(sample_img_shapes)

        encoder_hidden_states = [prompt_embeds[idx : idx + 1] for idx in range(len(prompts))]
        if prompt_embeds_mask is None:
            encoder_hidden_states_mask = [None] * len(prompts)
        else:
            encoder_hidden_states_mask = [prompt_embeds_mask[idx : idx + 1] for idx in range(len(prompts))]

        condition = {
            "latents": latents_list,
            "encoder_hidden_states": encoder_hidden_states,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "img_shapes": img_shapes_list,
        }

        if self.config.enable_edit:
            edit_latents_list = []
            edit_img_shapes_list = []
            for sample_edit_images in edit_images_per_sample:
                sample_edit_images = self._as_list(sample_edit_images)
                sample_edit_params = []
                sample_edit_shapes = []
                for edit_image in sample_edit_images:
                    params, shapes = self._encode_image_to_latents(edit_image)
                    sample_edit_params.append(params)
                    sample_edit_shapes.append(shapes[0])
                edit_latents_list.append(sample_edit_params)
                edit_img_shapes_list.append(sample_edit_shapes)
            condition["edit_latents"] = edit_latents_list
            condition["edit_img_shapes"] = edit_img_shapes_list

        return condition

    def rng_state_dict(self) -> dict[str, torch.Tensor]:
        """Snapshot the noise/timestep generator for the job-level checkpoint.

        ``process_condition`` draws both the noise and the timestep ids from
        ``self.generator``, so a resumed run that does not restore it replays the
        stream from the construction-time seed instead of continuing it.
        """
        return {"generator": self.generator.get_state()}

    def load_rng_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore a snapshot produced by :meth:`rng_state_dict`."""
        self.generator.set_state(state["generator"])

    def process_condition(
        self,
        latents=None,
        encoder_hidden_states=None,
        encoder_hidden_states_mask=None,
        img_shapes=None,
        edit_latents=None,
        edit_img_shapes=None,
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
            if edit_latents is not None:
                ready_inputs["edit_latents"] = edit_latents
            if edit_img_shapes is not None:
                ready_inputs["edit_img_shapes"] = edit_img_shapes
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
        edit_latents_list = self._as_list(edit_latents, len(latents_list))
        edit_img_shapes_list = self._as_list(edit_img_shapes, len(latents_list))
        if not latents_list or any(
            len(items) != len(latents_list)
            for items in (
                encoder_hidden_states_list,
                encoder_hidden_states_mask_list,
                img_shapes_list,
                edit_latents_list,
                edit_img_shapes_list,
            )
        ):
            raise ValueError("Qwen-Image condition fields must have the same nonzero sample count.")

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
            if self.config.training_recipe == "diffsynth":
                self.scheduler.set_timesteps(
                    sigmas=torch.linspace(1, 0, self.config.num_train_timesteps + 1)[:-1].numpy(),
                    device=self.generator.device,
                    mu=0.8,
                )
                timesteps = self.scheduler.timesteps
                weights = torch.exp(-2 * ((timesteps - 500) / 1000) ** 2)
                weights = weights - weights.min()
                weights = weights * (1000 / weights.sum())
                if len(timesteps) != 1000:
                    weights = weights * (len(timesteps) / 1000)
                    weights = weights + weights[1]
                self._training_weights = weights
            else:
                self.scheduler.set_timesteps(self.config.num_train_timesteps, device=self.generator.device, mu=mu)
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
        if self.config.training_recipe == "diffsynth":
            packed_conditions["loss_weights"] = []

        for sample_params, sample_context, sample_context_mask, sample_img_shapes, sample_edit_latents, sample_edit_img_shapes in zip(
            latents_list,
            encoder_hidden_states_list,
            encoder_hidden_states_mask_list,
            img_shapes_list,
            edit_latents_list,
            edit_img_shapes_list,
        ):
            if sample_params.ndim != 5 or sample_params.shape[1] != 2 * self.vae_config["z_dim"]:
                raise ValueError("Qwen-Image cache must contain raw VAE posterior parameters [B, 2*z_dim, 1, H, W].")
            sample_latents_raw = DiagonalGaussianDistribution(sample_params).mode()
            sample_latents_norm = self._normalize_latents(sample_latents_raw).to(self.generator.device)
            sample_latents = self._pack_latents(sample_latents_norm)
            expected_grid = [(1, sample_params.shape[-2] // 2, sample_params.shape[-1] // 2)]
            if [tuple(grid) for grid in sample_img_shapes] != expected_grid:
                raise ValueError(f"Qwen-Image img_shapes must match the packed latent grid {expected_grid}.")

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
            timestep = self.scheduler.timesteps[timestep_ids]
            # Index sigmas directly: low precision cached latents must not round
            # timesteps before the scheduler looks up their corresponding sigma.
            sigma = self.scheduler.sigmas[timestep_ids].to(sample_latents.dtype).view(-1, 1, 1)
            noisy_latents = (1 - sigma) * sample_latents + sigma * noise
            training_target = noise - sample_latents

            # Qwen-Image-Edit: concatenate clean source-image latents alongside the
            # noisy target latent. ``zero_cond_t`` in the transformer modulates the
            # source tokens with timestep 0 via ``modulate_index``.
            if sample_edit_latents is not None:
                sample_edit_latents = self._as_list(sample_edit_latents)
                edit_packed = []
                for edit_params, edit_shape in zip(sample_edit_latents, sample_edit_img_shapes):
                    if edit_params.ndim != 5 or edit_params.shape[1] != 2 * self.vae_config["z_dim"]:
                        raise ValueError(
                            "Qwen-Image edit cache must contain raw VAE posterior parameters [B, 2*z_dim, 1, H, W]."
                        )
                    edit_raw = DiagonalGaussianDistribution(edit_params).mode()
                    edit_norm = self._normalize_latents(edit_raw).to(self.generator.device)
                    expected_edit_grid = (1, edit_params.shape[-2] // 2, edit_params.shape[-1] // 2)
                    if tuple(edit_shape) != expected_edit_grid:
                        raise ValueError(
                            f"Qwen-Image edit img_shapes must match the packed latent grid {expected_edit_grid}."
                        )
                    edit_packed.append(self._pack_latents(edit_norm))
                hidden_states = torch.cat([noisy_latents] + edit_packed, dim=1)
                combined_shapes = sample_img_shapes + sample_edit_img_shapes
            else:
                hidden_states = noisy_latents
                combined_shapes = sample_img_shapes

            packed_conditions["hidden_states"].append(hidden_states)
            packed_conditions["timestep"].append(timestep / 1000)
            packed_conditions["encoder_hidden_states"].append(sample_context.to(sample_latents.device))
            if sample_context_mask is None:
                packed_conditions["encoder_hidden_states_mask"].append(None)
            else:
                packed_conditions["encoder_hidden_states_mask"].append(sample_context_mask.to(sample_latents.device))
            packed_conditions["img_shapes"].append(combined_shapes)
            packed_conditions["training_target"].append(training_target)
            packed_conditions["latents"].append(sample_latents)
            if self.config.training_recipe == "diffsynth":
                packed_conditions["loss_weights"].append(self._training_weights[timestep_ids])
            if self.config.guidance_scale is not None:
                guidance = torch.full(
                    [sample_latents.shape[0]],
                    self.config.guidance_scale,
                    device=sample_latents.device,
                    dtype=torch.float32,
                )
                packed_conditions["guidance"].append(guidance)

        return packed_conditions

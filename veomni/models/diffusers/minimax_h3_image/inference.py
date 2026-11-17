"""Text-to-image and image-editing inference with MiniMax-H3-Image.

The released checkpoint is guidance-distilled and samples on a fixed six-step
schedule, with one PDD output head per step. This pipeline mirrors the
reference layout: one packed sequence of text rows, reference-image rows and
generated-image rows, with the shared MiniMax-H3 transformer running without
its audio branch.
"""

from __future__ import annotations

import json
import os
from itertools import chain
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from veomni.arguments.arguments_types import OpsImplementationConfig
from veomni.models.checkpoint.weights import init_empty_weights, load_model_weights
from veomni.utils import logging
from veomni.utils.device import get_device_type, get_torch_device

from ..minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import MiniMaxH3DiTModelConfig
from ..minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import MiniMaxH3DiTModel
from .layout import (
    MINIMAX_H3_IMAGE_TAG,
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_image_latents,
    unpatchify_image_rows,
)
from .minimax_h3_image_condition.configuration_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModelConfig,
)
from .minimax_h3_image_condition.modeling_minimax_h3_image_condition import MiniMaxH3ImageConditionModel


logger = logging.get_logger(__name__)

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)
DEFAULT_SIGMAS = (1.0, 0.9675236940383911, 0.8875485062599182, 0.75, 0.5327739119529724, 0.23200887441635132, 0.0)


class MiniMaxH3ImageScheduler:
    """Fixed rectified-flow schedule used by the released PDD heads."""

    def __init__(self, sigmas=DEFAULT_SIGMAS):
        self.sigmas = torch.tensor(sigmas, dtype=torch.float32)
        self.timesteps = 1.0 - self.sigmas[:-1]

    def scale_noise(self, sample: torch.Tensor, timestep: float, noise: torch.Tensor) -> torch.Tensor:
        return timestep * sample + (1.0 - timestep) * noise

    def step(self, model_output: torch.Tensor, index: int, sample: torch.Tensor) -> torch.Tensor:
        sigma = self.sigmas[index].to(sample.device)
        sigma_next = self.sigmas[index + 1].to(sample.device)
        return (sample.to(torch.float32) + (sigma - sigma_next) * model_output.to(torch.float32)).to(sample.dtype)


def _bound_short_edge(
    width: int, height: int, min_short_edge: int, max_short_edge: int, multiple: int
) -> tuple[int, int]:
    short_edge = min(width, height)
    if min_short_edge <= short_edge <= max_short_edge:
        return width, height
    scale = (min_short_edge if short_edge < min_short_edge else max_short_edge) / short_edge
    return (
        max(multiple, round(width * scale / multiple) * multiple),
        max(multiple, round(height * scale / multiple) * multiple),
    )


def _resize_reference(image: Image.Image, min_short_edge: int, max_short_edge: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if min(image.size) < 32:
        raise ValueError(f"A reference image needs at least 32 pixels on each side, got {image.size}.")
    width, height = _bound_short_edge(*image.size, min_short_edge, max_short_edge, 32)
    if image.size == (width, height):
        return image
    return image.resize((width, height), Image.Resampling.BICUBIC)


class MiniMaxH3ImagePipeline(torch.nn.Module):
    """MiniMax-H3-Image text-to-image and image-editing pipeline."""

    def __init__(self, device=None, torch_dtype=torch.bfloat16):
        super().__init__()
        self.device = self._resolve_device(device)
        self.torch_dtype = torch_dtype
        self.condition_model: MiniMaxH3ImageConditionModel | None = None
        self.text_encoder = None
        self.vae = None
        self.dit: MiniMaxH3DiTModel | None = None
        self.tokenizer = None
        self.processor = None
        self.scheduler = MiniMaxH3ImageScheduler()
        self.patch_size = (1, 2, 2)
        self.vae_spatial_compression_ratio = 16
        self.latent_channels = 24
        self.reference_noise_aug = 0.999
        self.max_sequence_length = 4096

    @staticmethod
    def _resolve_device(device) -> torch.device:
        resolved = torch.device(device) if device is not None else torch.device(get_device_type())
        if resolved.index is None and resolved.type != "cpu":
            try:
                resolved = torch.device(resolved.type, get_torch_device().current_device())
            except (AttributeError, RuntimeError):
                pass
        return resolved

    @staticmethod
    def from_pretrained(
        model_path: str,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: str | torch.device | None = None,
        condition_model_cfg: dict | None = None,
        transformer_config_kwargs: dict | None = None,
        ops_implementation=None,
    ) -> "MiniMaxH3ImagePipeline":
        model_path = str(Path(model_path))
        pipe = MiniMaxH3ImagePipeline(device=device, torch_dtype=torch_dtype)
        if ops_implementation is None and pipe.device.type == "cpu":
            ops_implementation = OpsImplementationConfig(
                attn_implementation="eager",
                moe_implementation="eager",
                cross_entropy_loss_implementation="eager",
                rms_norm_implementation="eager",
                swiglu_mlp_implementation="eager",
                rotary_pos_emb_implementation="eager",
                load_balancing_loss_implementation="eager",
                rms_norm_gated_implementation="eager",
                causal_conv1d_implementation="eager",
                chunk_gated_delta_rule_implementation="eager",
            )

        pdd_path = os.path.join(model_path, "pdd_fused_heads.safetensors")
        if not os.path.isfile(pdd_path):
            raise FileNotFoundError(f"MiniMax-H3-Image PDD heads not found: {pdd_path}")
        from safetensors.torch import load_file

        pdd_state = load_file(pdd_path)
        num_pdd_heads = int(pdd_state["video_out.weight"].shape[0])
        pdd_config_path = os.path.join(model_path, "pdd_config.json")
        sigmas = DEFAULT_SIGMAS
        if os.path.isfile(pdd_config_path):
            with open(pdd_config_path) as handle:
                sigmas = json.load(handle).get("sigmas", DEFAULT_SIGMAS)
        if len(sigmas) - 1 != num_pdd_heads:
            raise ValueError(
                f"Sigma schedule has {len(sigmas) - 1} transitions but the checkpoint has {num_pdd_heads} PDD heads."
            )
        pipe.scheduler = MiniMaxH3ImageScheduler(sigmas)

        condition_cfg = dict(condition_model_cfg or {})
        condition_cfg.setdefault("base_model_path", model_path)
        condition_cfg.setdefault("skip_encoder_load", False)
        condition_config = MiniMaxH3ImageConditionModelConfig(**condition_cfg)
        condition_config._apply_ops_overrides(ops_implementation=ops_implementation, _validate=False)
        condition_model = MiniMaxH3ImageConditionModel._from_config(condition_config)
        condition_model.eval()
        pipe.__dict__["condition_model"] = condition_model
        pipe.text_encoder = condition_model.text_encoder
        pipe.vae = condition_model.vae
        pipe.tokenizer = condition_model.tokenizer
        pipe.processor = condition_model.processor
        pipe.max_sequence_length = condition_config.max_sequence_length
        if pipe.vae is not None:
            pipe.vae_spatial_compression_ratio = pipe.vae.spatial_compression_ratio
            pipe.latent_channels = pipe.vae.config.latent_channels

        transformer_config_kwargs = dict(transformer_config_kwargs or {})
        transformer_config_kwargs.setdefault("num_pdd_heads", num_pdd_heads)
        transformer_config_kwargs.setdefault("audio_enabled", False)
        transformer_config_path = os.path.join(model_path, "transformer", "config.json")
        dit_config = MiniMaxH3DiTModelConfig.from_pretrained(
            transformer_config_path, _validate_ops=False, **transformer_config_kwargs
        )
        dit_config._apply_ops_overrides(ops_implementation=ops_implementation, _validate=False)
        with init_empty_weights():
            pipe.dit = MiniMaxH3DiTModel._from_config(dit_config, torch_dtype=torch_dtype)
        pipe._keep_dit_fp32_modules()
        load_model_weights(pipe.dit, os.path.join(model_path, "transformer"), init_device="cpu")
        pipe.dit.eval()
        pipe._load_pdd_heads(pdd_state)

        pipe._offload(["text_encoder", "vae", "dit"])
        return pipe

    def _load_pdd_heads(self, state: dict) -> None:
        weight = state["video_out.weight"]
        bias = state["video_out.bias"]
        final_layer = self.dit.dit.final_layer
        if final_layer.pdd_out is None:
            raise RuntimeError("The transformer config does not declare any PDD output heads.")
        if len(final_layer.pdd_out) != weight.shape[0]:
            raise ValueError(
                f"Transformer has {len(final_layer.pdd_out)} PDD heads but the checkpoint has {weight.shape[0]}."
            )
        for index, head in enumerate(final_layer.pdd_out):
            head.weight.data = weight[index].clone()
            head.bias.data = bias[index].clone()

    def _keep_dit_fp32_modules(self) -> None:
        """Preserve the released checkpoint's FP32 projections across loading."""
        modules = (
            self.dit.dit.video_patch_proj,
            self.dit.dit.time_embedder,
            self.dit.dit.final_layer.video_out,
            self.dit.dit.final_layer.pdd_out,
        )
        for module in modules:
            if module is not None:
                module.to(torch.float32)

    @staticmethod
    def _parameter_on_device(parameter: torch.Tensor, device: torch.device) -> bool:
        if parameter.device.type != device.type:
            return False
        if parameter.device.index is None or device.index is None:
            return True
        return parameter.device.index == device.index

    def _stage(self, names) -> None:
        for name in names:
            module = getattr(self, name)
            if module is None:
                continue
            if any(
                not self._parameter_on_device(tensor, self.device)
                for tensor in chain(module.parameters(), module.buffers())
            ):
                module.to(self.device)

    @staticmethod
    def _offload_names(pipe, names) -> None:
        for name in names:
            module = getattr(pipe, name)
            if module is None:
                continue
            if any(tensor.device.type != "cpu" for tensor in chain(module.parameters(), module.buffers())):
                module.to("cpu")

    def _offload(self, names) -> None:
        self._offload_names(self, names)

    @torch.no_grad()
    def __call__(
        self,
        prompt: str,
        height: int = 2048,
        width: int = 2048,
        references: list[Image.Image] | Image.Image | None = None,
        seed: int = 0,
        min_generation_area: int = 2048 * 2048,
        output_type: str = "pil",
    ):
        """Generate one image from text or edit one or more reference images."""
        if self.dit is None or self.condition_model is None:
            raise RuntimeError("MiniMax-H3-Image pipeline is not loaded.")
        if height <= 0 or width <= 0:
            raise ValueError(f"height and width must be positive, got {height}x{width}.")
        if not 1024 <= min_generation_area <= 4096 * 4096:
            raise ValueError(f"min_generation_area must be within [1024, {4096 * 4096}].")

        canvas_height, canvas_width = get_image_canvas_size(height, width, min_generation_area, 32)
        latent_height = canvas_height // self.vae_spatial_compression_ratio
        latent_width = canvas_width // self.vae_spatial_compression_ratio

        reference_images = [] if references is None else references
        if isinstance(reference_images, Image.Image):
            reference_images = [reference_images]
        else:
            reference_images = list(reference_images)
        vae_references = [_resize_reference(image, 1024, 2048) for image in reference_images]
        vision_references = [_resize_reference(image, 1024, 1024) for image in reference_images]

        generator = torch.Generator(device="cpu").manual_seed(seed)
        reference_latents = []
        try:
            self._stage(["text_encoder"])
            prompt_embeds, text_token_tags = self.condition_model.encode_text(
                prompt,
                reference_images=vision_references or None,
                device=self.device,
                dtype=self.torch_dtype,
            )
            prompt_embeds = prompt_embeds.to(self.device)
            if vae_references:
                self._stage(["vae"])
                reference_latents = self.condition_model.encode_reference_images(
                    vae_references, generator=generator, device=self.device
                )
        finally:
            self._offload(["text_encoder", "vae"])

        reference_grids = [(latents.shape[-2], latents.shape[-1]) for latents in reference_latents]
        position_ids, token_tags, image_indices, text_indices, num_reference_rows = build_image_packed_sequence(
            text_token_tags,
            reference_grids,
            latent_height,
            latent_width,
            self.patch_size,
            image_tag=MINIMAX_H3_IMAGE_TAG,
        )
        position_ids = position_ids.to(self.device, torch.float32)
        token_tags = token_tags.to(self.device)
        image_indices = image_indices.to(self.device)
        text_indices = text_indices.to(self.device)

        target_latents = torch.randn(
            (1, self.latent_channels, 1, latent_height, latent_width),
            generator=generator,
            dtype=torch.float32,
        ).to(self.device)
        target_rows = patchify_image_latents(target_latents, self.patch_size)
        condition_rows = None
        if reference_latents:
            condition_rows = []
            for latents in reference_latents:
                noise = torch.randn(latents.shape, generator=generator, dtype=torch.float32).to(self.device)
                noised = self.scheduler.scale_noise(latents.to(self.device), self.reference_noise_aug, noise)
                condition_rows.append(patchify_image_latents(noised[:, :, None], self.patch_size))
            condition_rows = torch.cat(condition_rows, dim=0)

        rows = torch.cat([condition_rows, target_rows], dim=0) if condition_rows is not None else target_rows
        sequence_length = position_ids.shape[0]
        num_steps = len(self.scheduler.sigmas) - 1

        try:
            self._stage(["dit"])
            for step in range(num_steps):
                timestep = float(1.0 - self.scheduler.sigmas[step])
                if reference_latents:
                    unique_timesteps = torch.tensor(
                        [self.reference_noise_aug, timestep], dtype=torch.float32, device=self.device
                    )
                    timestep_indices = torch.ones(sequence_length, dtype=torch.long, device=self.device)
                    timestep_indices[image_indices[:num_reference_rows]] = 0
                else:
                    unique_timesteps = torch.tensor([timestep], dtype=torch.float32, device=self.device)
                    timestep_indices = torch.zeros(sequence_length, dtype=torch.long, device=self.device)

                noise_pred = self.dit.forward_image(
                    x=rows,
                    prompt_embeds=prompt_embeds,
                    unique_timesteps=unique_timesteps,
                    timestep_indices=timestep_indices,
                    token_tags=token_tags,
                    position_ids=position_ids,
                    image_indices=image_indices,
                    text_indices=text_indices,
                    pdd_step_index=step,
                )
                target_rows = self.scheduler.step(noise_pred[num_reference_rows:], step, target_rows)
                rows = torch.cat([condition_rows, target_rows], dim=0) if condition_rows is not None else target_rows
        finally:
            self._offload(["dit"])

        latents = unpatchify_image_rows(
            target_rows,
            latent_height,
            latent_width,
            self.latent_channels,
            self.patch_size,
        )
        self._stage(["vae"])
        latents_mean = torch.tensor(self.vae.config.latents_mean, device=self.device).view(1, -1, 1, 1)
        latents_std = torch.tensor(self.vae.config.latents_std, device=self.device).view(1, -1, 1, 1)
        image = self.vae.decode(latents * latents_std + latents_mean, return_dict=False)[0]

        pixel_mean = torch.tensor(PIXEL_MEAN, device=self.device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=self.device).view(1, -1, 1, 1)
        image = (image.float() * pixel_std + pixel_mean).clamp(0, 1)
        image = (image * 255).clamp(0, 255).to(torch.uint8)
        image = image[0].permute(1, 2, 0).cpu().numpy()
        result = Image.fromarray(image)
        if (height, width) != (canvas_height, canvas_width):
            result = result.resize((width, height), Image.Resampling.BICUBIC)
        if output_type == "pil":
            return result
        if output_type == "np":
            return np.asarray(result)
        raise ValueError(f"Unsupported output_type {output_type!r}.")

"""MiniMax-H3-Image text and image conditioning."""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file
from transformers import AutoProcessor

from veomni.models.modeling_base import VeOmniPreTrainedModel as PreTrainedModel
from veomni.utils import logging

from ..layout import (
    MINIMAX_H3_IMAGE_TAG,
    MINIMAX_H3_TEXT_TAG,
    build_image_packed_sequence,
)
from ..minimax_h3_image_vae.configuration_minimax_h3_image_vae import MiniMaxH3ImageVAEConfig
from ..minimax_h3_image_vae.modeling_minimax_h3_image_vae import MiniMaxH3ImageVAE
from .configuration_minimax_h3_image_condition import MiniMaxH3ImageConditionModelConfig


logger = logging.get_logger(__name__)

PIXEL_MEAN = (0.485, 0.456, 0.406)
PIXEL_STD = (0.229, 0.224, 0.225)


def _init_empty_weights():
    from ...minimax_h3.minimax_h3_condition._module_utils import init_empty_weights

    return init_empty_weights()


def _state_dict_iterator(filepath: str):
    if filepath.endswith(".safetensors"):
        with safe_open(filepath, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                yield key, handle.get_tensor(key)
    else:
        state_dict = torch.load(filepath, map_location="cpu", weights_only=True, mmap=True)
        for key, value in state_dict.items():
            yield key, value


class MiniMaxH3ImageConditionModel(PreTrainedModel):
    config_class = MiniMaxH3ImageConditionModelConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: MiniMaxH3ImageConditionModelConfig, **kwargs):
        super().__init__(config, **kwargs)
        self._text_encoder = None
        self._vae = None
        self._processor = None
        self._tokenizer = None
        if not config.skip_encoder_load:
            self._init_encoders(config)

    @property
    def text_encoder(self):
        return self._text_encoder

    @property
    def vae(self):
        return self._vae

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def processor(self):
        return self._processor

    @property
    def _execution_device(self):
        if self._text_encoder is not None:
            return next(self._text_encoder.parameters()).device
        return torch.device("cpu")

    def _init_encoders(self, config: MiniMaxH3ImageConditionModelConfig):
        from ...minimax_h3.minimax_h3_core.minimax_h3_text_encoder import MiniMaxH3TextEncoder

        base = config.base_model_path
        if not base:
            return

        processor_path = os.path.join(base, config.processor_subfolder)
        self._processor = AutoProcessor.from_pretrained(processor_path)
        self._tokenizer = self._processor.tokenizer

        vae_config = MiniMaxH3ImageVAEConfig.from_pretrained(os.path.join(base, config.vae_subfolder))
        with _init_empty_weights():
            self._vae = MiniMaxH3ImageVAE(vae_config)
        vae_path = os.path.join(base, config.vae_subfolder, "diffusion_pytorch_model.safetensors")
        if not os.path.isfile(vae_path):
            raise FileNotFoundError(f"MiniMax-H3-Image VAE weights not found: {vae_path}")
        self._vae.load_state_dict(load_file(vae_path), assign=True, strict=True)
        self._vae.to(torch.float32)
        self._vae.requires_grad_(False)
        logger.info_rank0("MiniMax-H3-Image VAE loaded.")

        with _init_empty_weights():
            self._text_encoder = MiniMaxH3TextEncoder(num_retained_layers=config.text_encoder_num_retained_layers)
        text_path = os.path.join(base, config.text_encoder_subfolder)
        filter_fn = self._text_encoder_filter(config.text_encoder_num_retained_layers)
        loaded = self._stream_load_weights(self._text_encoder, text_path, torch.bfloat16, filter_fn=filter_fn)
        if loaded == 0:
            raise RuntimeError(f"MiniMax-H3-Image text encoder loaded 0 tensors from {text_path}.")
        self._text_encoder.to(torch.bfloat16)
        self._text_encoder.requires_grad_(False)
        logger.info_rank0("MiniMax-H3-Image text encoder loaded (%d tensors).", loaded)

        for name, module in (("VAE", self._vae), ("text encoder", self._text_encoder)):
            meta_parameters = [key for key, parameter in module.named_parameters() if parameter.device.type == "meta"]
            if meta_parameters:
                raise RuntimeError(
                    f"MiniMax-H3-Image {name} has {len(meta_parameters)} parameter(s) on the meta device after "
                    f"loading (first: {meta_parameters[0]!r}); the checkpoint is incomplete."
                )

    @staticmethod
    def _text_encoder_filter(num_retained_layers: int):
        def keep(key: str) -> bool:
            return (
                key.startswith("lm_head.")
                or key.startswith("model.language_model.norm.")
                or any(
                    key.startswith(f"model.language_model.layers.{index}.")
                    for index in range(num_retained_layers, 100)
                )
            )

        return keep

    @staticmethod
    def _stream_load_weights(model, path: str, dtype: torch.dtype, filter_fn=None) -> int:
        total = 0
        for filename in sorted(os.listdir(path)):
            if not filename.endswith(".safetensors"):
                continue
            state = OrderedDict()
            for key, value in _state_dict_iterator(os.path.join(path, filename)):
                if filter_fn is not None and filter_fn(key):
                    continue
                state[key] = value.to(dtype)
            if state:
                model.load_state_dict(state, assign=True, strict=False)
                total += len(state)
        return total

    def _text_ids(self, text: str, max_length: int | None = None) -> list[int]:
        ids = self._tokenizer(text, add_special_tokens=False)["input_ids"]
        return ids[:max_length] if max_length is not None else ids

    def _vision_block_ids(self, count: int) -> list[int]:
        tokenizer = self._tokenizer
        return (
            [tokenizer.convert_tokens_to_ids("<|vision_start|>")]
            + [tokenizer.convert_tokens_to_ids("<|image_pad|>")] * int(count)
            + [tokenizer.convert_tokens_to_ids("<|vision_end|>")]
        )

    @torch.no_grad()
    def encode_text(
        self,
        prompt: str,
        reference_images: list[Image.Image] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a t2i or edit presentation with the 50-layer Qwen3-VL stack."""
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("MiniMax-H3-Image requires a single non-empty prompt string.")
        if self._text_encoder is None or self._tokenizer is None:
            raise RuntimeError("MiniMax-H3-Image text encoder is not loaded.")

        device = device or self._execution_device
        dtype = dtype or next(self._text_encoder.parameters()).dtype
        max_length = self.config.max_sequence_length
        token_ids: list[int] = []
        token_tags: list[int] = []
        vision = None

        if reference_images:
            vision = self._processor.image_processor(images=reference_images, return_tensors="pt")
            merge_size = int(self._processor.image_processor.merge_size) ** 2
            for index, grid in enumerate(vision["image_grid_thw"]):
                label_ids = self._text_ids(f"Picture {index + 1}: ", max_length=max_length)
                vision_ids = self._vision_block_ids(int(grid.prod()) // merge_size)
                token_ids += label_ids + vision_ids
                token_tags += [MINIMAX_H3_TEXT_TAG] * len(label_ids)
                token_tags += [MINIMAX_H3_IMAGE_TAG] * len(vision_ids)

        prompt_ids = self._text_ids(prompt, max_length=max_length)
        token_ids += prompt_ids
        token_tags += [MINIMAX_H3_TEXT_TAG] * len(prompt_ids)

        input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if vision is not None:
            kwargs["pixel_values"] = vision["pixel_values"].to(device, dtype)
            kwargs["image_grid_thw"] = vision["image_grid_thw"].to(device, torch.long)

        prompt_embeds = self._text_encoder(**kwargs)
        return prompt_embeds.unsqueeze(0), torch.tensor(token_tags, dtype=torch.long)

    @torch.no_grad()
    def encode_reference_images(
        self,
        images: list[Image.Image],
        generator: torch.Generator | None = None,
        device: torch.device | None = None,
    ) -> list[torch.Tensor]:
        """Encode edit references into normalized conditioning latents."""
        if self._vae is None:
            raise RuntimeError("MiniMax-H3-Image VAE is not loaded.")
        device = device or self._vae.device
        ratio = self._vae.spatial_compression_ratio
        patch_size = (1, 2, 2)
        pixel_mean = torch.tensor(PIXEL_MEAN, device=device).view(1, -1, 1, 1)
        pixel_std = torch.tensor(PIXEL_STD, device=device).view(1, -1, 1, 1)
        latents_mean = torch.tensor(self._vae.config.latents_mean, device=device).view(1, -1, 1, 1)
        latents_std = torch.tensor(self._vae.config.latents_std, device=device).view(1, -1, 1, 1)

        reference_latents = []
        for image in images:
            pixels = torch.from_numpy(np.array(image)).to(device).permute(2, 0, 1)[None]
            pixels = pixels[..., : pixels.shape[-2] // ratio * ratio, : pixels.shape[-1] // ratio * ratio]
            pixels = (pixels.to(torch.float32).div(255.0) - pixel_mean) / pixel_std
            posterior = self._vae.encode(pixels, return_dict=False)[0]
            if generator is not None and generator.device.type != device.type:
                noise = torch.randn(posterior.mean.shape, generator=generator, device="cpu", dtype=torch.float32).to(
                    device
                )
                latents = posterior.mean + posterior.std * noise
            else:
                latents = posterior.sample(generator=generator)
            latents = latents.to(torch.float16).float()
            latents = latents[
                ...,
                : latents.shape[-2] // patch_size[1] * patch_size[1],
                : latents.shape[-1] // patch_size[2] * patch_size[2],
            ]
            reference_latents.append((latents - latents_mean) / latents_std)
        return reference_latents

    def build_layout(
        self,
        text_token_tags: torch.Tensor,
        reference_grids: list[tuple[int, int]],
        latent_height: int,
        latent_width: int,
        patch_size: tuple[int, int, int] = (1, 2, 2),
    ):
        return build_image_packed_sequence(
            text_token_tags,
            reference_grids,
            latent_height,
            latent_width,
            patch_size,
        )

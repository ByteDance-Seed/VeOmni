"""BAGEL latent VAE module.

Tensor call-sites:
* ``bagel_vae.encode`` maps normalized image tensors to scaled latent grids.
* ``bagel_vae.decode`` maps generated latent grids to decoded image tensors.

The VAE module is a codec boundary only. Flow timestep/noise sampling, latent
patchification, packed MoT indexes, and conversation-carrier mutation belong to
the Bagel VAE module mixin and downstream Bagel nodes.

``encode_context()`` (image-understanding VAE encode) and ``decode_generated()``
(final generated-latent decode) are the FSM generation-node methods, native here
so pure HF inference (no accelerated mixins) can run BAGEL's generation graphs.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange
from torch import Tensor

from ....mixins.offline_encoding_mixin import OfflineEncodingMixin
from ....omni_pretrained_model import OmniPreTrainedModel
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from ..sources import BAGEL_GENERATED_LATENT, BAGEL_VAE_CONTEXT
from .configuration import BagelVAEConfig
from .processing import BAGEL_VAE_PIXEL_SHAPE, BagelVAEPreprocessor, BagelVAEProcessor, crop_latent_to_image_shape


def select_bagel_vae_context_items(
    conversation_list: list[list[ConversationItem]] | None,
    *,
    exclude_dummy: bool = False,
) -> list[ConversationItem]:
    """Select VAE context image items (training includes dummies; inference skips them)."""
    if conversation_list is None:
        raise ValueError("BagelVAE requires conversation_list to select VAE context items.")

    items = list(iter_desired_items(conversation_list, types=["image"], sources=[BAGEL_VAE_CONTEXT]))
    if exclude_dummy:
        return [item for item in items if not is_dummy(item)]
    return items


class InferenceMixin:
    """FSM ``encode_context`` / ``decode_generated`` — HF ``GenerationMixin`` analog.

    Listed *before* :class:`~....omni_pretrained_model.OmniPreTrainedModel` in
    :class:`BagelVAE`'s bases for consistency with every other module's
    native / accelerated split (see ``janus/llama/modeling.py`` for the full
    MRO rationale where a competing no-op default exists — this module
    doesn't override ``reset_local_inference_state`` / ``finalize`` so there
    is nothing to shadow here).
    """

    def encode_context(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """FSM node: encode image-understanding context images into VAE latents."""
        del generation_kwargs, kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}

        batched = [conversation_list]
        image_items = select_bagel_vae_context_items(batched, exclude_dummy=True)
        if not image_items:
            return {"conversation_list": conversation_list}

        pixel_values = torch.stack([item.value for item in image_items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        outputs = self.encode(pixel_values=pixel_values)
        for image_item, latent in zip(image_items, outputs["latents"], strict=True):
            latent = crop_latent_to_image_shape(
                latent,
                image_item.meta.get(BAGEL_VAE_PIXEL_SHAPE),
                downsample=int(self.config.downsample),
            )
            image_item.value = latent.to(device=self.device, dtype=self.dtype)
            image_item.source = BAGEL_VAE_CONTEXT
        return {"conversation_list": conversation_list}

    def decode_generated(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """FSM node: decode the flow connector's completed latent into a PIL image."""
        del generation_kwargs, kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}

        batched = [conversation_list]
        decode_items = self._select_vae_decode_items(batched)
        if not decode_items:
            return {"conversation_list": conversation_list}

        latents = []
        for item in decode_items:
            latents.append(
                item.value.detach().squeeze(0)
                if item.value.dim() == 4 and item.value.shape[0] == 1
                else item.value.detach()
            )
        latents = torch.stack(latents, dim=0).to(device=self.device, dtype=self.dtype)

        outputs = self.decode(latents=latents)
        pixel_values = outputs["pixel_values"]
        for item, image in zip(decode_items, pixel_values, strict=True):
            item.type = "image"
            item.value = image.to(device=self.device, dtype=self.dtype)
        if self._image_processor is None:
            raise RuntimeError(
                "BagelVAE: cannot postprocess decoded image — no image processor was "
                "loaded. Ensure `preprocessor_config.json` ships next to the weights."
            )
        return {
            "conversation_list": conversation_list,
            "generated": {"type": "image", "value": self._image_processor.postprocess(pixel_values[-1])[0]},
        }

    def _select_vae_decode_items(
        self, conversation_list: list[list[ConversationItem]] | None
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelVAE decode requires conversation_list to select latent items.")

        # Final image decode consumes the completed latent emitted by the flow connector.
        decode_items: list[ConversationItem] = []
        for item in iter_desired_items(conversation_list, types=["output"], sources=[BAGEL_GENERATED_LATENT]):
            if not is_dummy(item):
                decode_items.append(item)
        return decode_items


class BagelVAE(InferenceMixin, OmniPreTrainedModel):
    config_class = BagelVAEConfig
    image_processor_class = BagelVAEProcessor
    preprocessor_class = BagelVAEPreprocessor
    base_model_prefix = "bagel_vae"
    main_input_name = "pixel_values"
    _no_split_modules: list[str] = ["ResnetBlock", "AttnBlock"]
    supports_gradient_checkpointing = False

    def __init__(self, config: BagelVAEConfig, **kwargs: Any) -> None:
        super().__init__(config, **kwargs)
        cache_mode = OfflineEncodingMixin.validated_cache_mode(config)
        if cache_mode in {"full", "encode_only"}:
            self.encoder = Encoder(
                resolution=config.resolution,
                in_channels=config.in_channels,
                ch=config.ch,
                ch_mult=config.ch_mult,
                num_res_blocks=config.num_res_blocks,
                z_channels=config.z_channels,
            )
        if cache_mode == "full":
            self.decoder = Decoder(
                resolution=config.resolution,
                in_channels=config.in_channels,
                ch=config.ch,
                out_ch=config.out_ch,
                ch_mult=config.ch_mult,
                num_res_blocks=config.num_res_blocks,
                z_channels=config.z_channels,
            )
        self._image_processor: BagelVAEProcessor | None = None
        self.post_init()

    def freeze_model(self) -> None:
        if self.config.freeze:
            self.eval()
            self.requires_grad_(False)

    def _require_encoder(self) -> Encoder:
        encoder = getattr(self, "encoder", None)
        if encoder is None:
            raise RuntimeError(
                f"BagelVAE requires the VAE encoder; cache_mode={OfflineEncodingMixin.validated_cache_mode(self.config)!r}."
            )
        return encoder

    def _require_decoder(self) -> Decoder:
        decoder = getattr(self, "decoder", None)
        if decoder is None:
            raise RuntimeError(
                f"BagelVAE requires the VAE decoder; cache_mode={OfflineEncodingMixin.validated_cache_mode(self.config)!r}."
            )
        return decoder

    @property
    def _encoder_device(self) -> torch.device:
        return self._require_encoder().conv_in.weight.device

    @property
    def _decoder_device(self) -> torch.device:
        return self._require_decoder().conv_in.weight.device

    @contextmanager
    def _runtime_context(self, tensor: torch.Tensor):
        grad_context = torch.enable_grad()
        if self.config.freeze:
            grad_context = torch.no_grad()

        autocast_context = nullcontext()
        if self.dtype != torch.float32:
            autocast_context = torch.amp.autocast(tensor.device.type, enabled=True, dtype=self.dtype)

        with grad_context, autocast_context:
            yield

    def encode(
        self,
        pixel_values: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        pixel_values = pixel_values.to(device=self._encoder_device, dtype=self.dtype)
        with self._runtime_context(pixel_values):
            posterior = self._encode_posterior(pixel_values)
            latents = self._sample_scaled_latents(posterior)
        return {"latents": latents.to(dtype=self.dtype)}

    def decode(
        self,
        latents: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        decoder = self._require_decoder()
        latents = latents.to(device=self._decoder_device, dtype=self.dtype)
        latents = latents / self.config.scale_factor + self.config.shift_factor
        with self._runtime_context(latents):
            pixel_values = decoder(latents)
        return {"pixel_values": pixel_values.to(dtype=self.dtype)}

    def _encode_posterior(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoder = self._require_encoder()
        return torch.chunk(encoder(pixel_values), 2, dim=1)

    def _sample_scaled_latents(self, posterior: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        mean, logvar = posterior
        latents = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        return self.config.scale_factor * (latents - self.config.shift_factor)


def swish(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class AttnBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.norm = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def attention(self, hidden_states: Tensor) -> Tensor:
        hidden_states = self.norm(hidden_states)
        query = self.q(hidden_states)
        key = self.k(hidden_states)
        value = self.v(hidden_states)

        batch, channels, height, width = query.shape
        query = rearrange(query, "b c h w -> b 1 (h w) c").contiguous()
        key = rearrange(key, "b c h w -> b 1 (h w) c").contiguous()
        value = rearrange(value, "b c h w -> b 1 (h w) c").contiguous()
        hidden_states = nn.functional.scaled_dot_product_attention(query, key, value)
        return rearrange(hidden_states, "b 1 (h w) c -> b c h w", h=height, w=width, c=channels, b=batch)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.proj_out(self.attention(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        hidden_states = self.norm1(x)
        hidden_states = swish(hidden_states)
        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = swish(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + hidden_states


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.pad(x, (0, 1, 0, 1), mode="constant", value=0)
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        x = nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class Encoder(nn.Module):
    def __init__(
        self,
        resolution: int,
        in_channels: int,
        ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.conv_in = nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        block_in = self.ch
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in)
            self.down.append(down)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, 2 * z_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: Tensor) -> Tensor:
        hidden_stack = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                block = self.down[i_level].block[i_block]
                hidden_states = (
                    self._gradient_checkpointing_func(block.__call__, hidden_stack[-1])
                    if self.gradient_checkpointing and self.training
                    else block(hidden_stack[-1])
                )
                if len(self.down[i_level].attn) > 0:
                    attn = self.down[i_level].attn[i_block]
                    hidden_states = (
                        self._gradient_checkpointing_func(attn.__call__, hidden_states)
                        if self.gradient_checkpointing and self.training
                        else attn(hidden_states)
                    )
                hidden_stack.append(hidden_states)
            if i_level != self.num_resolutions - 1:
                downsample = self.down[i_level].downsample
                hidden_stack.append(
                    self._gradient_checkpointing_func(downsample.__call__, hidden_stack[-1])
                    if self.gradient_checkpointing and self.training
                    else downsample(hidden_stack[-1])
                )

        hidden_states = hidden_stack[-1]
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2.__call__, hidden_states)
        else:
            hidden_states = self.mid.block_1(hidden_states)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states)
        hidden_states = self.norm_out(hidden_states)
        hidden_states = swish(hidden_states)
        return self.conv_out(hidden_states)


class Decoder(nn.Module):
    def __init__(
        self,
        ch: int,
        out_ch: int,
        ch_mult: list[int],
        num_res_blocks: int,
        in_channels: int,
        resolution: int,
        z_channels: int,
    ) -> None:
        super().__init__()
        self.gradient_checkpointing = False
        self.ch = ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.ffactor = 2 ** (self.num_resolutions - 1)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)

        self.conv_in = nn.Conv2d(z_channels, block_in, kernel_size=3, stride=1, padding=1)

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for _ in range(self.num_res_blocks + 1):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
            self.up.insert(0, up)

        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=block_in, eps=1e-6, affine=True)
        self.conv_out = nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, z: Tensor) -> Tensor:
        hidden_states = self.conv_in(z)

        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(self.mid.block_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.attn_1.__call__, hidden_states)
            hidden_states = self._gradient_checkpointing_func(self.mid.block_2.__call__, hidden_states)
        else:
            hidden_states = self.mid.block_1(hidden_states)
            hidden_states = self.mid.attn_1(hidden_states)
            hidden_states = self.mid.block_2(hidden_states)

        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                block = self.up[i_level].block[i_block]
                hidden_states = (
                    self._gradient_checkpointing_func(block.__call__, hidden_states)
                    if self.gradient_checkpointing and self.training
                    else block(hidden_states)
                )
                if len(self.up[i_level].attn) > 0:
                    attn = self.up[i_level].attn[i_block]
                    hidden_states = (
                        self._gradient_checkpointing_func(attn.__call__, hidden_states)
                        if self.gradient_checkpointing and self.training
                        else attn(hidden_states)
                    )
            if i_level != 0:
                upsample = self.up[i_level].upsample
                hidden_states = (
                    self._gradient_checkpointing_func(upsample.__call__, hidden_states)
                    if self.gradient_checkpointing and self.training
                    else upsample(hidden_states)
                )

        hidden_states = self.norm_out(hidden_states)
        hidden_states = swish(hidden_states)
        return self.conv_out(hidden_states)


__all__ = [
    "BagelVAE",
    "BagelVAEConfig",
    "InferenceMixin",
    "select_bagel_vae_context_items",
]

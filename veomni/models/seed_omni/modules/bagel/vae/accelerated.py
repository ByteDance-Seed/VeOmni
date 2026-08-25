"""SeedOmni V2 carrier hooks for BAGEL's latent VAE module — training-graph hooks only.

``encode_context()`` and ``decode_generated()`` (the FSM generation nodes) live
natively on :class:`~.modeling.BagelVAE` — this file only carries the training
pre/forward/post hooks and offline-encoding-cache endpoints.
"""

from __future__ import annotations

import shutil
from typing import Any

import torch

from veomni.utils.device import get_device_id, get_device_type

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs, slice_input_tensor
from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.offline_encoding_mixin import OfflineEncodingMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem
from ..sources import BAGEL_VAE_CONTEXT
from .configuration import BagelVAEConfig
from .modeling import BagelVAE, select_bagel_vae_context_items
from .processing import BAGEL_VAE_PIXEL_SHAPE, crop_latent_to_image_shape


def _posterior_from_cache(
    cache: torch.Tensor,
    *,
    z_channels: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], bool]:
    if cache.dim() == 5 and int(cache.shape[0]) == 1:
        cache = cache.squeeze(0)
    if cache.dim() == 4 and int(cache.shape[0]) == 2 and int(cache.shape[1]) == z_channels:
        return (cache[0].unsqueeze(0), cache[1].unsqueeze(0)), True
    raise ValueError(
        "BAGEL VAE posterior cache tensor must be shaped (2, C, H, W) "
        f"or singleton-batched (1, 2, C, H, W); got {tuple(cache.shape)}."
    )


class BagelVAEOfflineMixin:
    """Offline-cache tensor endpoints and training-graph hooks for VAE encode / process."""

    config: BagelVAEConfig
    device: torch.device
    dtype: torch.dtype
    _conversation_carrier: list[list[ConversationItem]] | None
    _encode_items: list[ConversationItem]
    _sp_encode_count: int | None

    def offline_encode(
        self,
        pixel_values: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        pixel_values = pixel_values.to(device=self._encoder_device, dtype=self.dtype)
        with self._runtime_context(pixel_values):
            posterior = self._encode_posterior(pixel_values)
        return {"encoded_cache": torch.cat(posterior, dim=1).to(dtype=self.dtype)}

    def online_process(
        self,
        encoded_cache: torch.Tensor | list[torch.Tensor],
        **kwargs: object,
    ) -> dict[str, Any]:
        del kwargs
        if not isinstance(encoded_cache, list):
            encoded_cache = [encoded_cache]

        latents = []
        for cache in encoded_cache:
            posterior, is_item_cache = _posterior_from_cache(cache, z_channels=int(self.config.z_channels))
            latent = self._sample_scaled_latents(posterior)
            if is_item_cache:
                latent = latent.squeeze(0)
            latents.append(latent.to(dtype=cache.dtype))
        return {"latents": latents}

    @post_forward("offline_encode")
    def offline_encode_post(self, encoded_cache: torch.Tensor) -> dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            # Keep the CNN's spatial outputs intact through gather; crop each
            # cache item from its original pixel shape only after batch unpadding.
            encoded_cache = self._gather_encode_output(encoded_cache)

        conversation = self._conversation_carrier
        encode_items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []
        self._sp_encode_count = None

        for item, cache_tensor in zip(encode_items, encoded_cache, strict=True):
            item.type = "image"
            cache_tensor = crop_latent_to_image_shape(
                cache_tensor,
                item.meta.get(BAGEL_VAE_PIXEL_SHAPE),
                downsample=int(self.config.downsample),
            )
            z_channels = int(self.config.z_channels)
            if cache_tensor.dim() == 3 and int(cache_tensor.shape[0]) == 2 * z_channels:
                cache_tensor = cache_tensor.reshape(2, z_channels, *cache_tensor.shape[-2:])
            item.value = cache_tensor.detach().to(device=self.device, dtype=self.dtype)
            item.source = BAGEL_VAE_CONTEXT
            item.meta = {}
        return {"conversation_list": conversation}

    @pre_forward("online_process")
    def online_process_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._encode_items = []
        self._sp_encode_count = None

        self._encode_items = select_bagel_vae_context_items(conversation_list)
        encoded_cache: list[torch.Tensor] = []
        for item in self._encode_items:
            cache = item.value
            if not torch.is_tensor(cache):
                raise ValueError("BAGEL VAE online_process requires tensor posterior cache.")
            encoded_cache.append(cache.detach().to(device=self._online_process_device()))
        return {"encoded_cache": encoded_cache}

    def _online_process_dtype(self) -> torch.dtype:
        config_dtype = getattr(self.config, "dtype", None) or getattr(self.config, "torch_dtype", None)
        if isinstance(config_dtype, torch.dtype):
            return config_dtype
        if isinstance(config_dtype, str) and hasattr(torch, config_dtype):
            dtype = getattr(torch, config_dtype)
            if isinstance(dtype, torch.dtype):
                return dtype
        return torch.get_default_dtype()

    def _online_process_device(self) -> torch.device:
        device_type = get_device_type()
        if device_type == "cpu":
            return torch.device("cpu")
        return torch.device(device_type, get_device_id())


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`BagelVAE` modeling APIs."""

    config: BagelVAEConfig
    device: torch.device
    dtype: torch.dtype
    _conversation_carrier: list[list[ConversationItem]] | None
    _encode_items: list[ConversationItem]
    _sp_encode_count: int | None

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._encode_items: list[ConversationItem] = []
        self._sp_encode_count: int | None = None

    @pre_forward("encode", "offline_encode")
    def encode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list

        self._encode_items = select_bagel_vae_context_items(conversation_list)

        self._metric_meter_stash_latent_tokens(self._vae_latent_token_lengths())

        pixel_values = torch.stack([item.value for item in self._encode_items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_encode_count = None
            return {"pixel_values": pixel_values}

        if ps.cp_size != 1:
            raise ValueError(f"BAGEL VAE training supports Ulysses groups only; got cp_size={ps.cp_size}.")

        self._sp_encode_count = int(pixel_values.shape[0])
        if self._sp_encode_count == 0:
            raise ValueError("BAGEL VAE SP requires at least one real or dummy image.")
        # Slice only the image-batch dimension. Processor-added spatial zero
        # padding is part of each image canvas and must pass through the CNN.
        pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=ps.sp_group)
        return {"pixel_values": pixel_values}

    @post_forward("encode", "online_process")
    def encode_post(self, latents: torch.Tensor | list[torch.Tensor]) -> dict[str, Any]:
        if not isinstance(latents, list) and get_parallel_state().sp_size > 1:
            # Remove SP's batch-padding images first. Per-image spatial latent
            # padding is cropped below from BAGEL_VAE_PIXEL_SHAPE after gather.
            latents = self._gather_encode_output(latents)

        conversation = self._conversation_carrier
        encode_items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []
        self._sp_encode_count = None

        if isinstance(latents, list):
            for item, latent in zip(encode_items, latents, strict=True):
                item.type = "image"
                item.value = latent
                item.source = BAGEL_VAE_CONTEXT
        else:
            for item, latent in zip(encode_items, latents, strict=True):
                item.type = "image"
                latent = crop_latent_to_image_shape(
                    latent,
                    item.meta.get(BAGEL_VAE_PIXEL_SHAPE),
                    downsample=int(self.config.downsample),
                )
                item.value = latent
                item.source = BAGEL_VAE_CONTEXT
        return {"conversation_list": conversation}

    def _gather_encode_output(self, output: torch.Tensor) -> torch.Tensor:
        if self._sp_encode_count is None:
            raise RuntimeError("BAGEL VAE SP image count was not initialized.")
        output = gather_outputs(output, gather_dim=0, group=get_parallel_state().sp_group)
        return output.narrow(0, 0, self._sp_encode_count)

    def _vae_latent_token_lengths(self) -> list[int]:
        # Latent tokens per image = (H // ds) * (W // ds), using the real pixel shape
        ds = max(int(self.config.downsample), 1)
        lengths: list[int] = []
        for item in self._encode_items:
            shape = item.meta.get(BAGEL_VAE_PIXEL_SHAPE)
            if torch.is_tensor(shape):
                dims = shape.detach().reshape(-1).tolist()
                height, width = int(dims[0]), int(dims[1])
            else:
                height, width = int(item.value.shape[-2]), int(item.value.shape[-1])
            lengths.append((height // ds) * (width // ds))
        return lengths

    def _metric_meter_stash_latent_tokens(self, lengths: list[int]) -> None:
        # currently use the same counts for encode and offline_encode
        self.metric_meter_set_seqlens("encode", lengths)
        self.metric_meter_set_seqlens("offline_encode", lengths)


class MeterMixin(MetricMeterMixin):
    """Per-module training meter for BAGEL's latent VAE codec (FLUX-style conv AE)."""

    config: BagelVAEConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config

        if getattr(cfg, "freeze", True):
            return 0.0

        ch = int(cfg.ch)
        ch_mult = [int(m) for m in cfg.ch_mult]
        in_ch_mult = [1, *ch_mult]
        num_res = len(ch_mult)
        num_res_blocks = int(cfg.num_res_blocks)
        in_channels = int(cfg.in_channels)
        z_channels = int(cfg.z_channels)

        # Conv MACs per latent token (linear term): area(level i) = N · 4**(num_res-1-i).
        def area_factor(level: int) -> int:
            return 4 ** (num_res - 1 - level)

        lin_macs_per_token = in_channels * ch * 9 * area_factor(0)  # conv_in (3×3)

        for i in range(num_res):
            af = area_factor(i)
            block_in = ch * in_ch_mult[i]
            block_out = ch * ch_mult[i]
            for j in range(num_res_blocks):
                cin = block_in if j == 0 else block_out
                lin_macs_per_token += cin * block_out * 9 * af  # ResnetBlock.conv1
                lin_macs_per_token += block_out * block_out * 9 * af  # ResnetBlock.conv2
                if j == 0 and block_in != block_out:
                    lin_macs_per_token += block_in * block_out * af  # 1×1 nin_shortcut
            if i != num_res - 1:
                # Strided Downsample (3×3) emits the next level's spatial area.
                lin_macs_per_token += block_out * block_out * 9 * area_factor(i + 1)

        # Mid block at the latent resolution (area factor 1): two ResnetBlocks + AttnBlock.
        bc = ch * ch_mult[-1]
        lin_macs_per_token += 4 * (bc * bc * 9)  # block_1 + block_2 (conv1 + conv2 each)
        lin_macs_per_token += 4 * (bc * bc)  # attn q/k/v/proj_out are 1×1 convs
        lin_macs_per_token += bc * (2 * z_channels) * 9  # conv_out (3×3)

        tokens = sum(seqlens)
        # Mid self-attention (single head, dim = bc): QK^T + attn·V ⇒ 2·N**2·bc MACs.
        attn_macs = sum(2 * (n * n) * bc for n in seqlens)

        total_macs = lin_macs_per_token * tokens + attn_macs
        return 6 * total_macs / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, BagelVAEOfflineMixin, OfflineEncodingMixin, MeterMixin):
    """Carrier hooks for raw-image VAE encode and latent decode.

    ``encode_context()`` / ``decode_generated()`` already live on the native
    :class:`~.modeling.BagelVAE` (via its own :class:`~.modeling.InferenceMixin`),
    so no ``InferenceMixin`` is needed here.
    """

    config: BagelVAEConfig

    def save_full_hf_checkpoint(self, output_dir: str, *, source_path: str, trainer: Any, state: Any) -> None:
        del trainer, state
        shutil.copytree(source_path, output_dir, dirs_exist_ok=True)


class BagelVAEAccelerated(VeOmniMixin, BagelVAE):
    pass


__all__ = ["BAGEL_VAE_PIXEL_SHAPE", "BagelVAEAccelerated", "BagelVAEOfflineMixin"]

"""SeedOmni V2 carrier hooks for BAGEL's latent VAE module."""

from __future__ import annotations

import shutil
from typing import Any

import torch

from veomni.utils.device import get_device_id, get_device_type

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import CPUPreprocessor, ModuleMixin, post_forward, pre_forward
from ....mixins.offline_encoding import OfflineEncodingMixin
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from ..sources import BAGEL_GENERATED_LATENT, BAGEL_VAE_CONTEXT
from .configuration import BagelVAEConfig
from .processing import crop_latent_to_image_shape, route_image_sources


BAGEL_VAE_PIXEL_SHAPE = "bagel_vae_pixel_shape"


class BagelVAECPUPreprocessor(CPUPreprocessor):
    """Worker-side image normalize for BAGEL VAE context/target images."""

    def __init__(
        self,
        image_processor: Any,
        dtype: torch.dtype,
        dummy_pixel_values: torch.Tensor,
        dummy_pixel_shape: torch.Tensor,
    ) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values
        self._dummy_pixel_shape = dummy_pixel_shape

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        infer_type = None if generation_kwargs is None else generation_kwargs.get("infer_type")
        route_image_sources(conversation_list, inference=inference, infer_type=infer_type)

        image_items: list[ConversationItem] = []
        missing_samples: list[list[ConversationItem]] = []
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], sources=[BAGEL_VAE_CONTEXT]))
            if sample_image_items:
                image_items.extend(sample_image_items)
            elif not inference:
                missing_samples.append(sample)

        if image_items:
            inputs = self._image_processor(
                images=[item.value for item in image_items], return_tensors="pt", dtype=self._dtype
            )
            for item, pixels, pixel_shape in zip(
                image_items, inputs["pixel_values"], inputs["pixel_shapes"], strict=True
            ):
                item.value = pixels.to(dtype=self._dtype)
                item.source = BAGEL_VAE_CONTEXT
                item.meta[BAGEL_VAE_PIXEL_SHAPE] = pixel_shape.to(dtype=torch.long)

        if not missing_samples:
            return

        if image_items:
            # VAE encode stacks every context item in the micro-batch, so dummies must
            # match a real processed image shape whenever the batch has one.
            reference_item = image_items[0]
            dummy_pixel_values = torch.zeros_like(reference_item.value, dtype=self._dtype)
            dummy_pixel_shape = reference_item.meta[BAGEL_VAE_PIXEL_SHAPE].to(dtype=torch.long)
        else:
            dummy_pixel_values = self._dummy_pixel_values.to(dtype=self._dtype)
            dummy_pixel_shape = self._dummy_pixel_shape.to(dtype=torch.long)

        for sample in missing_samples:
            sample.append(
                ConversationItem(
                    type="image",
                    value=dummy_pixel_values.clone(),
                    role="dummy",
                    source=BAGEL_VAE_CONTEXT,
                    meta={
                        BAGEL_VAE_PIXEL_SHAPE: dummy_pixel_shape.clone(),
                    },
                )
            )


class BagelVAEModuleMixin(OfflineEncodingMixin, ModuleMixin):
    """Carrier hooks for raw-image VAE encode and latent decode."""

    config: BagelVAEConfig

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._encode_items: list[ConversationItem] = []

    def build_cpu_preprocessor(self) -> CPUPreprocessor | None:
        # Full training and offline-cache production preprocess raw images here;
        # process-only training reads preprocessed cached conversations instead.
        if self.cache_mode == "process_only":
            return None
        if getattr(self, "_image_processor", None) is None:
            return None
        size = max(int(self.config.image_stride), int(self.config.downsample))
        dummy = torch.zeros(int(self.config.in_channels), size, size, dtype=self.dtype)
        dummy_shape = torch.tensor([size, size], dtype=torch.long)
        return BagelVAECPUPreprocessor(self._image_processor, self.dtype, dummy, dummy_shape)

    # ── Graph Entrypoints ──────────────────────────────────

    def encode_context(
        self,
        conversation_list: list[ConversationItem] | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}

        batched = [conversation_list]
        image_items = self._select_vae_context_image_items(batched)
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
        return {
            "conversation_list": conversation_list,
            "generated": {"type": "image", "value": self._image_processor.postprocess(pixel_values[-1])[0]},
        }

    # ── Training hooks ──────────────────────────────────

    @pre_forward("encode", "offline_encode")
    def encode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list

        self._encode_items = self._select_vae_context_items(conversation_list)

        self._metric_meter_stash_latent_tokens(self._vae_latent_token_lengths())

        pixel_values = torch.stack([item.value for item in self._encode_items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        return {"pixel_values": pixel_values}

    @post_forward("encode", "online_process")
    def encode_post(self, latents: torch.Tensor | list[torch.Tensor]) -> dict[str, Any]:
        conversation = self._conversation_carrier
        encode_items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []

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

    @post_forward("offline_encode")
    def offline_encode_post(self, encoded_cache: torch.Tensor) -> dict[str, Any]:
        conversation = self._conversation_carrier
        encode_items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []

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

        self._encode_items = self._select_vae_context_items(conversation_list)
        encoded_cache: list[torch.Tensor] = []
        for item in self._encode_items:
            cache = item.value
            if not torch.is_tensor(cache):
                raise ValueError("BAGEL VAE online_process requires tensor posterior cache.")
            encoded_cache.append(cache.detach().to(device=self._online_process_device()))
        return {"encoded_cache": encoded_cache}

    # ── Internal helpers ──────────────────────────────────

    def _select_vae_context_items(
        self, conversation_list: list[list[ConversationItem]] | None
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelVAE requires conversation_list to select VAE context items.")

        return list(iter_desired_items(conversation_list, types=["image"], sources=[BAGEL_VAE_CONTEXT]))

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

    def _select_vae_context_image_items(
        self, conversation_list: list[list[ConversationItem]] | None
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelVAE encode_context requires conversation_list to select context images.")

        image_items: list[ConversationItem] = []
        for item in iter_desired_items(
            conversation_list,
            types=["image"],
            sources=[BAGEL_VAE_CONTEXT],
        ):
            if not is_dummy(item):
                image_items.append(item)
        return image_items

    def _vae_latent_token_lengths(self) -> list[int]:
        # Latent tokens per image = (H // ds) * (W // ds), using the real pixel shape
        ds = max(int(self.config.downsample), 1)
        lengths: list[int] = []
        for item in self._encode_items:
            if is_dummy(item):
                continue
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

    def save_full_hf_checkpoint(self, output_dir: str, *, source_path: str, trainer: Any, state: Any) -> None:
        del trainer, state
        shutil.copytree(source_path, output_dir, dirs_exist_ok=True)


class BagelVAEMetricMeterMixin(MetricMeterMixin):
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


__all__ = [
    "BagelVAECPUPreprocessor",
    "BagelVAEModuleMixin",
    "BagelVAEMetricMeterMixin",
]

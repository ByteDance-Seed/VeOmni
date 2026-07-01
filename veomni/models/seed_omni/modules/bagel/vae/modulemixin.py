"""SeedOmni V2 carrier hooks for BAGEL's latent VAE module."""

from __future__ import annotations

import shutil
from typing import Any

import torch

from veomni.utils.device import get_device_id, get_device_type

from ....mixins.modulemixin import CPUPreprocessor, ModuleMixin, post_forward, pre_forward
from ....mixins.offline_encoding import ENCODED_CACHE_KIND_META_KEY, OfflineEncodingMixin
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from ..sources import BAGEL_GENERATED_LATENT, BAGEL_VAE_CONTEXT
from .cache import BAGEL_VAE_POSTERIOR_CACHE_KIND
from .configuration import BagelVAEConfig
from .processing import crop_latent_to_image_shape, route_image_sources


BAGEL_VAE_PIXEL_SHAPE = "bagel_vae_pixel_shape"


class BagelVAECPUPreprocessor(CPUPreprocessor):
    """Worker-side image normalize for BAGEL VAE context/target images."""

    def __init__(self, image_processor: Any, dtype: torch.dtype) -> None:
        self._image_processor = image_processor
        self._dtype = dtype

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        infer_type = None if generation_kwargs is None else generation_kwargs.get("infer_type")
        route_image_sources(conversation_list, inference=inference, infer_type=infer_type)

        image_items = []
        for item in iter_desired_items(conversation_list, types=["image"], sources=[BAGEL_VAE_CONTEXT]):
            if not is_dummy(item):
                image_items.append(item)
        if not image_items:
            return

        inputs = self._image_processor(
            images=[item.value for item in image_items], return_tensors="pt", dtype=self._dtype
        )
        for item, pixels, pixel_shape in zip(image_items, inputs["pixel_values"], inputs["pixel_shapes"], strict=True):
            item.value = pixels.to(dtype=self._dtype)
            item.source = BAGEL_VAE_CONTEXT
            item.meta[BAGEL_VAE_PIXEL_SHAPE] = pixel_shape.to(dtype=torch.long)


class BagelVAEModuleMixin(OfflineEncodingMixin, ModuleMixin):
    """Carrier hooks for raw-image VAE encode and latent decode."""

    config: BagelVAEConfig

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._encode_items: list[ConversationItem] = []
        self._decode_items: list[ConversationItem] = []
        self._encode_is_dummy = False
        self._decode_is_dummy = False

    def build_cpu_preprocessor(self) -> CPUPreprocessor | None:
        # Full training and offline-cache production preprocess raw images here;
        # process-only training reads preprocessed cached conversations instead.
        if getattr(self.config, "cache_mode", "full") == "process_only":
            return None
        if getattr(self, "_image_processor", None) is None:
            return None
        return BagelVAECPUPreprocessor(self._image_processor, self.dtype)

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
        self._encode_is_dummy = False

        self._encode_items = self._select_vae_encode_items(conversation_list)
        if not self._encode_items:
            self._encode_is_dummy = True
            return self.dummy_inputs(kind="encode")

        pixel_values = torch.stack([item.value for item in self._encode_items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        return {"pixel_values": pixel_values}

    @post_forward("encode", "online_process")
    def encode_post(self, latents: torch.Tensor) -> dict[str, Any]:
        conversation = self._conversation_carrier
        encode_items = self._encode_items
        encode_is_dummy = self._encode_is_dummy
        self._conversation_carrier = None
        self._encode_items = []
        self._encode_is_dummy = False

        if encode_is_dummy:
            if conversation is not None:
                value = latents.squeeze(0) if latents.dim() == 4 and latents.shape[0] == 1 else latents
                for sample in conversation:
                    sample.append(
                        ConversationItem(
                            type="image",
                            value=value,
                            role="dummy",
                            source=BAGEL_VAE_CONTEXT,
                            meta={"source": "bagel_vae"},
                        )
                    )
            return {"conversation_list": conversation}

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
        encode_is_dummy = self._encode_is_dummy
        self._conversation_carrier = None
        self._encode_items = []
        self._encode_is_dummy = False

        if encode_is_dummy:
            return {"conversation_list": conversation}

        for item, cache_tensor in zip(encode_items, encoded_cache, strict=True):
            item.type = "image"
            cache_tensor = crop_latent_to_image_shape(
                cache_tensor,
                item.meta.get(BAGEL_VAE_PIXEL_SHAPE),
                downsample=int(self.config.downsample),
            )
            item.value = cache_tensor.detach().to(device=self.device, dtype=self.dtype)
            item.source = BAGEL_VAE_CONTEXT
            item.meta = {ENCODED_CACHE_KIND_META_KEY: BAGEL_VAE_POSTERIOR_CACHE_KIND}
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
        self._encode_is_dummy = False

        self._encode_items = self._select_vae_posterior_cache_items(conversation_list)
        if not self._encode_items:
            self._encode_is_dummy = True
            return self.dummy_inputs(kind="online_process")

        encoded_cache: list[torch.Tensor] = []
        for item in self._encode_items:
            cache = item.value
            if not torch.is_tensor(cache):
                raise ValueError("BAGEL VAE online_process requires tensor posterior cache.")
            encoded_cache.append(cache.detach().to(device=self._online_process_device()))
        return {"encoded_cache": encoded_cache}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._decode_items = []
        self._decode_is_dummy = False

        self._decode_items = self._select_vae_decode_items(conversation_list)
        if not self._decode_items:
            self._decode_is_dummy = True
            return self.dummy_inputs(kind="decode")

        latents = []
        for item in self._decode_items:
            latents.append(
                item.value.detach().squeeze(0)
                if item.value.dim() == 4 and item.value.shape[0] == 1
                else item.value.detach()
            )
        return {"latents": torch.stack(latents, dim=0).to(device=self.device, dtype=self.dtype)}

    @post_forward("decode")
    def decode_post(self, pixel_values: torch.Tensor) -> dict[str, Any]:
        conversation = self._conversation_carrier
        decode_items = self._decode_items
        decode_is_dummy = self._decode_is_dummy
        self._conversation_carrier = None
        self._decode_items = []
        self._decode_is_dummy = False

        if decode_is_dummy:
            return {"conversation_list": conversation}

        for item, image in zip(decode_items, pixel_values, strict=True):
            item.type = "image"
            item.value = image.to(device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation}

    # ── Dummy helpers ──────────────────────────────────

    def dummy_inputs(self, kind: str = "encode") -> dict[str, Any]:
        if kind == "decode":
            size = max(1, int(self.config.resolution) // max(int(self.config.downsample), 1))
            return {
                "latents": torch.zeros(
                    1, int(self.config.z_channels), size, size, device=self.device, dtype=self.dtype
                )
            }

        if kind == "online_process":
            size = max(1, int(self.config.resolution) // max(int(self.config.downsample), 1))
            return {
                "encoded_cache": torch.zeros(
                    1,
                    2,
                    int(self.config.z_channels),
                    size,
                    size,
                    device=self._online_process_device(),
                    dtype=self._online_process_dtype(),
                ),
            }

        size = max(int(self.config.image_stride), int(self.config.downsample))
        return {
            "pixel_values": torch.zeros(
                1, int(self.config.in_channels), size, size, device=self.device, dtype=self.dtype
            )
        }

    # ── Internal helpers ──────────────────────────────────

    def _select_vae_encode_items(
        self, conversation_list: list[list[ConversationItem]] | None
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelVAE encode requires conversation_list to select image items.")

        encode_items: list[ConversationItem] = []
        for item in iter_desired_items(conversation_list, types=["image"], sources=[BAGEL_VAE_CONTEXT]):
            if not is_dummy(item):
                encode_items.append(item)
        return encode_items

    def _select_vae_posterior_cache_items(
        self, conversation_list: list[list[ConversationItem]] | None
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelVAE online_process requires conversation_list to select posterior cache items.")

        cached_items: list[ConversationItem] = []
        for item in iter_desired_items(
            conversation_list,
            types=["image"],
            sources=[BAGEL_VAE_CONTEXT],
            meta_keys=[ENCODED_CACHE_KIND_META_KEY],
        ):
            if is_dummy(item):
                continue
            kind = item.meta.get(ENCODED_CACHE_KIND_META_KEY)
            if kind != BAGEL_VAE_POSTERIOR_CACHE_KIND:
                raise ValueError(
                    f"BAGEL VAE expected encoded cache kind {BAGEL_VAE_POSTERIOR_CACHE_KIND!r}, got {kind!r}."
                )
            cached_items.append(item)
        return cached_items

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


__all__ = ["BAGEL_VAE_PIXEL_SHAPE", "BagelVAECPUPreprocessor", "BagelVAEModuleMixin"]

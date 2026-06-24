"""SeedOmni V2 carrier hooks for BAGEL's latent VAE module."""

from __future__ import annotations

from typing import Any

import torch

from ....mixins.modulemixin import CPUPreprocessor, ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from ..sources import BAGEL_GENERATED_LATENT, BAGEL_VAE_CONTEXT
from .configuration import BagelVAEConfig


_OMNI_PIXELS = "bagel_vae_pixels"


class BagelVAECPUPreprocessor(CPUPreprocessor):
    """Worker-side image normalize for BAGEL training VAE targets."""

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
        del generation_kwargs
        if inference:
            return
        image_items = []
        # Training data routes image branches by role: user images feed SigLIP,
        # assistant images feed VAE.
        for item in iter_desired_items(conversation_list, types=["image"], roles=["assistant"]):
            if not is_dummy(item) and not item.meta.get(_OMNI_PIXELS):
                image_items.append(item)
        if not image_items:
            return

        inputs = self._image_processor(
            images=[item.value for item in image_items], return_tensors="pt", dtype=self._dtype
        )
        for item, pixels in zip(image_items, inputs["pixel_values"], strict=True):
            item.value = pixels.to(dtype=self._dtype)
            item.source = BAGEL_VAE_CONTEXT
            item.meta[_OMNI_PIXELS] = True


class BagelVAEModuleMixin(ModuleMixin):
    """Carrier hooks for raw-image VAE encode and latent decode."""

    config: BagelVAEConfig

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._encode_items: list[ConversationItem] = []
        self._decode_items: list[ConversationItem] = []
        self._encode_is_dummy = False
        self._decode_is_dummy = False

    def build_cpu_preprocessor(self) -> CPUPreprocessor | None:
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

        inputs = self._image_processor(
            images=[item.value for item in image_items],
            return_tensors="pt",
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.encode(pixel_values=inputs["pixel_values"])
        for image_item, latent in zip(image_items, outputs["latents"], strict=True):
            # Edit keeps the raw user image in place for SigLIP and inserts the
            # VAE context latent immediately before that image for flow context.
            conversation_list.insert(
                conversation_list.index(image_item),
                ConversationItem(
                    type="output",
                    value=latent.to(device=self.device, dtype=self.dtype),
                    role="assistant",
                    source=BAGEL_VAE_CONTEXT,
                    meta={},
                ),
            )
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

    @pre_forward("encode")
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

        if all(item.meta.get(_OMNI_PIXELS) for item in self._encode_items):
            pixel_values = torch.stack([item.value for item in self._encode_items], dim=0).to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            return {"pixel_values": pixel_values}

        return self._image_processor(
            images=[item.value for item in self._encode_items],
            return_tensors="pt",
            device=self.device,
            dtype=self.dtype,
        )

    @post_forward("encode")
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
                            type="output",
                            value=value,
                            role="dummy",
                            meta={"source": "bagel_vae"},
                        )
                    )
            return {"conversation_list": conversation}

        for item, latent in zip(encode_items, latents, strict=True):
            item.type = "output"
            item.value = latent.to(device=self.device, dtype=self.dtype)
            item.source = BAGEL_VAE_CONTEXT
        return {"conversation_list": conversation}

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

        # Training VAE encode consumes generation target images. Data-level role
        # routing keeps these separate from SigLIP's user-image branch.
        encode_items: list[ConversationItem] = []
        for item in iter_desired_items(conversation_list, types=["image"], roles=["assistant"]):
            if not is_dummy(item):
                encode_items.append(item)
        return encode_items

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

        # Same staged source contract as SigLIP: once inference prompt
        # preprocessing materializes branch sources, raw edit images can arrive
        # tagged as BAGEL_VAE_CONTEXT. Today build_conversation() still creates
        # prompt images with source=None, so keep that fallback.
        image_items: list[ConversationItem] = []
        for item in iter_desired_items(
            conversation_list,
            types=["image"],
            roles=["user"],
            sources=[None, BAGEL_VAE_CONTEXT],
        ):
            if not is_dummy(item):
                image_items.append(item)
        return image_items


__all__ = ["BagelVAECPUPreprocessor", "BagelVAEModuleMixin"]

"""SeedOmni V2 carrier hooks for BAGEL's latent VAE module."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem
from ....module import ModuleMixin, post_forward, pre_forward
from ..carrier_updates import append as carrier_append
from ..carrier_updates import materialize_carrier_updates
from .configuration import BagelVAEConfig
from .processing import (
    as_batched_decode_conversation,
    context_encode_image_items,
    decoded_tensor_to_pil,
    insert_context_encoded_latents,
    latent_decode_items,
    prepare_decode_inputs,
    prepare_encode_inputs,
    raw_context_image_items,
    raw_image_encode_items,
    scatter_decoded_images,
    scatter_encoded_latents,
)


class BagelVAEModuleMixin(ModuleMixin):
    """Carrier hooks for raw-image VAE encode and latent decode."""

    config: BagelVAEConfig

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._encode_items: list[ConversationItem] = []
        self._decode_items: list[ConversationItem] = []

    def decode(
        self,
        latents: torch.Tensor | None = None,
        conversation_list: Any | None = None,
        generation_kwargs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del generation_kwargs, kwargs
        if latents is not None:
            return self._decode_latents(latents)
        if conversation_list is not None:
            return self._decode_conversation(conversation_list)
        dummy = self.dummy_inputs(kind="decode")
        outputs = self._decode_latents(dummy["latents"])
        outputs["is_dummy"] = True
        return outputs

    def _decode_conversation(self, conversation_list: Any) -> dict[str, Any]:
        batched = as_batched_decode_conversation(conversation_list)
        decode_items = latent_decode_items(batched)
        if not decode_items:
            return {"conversation_list": conversation_list}
        inputs = prepare_decode_inputs(decode_items, device=self.device, dtype=self.dtype)
        outputs = self._decode_latents(inputs["latents"])
        pixel_values = outputs["pixel_values"]
        scatter_decoded_images(decode_items, pixel_values, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation_list}

    def _encode_context_conversation(self, conversation_list: Any) -> dict[str, Any]:
        batched = as_batched_decode_conversation(conversation_list)
        context_items = raw_context_image_items(batched)
        if not context_items:
            return {"conversation_list": conversation_list}

        image_items = context_encode_image_items(context_items)
        inputs = prepare_encode_inputs(image_items, config=self.config, device=self.device, dtype=self.dtype)
        outputs = self._encode_pixel_values(inputs["pixel_values"])
        insert_context_encoded_latents(
            context_items,
            outputs["latents"],
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation_list}

    @pre_forward("encode")
    def encode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._encode_items = raw_image_encode_items(conversation_list)
        if not self._encode_items:
            return {"pixel_values": None}

        return prepare_encode_inputs(
            self._encode_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )

    @post_forward("encode")
    def encode_post(self, latents: torch.Tensor, is_dummy: bool = False) -> dict[str, Any]:
        conversation = self._conversation_carrier
        encode_items = self._encode_items
        self._conversation_carrier = None
        self._encode_items = []

        if is_dummy:
            if conversation is not None:
                value = latents.squeeze(0) if latents.dim() == 4 and latents.shape[0] == 1 else latents
                materialize_carrier_updates(
                    conversation,
                    [
                        carrier_append(
                            sample,
                            ConversationItem(
                                type="output",
                                value=value,
                                role="dummy",
                                meta={"source": "bagel_vae"},
                            ),
                        )
                        for sample in conversation
                    ],
                )
            return {"conversation_list": conversation}

        scatter_encoded_latents(encode_items, latents, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation}

    @pre_forward("decode")
    def decode_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._decode_items = latent_decode_items(conversation_list)
        if not self._decode_items:
            return {"latents": None}
        return prepare_decode_inputs(self._decode_items, device=self.device, dtype=self.dtype)

    @post_forward("decode")
    def decode_post(self, pixel_values: torch.Tensor, is_dummy: bool = False) -> dict[str, Any]:
        conversation = self._conversation_carrier
        decode_items = self._decode_items
        self._conversation_carrier = None
        self._decode_items = []

        if is_dummy:
            return {"conversation_list": conversation}

        scatter_decoded_images(decode_items, pixel_values, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation}

    def dummy_inputs(self, kind: str = "encode") -> dict[str, Any]:
        if kind == "decode":
            size = max(1, int(self.config.resolution) // max(int(self.config.downsample), 1))
            return {
                "latents": torch.zeros(
                    1,
                    int(self.config.z_channels),
                    size,
                    size,
                    device=self.device,
                    dtype=self.dtype,
                )
            }
        size = max(int(self.config.image_stride), int(self.config.downsample))
        return {
            "pixel_values": torch.zeros(
                1,
                int(self.config.in_channels),
                size,
                size,
                device=self.device,
                dtype=self.dtype,
            )
        }

    def finalize(self, *, ctx: dict[str, Any]) -> dict[str, Any]:
        for sample in reversed(as_batched_decode_conversation(ctx.get("conversation_list", []))):
            for item in reversed(sample):
                if item.type != "image" or item.role != "assistant" or not torch.is_tensor(item.value):
                    continue
                return {"generated": {"type": "image", "value": decoded_tensor_to_pil(item.value)}}
        return {}


__all__ = ["BagelVAEModuleMixin"]

"""SeedOmni V2 hooks for BAGEL's SigLIP NaViT vision encoder."""

from __future__ import annotations

from typing import Any

import torch

from ....mixins.modulemixin import CPUPreprocessor, ModuleMixin, post_forward, pre_forward
from ....mixins.tracemixin import TraceMixin
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT
from .configuration import BagelSiglipNavitConfig


_OMNI_PATCHES = "bagel_siglip_navit_patches"
_OMNI_POSITION_IDS = "bagel_siglip_navit_position_ids"
_OMNI_TOKEN_LEN = "bagel_siglip_navit_token_len"


class BagelSiglipNavitCPUPreprocessor(CPUPreprocessor):
    """Worker-side image patchify for BAGEL SigLIP NaViT training inputs."""

    def __init__(self, image_processor: Any, dtype: torch.dtype) -> None:
        self._image_processor = image_processor
        self._dtype = dtype

    def __call__(self, conversation_list: list[list[ConversationItem]]) -> None:
        image_items: list[ConversationItem] = []
        # Training data routes image branches by role: user images feed SigLIP,
        # assistant images feed VAE.
        for item in iter_desired_items(conversation_list, types=["image"], roles=["user"]):
            if item.meta.get(_OMNI_PATCHES):
                continue
            image_items.append(item)

        if not image_items:
            return

        inputs = self._image_processor(
            images=[item.value for item in image_items], return_tensors="pt", dtype=self._dtype
        )
        lengths = inputs["token_lens"].detach().cpu().reshape(-1).tolist()
        pixel_chunks = torch.split(inputs["patchified_pixel_values"], lengths, dim=0)
        position_chunks = torch.split(inputs["patchified_position_ids"], lengths, dim=0)
        for item, pixels, position_ids, length in zip(
            image_items, pixel_chunks, position_chunks, lengths, strict=True
        ):
            item.value = pixels.to(dtype=self._dtype)
            item.source = BAGEL_SIGLIP_CONTEXT
            item.meta[_OMNI_PATCHES] = True
            item.meta[_OMNI_POSITION_IDS] = position_ids.to(dtype=torch.long)
            item.meta[_OMNI_TOKEN_LEN] = int(length)


class BagelSiglipNavitModuleMixin(ModuleMixin):
    """Carrier hooks for BAGEL visual-understanding image features."""

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._image_items: list[ConversationItem] = []
        self._forward_is_dummy = False

    def build_cpu_preprocessor(self) -> CPUPreprocessor | None:
        """Worker-side image patchify for training batches."""
        if getattr(self, "_image_processor", None) is None:
            return None
        return BagelSiglipNavitCPUPreprocessor(self._image_processor, self.dtype)

    # ── Graph Entrypoints ──────────────────────────────────

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        batched = [conversation_list]
        image_items = self._select_siglip_image_items(batched)
        if not image_items:
            return {"conversation_list": conversation_list}

        inputs = self._image_processor(
            images=[item.value for item in image_items],
            return_tensors="pt",
            device=self.device,
            dtype=self.dtype,
        )
        outputs = self.forward(**inputs)
        token_lens = outputs.get("token_lens", inputs["token_lens"])
        self._scatter_image_embeds(image_items, outputs["image_embeds"], token_lens)
        return {"conversation_list": batched[0]}

    # ── Training hooks ──────────────────────────────────

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._forward_is_dummy = False

        self._image_items = self._select_siglip_image_items(conversation_list)
        if not self._image_items:
            self._forward_is_dummy = True
            return self.dummy_inputs()

        if all(item.meta.get(_OMNI_PATCHES) for item in self._image_items):
            return self._inputs_from_preprocessed_items(self._image_items)

        inputs = self._image_processor(
            images=[item.value for item in self._image_items],
            return_tensors="pt",
            device=self.device,
            dtype=self.dtype,
        )
        return inputs

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        token_lens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        conversation = self._conversation_carrier
        image_items = self._image_items
        is_dummy = self._forward_is_dummy
        self._conversation_carrier = None
        self._image_items = []
        self._forward_is_dummy = False

        if is_dummy:
            value = image_embeds.squeeze(0) if image_embeds.dim() == 3 and image_embeds.shape[0] == 1 else image_embeds
            for sample in conversation:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=value,
                        role="dummy",
                        source=BAGEL_SIGLIP_CONTEXT,
                        meta={"source": "bagel_siglip_navit"},
                    )
                )
            return {"conversation_list": conversation}

        if token_lens is None:
            raise ValueError("BagelSiglipNavit.forward_post requires token_lens for non-dummy outputs.")

        self._scatter_image_embeds(image_items, image_embeds, token_lens)
        return {"conversation_list": conversation}

    # ── Dummy helpers ──────────────────────────────────

    def dummy_inputs(self) -> dict[str, Any]:
        patch_dim = self.config.num_channels * self.config.patch_size * self.config.patch_size
        token_lens = torch.tensor([1], dtype=torch.int32, device=self.device)
        return {
            "patchified_pixel_values": torch.zeros(1, patch_dim, device=self.device, dtype=self.dtype),
            "patchified_position_ids": torch.zeros(1, dtype=torch.long, device=self.device),
            "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            "max_seqlen": 1,
            "token_lens": token_lens,
        }

    # ── Internal helpers ──────────────────────────────────

    def _select_siglip_image_items(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelSiglipNavit requires conversation_list to select image items.")

        # Training batches normally reach this module after
        # BagelSiglipNavitCPUPreprocessor, which patchifies raw images and tags
        # them as BAGEL_SIGLIP_CONTEXT. Inference does not have that prompt
        # preprocessor yet: edit graphs currently serialize
        # bagel_vae.encode_context -> bagel_siglip_navit with explicit edges
        # instead of materializing branch sources up front, so raw prompt images
        # may still arrive with source=None.
        image_items: list[ConversationItem] = []
        for item in iter_desired_items(
            conversation_list,
            types=["image"],
            roles=["user"],
            sources=[None, BAGEL_SIGLIP_CONTEXT],
        ):
            image_items.append(item)
        return image_items

    def _inputs_from_preprocessed_items(
        self,
        image_items: list[ConversationItem],
    ) -> dict[str, Any]:
        token_lens = torch.tensor(
            [int(item.meta[_OMNI_TOKEN_LEN]) for item in image_items],
            dtype=torch.int32,
            device=self.device,
        )
        return {
            "patchified_pixel_values": torch.cat([item.value for item in image_items], dim=0).to(
                device=self.device, dtype=self.dtype, non_blocking=True
            ),
            "patchified_position_ids": torch.cat(
                [item.meta[_OMNI_POSITION_IDS] for item in image_items],
                dim=0,
            ).to(device=self.device, dtype=torch.long, non_blocking=True),
            "cu_seqlens": torch.nn.functional.pad(torch.cumsum(token_lens, dim=0), (1, 0)).to(torch.int32),
            "max_seqlen": int(token_lens.max().item()),
            "token_lens": token_lens,
        }

    def _scatter_image_embeds(
        self,
        image_items: list[ConversationItem],
        image_embeds: torch.Tensor,
        token_lens: torch.Tensor,
    ) -> None:
        offset = 0
        lengths = token_lens.detach().cpu().reshape(-1).tolist()
        for item, length in zip(image_items, lengths, strict=True):
            item.value = image_embeds[offset : offset + int(length)].to(device=self.device, dtype=self.dtype)
            item.source = BAGEL_SIGLIP_CONTEXT
            offset += int(length)

        if offset != int(image_embeds.shape[0]):
            raise RuntimeError("BAGEL SigLIP token count mismatch during feature scatter.")


class BagelSiglipNavitTraceMixin(TraceMixin):
    """Per-module training trace for BAGEL SigLIP NaViT."""

    config: BagelSiglipNavitConfig

    def trace_token_lengths(self, method: str, data: dict[str, Any]) -> list[int]:
        if method != "forward":
            return []
        token_lens = data.get("token_lens")
        if token_lens is None:
            return []
        return [int(value) for value in token_lens.detach().cpu().reshape(-1).tolist()]

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config
        dim = cfg.hidden_size
        heads = cfg.num_attention_heads
        head_dim = dim // heads
        patch_embed_n = dim * cfg.num_channels * cfg.patch_size * cfg.patch_size
        attn_linear_n = dim * 4 * dim
        mlp_n = dim * cfg.intermediate_size * 2
        connector_n = dim * cfg.output_size + cfg.output_size * cfg.output_size
        dense_n = patch_embed_n + (attn_linear_n + mlp_n) * cfg.num_hidden_layers + connector_n
        tokens = sum(seqlens)
        seqlen_sq = sum(length * length for length in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * heads * cfg.num_hidden_layers
        return (dense_flops + attn_flops) / 1e12


__all__ = ["BagelSiglipNavitCPUPreprocessor", "BagelSiglipNavitModuleMixin", "BagelSiglipNavitTraceMixin"]

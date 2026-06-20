"""SeedOmni V2 hooks for BAGEL's SigLIP NaViT vision encoder."""

from __future__ import annotations

from typing import Any

import torch

from ....conversation import ConversationItem
from ....module import ModuleMixin, post_forward, pre_forward
from ....tracemixin import TraceMixin
from .configuration import BagelSiglipNavitConfig
from .processing import prepare_image_batch, scatter_image_embeds, user_raw_image_items


class BagelSiglipNavitModuleMixin(ModuleMixin):
    """Carrier hooks for BAGEL visual-understanding image features."""

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._image_items: list[ConversationItem] = []
        self._image_token_lens: torch.Tensor | None = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list
        self._image_items = user_raw_image_items(
            conversation_list or [],
            output_size=int(self.config.output_size),
        )
        self._image_token_lens = None
        if not self._image_items:
            return {"patchified_pixel_values": None}

        inputs = prepare_image_batch(
            self._image_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )
        self._image_token_lens = inputs["token_lens"]
        return inputs

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        token_lens: torch.Tensor | None = None,
        is_dummy: bool = False,
    ) -> dict[str, Any]:
        conversation = self._conversation_carrier
        image_items = self._image_items
        self._conversation_carrier = None
        self._image_items = []
        self._image_token_lens = None

        if is_dummy:
            if conversation is not None:
                value = (
                    image_embeds.squeeze(0) if image_embeds.dim() == 3 and image_embeds.shape[0] == 1 else image_embeds
                )
                for sample in conversation:
                    sample.append(
                        ConversationItem(
                            type="image",
                            value=value,
                            role="dummy",
                            meta={"source": "bagel_siglip_navit"},
                        )
                    )
            return {"conversation_list": conversation}

        if token_lens is None:
            token_lens = self._image_token_lens
        if token_lens is None:
            raise ValueError("BagelSiglipNavit.forward_post requires token_lens for non-dummy outputs.")

        scatter_image_embeds(image_items, image_embeds, token_lens, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation}

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

    def generate(
        self,
        conversation_list: list[ConversationItem] | list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        if conversation_list is None:
            return {"conversation_list": conversation_list}

        image_items = user_raw_image_items(conversation_list, output_size=self.config.output_size)
        if not image_items:
            return {"conversation_list": conversation_list}

        inputs = prepare_image_batch(
            image_items,
            config=self.config,
            device=self.device,
            dtype=self.dtype,
        )
        with torch.inference_mode():
            outputs = self.forward(**inputs)
        token_lens = outputs.get("token_lens")
        if token_lens is None:
            token_lens = inputs["token_lens"]
        scatter_image_embeds(image_items, outputs["image_embeds"], token_lens, device=self.device, dtype=self.dtype)
        return {"conversation_list": conversation_list}


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


__all__ = ["BagelSiglipNavitModuleMixin", "BagelSiglipNavitTraceMixin"]

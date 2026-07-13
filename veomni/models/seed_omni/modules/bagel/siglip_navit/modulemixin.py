"""SeedOmni V2 hooks for BAGEL's SigLIP NaViT vision encoder."""

from __future__ import annotations

from typing import Any

import torch

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import CPUPreprocessor, ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT
from .configuration import BagelSiglipNavitConfig


_OMNI_POSITION_IDS = "bagel_siglip_navit_position_ids"
_OMNI_TOKEN_LEN = "bagel_siglip_navit_token_len"


class BagelSiglipNavitCPUPreprocessor(CPUPreprocessor):
    """Worker-side image patchify for BAGEL SigLIP NaViT context images."""

    def __init__(self, image_processor: Any, dtype: torch.dtype, dummy_pixel_values: torch.Tensor) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values

    def __call__(
        self,
        conversation_list: list[list[ConversationItem]],
        *,
        inference: bool = False,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        del generation_kwargs

        image_items: list[ConversationItem] = []
        for sample in conversation_list:
            sample_image_items = list(iter_desired_items([sample], types=["image"], sources=[BAGEL_SIGLIP_CONTEXT]))
            if sample_image_items:
                image_items.extend(sample_image_items)
            elif not inference:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values.to(dtype=self._dtype).clone(),
                        role="dummy",
                        source=BAGEL_SIGLIP_CONTEXT,
                        meta={
                            _OMNI_POSITION_IDS: torch.zeros(1, dtype=torch.long),
                            _OMNI_TOKEN_LEN: 1,
                        },
                    )
                )

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
            item.meta[_OMNI_POSITION_IDS] = position_ids.to(dtype=torch.long)
            item.meta[_OMNI_TOKEN_LEN] = int(length)


class BagelSiglipNavitModuleMixin(ModuleMixin):
    """Carrier hooks for BAGEL visual-understanding image features."""

    def init_omni_state(self) -> None:
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._image_items: list[ConversationItem] = []

    def build_cpu_preprocessor(self) -> CPUPreprocessor | None:
        """Worker-side image patchify for training batches."""
        if getattr(self, "_image_processor", None) is None:
            return None
        patch_dim = self.config.num_channels * self.config.patch_size * self.config.patch_size
        dummy = torch.zeros(1, patch_dim, dtype=self.dtype)
        return BagelSiglipNavitCPUPreprocessor(self._image_processor, self.dtype, dummy)

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

        inputs = self._inputs_from_preprocessed_items(image_items)
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

        self._image_items = self._select_siglip_image_items(conversation_list)
        out = self._inputs_from_preprocessed_items(self._image_items)
        self.metric_meter_set_seqlens("forward", self._metric_sample_token_lens(conversation_list))
        return out

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        token_lens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        conversation = self._conversation_carrier
        image_items = self._image_items
        self._conversation_carrier = None
        self._image_items = []

        if token_lens is None:
            raise ValueError("BagelSiglipNavit.forward_post requires token_lens for non-dummy outputs.")

        self._scatter_image_embeds(image_items, image_embeds, token_lens)
        return {"conversation_list": conversation}

    # ── Internal helpers ──────────────────────────────────

    def _select_siglip_image_items(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
    ) -> list[ConversationItem]:
        if conversation_list is None:
            raise ValueError("BagelSiglipNavit requires conversation_list to select image items.")

        image_items: list[ConversationItem] = []
        for item in iter_desired_items(
            conversation_list,
            types=["image"],
            sources=[BAGEL_SIGLIP_CONTEXT],
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

    def _metric_sample_token_lens(
        self,
        conversation_list: list[list[ConversationItem]],
    ) -> list[int]:
        sample_lens: list[int] = []
        for sample in conversation_list:
            sample_len = 0
            # SigLIP encodes per image, including dummy carriers, but
            # multi-source metering needs one aggregated length per sample.
            for item in iter_desired_items([sample], types=["image"], sources=[BAGEL_SIGLIP_CONTEXT]):
                sample_len += int(item.meta[_OMNI_TOKEN_LEN])
            sample_lens.append(sample_len)
        return sample_lens

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


class BagelSiglipNavitMetricMeterMixin(MetricMeterMixin):
    """Per-module training trace for BAGEL SigLIP NaViT."""

    config: BagelSiglipNavitConfig

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


__all__ = ["BagelSiglipNavitCPUPreprocessor", "BagelSiglipNavitModuleMixin", "BagelSiglipNavitMetricMeterMixin"]

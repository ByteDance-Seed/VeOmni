"""SeedOmni V2 hooks for BAGEL's SigLIP NaViT vision encoder."""

from __future__ import annotations

from typing import Any

import torch

from ......distributed.parallel_state import get_parallel_state
from ......distributed.sequence_parallel import gather_outputs
from ....mixins.base_mixin import BaseMixin
from ....mixins.inference_module_mixin import InferenceModuleMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, iter_desired_items
from ..sources import BAGEL_SIGLIP_CONTEXT
from .configuration import BagelSiglipNavitConfig
from .processing import _OMNI_POSITION_IDS, _OMNI_TOKEN_LEN, BagelSiglipNavitPreprocessor


def select_siglip_image_items(
    conversation_list: list[list[ConversationItem]] | None,
) -> list[ConversationItem]:
    """Select BAGEL SigLIP image carriers (training + inference)."""
    if conversation_list is None:
        raise ValueError("BagelSiglipNavit requires conversation_list to select image items.")

    return list(
        iter_desired_items(
            conversation_list,
            types=["image"],
            sources=[BAGEL_SIGLIP_CONTEXT],
        )
    )


def siglip_inputs_from_preprocessed_items(
    image_items: list[ConversationItem],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Build NaViT forward inputs from CPU-preprocessed image items."""
    token_lens = torch.tensor(
        [int(item.meta[_OMNI_TOKEN_LEN]) for item in image_items],
        dtype=torch.int32,
        device=device,
    )
    return {
        "patchified_pixel_values": torch.cat([item.value for item in image_items], dim=0).to(
            device=device, dtype=dtype, non_blocking=True
        ),
        "patchified_position_ids": torch.cat(
            [item.meta[_OMNI_POSITION_IDS] for item in image_items],
            dim=0,
        ).to(device=device, dtype=torch.long, non_blocking=True),
        "cu_seqlens": torch.nn.functional.pad(torch.cumsum(token_lens, dim=0), (1, 0)).to(torch.int32),
        "max_seqlen": int(token_lens.max().item()),
        "token_lens": token_lens,
    }


def scatter_siglip_image_embeds(
    image_items: list[ConversationItem],
    image_embeds: torch.Tensor,
    token_lens: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Scatter NaViT outputs back onto conversation image items."""
    offset = 0
    lengths = token_lens.detach().cpu().reshape(-1).tolist()
    for item, length in zip(image_items, lengths, strict=True):
        item.value = image_embeds[offset : offset + int(length)].to(device=device, dtype=dtype)
        item.source = BAGEL_SIGLIP_CONTEXT
        offset += int(length)

    if offset != int(image_embeds.shape[0]):
        raise RuntimeError("BAGEL SigLIP token count mismatch during feature scatter.")


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`BagelSiglipNavit` modeling APIs."""

    config: BagelSiglipNavitConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: list[list[ConversationItem]] | None = None
        self._image_items: list[ConversationItem] = []
        self._sp_image_count: int | None = None
        self._sp_token_count: int | None = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: list[list[ConversationItem]] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        self._conversation_carrier = conversation_list

        self._image_items = select_siglip_image_items(conversation_list)
        out = siglip_inputs_from_preprocessed_items(self._image_items, device=self.device, dtype=self.dtype)
        self.metric_meter_set_seqlens("forward", self._metric_sample_token_lens(conversation_list))

        ps = get_parallel_state()
        if ps.sp_size == 1:
            self._sp_image_count = None
            self._sp_token_count = None
            return out

        if ps.cp_size != 1:
            raise ValueError(f"BAGEL SigLIP NaViT training supports Ulysses groups only; got cp_size={ps.cp_size}.")

        token_lens = out["token_lens"]
        image_count = int(token_lens.numel())
        if image_count == 0:
            raise ValueError("BAGEL SigLIP NaViT SP requires at least one real or dummy image.")
        self._sp_image_count = image_count
        self._sp_token_count = int(token_lens.sum().item())

        pad_images = (-image_count) % ps.sp_size
        if pad_images:
            token_lens = torch.cat((token_lens, token_lens.new_ones(pad_images)), dim=0)
            out["patchified_pixel_values"] = torch.cat(
                (
                    out["patchified_pixel_values"],
                    out["patchified_pixel_values"].new_zeros((pad_images, *out["patchified_pixel_values"].shape[1:])),
                ),
                dim=0,
            )
            out["patchified_position_ids"] = torch.cat(
                (
                    out["patchified_position_ids"],
                    out["patchified_position_ids"].new_zeros((pad_images, *out["patchified_position_ids"].shape[1:])),
                ),
                dim=0,
            )

        images_per_rank = int(token_lens.numel()) // ps.sp_size
        image_start = ps.sp_rank * images_per_rank
        image_end = image_start + images_per_rank
        full_cu_seqlens = torch.nn.functional.pad(torch.cumsum(token_lens, dim=0), (1, 0))
        token_start = int(full_cu_seqlens[image_start].item())
        token_end = int(full_cu_seqlens[image_end].item())
        local_token_lens = token_lens[image_start:image_end].contiguous()
        return {
            "patchified_pixel_values": out["patchified_pixel_values"][token_start:token_end].contiguous(),
            "patchified_position_ids": out["patchified_position_ids"][token_start:token_end].contiguous(),
            "cu_seqlens": torch.nn.functional.pad(torch.cumsum(local_token_lens, dim=0), (1, 0)).to(torch.int32),
            "max_seqlen": int(local_token_lens.max().item()),
            "token_lens": local_token_lens,
        }

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        token_lens: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        if token_lens is None:
            raise ValueError("BagelSiglipNavit.forward_post requires token_lens for non-dummy outputs.")

        ps = get_parallel_state()
        if ps.sp_size > 1:
            if self._sp_image_count is None or self._sp_token_count is None:
                raise RuntimeError("BAGEL SigLIP NaViT SP image shape was not initialized.")
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=ps.sp_group)
            token_lens = gather_outputs(token_lens, gather_dim=0, group=ps.sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._sp_token_count)
            token_lens = token_lens.narrow(0, 0, self._sp_image_count)

        conversation = self._conversation_carrier
        image_items = self._image_items
        self._conversation_carrier = None
        self._image_items = []
        self._sp_image_count = None
        self._sp_token_count = None

        scatter_siglip_image_embeds(
            image_items,
            image_embeds,
            token_lens,
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": conversation}

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


class InferenceMixin(InferenceModuleMixin):
    config: BagelSiglipNavitConfig
    device: torch.device
    dtype: torch.dtype

    def forward(
        self,
        patchified_pixel_values: torch.Tensor,
        patchified_position_ids: torch.LongTensor,
        cu_seqlens: torch.IntTensor,
        max_seqlen: int,
        token_lens: torch.IntTensor,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """IDE stub — implemented on :class:`BagelSiglipNavit` in ``modeling.py``."""
        ...

    def generate(
        self,
        conversation_list: list[ConversationItem] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        del kwargs
        batched = [conversation_list]
        image_items = select_siglip_image_items(batched)
        if not image_items:
            return {"conversation_list": conversation_list}

        inputs = siglip_inputs_from_preprocessed_items(image_items, device=self.device, dtype=self.dtype)
        outputs = self.forward(**inputs)
        token_lens = outputs.get("token_lens", inputs["token_lens"])
        scatter_siglip_image_embeds(
            image_items,
            outputs["image_embeds"],
            token_lens,
            device=self.device,
            dtype=self.dtype,
        )
        return {"conversation_list": batched[0]}


class MeterMixin(MetricMeterMixin):
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


class VeOmniMixin(BaseMixin, TrainingMixin, InferenceMixin, MeterMixin):
    """Carrier hooks for BAGEL visual-understanding image features."""

    preprocessor_class = BagelSiglipNavitPreprocessor


__all__ = [
    "InferenceMixin",
    "MeterMixin",
    "VeOmniMixin",
    "TrainingMixin",
]

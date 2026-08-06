"""VeOmni-accelerated Janus SigLIP — training hooks + FSDP dummy forward patch."""

from typing import Any, Dict, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

from ....mixins.base_mixin import BaseMixin
from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.training_module_mixin import TrainingModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from .configuration import JanusSiglipConfig
from .modeling import JanusSiglip
from .processing import JanusSiglipPreprocessor, JanusSiglipProcessor


_SOURCE = "janus_siglip"


class TrainingMixin(TrainingModuleMixin):
    """Training-graph hooks — depends on :class:`JanusSiglip` modeling APIs."""

    config: JanusSiglipConfig
    device: torch.device
    dtype: torch.dtype

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._conversation_carrier: Any = None
        self._sp_own_len: Optional[int] = None

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        items = list(iter_desired_items(conversation_list, types=["image"], sources=[_SOURCE]))
        pixel_values = torch.stack([it.value for it in items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        is_dummy_flag = all(is_dummy(it) for it in items)
        self._metric_meter_stash_tokens(int(pixel_values.shape[0]))

        if get_parallel_state().sp_size > 1:
            self._sp_own_len = pixel_values.size(0)
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=get_parallel_state().sp_group)
            return {"pixel_values": pixel_values}
        return {"pixel_values": pixel_values, "is_dummy": is_dummy_flag}

    def _metric_meter_stash_tokens(self, num_images: int) -> None:
        cfg = self.config.vision_config
        patches = (cfg.image_size // cfg.patch_size) ** 2
        self.metric_meter_set_seqlens("forward", [patches] * num_images)

    @post_forward("forward")
    def forward_post(self, image_embeds: torch.Tensor) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._sp_own_len)
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        items = list(iter_desired_items(conversation, types=["image"], sources=[_SOURCE]))
        for item, emb in zip(items, image_embeds, strict=True):
            item.value = emb
        return {"conversation_list": conversation}

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vision_config
        return {
            "pixel_values": torch.zeros(cfg.num_channels, cfg.image_size, cfg.image_size, dtype=self.dtype),
        }


class MeterMixin(MetricMeterMixin):
    config: JanusSiglipConfig

    def estimate_flops(self, seqlens: list[int]) -> float:
        cfg = self.config.vision_config
        dim = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        head_dim = dim // num_heads
        in_channels = getattr(cfg, "num_channels", 3)
        intermediate_size = int(dim * cfg.mlp_ratio)

        patch_embed_n = dim * in_channels * cfg.patch_size * cfg.patch_size
        attn_linear_n = dim * 4 * dim
        mlp_n = dim * intermediate_size * 2
        dense_n = patch_embed_n + (attn_linear_n + mlp_n) * num_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


class VeOmniMixin(BaseMixin, TrainingMixin, MeterMixin):
    config: JanusSiglipConfig
    _image_processor: JanusSiglipProcessor
    preprocessor_class = JanusSiglipPreprocessor


class JanusSiglipAccelerated(VeOmniMixin, JanusSiglip):
    """Training/runtime SigLIP — patches :meth:`forward` for FSDP dummy anchors."""

    def _dummy_image_embeds(self, pixel_values: torch.Tensor) -> torch.Tensor:
        cfg = self.config.vision_config
        b, _, h, w = pixel_values.shape
        num_patches = (h // cfg.patch_size) * (w // cfg.patch_size)
        return pixel_values.new_zeros(b, num_patches, cfg.projection_dim)

    def forward(
        self,
        pixel_values: Optional[torch.Tensor],
        is_dummy: bool = False,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        if is_dummy and not (self.training and get_parallel_state().fsdp_enabled):
            image_embeds = self._dummy_image_embeds(pixel_values)
        else:
            image_embeds = self._encode_pixel_values(pixel_values)
        return {"image_embeds": image_embeds}


__all__ = ["JanusSiglipAccelerated"]

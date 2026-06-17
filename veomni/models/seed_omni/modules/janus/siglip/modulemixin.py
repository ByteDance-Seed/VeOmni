from typing import Any, Dict, List, Optional

import torch

from ....conversation import ConversationItem, collect_desired_values, iter_desired_items
from ....module import ModuleMixin, post_forward, pre_forward
from ....tracemixin import TraceMixin
from .configuration import JanusSiglipConfig


class JanusSiglipModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None

    # Training hooks

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        pixel_values = self._pixels_from_raw_images(
            collect_desired_values(conversation_list, types=["image"], roles=["user"])
        )
        return {"pixel_values": pixel_values}

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        if is_dummy:
            assert image_embeds.shape[0] == 1
            image_embeds = image_embeds.squeeze(0)
            for sample in conversation:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=image_embeds,
                        role="dummy",
                        meta={"source": "janus_siglip"},
                    )
                )
        else:
            items = list(iter_desired_items(conversation, types=["image"], roles=["user"]))
            for item, emb in zip(items, image_embeds, strict=True):
                item.value = emb
        return {"conversation_list": conversation}

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> Optional[torch.Tensor]:
        if not raw_images:
            return None
        return self._image_processor(images=raw_images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vision_config or {}
        h = cfg.image_size
        c = cfg.num_channels
        return {
            "pixel_values": torch.zeros(1, c, h, h, device=self.device, dtype=self.dtype),
        }

    # Inference hooks
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        pending = [part for part in conversation_list if part.type == "image_output"]
        if not pending:
            pending = [part for part in conversation_list if part.type == "image" and part.role == "user"]

        if not pending:
            return {"conversation_list": conversation_list}

        embeds = self._encode_pixel_values(self._pixels_from_raw_images([part.value for part in pending]))
        for part, emb in zip(pending, embeds, strict=True):
            part.value = emb if emb.dim() == 2 else emb.squeeze(0)
            if part.type == "image_output":
                part.type = "image"
                assert part.role == "assistant"

        return {"conversation_list": conversation_list}


class JanusSiglipTraceMixin(TraceMixin):
    """Per-module training-trace for the SigLIP vision tower."""

    config: JanusSiglipConfig

    def trace_token_lengths(self, method: str, data: Dict[str, Any]) -> List[int]:
        # One ViT sequence per image; tokens = patches = (image/patch)**2.
        # (SP would slice the image batch dim, so this is the local count — the
        # vision tower has no full-length cu_seqlens to recover from.)
        pixel_values = data.get("pixel_values")
        if pixel_values is None:
            return []
        cfg = self.config.vision_config
        patches = (cfg.image_size // cfg.patch_size) ** 2
        return [patches] * int(pixel_values.shape[0])

    def estimate_flops(self, seqlens: List[int]) -> float:
        # SigLIP ViT: patch-embed conv + per-layer (q/k/v/o attn proj + GELU MLP)
        # + quadratic attention. fwd+bwd ⇒ 6x linear, 12x attention. The small
        # aligner MLP is negligible and omitted.
        cfg = self.config.vision_config
        dim = cfg.hidden_size
        num_layers = cfg.num_hidden_layers
        num_heads = cfg.num_attention_heads
        head_dim = dim // num_heads
        in_channels = getattr(cfg, "num_channels", 3)
        # JanusVisionConfig sizes the MLP via mlp_ratio (no `intermediate_size`).
        intermediate_size = int(dim * cfg.mlp_ratio)

        patch_embed_n = dim * in_channels * cfg.patch_size * cfg.patch_size
        attn_linear_n = dim * 4 * dim  # q, k, v, o
        mlp_n = dim * intermediate_size * 2  # fc1 + fc2 (GELU, no GLU)
        dense_n = patch_embed_n + (attn_linear_n + mlp_n) * num_layers

        tokens = sum(seqlens)
        seqlen_sq = sum(s * s for s in seqlens)
        dense_flops = 6 * dense_n * tokens
        attn_flops = 12 * seqlen_sq * head_dim * num_heads * num_layers
        return (dense_flops + attn_flops) / 1e12


__all__ = ["JanusSiglipModuleMixin", "JanusSiglipTraceMixin"]

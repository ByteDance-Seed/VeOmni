from typing import Any, Dict, List, Optional

import torch

from ....conversation import ConversationItem, iter_desired_items, worker_dummy_items
from ....module import (
    CPUPreprocessor,
    ModuleMixin,
    post_forward,
    pre_forward,
)
from ....tracemixin import TraceMixin
from .configuration import JanusSiglipConfig


_SOURCE = "janus_siglip"
# Module-specific sentinel on a real user-image item's meta marking that the
# worker already normalized its pixels (kept distinct per module per MR review).
_OMNI_PIXELS = "janus_siglip_pixels"


class JanusSiglipCPUPreprocessor(CPUPreprocessor):
    """Worker-side image normalize for the SigLIP (understanding) tower.

    Holds only the (picklable) HF image processor + a CPU zero-pixel template —
    never the model. Runs the same normalize as ``_pixels_from_raw_images`` but on
    **CPU** (bf16, to halve worker→main IPC); writes the pixel tensor back into
    each ``user``-image item. When a micro-batch has **no** user image, appends a
    ``role="dummy"`` placeholder per sample carrying the zero pixels, so the GPU
    forward never builds dummy inputs (the FSDP gradient anchor still runs there).
    """

    def __init__(self, image_processor: Any, dtype: Any, dummy_pixel_values: torch.Tensor) -> None:
        self._image_processor = image_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (C, H, W), model dtype

    def __call__(self, conversation_list: list[list[ConversationItem]]) -> None:
        todo = [
            it
            for it in iter_desired_items(conversation_list, types=["image"], roles=["user"])
            if not it.meta.get(_OMNI_PIXELS)
        ]
        if todo:
            pixel_values = self._image_processor(images=[it.value for it in todo], return_tensors="pt")["pixel_values"]
            for it, px in zip(todo, pixel_values, strict=True):
                it.value = px.to(dtype=self._dtype)
                it.meta[_OMNI_PIXELS] = True
        # Real user images present → no dummy needed.
        if any(iter_desired_items(conversation_list, types=["image"], roles=["user"])):
            return
        # No real user images anywhere → append one dummy placeholder per sample.
        if worker_dummy_items(conversation_list, _SOURCE):
            return
        for sample in conversation_list:
            sample.append(
                ConversationItem(
                    type="image",
                    value=self._dummy_pixel_values,
                    role="dummy",
                    meta={"source": _SOURCE, _OMNI_PIXELS: True},
                )
            )


class JanusSiglipModuleMixin(ModuleMixin):
    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None

    # Training hooks

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side image normalize (see :class:`JanusSiglipCPUPreprocessor`)."""
        if getattr(self, "_image_processor", None) is None:
            return None
        cfg = self.config.vision_config
        dummy = torch.zeros(cfg.num_channels, cfg.image_size, cfg.image_size, dtype=self.dtype)
        return JanusSiglipCPUPreprocessor(self._image_processor, self.dtype, dummy)

    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        if items and all(it.meta.get(_OMNI_PIXELS) for it in items):
            # Worker already normalized: just stack + move to device.
            pixel_values = torch.stack([it.value for it in items], dim=0).to(
                device=self.device, dtype=self.dtype, non_blocking=True
            )
            return {"pixel_values": pixel_values, "is_dummy": False}
        if items:
            # Eager / no-worker path with real images: normalize here.
            return {"pixel_values": self._pixels_from_raw_images([it.value for it in items]), "is_dummy": False}
        # No real user images: consume the worker-built dummy placeholder if present
        # (one dummy forward, batch 1); else fall back to the modeling device dummy.
        dummy_items = worker_dummy_items(conversation_list, _SOURCE)
        if dummy_items:
            pixel_values = (
                dummy_items[0].value.unsqueeze(0).to(device=self.device, dtype=self.dtype, non_blocking=True)
            )
            return {"pixel_values": pixel_values, "is_dummy": True}
        return {"pixel_values": None, "is_dummy": None}

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
            dummy_items = worker_dummy_items(conversation, _SOURCE)
            if dummy_items:
                # Worker pre-created the placeholders: overwrite zero pixels → embed.
                for item in dummy_items:
                    item.value = image_embeds
            else:
                # Eager / no-worker fallback: append the dummy placeholder per sample.
                for sample in conversation:
                    sample.append(
                        ConversationItem(
                            type="image",
                            value=image_embeds,
                            role="dummy",
                            meta={"source": _SOURCE},
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
        if isinstance(cfg, dict):
            h = cfg["image_size"]
            c = cfg["num_channels"]
        else:
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

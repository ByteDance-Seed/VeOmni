from typing import Any, Dict, List, Optional

import torch

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.modulemixin import (
    CPUPreprocessor,
    ModuleMixin,
    post_forward,
    pre_forward,
)
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from .configuration import JanusSiglipConfig
from .processing import JanusSiglipProcessor


_SOURCE = "janus_siglip"


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

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        if image_items:
            # Real user images present → normalize them; no dummy needed. Tag with
            # the module source so forward_pre/post can pick up real images and
            # dummies uniformly (single ``source == _SOURCE`` filter).
            pixel_values = self._image_processor(images=[it.value for it in image_items], return_tensors="pt")[
                "pixel_values"
            ]
            for it, px in zip(image_items, pixel_values, strict=True):
                it.value = px.to(dtype=self._dtype)
                it.source = _SOURCE
        elif not inference:
            # Training only: a sample with no user image still must run the ViT
            # (FSDP gradient anchor), so inject a dummy. Inference skips it — a
            # text-only request just leaves ``generate`` nothing to encode.
            for sample in conversation_list:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=self._dummy_pixel_values,
                        role="dummy",
                        source=_SOURCE,
                    )
                )


class JanusSiglipModuleMixin(ModuleMixin):
    config: JanusSiglipConfig
    _image_processor: JanusSiglipProcessor

    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side image normalize (see :class:`JanusSiglipCPUPreprocessor`)."""
        dummy = self.dummy_inputs()["pixel_values"]
        return JanusSiglipCPUPreprocessor(self._image_processor, self.dtype, dummy)

    # Training hooks
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        # Real user images and worker-built dummies both carry source == _SOURCE
        # (a batch is all-real or all-dummy, normalized on CPU by the
        # JanusSiglipCPUPreprocessor); stack + move. The dummy flag just tags the
        # batch so modeling.forward can skip the ViT off-FSDP.
        items = list(iter_desired_items(conversation_list, types=["image"], sources=[_SOURCE]))
        pixel_values = torch.stack([it.value for it in items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        return {"pixel_values": pixel_values, "is_dummy": is_dummy(items[0])}

    @post_forward("forward")
    def forward_post(self, image_embeds: torch.Tensor) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        self._conversation_carrier = None
        # forward returns one embed row per fed item, in source order; scatter them
        # back onto the same source items (real or dummy alike).
        items = list(iter_desired_items(conversation, types=["image"], sources=[_SOURCE]))
        for item, emb in zip(items, image_embeds, strict=True):
            item.value = emb
        return {"conversation_list": conversation}

    def dummy_inputs(self) -> Dict[str, Any]:
        # Per-image (C, H, W) zero template on CPU: it seeds the worker-side
        # CPUPreprocessor's dummy item, which is pickled into the batch (a CUDA
        # tensor would crash the DataLoader worker), and forward_pre stacks it
        # exactly like a real per-image pixel tensor.
        cfg = self.config.vision_config
        return {
            "pixel_values": torch.zeros(cfg.num_channels, cfg.image_size, cfg.image_size, dtype=self.dtype),
        }

    # Inference hooks
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        pending = [part for part in conversation_list if part.type == "image_output"]
        if pending:
            # Mid-loop generated images (raw PIL) — produced during the FSM, so the
            # pre-FSM preprocessor never saw them; normalize them on the fly here.
            pixel_values = self._pixels_from_raw_images([part.value for part in pending])
        else:
            pending = [part for part in conversation_list if part.type == "image" and part.role == "user"]
            if not pending:
                return {"conversation_list": conversation_list}
            # User images were normalized by the inference CPU preprocessor before
            # the FSM (``item.value`` already holds the pixel tensor); just stack + move.
            pixel_values = torch.stack([part.value for part in pending], dim=0).to(self.device, self.dtype)

        embeds = self._encode_pixel_values(pixel_values)
        for part, emb in zip(pending, embeds, strict=True):
            part.value = emb if emb.dim() == 2 else emb.squeeze(0)
            if part.type == "image_output":
                part.type = "image"
                assert part.role == "assistant"

        return {"conversation_list": conversation_list}

    def _pixels_from_raw_images(self, raw_images: list[Any]) -> Optional[torch.Tensor]:
        if not raw_images:
            return None
        return self._image_processor(images=raw_images, return_tensors="pt")["pixel_values"].to(
            device=self.device, dtype=self.dtype
        )


class JanusSiglipMetricMeterMixin(MetricMeterMixin):
    """Per-module training meter for the SigLIP vision tower."""

    config: JanusSiglipConfig

    def metric_meter_token_lengths(self, method: str, data: Dict[str, Any]) -> List[int]:
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


__all__ = ["JanusSiglipModuleMixin", "JanusSiglipMetricMeterMixin"]

from typing import Any, Dict, List, Optional

import torch

from veomni.distributed.parallel_state import get_parallel_state
from veomni.distributed.sequence_parallel import gather_outputs, slice_input_tensor

from ....mixins.metric_meter_mixin import MetricMeterMixin
from ....mixins.module_mixin import ModuleMixin, post_forward, pre_forward
from ....utils.conversation import ConversationItem, is_dummy, iter_desired_items
from .configuration import JanusSiglipConfig
from .processing import JanusSiglipPreprocessor, JanusSiglipProcessor


_SOURCE = "janus_siglip"


class JanusSiglipModuleMixin(ModuleMixin):
    config: JanusSiglipConfig
    _image_processor: JanusSiglipProcessor
    preprocessor_class = JanusSiglipPreprocessor

    def init_omni_state(self) -> None:
        # Training state
        self._conversation_carrier: Any = None
        # Active sample's image count. Under SP the output-gather hook
        # (``forward_sp_post``) narrows the all-gathered (batch-padded) embeds to it.
        self._sp_own_len: Optional[int] = None

    # Training hooks
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        # Real user images and worker-built dummies both carry source == _SOURCE
        # (normalized on CPU by the JanusSiglipPreprocessor); stack + move.
        items = list(iter_desired_items(conversation_list, types=["image"], sources=[_SOURCE]))
        pixel_values = torch.stack([it.value for it in items], dim=0).to(
            device=self.device, dtype=self.dtype, non_blocking=True
        )
        # Single batch-level dummy flag: True only when *every* fed image is a
        # dummy (a worker-injected zero-pixel placeholder), i.e. the whole batch
        # is dummy; if any image is real it is False. Passed as a scalar so the
        # modeling forward can short-circuit to a dummy output when appropriate.
        is_dummy_flag = all(is_dummy(it) for it in items)
        # Metering: this rank's OWN image count, stashed BEFORE the SP slice below.
        # The meter sums over the DP group only, so each rank reports just its own
        # images (not the SP peers that hold the same replicated batch) — identical
        # to the non-SP run for both SP-disabled and uniform SP.
        self._metric_meter_stash_tokens(int(pixel_values.shape[0]))

        if get_parallel_state().sp_size > 1:
            # SP input-slice: hand this rank its ``1/sp_size`` slice of the image
            # batch (SigLIP shards the batch dim — its ViT attention is not Ulysses;
            # slicing the replicated batch is exactly per-rank image balance). Every
            # SP rank already holds the same image batch (the dataloader replicates
            # each shard); pad it to a multiple of ``sp_size`` and take this rank's
            # contiguous chunk. ``is_dummy`` is dropped: the SP path is a training
            # path that always runs the ViT (the FSDP grad anchor). ``forward_post``
            # all-gathers the shards back.
            self._sp_own_len = pixel_values.size(0)
            pixel_values = slice_input_tensor(pixel_values, dim=0, padding=True, group=get_parallel_state().sp_group)
            return {"pixel_values": pixel_values}
        # ``is_dummy`` stays for the plain (non-SP) forward: a non-SP inference/eval
        # short-circuit that skips the ViT for an all-dummy batch with no anchor.
        return {"pixel_values": pixel_values, "is_dummy": is_dummy_flag}

    def _metric_meter_stash_tokens(self, num_images: int) -> None:
        # One ViT sequence per image; tokens = patches = (image/patch)**2.
        cfg = self.config.vision_config
        patches = (cfg.image_size // cfg.patch_size) ** 2
        self.metric_meter_set_seqlens("forward", [patches] * num_images)

    @post_forward("forward")
    def forward_post(self, image_embeds: torch.Tensor) -> Dict[str, Any]:
        if get_parallel_state().sp_size > 1:
            # SP output-gather: all-gather batch shards back to the full image batch
            # (autograd-aware; backward sums grads across the SP group), then drop
            # the SP pad tail so the count matches its conversation items below.
            image_embeds = gather_outputs(image_embeds, gather_dim=0, group=get_parallel_state().sp_group)
            image_embeds = image_embeds.narrow(0, 0, self._sp_own_len)
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
        # Preprocessor's dummy item, which is pickled into the batch (a CUDA
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

from typing import Any, Dict, List, Optional, Tuple

import torch

from ....conversation import ConversationItem, iter_desired_items
from ....module import ModuleMixin, post_forward, pre_forward


class Qwen3VLVisionEncoderModuleMixin(ModuleMixin):
    """Graph hooks for the Qwen3-VL vision tower (images + videos).

    Both ``image`` and ``video`` user items are encoded by the **same** ViT in a
    single forward (image patches + video patches concatenated) so the FSDP unit
    runs exactly one visual forward per step regardless of which modality a
    micro-batch holds.  Each item's merged patch tokens are written back onto its
    ``value``; per-item ``grid_thw`` (for backbone M-RoPE) and ``deepstack``
    features (for interior-layer injection) ride on ``item.meta``.

    Video frames come pre-decoded as a :class:`VideoInputs` bundle (``item.value``)
    and go through the dedicated ``Qwen3VLVideoProcessor`` (temporal patchify);
    images use the ``Qwen2VLImageProcessor``.  Qwen3-VL has no audio modality.
    """

    def init_omni_state(self) -> None:
        self._conversation_carrier: Any = None
        self._visual_specs: Optional[list] = None

    # ── Training hooks ──────────────────────────────────────────────────────
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        pixel_values, grid_thw, vit_metadata, specs = self._process_visual_items(image_items, video_items)
        self._visual_specs = specs
        return {"pixel_values": pixel_values, "image_grid_thw": grid_thw, "vit_metadata": vit_metadata}

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        deepstack_features: List[torch.Tensor],
        image_grid_thw: torch.Tensor,
        is_dummy: bool = False,
    ) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        specs = self._visual_specs
        self._conversation_carrier = None
        self._visual_specs = None

        if is_dummy:
            for sample in conversation:
                sample.append(
                    ConversationItem(
                        type="image",
                        value=image_embeds,
                        role="dummy",
                        meta={"source": "qwen3vl_vision", "deepstack": deepstack_features},
                    )
                )
            return {"conversation_list": conversation}

        self._scatter_visual_embeds(specs, image_embeds, deepstack_features)
        return {"conversation_list": conversation}

    def _scatter_visual_embeds(
        self,
        specs: list,
        embeds: torch.Tensor,
        deepstack_features: List[torch.Tensor],
    ) -> None:
        sizes = [n for (_item, _grid, n) in specs]
        embeds_split = torch.split(embeds, sizes, dim=0)
        deepstack_split = [torch.split(layer, sizes, dim=0) for layer in deepstack_features]
        for idx, (item, grid, _n) in enumerate(specs):
            item.value = embeds_split[idx]
            item.source = "qwen3vl_vision"
            item.meta["grid_thw"] = grid
            item.meta["deepstack"] = [layer[idx] for layer in deepstack_split]

    def _process_visual_items(
        self,
        image_items: list,
        video_items: list,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[Dict[str, Any]], list]:
        """Process images + videos into one ViT batch. Returns (pixel_values,
        grid_thw, vit_metadata, specs) where ``specs`` is an ordered list of
        ``(item, grid_tensor, num_merged_tokens)`` matching the concat order."""
        if not image_items and not video_items:
            return None, None, None, []

        pv_list: list[torch.Tensor] = []
        grid_rows: list[list[int]] = []
        specs: list = []
        merge_area = self.config.vision_config.spatial_merge_size**2

        if image_items:
            out = self._image_processor(images=[it.value for it in image_items], return_tensors="pt")
            pv_list.append(out["pixel_values"])
            for it, g in zip(image_items, out["image_grid_thw"].tolist()):
                grid_rows.append(g)
                specs.append(
                    (it, torch.tensor(g, dtype=torch.long, device=self.device), int(g[0] * g[1] * g[2]) // merge_area)
                )

        if video_items:
            frames = [it.value.video for it in video_items]
            out = self._video_processor(videos=frames, return_tensors="pt")
            pv_list.append(out["pixel_values_videos"])
            for it, g in zip(video_items, out["video_grid_thw"].tolist()):
                grid_rows.append(g)
                specs.append(
                    (it, torch.tensor(g, dtype=torch.long, device=self.device), int(g[0] * g[1] * g[2]) // merge_area)
                )

        vit_metadata = self._build_vit_metadata(grid_rows)
        pixel_values = torch.cat(pv_list, dim=0).to(device=self.device, dtype=self.dtype)
        grid_thw = torch.tensor(grid_rows, dtype=torch.long, device=self.device)
        return pixel_values, grid_thw, vit_metadata, specs

    @staticmethod
    def _build_vit_metadata(grid_thw_list: list[list[int]]) -> Dict[str, Any]:
        """Host-side ViT metadata so the patched ViT skips per-forward syncs.

        Mirrors the fallback inside ``Qwen3VLVisionModel.forward`` (no SP path —
        the V2 backbone does not support sequence parallel yet).
        """
        cu: list[int] = [0]
        for t, h, w in grid_thw_list:
            frame_len = h * w
            for _ in range(t):
                cu.append(cu[-1] + frame_len)
        max_seqlen = max((c2 - c1 for c1, c2 in zip(cu, cu[1:])), default=0)
        return {
            "grid_thw_list": grid_thw_list,
            "cu_seqlens": torch.tensor(cu, dtype=torch.int32, device="cpu"),
            "max_seqlen": max_seqlen,
        }

    def dummy_inputs(self) -> Dict[str, Any]:
        cfg = self.config.vision_config
        merge = cfg.spatial_merge_size
        t, h, w = 1, 2 * merge, 2 * merge
        pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        pixel_values = torch.zeros(t * h * w, pixel_row, device=self.device, dtype=self.dtype)
        image_grid_thw = torch.tensor([[t, h, w]], dtype=torch.long, device=self.device)
        vit_metadata = self._build_vit_metadata([[t, h, w]])
        return {"pixel_values": pixel_values, "image_grid_thw": image_grid_thw, "vit_metadata": vit_metadata}

    # ── Inference hooks ─────────────────────────────────────────────────────
    def generate(
        self,
        conversation_list: Optional[List[ConversationItem]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        del kwargs
        image_items = [p for p in conversation_list if p.type == "image" and p.role == "user"]
        video_items = [p for p in conversation_list if p.type == "video" and p.role == "user"]
        if not image_items and not video_items:
            return {"conversation_list": conversation_list}

        pixel_values, grid_thw, vit_metadata, specs = self._process_visual_items(image_items, video_items)
        image_embeds, deepstack_features = self._encode(pixel_values, grid_thw, vit_metadata)
        self._scatter_visual_embeds(specs, image_embeds, deepstack_features)
        return {"conversation_list": conversation_list}


__all__ = ["Qwen3VLVisionEncoderModuleMixin"]

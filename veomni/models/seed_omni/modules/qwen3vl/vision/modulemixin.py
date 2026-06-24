from typing import Any, Dict, List, Optional, Tuple

import torch

from ....mixins.modulemixin import (
    CPUPreprocessor,
    ModuleMixin,
    post_forward,
    pre_forward,
)
from ....utils.conversation import ConversationItem, iter_desired_items


# qwen3vl-specific meta key: per-item ``grid_thw`` stashed alongside the
# normalized patches (which go onto ``item.value``, like siglip/vqvae) by the
# CPU preprocessor (DataLoader worker for training, pre-FSM pass for inference).
# ``_pixels_and_grid`` pops it on the main process.
_OMNI_GRID = "_omni_grid"
_SOURCE = "qwen3vl_vision"


def _video_metadata(items: list, frames: list) -> list[dict]:
    """HF ``video_metadata`` for the handed-over (already decoded) frames.

    The data layer (``seed_omni/video_utils.load_video``) pre-trims each clip to
    ``mm_configs.fps`` purely as a memory bound, so the frames passed here are a
    self-contained clip whose *source* fps **is** ``VideoInputs.video_fps``. We
    forward that as the metadata fps; the HF ``Qwen3VLVideoProcessor`` then
    sub-samples to its own authoritative ``self.fps`` (the model's target rate).
    Without metadata it would default to ``fps=24`` and mangle the clip.
    """
    return [{"total_num_frames": f.shape[0], "fps": it.value.video_fps} for it, f in zip(items, frames)]


def _store_patches(items: list, pixel_values: torch.Tensor, grid_thw: torch.Tensor, dtype: Any) -> None:
    """Split flat ViT patches by per-item grid; stash patches on ``value`` + grid on ``meta``.

    Used by the CPU preprocessor (both training and inference share it) so items
    are left in the preprocessed form that ``_pixels_and_grid`` reads back.
    """
    grids = grid_thw.tolist()
    sizes = [g[0] * g[1] * g[2] for g in grids]
    chunks = torch.split(pixel_values, sizes, dim=0)
    for it, px, g in zip(items, chunks, grids, strict=True):
        it.value = px.to(dtype=dtype)
        it.meta[_OMNI_GRID] = g


class Qwen3VLVisionCPUPreprocessor(CPUPreprocessor):
    """Worker-side image/video patchify+normalize for the Qwen3-VL vision tower.

    Holds only the (picklable) HF image / video processors + a CPU zero-patch
    template — never the model. Runs them on **CPU** (bf16, to halve IPC), writes
    the per-item normalized patches onto ``item.value`` and stashes ``grid_thw`` on
    ``meta``. When a micro-batch has **no** user image/video, appends a
    ``role="dummy"`` placeholder per sample
    carrying the zero patches + grid (the merger still runs on it in the GPU
    forward for the FSDP gradient anchor).
    """

    def __init__(
        self,
        image_processor: Any,
        video_processor: Any,
        dtype: Any,
        dummy_pixel_values: torch.Tensor,
        dummy_grid: list,
    ) -> None:
        self._image_processor = image_processor
        self._video_processor = video_processor
        self._dtype = dtype
        self._dummy_pixel_values = dummy_pixel_values  # CPU (t*h*w, pixel_row), model dtype
        self._dummy_grid = dummy_grid  # [t, h, w]

    def __call__(
        self, conversation_list: list[list[ConversationItem]], inference: bool = False, **kwargs: Any
    ) -> None:
        del kwargs  # generation_kwargs unused: prep is kwarg-independent
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        if image_items and self._image_processor is not None:
            out = self._image_processor(images=[it.value for it in image_items], return_tensors="pt")
            self._store(image_items, out["pixel_values"], out["image_grid_thw"])
        if video_items and self._video_processor is not None:
            frames = [it.value.video for it in video_items]
            out = self._video_processor(
                videos=frames, video_metadata=_video_metadata(video_items, frames), return_tensors="pt"
            )
            self._store(video_items, out["pixel_values_videos"], out["video_grid_thw"])
        if image_items or video_items or inference:
            # Real image/video present → no dummy needed. Inference also skips the
            # dummy: a text-only request just leaves ``generate`` nothing to encode
            # (no FSDP gradient anchor required at inference).
            return
        for sample in conversation_list:
            sample.append(
                ConversationItem(
                    type="image",
                    value=self._dummy_pixel_values,
                    role="dummy",
                    source=_SOURCE,
                    meta={_OMNI_GRID: self._dummy_grid},
                )
            )

    def _store(self, items: list, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> None:
        _store_patches(items, pixel_values, grid_thw, self._dtype)


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

    def build_cpu_preprocessor(self) -> Optional[CPUPreprocessor]:
        """Worker-side patchify+normalize (see :class:`Qwen3VLVisionCPUPreprocessor`)."""
        image_processor = getattr(self, "_image_processor", None)
        video_processor = getattr(self, "_video_processor", None)
        if image_processor is None and video_processor is None:
            return None
        cfg = self.config.vision_config
        merge = cfg.spatial_merge_size
        t, h, w = 1, 2 * merge, 2 * merge
        pixel_row = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size
        dummy_pixels = torch.zeros(t * h * w, pixel_row, dtype=self.dtype)
        return Qwen3VLVisionCPUPreprocessor(image_processor, video_processor, self.dtype, dummy_pixels, [t, h, w])

    # ── Training hooks ──────────────────────────────────────────────────────
    @pre_forward("forward")
    def forward_pre(
        self,
        conversation_list: Optional[list[list[ConversationItem]]] = None,
    ) -> Dict[str, Any]:
        self._conversation_carrier = conversation_list
        image_items = list(iter_desired_items(conversation_list, types=["image"], roles=["user"]))
        video_items = list(iter_desired_items(conversation_list, types=["video"], roles=["user"]))
        dummy = not (image_items or video_items)
        if dummy:
            # No real visual input: feed the worker-built dummy placeholders (one
            # per sample) through the same path as real images — they carry
            # patches + ``_OMNI_GRID`` like real items. Under FSDP these run the
            # ViT (gradient anchor); without FSDP modeling.forward emits
            # real-shaped zeros, so the batch stays uniform either way.
            image_items = list(
                iter_desired_items(conversation_list, types=["image"], roles=["dummy"], sources=[_SOURCE])
            )
        pixel_values, grid_thw, vit_metadata, specs = self._process_visual_items(image_items, video_items)
        self._visual_specs = specs
        return {
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "vit_metadata": vit_metadata,
            "is_dummy": dummy,
        }

    @post_forward("forward")
    def forward_post(
        self,
        image_embeds: torch.Tensor,
        deepstack_features: List[torch.Tensor],
        image_grid_thw: torch.Tensor,
    ) -> Dict[str, Any]:
        conversation = self._conversation_carrier
        specs = self._visual_specs
        self._conversation_carrier = None
        self._visual_specs = None
        # forward returns merged tokens in spec order (real or dummy alike);
        # scatter them back onto the originating items.
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
            item.source = _SOURCE
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
            pixel_values, grids = self._pixels_and_grid(image_items)
            pv_list.append(pixel_values)
            for it, g in zip(image_items, grids):
                grid_rows.append(g)
                specs.append(
                    (it, torch.tensor(g, dtype=torch.long, device=self.device), int(g[0] * g[1] * g[2]) // merge_area)
                )

        if video_items:
            pixel_values, grids = self._pixels_and_grid(video_items)
            pv_list.append(pixel_values)
            for it, g in zip(video_items, grids):
                grid_rows.append(g)
                specs.append(
                    (it, torch.tensor(g, dtype=torch.long, device=self.device), int(g[0] * g[1] * g[2]) // merge_area)
                )

        vit_metadata = self._build_vit_metadata(grid_rows)
        pixel_values = torch.cat(pv_list, dim=0).to(device=self.device, dtype=self.dtype)
        grid_thw = torch.tensor(grid_rows, dtype=torch.long, device=self.device)
        return pixel_values, grid_thw, vit_metadata, specs

    def _pixels_and_grid(self, items: list) -> Tuple[torch.Tensor, list[list[int]]]:
        """Read back the preprocessed patches (``item.value``) + grid (``meta``).

        Items are always already patchified by the CPU preprocessor — inside the
        DataLoader worker for training, or once over the request before the FSM for
        inference — so this only concatenates ``value`` and pops the stashed
        ``_OMNI_GRID``.
        """
        pixel_values = torch.cat([it.value for it in items], dim=0)
        grids = [it.meta.pop(_OMNI_GRID) for it in items]
        return pixel_values, grids

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

        # Items were patchified by the inference CPU preprocessor before the FSM
        # (patches on ``value`` + grid on ``meta``, exactly as in training), so this
        # only reads them back, runs the ViT, and scatters the merged tokens.
        pixel_values, grid_thw, vit_metadata, specs = self._process_visual_items(image_items, video_items)
        image_embeds, deepstack_features = self._encode(pixel_values, grid_thw, vit_metadata)
        self._scatter_visual_embeds(specs, image_embeds, deepstack_features)
        return {"conversation_list": conversation_list}


__all__ = ["Qwen3VLVisionEncoderModuleMixin"]

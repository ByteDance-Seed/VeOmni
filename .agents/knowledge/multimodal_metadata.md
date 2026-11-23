# Multimodal Metadata Precompute

Contract for precomputing VLM/Omni forward metadata on CPU dataloader workers so `Model.forward` / `ViT.forward` stay free of host↔device syncs (gated by `tests/models/transformers/test_model_forward_no_implicit_sync.py`).

## Ownership & Single Key

- ViT metadata formulas are model-specific (`grid_thw` → `cu_seqlens`, `cu_window_seqlens`, `audio_seqlens`), so **the model owns the derivation** via picklable hooks while `MainCollator` stays generic.
- End-to-end key: HF processor's CPU `LongTensor` `image_grid_thw` / `video_grid_thw` (packed along `pack_dim=0` to `(total_images, 3)`). The model's hook calls `.tolist()` once on CPU at collate time; the packed tensor stays in the batch for `get_image_features`.

## Model Hooks (on `…ForConditionalGeneration` in `modeling_<model>.py`)

1. **`get_metadata_collate_func() -> Callable | None`**: returns a picklable module-level function (or `partial` over config constants, never an `nn.Module`) with signature `fn(batch: dict, sp_pad: dict[str, int]) -> None`. Mutates `batch["multimodal_metadata"]` in place. Invoked by `PackingCollator` (`sp_pad={...: 0}`) or `SequenceParallelCollator` (once, **after** SP padding).
2. **`get_extra_collate_infos() -> dict`** (optional): returns model-specific `DataCollateInfo` tuples `(pack_dim, sp_slice, sp_pad_value, sp_pad_scale)` merged into the collator (e.g. Omni audio tensors `input_features`, `audio_feature_lengths`, `audio_mask`).

`vlm_trainer._build_collate_fn` resolves both via `getattr`; models without them fall back to runtime derivation.

## `multimodal_metadata` Dict Contract

All keys are optional (missing keys trigger runtime fallback):

| Key | Type | Notes |
|---|---|---|
| `image_grid_thw_list` / `video_grid_thw_list` | `list[list[int]]` | `[t, h, w]` triplets on CPU; never sent to GPU |
| `vit_{image,video}_cu_seqlens` | `torch.IntTensor` | CPU tensor auto-moved to device; includes SP-pad tail as one extra synthetic image |
| `vit_{image,video}_max_seqlen` | `int` | Includes SP-pad |
| `vit_{image,video}_cu_window_seqlens` | `torch.IntTensor` | Window-attention ViTs (`qwen2_5_vl`, `qwen2_5_omni`); `unique_consecutive`'d, includes SP-pad |
| `vit_{image,video}_window_index` | `torch.LongTensor` | Host-side port of `get_window_index` permutation |
| `vit_{image,video}_win_max_seqlen` | `int` | Max window segment length (`qwen2_5_omni` omits this) |

- Do **not** carry `n_image_tokens` / `n_video_tokens` (derive from `*_grid_thw_list`), `position_ids` (top-level kwarg), or `rope_deltas` (generation-only; dropped by data transform).

## Model → ViT Handoff & Fallback

- `Model.forward` pops `multimodal_metadata`, slices per modality into a single `vit_metadata={"grid_thw_list", "cu_seqlens", "max_seqlen", ...}` kwarg, and passes it to `get_image_features` / `get_video_features`.
- `ViT.forward` pops `vit_metadata` and **must keep a runtime fallback** when absent (supports third-party collators and manual inference).
- `dummy_forward` (FSDP path for ranks without media) constructs the same `vit_metadata` host-side from Python-int dummy grids.
- **SP-pad tail**: when `SequenceParallelCollator` pads `pixel_values` to SP-divisible, `collate_multimodal_metadata` appends one synthetic `cu_seqlens` entry for `sp_pad` patches (embeddings discarded after per-rank slice).

## Wired Models

`qwen3_vl`, `qwen3_vl_moe`, `qwen3_omni_moe` (+ audio `get_extra_collate_infos`), `qwen3_5`, `qwen3_5_moe`, `qwen2_vl`, `qwen2_5_vl` (window ViT), `qwen2_5_omni` (window ViT + audio; thinker exposes hook, outer model delegates), `qwen4_exp`.

## Adding to a New Model

1. Define module-level `collate_multimodal_metadata(batch, sp_pad)` in `veomni/models/transformers/<model>/modeling_<model>.py` (append SP-pad tail per `sp_pad`; port `get_window_index` host-side for window ViTs).
2. Implement `get_metadata_collate_func()` (returning the helper or a `partial` with config dims) and optional `get_extra_collate_infos()`.
3. Wire `multimodal_metadata` → `vit_metadata` in `Model.forward`, `ViT.forward` (with runtime fallback), and `dummy_forward`.
4. Add the model to `CASES` and `_MM_METADATA_WIRED_CASES` in `tests/models/transformers/test_model_forward_no_implicit_sync.py` (tests sync-free forward and bitwise parity against fallback); see `tests/data/test_mm_metadata.py` for collator-hook/picklability tests.

# Multimodal Metadata Precompute

Contract for **precomputing multimodal forward metadata on CPU workers** instead of
deriving it inside Model.forward / ViT.forward (which forces host↔device syncs).

## Why this exists

For Qwen multimodal models the runtime ViT and Model forwards historically derived
metadata values per step from GPU tensors:

- mrope `position_ids` + `rope_deltas` (per-sample variable-shape mrope algorithm)
- ViT `cu_seqlens` + `max_seqlen` (from `image_grid_thw`)

Each derivation costs 1+ host↔device syncs (see `debug-cuda-sync` skill for the
inventory). Moving the derivation off the GPU critical path — into the **collator**,
which runs in CPU dataloader workers — eliminates the syncs entirely and lets the
model forward consume CPU-int / CPU-tensor values that are batched up onto the
device by the normal trainer `.to(device)` step.

## The derivation is model-owned, not framework-generic

The ViT metadata formula is **Qwen-VL-family-specific**, not a framework-wide
multimodal contract:

- Different modalities/models use different metadata (`grid_thw` → `cu_seqlens` vs
  `cu_window_seqlens` for qwen2.5-vl, `audio_seqlens` for omni audio, …).
- The formulas depend on model config (`spatial_merge_size`, `window_size`, …)
  that the collator does not have.

So the derivation lives **on the model**, exposed as a picklable hook — exactly
mirroring how `get_position_id_func` exposes the per-sample position-id closure.
The collator stays generic: it packs / SP-pads / SP-slices and then *invokes* the
model's hook; it never contains model-specific metadata logic.

## Single key end-to-end

There is **one representation per modality** through the whole pipeline: the HF
processor's `image_grid_thw` / `video_grid_thw` CPU `LongTensor`.

- The data transform emits only those tensors (no Python-list sidecar).
- The collator packs them via the existing `DataCollateInfo` (`pack_dim=0`,
  `torch.cat` → `(total_images, 3)`); no special-case keys.
- The model's `collate_multimodal_metadata` hook reads the packed tensor and does
  `.tolist()` **once** at batch time (CPU — the collator runs in dataloader
  workers, so no host-device sync).

The grid tensor stays in the batch (Model.forward still needs it as the HF
`get_image_features` arg); the hook only reads it.

## Two model hooks

Patched onto each wired `…ForConditionalGeneration` via patchgen `override_method`:

### `get_metadata_collate_func() -> Callable | None`

Returns a **picklable** callable (a bare module-level helper, or a `partial` over
one closed over config constants — never an `nn.Module`) with signature:

```python
metadata_collate_func(batch: dict, sp_pad: dict[str, int]) -> None
```

It mutates `batch` in place, writing `batch["multimodal_metadata"]`. `sp_pad` maps
`pixel_values` / `pixel_values_videos` to the number of patch rows the SP collator
appended. The collator invokes it **once, after SP padding** — the same timing as
the text-side flash-attention kwargs (`add_flash_attention_kwargs_from_position_ids`).

For the qwen3-VL family the helper is `collate_multimodal_metadata`, a module-level
patchgen helper (`@config.add_helper`) emitted into each generated modeling file.
Its formula (temporal-unroll of `h*w` patch counts) needs no model config, so the
hook is the bare helper reference.

### `get_extra_collate_infos() -> dict`

Optional. Returns model-specific `DataCollateInfo` entries (as tuples
`(pack_dim, sp_slice, sp_pad_value, sp_pad_scale)`) that the collator merges into
its collate-info table. Omni models use it to declare the audio feature tensors
(`input_features`, `audio_feature_lengths`, `audio_mask`).

Both hooks are resolved by `vlm_trainer._build_collate_fn` via `getattr`: a model
that doesn't expose them simply gets the runtime fallback (see below).

## The `multimodal_metadata` dict contract

`collate_multimodal_metadata` writes `batch["multimodal_metadata"]`, a dict with
(all keys optional — missing keys make the consumer fall back to runtime derivation):

| Key | Type | Notes |
|---|---|---|
| `image_grid_thw_list` | `list[list[int]]` | `[t, h, w]` triplets — the hook's `.tolist()` of the packed `image_grid_thw`. Python list, never sent to GPU. |
| `video_grid_thw_list` | `list[list[int]]` | same, for video. |
| `vit_image_cu_seqlens` | `torch.IntTensor` (CPU, then auto-moved) | varlen-attention cu_seqlens; already includes the SP-pad tail as one extra "image" entry. |
| `vit_image_max_seqlen` | `int` (Python) | already includes SP-pad. |
| `vit_video_cu_seqlens` / `vit_video_max_seqlen` | same | for video. |

Window-attention ViTs (qwen2.5-VL / qwen2.5-omni) add three more keys per
modality — the host-side port of `get_window_index`:

| Key | Type | Notes |
|---|---|---|
| `vit_image_cu_window_seqlens` | `torch.IntTensor` (CPU) | window-attention cu_seqlens; `unique_consecutive`'d, includes the SP-pad tail. |
| `vit_image_window_index` | `torch.LongTensor` (CPU) | the `get_window_index` permutation that reorders the pre-merger ViT tokens. |
| `vit_image_win_max_seqlen` | `int` (Python) | max window segment length, includes SP-pad. qwen2.5-omni omits this (its vision attention takes `.max()` on-device). |

(`vit_video_*` equivalents for video.) The hook needs the vision config —
`get_metadata_collate_func` returns `partial(collate_multimodal_metadata,
window_size=..., spatial_merge_size=..., patch_size=...)`.

`n_image_tokens` / `n_video_tokens` are **not** carried — they depend on
`spatial_merge_size` and the model derives them from `*_grid_thw_list` with a
one-line `sum(...)` when needed.

`position_ids` is **NOT** in the dict — it remains a top-level model.forward kwarg.

`rope_deltas` is **NOT** carried: HF's `get_rope_index` returns it for the
generation/KV-cache decode path, but the training forward always receives a
precomputed `position_ids` and never derives or reads `rope_deltas`. The data
transform drops it.

## Model.forward → ViT: the `vit_metadata` sub-dict

The ViT processes either images or videos through the **same** module. Model.forward
selects the per-modality slice of `multimodal_metadata` and passes it as one
`vit_metadata` kwarg to `get_image_features` / `get_video_features`:

```python
multimodal_metadata = kwargs.pop("multimodal_metadata", None) or {}
image_vit_kwargs = {
    "vit_metadata": {
        "grid_thw_list": multimodal_metadata.get("image_grid_thw_list"),
        "cu_seqlens": multimodal_metadata.get("vit_image_cu_seqlens"),
        "max_seqlen": multimodal_metadata.get("vit_image_max_seqlen"),
    }
}
self.get_image_features(pixel_values, image_grid_thw, **image_vit_kwargs)
```

The patched ViT.forward pops the single `vit_metadata` kwarg:

```python
vit_metadata = kwargs.pop("vit_metadata", None) or {}
precomputed_grid_thw_list = vit_metadata.get("grid_thw_list")
precomputed_cu_seqlens = vit_metadata.get("cu_seqlens")
precomputed_max_seqlen = vit_metadata.get("max_seqlen")
```

A model whose ViT runs `dummy_forward` (FSDP path for ranks with no real images)
builds the same `vit_metadata` sub-dict host-side from its Python-int `t/h/w`.

## Producer flow (collator pipeline)

```
data transform (CPU worker, per-sample)
    └─ emits image_grid_thw / video_grid_thw CPU tensors (HF processor output)
       + position_ids (rope_deltas dropped — generation-only)

       ↓ DataLoader batch ↓

PackingCollator (CPU, generic)
    1. torch.cat per pack_dim — image_grid_thw packs to (total_images, 3)
    2. add_flash_attention_kwargs_from_position_ids (existing, non-SP only)
    3. non-SP: invoke metadata_collate_func(batch, sp_pad={...: 0})

       ↓ SP enabled ↓

SequenceParallelCollator (CPU, generic)
    1. sp_padding on input_ids / labels / position_ids / pixel_values{,_videos};
       records per-modality sp_pad patch counts
    2. add_flash_attention_kwargs_from_position_ids (existing)
    3. invoke metadata_collate_func(batch, sp_pad={pixel_values: N, ...})

       ↓ Trainer.preforward (.to_device, recurses into dicts) ↓

Model.forward(..., multimodal_metadata={...})
    Tensors in the dict get moved to GPU; Python ints/lists stay on CPU.
```

The collator carries `metadata_collate_func` (from `MainCollator`, see
`data_collator.MetadataCollateFunc`). When it is `None` — text models, third-party
pipelines — step (3) is skipped and no `multimodal_metadata` is produced.

## SP-pad tail handling

`collate_multimodal_metadata` receives the SP-pad patch counts via `sp_pad`. When
`SequenceParallelCollator` pads `pixel_values` to SP-divisible, the hook appends
one extra `cu_seqlens` entry for the padding patches — they become one synthetic
"image" so the ViT varlen attention treats them as an independent sequence. This
mirrors how text-side `cu_seq_lens` picks up SP-pad via the `position_ids == 0`
zero-tail convention. The embeddings for those positions are discarded after the
per-rank slice in Model.forward.

## Backwards compatibility

Every consumer site **must** keep a runtime fallback for when `multimodal_metadata`
is `None` or a key is missing. The patched ViT.forward derives the value in-forward
(the original `.tolist()` / cu_seqlens build) when `vit_metadata` is absent. This
guarantees:

1. Third-party collators / data pipelines that don't precompute keep working.
2. Inference scripts that construct inputs manually keep working.
3. Models without the `get_metadata_collate_func` hook keep working (they pay the
   runtime sync cost — caught by `tests/models/test_model_forward_no_implicit_sync.py`).

## Model coverage

| Model | Status | Notes |
|---|---|---|
| qwen3_vl | ✅ wired | Canonical. `collate_multimodal_metadata` helper in the gpu config; npu reuses it. |
| qwen3_vl_moe | ✅ wired | Reuses qwen3_vl's helper + hook. |
| qwen3_omni_moe | ✅ wired | Same ViT metadata. Also exposes `get_extra_collate_infos` (audio). |
| qwen3_5 | ✅ wired | Own `collate_multimodal_metadata` (identical formula). |
| qwen3_5_moe | ✅ wired | Reuses qwen3_5's ViT forward; own `collate_multimodal_metadata`. |
| qwen2_vl | ✅ wired | Own `collate_multimodal_metadata` (non-window ViT, same formula as qwen3_vl). |
| qwen2_5_vl | ✅ wired | Window-attention ViT. `collate_multimodal_metadata` ports `get_window_index` host-side; `get_metadata_collate_func` `partial`-closes the vision-config dims. |
| qwen2_5_omni | ✅ wired | Same window-attention ViT as qwen2_5_vl. Also exposes `get_extra_collate_infos` (audio); `get_metadata_collate_func` is patched on the thinker, the top-level model delegates. |

## Adding the hook to a new model (checklist)

1. Add a module-level `collate_multimodal_metadata(batch, sp_pad)` helper
   (`@config.add_helper`) — read `batch["image_grid_thw"]` / `["video_grid_thw"]`,
   `.tolist()`, derive `vit_*_cu_seqlens` / `vit_*_max_seqlen`, write
   `batch["multimodal_metadata"]`. Append the SP-pad tail per `sp_pad`.
   Window-attention ViTs also emit `vit_*_cu_window_seqlens` /
   `vit_*_window_index` (a host-side port of `get_window_index`).
2. Add a `get_metadata_collate_func` `override_method` returning that helper
   (or a `partial` over it if the formula needs config constants — e.g.
   `window_size` / `spatial_merge_size` / `patch_size` for window attention).
3. If the model has audio / extra feature tensors, add a `get_extra_collate_infos`
   `override_method`.
4. Model.forward: pop `multimodal_metadata`, build the per-modality `vit_metadata`
   sub-dict, pass to `get_image_features` / `get_video_features`.
5. ViT.forward: pop `vit_metadata`; consume `grid_thw_list` / `cu_seqlens` /
   `max_seqlen` (+ `cu_window_seqlens` / `window_index` for window attention)
   with a runtime fallback when absent.
6. `dummy_forward` (FSDP path): build the `vit_metadata` sub-dict host-side from
   the Python-int dummy grid.
7. Add the model to `tests/models/test_model_forward_no_implicit_sync.py`'s
   `CASES` + `_MM_METADATA_WIRED_CASES` so the sync gate feeds synthetic
   metadata and the bitwise-equivalence test
   (`test_multimodal_metadata_path_matches_fallback`) gates the collate hook
   against the in-forward fallback.

## Files

- `veomni/data/data_collator.py` — `MainCollator` carries `metadata_collate_func`;
  `PackingCollator` / `SequenceParallelCollator` invoke it after SP padding.
- `veomni/data/data_transform.py` — transforms emit the `*_grid_thw` tensors +
  `position_ids`.
- `veomni/trainer/vlm_trainer.py` — `_build_collate_fn` resolves the two model hooks.
- `veomni/models/transformers/<model>/<model>_{gpu,npu}_patch_gen_config.py` —
  `collate_multimodal_metadata` helper + `get_metadata_collate_func` /
  `get_extra_collate_infos` overrides; regenerated `generated/` files.
- `tests/data/test_mm_metadata.py` — collator-hook handoff + hook picklability.
- `tests/models/test_model_forward_no_implicit_sync.py` — sync gate; feeds synthetic
  `multimodal_metadata` for the wired cases.

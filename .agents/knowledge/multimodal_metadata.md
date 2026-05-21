# Multimodal Metadata Precompute

Contract for **precomputing multimodal forward metadata on CPU workers** instead of
deriving it inside Model.forward / ViT.forward (which forces host↔device syncs).

## Why this exists

For Qwen multimodal models the runtime ViT and Model forwards historically derived
~5-15 metadata values per step from GPU tensors:

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
(`input_features`, `audio_feature_lengths`, `audio_mask`) — previously hardcoded
by `model_type` in `vlm_trainer._build_collate_fn`.

Both hooks are resolved by `vlm_trainer._build_collate_fn` via `getattr`: a model
that doesn't expose them simply gets the runtime fallback (see below).

## The `multimodal_metadata` contract

Single optional kwarg on Model.forward / ViT.forward:

```python
def forward(self, ..., multimodal_metadata: dict | None = None, ...) -> ...:
```

The dict contains the following keys (all optional — missing keys fall back to the
existing runtime derivation):

| Key | Type | Producer | Consumer | Notes |
|---|---|---|---|---|
| `image_grid_thw_list` | `list[list[int]]` | `per_sample_metadata` (per-sample), `PackingCollator` flattens across the batch | ViT.forward, Model.forward | Python list of `[t, h, w]` triplets. Never sent to GPU. |
| `video_grid_thw_list` | `list[list[int]]` | same | same | same |
| `vit_image_cu_seqlens` | `torch.IntTensor` (CPU, then auto-moved) | `collate_multimodal_metadata` hook | Vision tower's varlen attention | Already includes the SP-pad tail as one extra "image" entry. |
| `vit_image_max_seqlen` | `int` (Python) | same | same | Already includes SP-pad. |
| `vit_video_cu_seqlens` | `torch.IntTensor` (CPU) | same | same | Same shape as image. |
| `vit_video_max_seqlen` | `int` (Python) | same | same | same |

`n_image_tokens` / `n_video_tokens` are **not** carried — they depend on
`spatial_merge_size` and the model derives them from `*_grid_thw_list` with a
one-line `sum(...)` when needed.

`position_ids` is **NOT** in the dict — it remains a top-level kwarg.

`rope_deltas` is **NOT** carried: HF's `get_rope_index` returns it for the
generation/KV-cache decode path, but the training forward always receives a
precomputed `position_ids` and never derives or reads `rope_deltas`. So the
data pipeline drops it at `merge_position_id_returns`.

## Producer flow (collator pipeline)

```
data transform (CPU worker, per-sample)
    └─ per_sample_metadata: emits image_grid_thw_list / video_grid_thw_list
       merge_position_id_returns: copies position_ids into the feature dict
       (these go into the per-sample feature dict)

       ↓ DataLoader batch ↓

PackingCollator (CPU, generic)
    1. torch.cat per pack_dim; *_grid_thw_list flattened across the batch
    2. add_flash_attention_kwargs_from_position_ids (existing, non-SP only)
    3. non-SP: invoke metadata_collate_func(batch, sp_pad={...: 0})

       ↓ SP enabled ↓

SequenceParallelCollator (CPU, generic)
    1. sp_padding on input_ids / labels / position_ids / pixel_values{,_videos};
       records per-modality sp_pad patch counts
    2. add_flash_attention_kwargs_from_position_ids (existing)
    3. invoke metadata_collate_func(batch, sp_pad={pixel_values: N, ...})

       ↓ Trainer.preforward (.to_device) ↓

Model.forward(..., multimodal_metadata={...})
    Tensors in the dict get auto-moved to GPU; Python ints/lists stay on CPU.
```

The collator carries `metadata_collate_func` (from `MainCollator`, see
`data_collator.MetadataCollateFunc`). When it is `None` — text models, third-party
pipelines — steps (3) are skipped and no `multimodal_metadata` is produced.

## SP-pad tail handling

`collate_multimodal_metadata` receives the SP-pad patch counts via `sp_pad`. When
`SequenceParallelCollator` pads `pixel_values` to SP-divisible, the hook appends
one extra `cu_seqlens` entry for the padding patches — they become one synthetic
"image" so the ViT varlen attention treats them as an independent sequence. This
mirrors how text-side `cu_seq_lens` picks up SP-pad via the `position_ids == 0`
zero-tail convention. The embeddings for those positions are discarded after the
per-rank slice in Model.forward.

## Consumer flow (model forward)

Every consumer site follows this template:

```python
precomputed_cu_seqlens = kwargs.pop("vit_cu_seqlens", None)
if precomputed_cu_seqlens is not None:
    cu_seqlens = precomputed_cu_seqlens.to(device, dtype=torch.int32, non_blocking=True)
else:
    # Runtime fallback — existing host-side build, still does one .tolist()
    cu_seqlens = _build_cu_seqlens_from(grid_thw)
```

## Backwards compatibility

Every consumer site **must** keep a runtime fallback for when `multimodal_metadata`
is `None` or a key is missing. This guarantees:

1. Third-party collators / data pipelines that don't precompute keep working.
2. Inference scripts that construct inputs manually keep working.
3. Models without the `get_metadata_collate_func` hook keep working (they pay the
   runtime sync cost — caught by `tests/models/test_model_forward_no_implicit_sync.py`).

## Model coverage

| Model | Status | Notes |
|---|---|---|
| qwen3_vl | ✅ wired | Canonical. `collate_multimodal_metadata` helper in the gpu config; npu reuses it. |
| qwen3_vl_moe | ✅ wired | Reuses qwen3_vl's helper + hook (`config.helpers.extend`). |
| qwen3_omni_moe | ✅ wired | Same ViT metadata. Also exposes `get_extra_collate_infos` (audio). |
| qwen3_5 | ✅ wired | Own `collate_multimodal_metadata` (identical formula). |
| qwen3_5_moe | ✅ wired | Own `collate_multimodal_metadata`. |
| qwen2_5_vl | ⚠️ fallback-only | Window-attention ViT (`cu_window_seqlens` from `get_window_index`, config-dependent). No `get_metadata_collate_func` yet — ViT keeps in-forward derivation. Full integration is a tracked follow-up (needs GPU `logits_equal_v5` verification). |
| qwen2_5_omni | ⚠️ fallback-only | Same window-attention ViT as qwen2_5_vl. Exposes `get_extra_collate_infos` (audio) but not `get_metadata_collate_func`. |

## Files

- `veomni/data/mm_metadata.py` — generic helpers only (`per_sample_metadata`,
  `merge_position_id_returns`). The batch-level derivation moved to the model hooks.
- `veomni/data/data_collator.py` — `MainCollator` carries `metadata_collate_func`;
  `PackingCollator` / `SequenceParallelCollator` invoke it after SP padding.
- `veomni/data/data_transform.py` — transforms call `per_sample_metadata` /
  `merge_position_id_returns`.
- `veomni/trainer/vlm_trainer.py` — `_build_collate_fn` resolves the two model hooks.
- `veomni/models/transformers/<model>/<model>_{gpu,npu}_patch_gen_config.py` —
  `collate_multimodal_metadata` helper + `get_metadata_collate_func` /
  `get_extra_collate_infos` overrides; regenerated `generated/` files.
- `tests/data/test_mm_metadata.py` — generic helpers + collator-hook handoff + hook
  picklability.
- `tests/models/test_model_forward_no_implicit_sync.py` — sync gate; feeds synthetic
  `multimodal_metadata` for the wired cases.

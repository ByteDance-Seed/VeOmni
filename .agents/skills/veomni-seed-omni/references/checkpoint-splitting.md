# Checkpoint Splitting

Read this when updating SeedOmni split-checkpoint conversion scripts.

## Source Files

Use the live model conversion scripts as examples:

- `veomni/models/seed_omni/modules/fake_model/convert_model.py` (minimal template)
- `veomni/models/seed_omni/modules/janus/convert_model.py`
- `veomni/models/seed_omni/modules/janus/convert_janus_weight_to_hf.py`
- `veomni/models/seed_omni/utils/convert_registry.py` (`convert_checkpoint`,
  `attach_module_assets`, `load_family_graphs`)

## Prefer an HF layout

When every module is a key-prefix slice of the upstream checkpoint, register an
`OmniHFLayout` in `modules/<family>/hf_layout.py` (reference:
`modules/qwen3/hf_layout.py`, machinery in `utils/hf_layout.py`) instead of a
converter. One declaration gives the offline convert, the direct load from the
upstream root (training and inference), and the HF export in the upstream
layout. Weight transforms beyond the rename go in the module's checkpoint
tensor converter, not the layout. List stored duplicates of tied weights in
`tied_source_keys`. Keep a converter only when the split needs more than
renames (several sources, re-packed tensors, generated assets).

Validate a layout with a round trip: load the upstream root, export with
`save_hf_source_checkpoint`, and compare keys, dtypes, shards and values
against the source (`tests/seed_omni/model/test_hf_layout.py`).

## Rules

- A family converter is registered in `OMNI_CONVERT_REGISTRY` under the upstream
  `model_type`, takes `(model_path, **kwargs)` and **returns**
  `{"modules": {name: module}, "training_graphs", "generation_graphs", "infer_type"}`.
  It writes nothing: `convert_checkpoint` saves the result through
  `OmniModel.save_pretrained`, so one run yields the full omni root (config,
  graph sidecars, module subfolders). Do not add a separate export / assemble
  script, and do not load a module on its own with `Module.from_pretrained(path)` —
  modules load through `OmniModel`.
- Each module instance carries its unique `model_type` config; the subfolder
  name is the module name.
- Hang module-owned assets on the instance with `attach_module_assets` so they
  save into the same subfolder:
  - tokenizer for text encoder modules,
  - image/video/audio processors for encoder modules,
  - codec processors for codec modules.
- Default graphs come from the family's `configs/seed_omni/<Family>/...` YAML via
  `load_family_graphs`; pass the `training_graph` / `generation_graph` kwargs
  through so CLI overrides still win.
- Filter monolithic checkpoint keys so each parameter belongs to exactly one
  module.
- If vocab layers move to a text encoder, remove `embed_tokens.*` and
  `lm_head.*` from the backbone state dict.
- Preserve tie-word-embedding semantics when splitting shared vocab weights.
- Write to local disk: FUSE-mounted HDFS rejects the safetensors writer
  (`os error 38`); copy the finished root afterwards.

## Validation

- Reload the root with `OmniModel.from_pretrained` and check module names,
  graph scenarios and `infer_type`.
- When re-converting, compare tensors against the previous checkpoint
  (keys, shapes, dtypes, values).
- Run graph visualization after pointing YAML at the new root.
- Run at least one training or inference smoke that exercises the new split.

# Checkpoint Splitting

Read this when updating SeedOmni split-checkpoint conversion scripts.

## Source Files

Use the live model conversion scripts as examples:

- `veomni/models/seed_omni/modules/fake_model/convert_model.py` (minimal template)
- `veomni/models/seed_omni/modules/<family>/convert_model.py`
- `veomni/models/seed_omni/modules/<family>/convert_<family>_weight_to_hf.py`
- `veomni/models/seed_omni/utils/convert_registry.py` (`convert_checkpoint`,
  `attach_module_assets`, `load_family_graphs`)

## Prefer an HF layout

When every module is a key-prefix slice of the upstream checkpoint, register an
`OmniHFLayout` in `modules/<family>/hf_layout.py` (machinery in
`utils/hf_layout.py`). One declaration gives the offline convert, the direct
load from the upstream root (training and inference), and the HF export in the
upstream layout. Put weight transforms beyond key-prefix renaming in the
module's checkpoint tensor converter, list stored duplicates of tied weights in
`tied_source_keys`, and use `convert_model.py` when combining multiple sources,
re-packing tensors, or generating assets.

Validate a layout with a round trip: load the upstream root, export with
`save_hf_source_checkpoint`, and compare keys, dtypes, shards and values
against the source (`tests/seed_omni/model/test_hf_layout.py`).

## Rules

- A family converter is registered in `OMNI_CONVERT_REGISTRY` under the upstream
  `model_type`, takes `(model_path, **kwargs)` and **returns**
  `{"modules": {name: module}, "training_graphs", "generation_graphs", "infer_type"}`.
  `convert_checkpoint` saves the result through `OmniModel.save_pretrained` so
  one run produces the full omni root (config, graph sidecars, module
  subfolders), and modules load through `OmniModel.from_pretrained`.
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
- Filter monolithic checkpoint keys so each parameter belongs to one module
  (for example, move `embed_tokens.*` and `lm_head.*` to the text encoder and
  preserve tie-word-embedding semantics).
- Write converted checkpoints to local disk first (since `safetensors` requires
  local seek/mmap support) and copy the finished root to remote storage
  afterwards.

## Validation

- Reload the root with `OmniModel.from_pretrained` and check module names,
  graph scenarios and `infer_type`.
- When re-converting, compare tensors against the previous checkpoint
  (keys, shapes, dtypes, values).
- Run at least one training or inference smoke test that exercises the split.

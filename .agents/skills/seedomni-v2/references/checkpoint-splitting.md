# Checkpoint Splitting

Read this when updating SeedOmni split-checkpoint conversion scripts.

## Source Files

Use the live model conversion scripts as examples:

- `veomni/models/seed_omni/modules/janus/convert_model.py`
- `veomni/models/seed_omni/modules/janus/convert_janus_weight_to_hf.py`

## Rules

- Each SeedOmni module gets its own output subfolder.
- Save `config.json` with the module's unique `model_type`.
- Save module-owned assets into the same subfolder:
  - tokenizer for text encoder modules,
  - image/video/audio processors for encoder modules,
  - codec processors for codec modules.
- Filter monolithic checkpoint keys so each parameter belongs to exactly one
  module output.
- If vocab layers move to a text encoder, remove `embed_tokens.*` and
  `lm_head.*` from the backbone state dict.
- Preserve tie-word-embedding semantics when splitting shared vocab weights.
- Print output paths in a final summary so config updates are mechanical.

## Validation

- Load each split module with `from_pretrained`.
- Verify strict state-dict load unless there is an intentional documented
  exception.
- Run graph visualization after pointing YAML at the split root.
- Run at least one training or inference smoke that exercises the new split.

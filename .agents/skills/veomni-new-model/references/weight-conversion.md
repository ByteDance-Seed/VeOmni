# Pretrained Weight Conversion

Loaded from `/veomni-new-model` when source checkpoint keys, tensor layouts, or component namespaces do not load directly into the target VeOmni model.

## 1. Decide & Confirm Naming Direction

- Inspect source keys/shapes/dtypes/tied weights against the meta-constructed target model's expected `state_dict()`. Separate backbone, encoder, VAE, EMA, and training-state namespaces.
- If source, checkpoint, and VeOmni keys conflict without an established convention, present concrete old/new keys and their loading, export, and DCP-resume consequences to the user before introducing renames or aliases.
- Never hide mismatches with `strict=False`.

## 2. Implement a Focused Converter

- Reuse an existing converter when possible; otherwise place a model-specific script at `scripts/model_conversion/<model_name>.py` (focused on source, destination, and real conversion flags — no generic framework).
- Handle and verify:
  - Prefix / component namespace renames
  - Layout transposes/permutes
  - Splitting fused tensors or concatenating multi-tensor inputs (track every input to a concat; per-tensor rules miss incomplete groups)
  - Tied weights / explicit aliases, deliberate dtype casts, exclusions, and newly initialized params (document reason for every dropped or initialized key)
- Reject duplicate destination keys, unexplained missing/extra keys, and shape/dtype mismatches. Respect shard indices and avoid processing any shard twice.
- Keep original checkpoints untouched, write to a new directory with required component configs, and stream/shard outputs for large models.

## 3. Verify End-to-End

- Verify full source accounting and target manifest coverage (mapping counts, exclusions, dtypes, fingerprints).
- Strictly load the converted checkpoint through the real VeOmni loader (honoring tied/nonpersistent contracts), then run functional and numerical parity checks against the source model in Phase 5. If export is requested, reload and execute in the target consumer.
- Note: weight import does not establish optimizer or DCP training-state resume.

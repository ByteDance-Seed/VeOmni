# Transformers v5 MoE Weight Format Migration Plan

## Background

VeOmni currently uses split MoE expert weights in several patched model paths:

- `gate_proj`: `[E, I, H]`
- `up_proj`: `[E, I, H]`
- `down_proj`: `[E, H, I]`

Transformers v5 standardizes expert weights to:

- `gate_up_proj`: `[E, 2I, H]`
- `down_proj`: `[E, H, I]`

This creates friction for loading HuggingFace safetensors and for downstream inference integrations.

### Agreed decisions

1. **Modeling format**: adopt transformers v5 format in VeOmni for relevant MoE models.
2. **Mismatch loading**: implement runtime conversion (per-model, pure PyTorch) in VeOmni loader.

## Goals

1. Remove mandatory offline checkpoint merge as a training prerequisite.
2. Keep fused MoE fast path efficient by supporting `gate_up_proj` directly.
3. Preserve backward compatibility for legacy checkpoints through runtime conversion.
4. Keep DCP-to-HF export and verl-veomni integration stable.

## Scope

In scope:

1. Loader runtime conversion for MoE expert weights.
2. Fused MoE API and kernel path support for `gate_up_proj`.
3. Modeling updates for transformers-v5 MoE paths.
4. EP sharding plan key updates.
5. Comprehensive docs and tests.

Out of scope:

1. Replacing HF-style generic conversion rule engines.
2. Removing offline conversion scripts entirely in first rollout.

## Implementation Plan

### Phase 0: Design and guardrails

1. Define exact conversion behavior per model family:
   - Legacy per-expert split keys -> merged `gate_up_proj` and `down_proj`.
   - Legacy tensor layout requiring transpose -> expected model layout.
2. Define strict validation rules:
   - Required key sets.
   - Expected shapes.
   - Clear error messages on mismatch.
3. Add feature flags/logging conventions for controlled rollout and debugging.

### Phase 1: Loader runtime conversion framework

1. Add a conversion hook framework in `veomni/models/module_utils.py` load loops.
2. Support streaming conversion during shard iteration.
3. Support buffered multi-key combine operations for per-expert keys.
4. Apply the same conversion logic to:
   - `load_model_weights`
   - `rank0_load_and_broadcast_weights`
5. Add conversion diagnostics:
   - model type
   - conversion mode chosen
   - number of converted tensors

### Phase 2: Per-model conversion implementations

1. Add explicit per-model converters (pure PyTorch only), initially for:
   - `qwen3_moe`
   - `qwen3_omni_moe` (thinker path)
   - `deepseek_v3` (MoE layers)
2. Handle legacy per-expert checkpoint keys and merged legacy keys.
3. Keep conversion code straightforward and independently unit-testable.

### Phase 3: Fused MoE interface migration

1. Extend `veomni.ops.fused_moe_forward` to accept `gate_up_proj` directly.
2. Keep temporary backward-compatible support for split FC1 arguments.
3. Update CUDA paths:
   - triton autograd path
   - torch grouped-mm path
4. Update NPU paths to consume merged FC1 without forward-time concat.
5. Remove avoidable runtime `split/transpose/contiguous` in model forward where possible.
6. Let's verify our implementation by asserting close bwtween `fused_moe_forward(..., fc1_1, fc1_2, None, ...)` and `fused_moe_forward(..., None, None, fc1_1_2, ...)` and eager mode in a unit test.

### Phase 4: Modeling and EP plan updates

1. Update MoE expert modules to parameterize:
   - `gate_up_proj`
   - `down_proj`
2. Update fused and eager forward logic to use merged FC1 representation.
3. Update `_init_weights` logic for new parameter names.
4. Update EP sharding plans from `gate_proj/up_proj` to `gate_up_proj` for migrated models.
5. Ensure DCP state dict save/load paths remain valid with updated names.

### Phase 5: Integration and migration cleanup

1. Verify `build_foundation_model` startup works from raw HF safetensors without offline merge.
2. Verify DCP checkpoint save/load and DCP-to-HF conversion with new layout.
3. Keep offline scripts as optional tools, with docs updated to "optional utility" status.
4. Update verl-veomni integration notes for both:
   - initial model loading
   - training-engine to inference-server weight transfer

## Documentation Deliverables

1. Update `docs/transformers_v5/transformers_v5_moe_weight_loading.md`:
   - new canonical VeOmni format
   - runtime conversion behavior and supported legacy formats
   - model-specific caveats
2. Add a dedicated runtime conversion doc section with:
   - conversion flow diagrams
   - error troubleshooting
   - performance notes
3. Add user migration guide:
   - before vs after workflow
   - when offline tools are still useful
4. Add maintainer guide:
   - how to add a new model converter
   - required tests for new converters

## Test Plan

### Unit tests

1. Converter tests for each supported model:
   - per-expert split -> merged
   - transpose conversion
   - missing key and shape mismatch failures
2. Loader tests for:
   - normal single-rank load
   - `rank0_load_and_broadcast_weights` path
3. Fused MoE kernel parity tests for merged FC1 path:
   - forward parity
   - backward parity

### Integration tests

1. Build and load model from representative HF checkpoint format without offline preprocessing.
2. EP/FSDP/TP smoke tests with migrated models.
3. Checkpoint save/load regression:
   - DCP save -> load
   - DCP -> HF export -> reload
4. verl-veomni interop validation for model initialization and weight transfer mapping.

### Regression tests

1. Existing non-MoE model load path unchanged.
2. Existing qwen3_vl_moe behavior preserved or improved.
3. Legacy split-format checkpoints still load successfully through runtime conversion.

## Success Metrics

### Functional metrics

1. `build_foundation_model` succeeds on target legacy MoE HF checkpoints without offline merge.
2. Fused and eager MoE produce numerically consistent outputs against pre-migration baseline tolerances.
3. DCP save/load and DCP-to-HF conversion pass for migrated models.

### Performance metrics

1. No significant throughput regression in fused MoE training/inference on CUDA and NPU.
2. Reduced forward-time tensor transform overhead for models already storing `gate_up_proj`.

### Quality metrics

1. New and updated docs fully cover user workflow and maintainer extension path.
2. Test suite includes unit + integration + regression coverage for conversion and kernels.
3. CI passes all affected test targets for migrated model families.

## Rollout Strategy

1. Land loader conversion framework and tests first.
2. Land fused MoE API updates with backward compatibility.
3. Migrate one model family first (`qwen3_moe`) as pilot.
4. Expand to `qwen3_omni_moe` and `deepseek_v3`.
5. Deprecate mandatory offline merge guidance in docs after pilot stabilization.

## Risks and Mitigations

1. Risk: silent conversion mistakes.
   - Mitigation: strict validation, verbose conversion logs, shape assertions, dedicated unit tests.
2. Risk: kernel regressions from API migration.
   - Mitigation: parity tests against reference path and staged rollout per backend.
3. Risk: checkpoint compatibility regressions.
   - Mitigation: end-to-end DCP/HF roundtrip tests and explicit legacy fixtures.

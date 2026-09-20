# Qwen3 Omni MoE: Model Integration Example

This guide maps the current thinker-training integration for contributors.
For runnable workflows, see the [Omni recipe](../../models/qwen/qwen3-omni.md)
and [offline audio/video recipe](../../models/qwen/qwen3-omni-offline-av.md).

Read the authored
[GPU patch configuration](../../../veomni/models/transformers/qwen3_omni_moe/qwen3_omni_moe_gpu_patch_gen_config.py),
[NPU configuration](../../../veomni/models/transformers/qwen3_omni_moe/qwen3_omni_moe_npu_patch_gen_config.py),
and [registration](../../../veomni/models/transformers/qwen3_omni_moe/__init__.py).
Do not edit generated outputs or apply the retired runtime modeling-patch procedure.
The [integration checklist](guide_and_checklist.md) covers the full workflow.

The generated top-level model trains the **thinker**. Talker/code2wav classes
are excluded, `has_talker` is disabled, and `enable_talker` raises. Audio inputs
are supported by the thinker path; this does not establish speech-generation
training support for the excluded towers.

## P1. Fix `tie_word_embeddings` (Config)

Inspect the thinker configuration and actual checkpoint keys before changing
weight tying. The current top-level patch also handles unused talker/code2wav
keys during loading. Keep these model-specific exclusions explicit; do not
suppress arbitrary unexpected keys in a general loader.

## P2. FSDP Dummy Forward (VLMs and Omni-modal)

Image, video, audio, and text-only samples can be distributed unevenly across
ranks. The patched towers and thinker forward must preserve collective ordering
for missing modalities. Cover the production path with the asymmetric-modality
distributed test rather than only calling each tower in isolation.

## P3. SP: Language Model Position Embeddings

The thinker text sequence consumes position information compatible with the
SP-sliced inputs and the selected attention backend. Validate packed samples,
multimodal rotary positions, and SP=1 versus SP>1 together.

## P4. SP: Vision Transformer Padding and Slicing

Vision patch lengths differ from language sequence lengths. The authored vision
patches own their padding and SP layout. When using precomputed metadata, do not
add the same SP padding a second time in the forward fallback.

## P5. SP: ViT-to-LM Fill-Back (3-Step Dance)

The thinker forward consumes modality masks and scatters visual/audio features
into the corresponding language-token slots. Trace the gather/scatter operations
in the actual patch configuration, including the async Ulysses path. Reusing the
VL-only fill-back code without audio handling is insufficient.

## P6. SP: Deepstack / Cross-Layer Visual Embeddings

The patched thinker text model handles deepstack features throughout the decoder.
Keep each deepstack tensor aligned with the rank's language sequence, including
ranks that did not receive visual inputs directly.

## P7. MoE: Fused Forward + Stacked Expert Weights

The thinker expert patch replaces the upstream experts implementation and routes
computation through VeOmni's fused-MoE interface. Registration attaches checkpoint
conversion for the fused parameter representation. Inspect the
[parallel plan](../../../veomni/models/transformers/qwen3_omni_moe/parallel_plan.py)
and [MoE loading guide](../../transformers_v5/transformers_v5_moe_weight_loading.md)
together: model tensors, checkpoint layout, and EP sharding must agree.

## P8. Pop Flash-Attention kwargs Before ViT Forward

Language attention metadata and vision/audio cumulative sequence lengths have
different meanings. The patched forwards must pass only the applicable metadata
to each tower. Validate each tower's actual current signature rather than copying
an older `kwargs` filtering block.

## P9. Pre-compute `max_seqlen` (Performance)

Expose model-owned metadata hooks to the collator and pass the per-modality
metadata through the thinker to each tower. Follow the
[metadata contract](../../developer/multimodal_metadata.md) and extend the
forward-no-implicit-sync regression cases. Retain the fallback for direct callers.

## P10. Position ID Transposition

Processor, packed-batch, and model rotary conventions need not use the same axes.
Check the shapes at each boundary against the authored position-ID helper.
A transpose that happens to work for text-only input can misalign multimodal
rotary axes after packing or SP slicing.

## P11. VeOmni Loss Utility

The patched thinker conditional-generation forward uses its configured loss
function and, where requested, the MoE auxiliary loss. Preserve the label-shift
and SP loss-reduction contracts. Test token masking and the selected eager/fused
loss path, not only the shape of returned logits.

## P12. `get_position_id_func` (Multimodal RoPE)

The top-level training model exposes position-ID, metadata, and additional
collation hooks through the thinker integration. The collator must receive these
hooks via the same runtime used by the real trainer. Check both registered
architecture variants where supported.

## Testing

### Three-Level Strategy

Use model-level numerical checks, distributed alignment checks, and a short
production-entry-point run. Each establishes different evidence; record device,
precision, parallel sizes, and source revision for hardware results.

### Level 1 — Unit Tests

#### Toy Config

Use `tests/toy_config/qwen3omni_toy` as the existing small-model reference.
Keep vocabulary, modality token IDs, and processor assumptions consistent.

#### Dummy Dataset

`veomni/data/dummy_dataset.py` provides the multimodal fixtures used by model and
parallel tests. Include missing-modality and mixed-modality batches.

#### Forward/Backward Patch Test

Extend `tests/models/test_models_patch.py` and relevant registry/converter cases.
Compare patched and upstream thinker behavior with matching weights and supported
attention/MoE combinations.

### Level 2 — Parallel Alignment Test

Extend the Qwen3 Omni cases in `tests/e2e/test_e2e_parallel.py`, and cover
asymmetric ranks in `tests/distributed/test_dummy_forward.py`. Test the intended
SP/EP configurations on real accelerators.

### Level 3 — End-to-End Training Test

Run a bounded training recipe through `tasks/train_vlm.py`. Verify data/processor
assets, finite loss, and checkpoint save/load behavior. This test must use the
same registered modeling and runtime as the user-facing recipe.

### What to Add Per Test Level

Follow [Testing a New Model](../../transformers_v5/testing_new_model.md) and the
[test selection rules](../../testing.md). A newly created test file must also be
selected by its owning CI workflow.

## Future Testing Gaps

The thinker tests do not validate talker/code2wav training or speech generation.
Full-size memory use, convergence, and additional hardware combinations require
separate runs; registration and toy tests are not evidence for those claims.

## Acknowledgements

This integration builds on Hugging Face's Qwen3 Omni MoE implementation. Compare
against the pinned Transformers version when reviewing generated differences.

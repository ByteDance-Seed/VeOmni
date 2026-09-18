# Qwen3 VL: Model Integration Example

This is a contributor's map of the current generated Qwen3 VL integration.
For training commands, use the [Qwen3 VL recipe](../../examples/qwen3_vl.md).
Follow the [integration checklist](guide_and_checklist.md) when adding a model.

The authoritative modeling input is the
[GPU patch configuration](../../../veomni/models/transformers/qwen3_vl/qwen3_vl_gpu_patch_gen_config.py).
The [NPU configuration](../../../veomni/models/transformers/qwen3_vl/qwen3_vl_npu_patch_gen_config.py)
and [MoE configuration](../../../veomni/models/transformers/qwen3_vl_moe/qwen3_vl_moe_gpu_patch_gen_config.py)
show how related targets share patches. Generated files are outputs, not edit points.

## 1. FSDP: Dummy ViT Forward

A text-only rank may receive no pixels while another rank executes the vision
tower. The patched `dummy_forward` keeps the required sharded tower operations
aligned across ranks. Review its use in the patched model forward together with
`_no_split_modules`; adding a dummy call outside the real forward does not prove
collective ordering is correct.

Exercise this path with `tests/distributed/test_dummy_forward.py`, including
rank-asymmetric image/video batches. The test must run on multiple accelerators;
a CPU import test cannot establish FSDP safety.

## 2. Sequence Parallelism

The text sequence and the vision patch sequence have different layouts. Trace
both through the patch configuration before changing a gather, padding rule,
mask, or position-ID transformation.

### 2.1 Language Model — Position Embeddings

The text forward consumes SP-sliced sequence inputs and compatible position
information. Attention operations use the VeOmni attention path; keep the
model's rotary handling consistent with that path and test packed boundaries.

### 2.2 Vision Transformer — Padding and Slicing

The vision forward accepts `vit_metadata` prepared by the model-owned collator
hook. It uses the precomputed grid information, cumulative lengths, and maximum
sequence length when available, with a runtime fallback for callers that bypass
the training data pipeline. SP tail padding must be reflected exactly once.

### 2.3 ViT-to-LM Fill-Back (3-Step Dance)

Visual embeddings must match the language sequence's image/video slots. Read the
patched model forward for the actual gather/scatter operations and mask handling;
copying a fill-back sequence from another family can duplicate or drop features.
Validate both SP=1 and SP>1, with mixed image/video/text-only samples.

### 2.4 Deepstack Visual Embeddings

Qwen3 VL carries visual features into multiple decoder layers. The SP handling
must cover those deepstack tensors as well as the initial input embedding.
A correct first-layer feature count is not sufficient to validate the full path.

## 3. Expert Parallelism

Dense Qwen3 VL and Qwen3 VL MoE share visual processing but differ in their expert
parameters and checkpoint conversion. Use the MoE sibling's authored configuration
and [registration](../../../veomni/models/transformers/qwen3_vl_moe/__init__.py)
when reviewing the expert path.

### 3.1 Parallel Plan

The [MoE parallel plan](../../../veomni/models/transformers/qwen3_vl_moe/parallel_plan.py)
selects expert parameter paths for ExtraParallel sharding. Match those paths to
the generated model and the checkpoint converter. See
[ExtraParallel](../../key_features/extra_parallel.md) for the mesh contract.

### 3.2 Fused MoE Forward

The MoE patch configuration replaces the upstream expert computation with the
VeOmni expert interface. Kernel selection belongs to the registry/dispatch layer;
the model does not hard-code one accelerator's fused implementation. Test the
supported fused/eager combinations and EP behavior using the existing model and
parallel test tables.

## 4. Performance: Pre-compute `max_seqlen`

The `collate_multimodal_metadata` helper and `get_metadata_collate_func` hook
allow the data pipeline to compute visual metadata on CPU. The forward consumes
that metadata instead of repeatedly reading lengths from device tensors.
The [metadata contract](../../developer/multimodal_metadata.md) describes hook
picklability, packing/SP order, and the fallback path.

Extend `tests/models/test_model_forward_no_implicit_sync.py` when wiring a new
family. Keep the fallback functional for inference and direct model callers;
only the precomputed training path promises to avoid these implicit syncs.

## 5. Model Registration

[Qwen3 VL registration](../../../veomni/models/transformers/qwen3_vl/__init__.py)
selects the generated GPU or NPU class and handles both the base model and
conditional-generation architecture. The registration decorator runs when the
package is imported; importing a generated module directly in a test does not
verify the registry route used by training.

Model-bound work runs through `VLMModelRuntime`; data and loop orchestration run
through `VLMTrainer`. See [Trainer](../trainer.md), then follow
[Testing a New Model](../../transformers_v5/testing_new_model.md) for registry,
numerical parity, freeze behavior, metadata, and distributed cases.

## Acknowledgements

This integration builds on Hugging Face's Qwen3 VL implementation. Keep comparisons
against the repository's pinned Transformers version so upstream changes can be
reviewed separately from VeOmni's patches.

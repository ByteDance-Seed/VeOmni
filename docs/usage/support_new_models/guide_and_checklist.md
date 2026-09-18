# Support New Models — Guide and Checklist

Use this workflow to add a Transformers-family model to VeOmni's generated
modeling, data pipeline, runtime, and tests. Start from the closest existing
model and the Transformers version pinned in `pyproject.toml`. Diffusion
architectures use the separate [DiT guide](dit_model_guide.md).

Worked examples: [Qwen3 VL](qwen3_vl_example.md) and
[Qwen3 Omni MoE](qwen3_omni_moe_example.md). The
[patchgen reference](../../design/patchgen.md) documents the generation API.

## Integration Complexity by Model Type

| Model | Integration surface |
| --- | --- |
| Dense text | GPU/NPU patch configuration, generated modeling, registration, training config, tests |
| MoE | Dense-model work plus fused expert layout, checkpoint conversion, and an ExtraParallel plan |
| VLM | Modeling plus processor/data transforms, position IDs, metadata hooks, asymmetric-modality FSDP tests |
| Omni | VLM work plus audio handling and an explicit definition of which output towers are trained |

## Step-by-Step Integration

### Step 0: Understand the Target Model

Record `model_type`, supported `architectures`, processor/tokenizer classes,
checkpoint parameter names, and the training task. Inspect the upstream source
from the pinned Transformers version. Identify attention variants, expert tensor
layout, and modalities before choosing an existing model to extend.

A matching `model_type` alone does not establish training support. Define the
intended accelerator/kernel combinations and which output heads are in scope.

### Step 1: Create the Model Directory

Keep authored inputs and generated outputs distinct:

```text
veomni/models/transformers/<model>/
├── __init__.py
├── <model>_gpu_patch_gen_config.py
├── <model>_npu_patch_gen_config.py    # when NPU support is implemented
├── parallel_plan.py                 # when additional sharding is needed
├── checkpoint_tensor_converter.py   # when checkpoint layout differs
└── generated/                       # written by patchgen only
```

Use [Qwen3](../../../veomni/models/transformers/qwen3/qwen3_gpu_patch_gen_config.py)
for dense text, or the worked examples above for multimodal models. Add custom
configuration/processor modules only where the upstream implementation needs
adaptation.

### Step 2: Register Your Model (`__init__.py`)

Register a modeling factory with `MODELING_REGISTRY` at module import time. The
factory selects GPU/NPU generated classes and returns the class matching the
requested architecture. Follow the actual
[Qwen3 registration](../../../veomni/models/transformers/qwen3/__init__.py).
Do not return an unpatched upstream class for a path that requires VeOmni hooks.

Use `MODEL_CONFIG_REGISTRY` or `MODEL_PROCESSOR_REGISTRY` only if custom classes
are required. Modeling/config registry keys match `config.json`'s `model_type`;
processor registry keys match the processor class name.

### Step 3: Add to Package `__init__.py`

Import the model package from
[the Transformers model package](../../../veomni/models/transformers/__init__.py)
so its registration is installed when VeOmni loads. Keep accelerator-specific
imports guarded; package import must also work without that accelerator runtime.

<span id="step-4-patch-the-model-modeling-py"></span>

### Step 4: Author and Generate the Modeling Patches

Write the patch configuration, using `PatchConfig` decorators for methods,
functions, imports, and class changes. Reuse a sibling configuration with
`name_map` where the structures match; verify renamed classes and attributes
against upstream source rather than assuming identical semantics.

Bind optimized operations through the current
[kernel registry and OpSlot contracts](../../design/unified_kernel_registry.md).
Do not add the retired `apply_veomni_patch()` modeling workflow or edit files in
`generated/` manually.

For example, inspect generation of the existing Qwen3 GPU configuration:

```bash
patchgen veomni.models.transformers.qwen3.qwen3_gpu_patch_gen_config --dry-run
```

For the new model, run `patchgen` with its own configuration module and `--diff`,
inspect the generated Python and diff, then run `make check-patchgen`. Generate
with the repository's pinned Transformers version and commit authored inputs
and generated outputs together. See [patchgen](../../design/patchgen.md) for
command options and discovery rules.

### Step 5: Define Expert Parallelism Plan (`parallel_plan.py`, MoE only)

Match the parameter paths and tensor dimensions of the generated model. A
fused `gate_up_proj` layout is not interchangeable with separate per-expert
linear modules. Use [Qwen3 MoE's plan](../../../veomni/models/transformers/qwen3_moe/parallel_plan.py)
and [ExtraParallel](../../key_features/extra_parallel.md) as references.

If the HF checkpoint representation differs, implement and test a
`CheckpointTensorConverter`; attach its factory in model registration as in
[Qwen3 MoE](../../../veomni/models/transformers/qwen3_moe/__init__.py).
Check load, HF export, and DCP resume separately. Never hide an unsupported
mapping by dropping missing/unexpected keys indiscriminately.

### Step 6: Patch the Processor (`processing_*.py`, multimodal only)

Compare upstream processor inputs with the data transform: singular/plural
argument names, empty modalities, image/video sizing, and audio sample rate.
Only register a custom processor when an adaptation is needed. Confirm that
model exports retain the same processor/tokenizer assets used for training.

### Step 7: Write the Data Transform Function

Register the transform in
[data_transform.py](../../../veomni/data/data_transform.py) or the appropriate
multimodal helper. Test the full input contract: tokenization, assistant-only
labels, modality masks, position IDs, and packing boundaries.

For VLM/Omni models, implement the model-owned metadata hooks described in
[multimodal metadata precompute](../../developer/multimodal_metadata.md).
Keep the collator generic and preserve its packing → SP padding → metadata
precompute → slicing order. Test ranks with different modality presence.

### Step 8: Hook into the Trainer

Choose the existing task entry point where possible. Model construction,
freezing/LoRA, sharding, weight loading, optimizer setup, and model checkpoint
I/O belong to `VeOmniModelRuntime` or a specialized runtime. The trainer owns
data, the job loop, callbacks, and scheduling. See the
[architecture guide](../../developer/architecture.md) and
[Trainer guide](../trainer.md).

For VLM integration, inspect `VLMModelRuntime` and `VLMTrainer` in
[vlm_trainer.py](../../../veomni/trainer/vlm_trainer.py). Extend only the relevant
runtime/data hooks; do not copy model construction back into the trainer.

### Step 9: Add a Config File

Add a YAML under the appropriate `configs/` modality. Keep model placement,
optimizer, and chat template under `model`; data under `data`; job scheduling,
logging, and checkpoint cadence under `train`. Check field names against the
[arguments reference](../arguments.md) and the actual dataclasses.

Add a recipe with prerequisites, preparation, launch, output checks, and limits,
then link its configuration in the [catalog](../../examples/index.md).

### Step 10: Test

Follow [Testing a New Model](../../transformers_v5/testing_new_model.md).
Extend existing cases for registry loading, patched/upstream numerical parity,
checkpoint conversion, and distributed training. For multimodal models, also
cover asymmetric modality batches and the forward metadata sync gate.

New tests must be selected by the owning CI workflow. State which hardware
combinations were actually run, and separate toy-model checks from full-size
training results. Run `make quality`, `make check-patchgen`, and the relevant
model/data/parallel tests before submission.

## Patch Reference (Quick Table)

| Concern | Reference |
| --- | --- |
| Generated modeling and shared patches | [Patchgen](../../design/patchgen.md) |
| Kernel binding | [Kernel registry](../../design/unified_kernel_registry.md) |
| Expert sharding | [ExtraParallel](../../key_features/extra_parallel.md) |
| Checkpoint representation | [MoE weight loading](../../transformers_v5/transformers_v5_moe_weight_loading.md) |
| VLM position IDs and metadata | [Metadata precompute](../../developer/multimodal_metadata.md) |
| Runtime ownership and training loop | [Architecture](../../developer/architecture.md) |

## Checklists

### Any New Model

- Registration selects the correct generated class on each supported device.
- Generated outputs reproduce from the checked-in configuration and pinned dependencies.
- Model, data, and checkpoint paths work together through a production task entry point.

### VLMs (image/video)

- Position IDs, masks, metadata, and visual feature fill-back agree under SP.
- Text-only ranks participate in the required FSDP tower operations.
- Processor assets and freeze/LoRA behavior are covered.

### MoE Models

- Expert layout, parallel plan, load conversion, and export agree.
- Fused/eager numerical parity and supported EP configurations are tested.

### Omni-modal (audio)

- Audio lengths/masks and missing-modality batches are covered.
- Training versus generation scope is explicit; unused towers are not accidentally wrapped.

### Testing (all models)

- Relevant existing suites include the model and CI selects the cases.
- Recipe documentation records validation limits instead of inferring support from registration.

## Common Pitfalls

- A generated model import succeeds while the registry still returns the upstream class.
- A fused expert shape changes without updating conversion or the parallel plan.
- A multimodal forward derives metadata on the GPU despite a collator precompute hook.
- An old config key is copied into a new recipe; unknown fields are rejected.
- A test runs locally but is not enumerated in the owning CI workflow.

## Key Imports

Use the public entry points in `veomni.models`, registries in
`veomni.models.loader`, the runtime in `veomni.models.model_runtime`, and
`PatchConfig` from `veomni.patchgen`. Consult their current implementations
rather than copying a model registration or a patch from an older release.

---
name: veomni-new-model
description: "Add support in VeOmni for an existing external model from Transformers, Diffusers, or another framework or standalone repository. Identify the source implementation, adapt modeling, weights, data and training interfaces, and validate the resulting integration. Use for new model support and external model migration; not for designing a model from scratch."
---

# VeOmni New-Model Onboarding

Own the lifecycle of bringing an existing external model into VeOmni. Keep one
entry point: inspect the implementation to choose the modeling route, then
complete the shared configuration, training, and validation steps.

Transformers modeling remains owned by `/veomni-patchgen-model`. An existing
model failure belongs to `/veomni-debug`; a patchgen-only refresh belongs to
`/veomni-patchgen-model`.

Repository paths below are relative to the VeOmni root. Optional resources are
relative to `.agents/skills/veomni-new-model/`.

## Plan the integration

Track the applicable phases with the running agent's plan mechanism:

1. Analyze the source model and identify integration gaps.
2. Implement modeling through the appropriate source-specific route.
3. Add model and training configuration.
4. Integrate data, trainer, and checkpoint interfaces.
5. Validate and document the supported behavior.

Record the source revision, intended training or inference scope, target
backend, affected files, and acceptance criteria. Keep the work proportional
to the request; a checkpoint conversion or hardware report is needed only when
the corresponding work is in scope.

## Phase 1: Analyze the source model

Read the actual config, model construction code, base classes, forward path,
and relevant training entry point. A HuggingFace hosting URL, repository name,
or installed dependency alone does not identify the model's implementation.

| Implementation evidence | Modeling route |
|---|---|
| The relevant model uses Transformers modeling and configuration contracts | Transformers |
| The relevant model uses Diffusers model components or pipelines | Diffusers |
| The model is implemented independently or depends on another framework's model or training abstractions | Other framework or standalone repository |

For mixed systems, classify components separately and record their tensor and
checkpoint boundaries. A Transformers encoder and an independent diffusion
backbone can follow different routes in the same integration. Resolve unclear
classification by inspecting construction and execution paths before editing.

Identify:

- Model category and the closest current VeOmni implementation: text, VLM,
  Omni, or DiT. Source framework and model category are separate decisions.
- Trainable and frozen components, forward inputs/outputs, loss construction,
  shape conventions, dtypes, and initialization or pretrained-weight needs.
- Raw data schema, preprocessing, conditioning, batching, and collation needed
  to reproduce the requested behavior.
- Source checkpoint format and parameter layout, including tied weights,
  component prefixes, and training-only state.
- Framework-specific operations, runtime dependencies, and required parallelism.

For a substantial external repository, read
[Source analysis and planning](references/migration-planning.md). The optional
local analyzer inventories classes, imports, configuration, and source revision;
it does not classify the model or generate implementation requirements.

## Phase 2: Implement modeling

Place DiT model integrations under `veomni/models/diffusers/<model_name>/`,
including independently implemented or other-framework diffusion backbones and
their model-specific conditioning components. This is a repository layout
convention, not a requirement to adopt Diffusers base classes or pipelines.
Choose implementation and loading contracts from the source evidence; reusable
Transformers encoder subcomponents still follow the Transformers route.

### Transformers

Create `veomni/models/transformers/<model_name>/`, then use
`/veomni-patchgen-model`. It owns patchgen configs, model registration, generated
modeling, ExtraParallel plans, required MoE tensor conversion, and model-level
tests. Reuse a suitable sibling patchgen config where possible. Never edit
`generated/` directly.

Return to this workflow for data, training configuration, and end-to-end
integration after the applicable loading and patch tests pass.

### Diffusers

Inspect the components actually needed for the requested task; an inference
pipeline alone does not define a training interface. Follow the closest
implementation under `veomni/models/diffusers/` and, for DiT work,
`docs/usage/support_new_models/dit_model_guide.md`.

Use the existing direct modeling or device-patch conventions. Preserve component
boundaries and pretrained-loading behavior where compatible. Identify the
trainable component and how conditioning, timestep/noise sampling, targets, and
loss reach it. Transformers subcomponents still use the Transformers route.

### Other framework or standalone repository

Read [External implementation mapping](references/migration-implementation.md)
for this route. Build an explicit source-to-VeOmni mapping:

1. Trace the source's model construction and execution path. Separate the model
   from its launcher, framework trainer, data pipeline, and checkpoint utilities.
   Inspect the training path when training is requested.
2. Choose the smallest compatible implementation strategy: wrap reusable
   PyTorch modules, port the required source subset, or translate operations
   whose source framework cannot run in VeOmni. Preserve architecture and
   numerical semantics; do not redesign the model during migration.
3. Place DiT integrations under `veomni/models/diffusers/<model_name>/` as
   specified above. For other model categories, follow the closest current
   VeOmni implementation; being non-Transformers alone does not imply a
   `diffusers/` location. Reuse the existing registration and loading contracts;
   add a loader extension only when they cannot express the source model.
4. Map construction/configuration, forward/loss, weight loading, and backend
   operations to VeOmni interfaces. Preserve parameter names and layouts where
   possible. Record deliberate differences and retain source revision, path,
   and license attribution for copied or translated code.
5. Establish a working eager/native path with a small representative input
   before introducing fused kernels or distributed execution. Compare against
   the source at the component boundary when a complete source run is costly.

For all routes, `parallel_plan.py` describes ExtraParallel sharding, not FSDP
wrapping. FSDP2 wraps generically in `build_parallelize_model()`. Add a plan only
when ExtraParallel is needed, including expert or embedding sharding. Keep
backend-specific imports isolated so supporting one backend does not break the
other backend's imports.

## Phase 3: Add configuration

- Follow current config/loader conventions, including
  `configs/model_configs/<model_family>/<ModelName>.json` where applicable.
  Translate upstream configuration fields explicitly when they differ.
- Put training YAML in the existing category-specific location:
  `configs/text/<model_name>.yaml`,
  `configs/multimodal/<model_name>/<model_name>.yaml`, or
  `configs/dit/<model_name>.yaml`.
- Verify fields against the current argument dataclasses and a working peer
  config. Include the required model, data, optimizer, accelerator, and
  checkpoint settings.

Use `assets/templates/data-config.yaml` and `assets/templates/train-config.yaml`
only as starting points when useful; adapt them to the chosen trainer. Do not
treat template fields or machine-specific paths as ready-to-run configuration.

## Phase 4: Integrate data, training, and checkpoints

For training, use `TextTrainer`, `VLMTrainer`, or `DitTrainer` as appropriate.
Adapt the source model to the existing trainer lifecycle before introducing
trainer extensions. Keep the source framework's launcher and training loop out
of the normal VeOmni execution path.

Reuse existing data transforms and collators where their semantics match. Add
custom handling only for an actual gap. Verify a real-format example through
preprocessing, batching, model inputs, and loss construction. For VLM/Omni
metadata precomputation, follow `.agents/knowledge/multimodal_metadata.md`.

When pretrained weights are used, first determine whether direct loading is
correct. If conversion is required, adapt `assets/templates/weight-converter.py`
or an appropriate existing converter. Account for every source and target
tensor, explain drops or initialization, reject collisions and unexplained
key/shape mismatches, and emit a conversion report. A `strict=False` load alone
does not establish correctness. Keep the original checkpoint recoverable.

Use `ModelCheckpointManager` in `veomni/models/checkpoint_manager.py` and the
existing trainer callbacks for VeOmni training checkpoints. Distinguish initial
loading of external model weights from resuming optimizer, scheduler, RNG, step,
and data state; a successful weight import does not prove training resume.

## Phase 5: Validate and document

Read `.agents/knowledge/testing.md` before selecting test locations. Establish
acceptance criteria from the requested behavior and the adaptations above,
then run applicable checks in increasing cost order:

| Trigger | Evidence |
|---|---|
| Every integration | Clean-process import, config/registry lookup, and representative forward behavior |
| Generated modeling changes | Applicable patch tests and patchgen drift check |
| Pretrained weights are used | Loading coverage, expected keys/shapes, and documented exceptions |
| Weight conversion is introduced | Complete mapping report, strict target loading, and converted-weight functional validation |
| Modeling, layouts, or operations are ported or changed | Fixed-input source-versus-VeOmni comparison with matched weights, dtype, execution mode, and stochastic inputs; predeclared tolerances |
| Training is in scope | Forward/backward, finite loss and gradients, optimizer update, and a short run through the intended data/trainer path |
| Convergence evidence is requested | Controlled tiny-overfit or other justified loss criterion, with recorded configuration and results |
| Resume is requested or checkpoint/state handling changes | Fresh-process resume compared with an uninterrupted run at the same next step, including applicable model and training state |
| GPU/NPU or distributed support is claimed | Execution on the claimed backend/topology; performance measurements only when requested or affected |

Where a source reference cannot run, state the limitation and the narrower
property any substitute check proves. Mark an unmet required check as blocked;
do not present it as passed. Inference-only work does not require a training
pipeline, and unrequested hardware support is not part of the completion claim.

Prefer existing enumerated model tests and `tests/e2e/test_e2e_parallel.py` when
they fit the model. Follow the testing guide for any new test location. Run
`make quality` and the focused checks appropriate to the actual diff. For numerical parity,
convergence, resume, or hardware evidence, read
[Validation methods](references/migration-validation.md) for the applicable check.

Document supported behavior, source revision, dependencies, configuration,
reproduction commands, validation results, and remaining limitations. Update
supported-model documentation where applicable. Use the migration checklist,
E2E report, or Ascend report templates under `assets/templates/` when the task
needs those deliverables; do not require the entire template set for every model.

---
name: veomni-new-model
description: "Integrate an existing external model into VeOmni, adapting modeling, data, training, and weight loading as needed. Use for new model support or migration from Transformers, Diffusers, or standalone implementations."
---

# Model migration into VeOmni

Bring an existing model into VeOmni while preserving its architecture and
numerical semantics. By default, complete a training integration: pretrained
weight loading, real data, a short training run, and training-state resume.
Honor an explicit inference-only or partial request by narrowing the phases and
acceptance checks. Debugging an already supported model uses `/veomni-debug`;
a patchgen-only refresh uses `/veomni-patchgen-model`.

## Responsibilities and scope

Use the language model's code-reading and implementation abilities to analyze
source, compare interfaces, choose an implementation route, generate configs,
write necessary scripts/tests, and explain results. The instructions below
specify evidence and VeOmni contracts rather than a fixed code scaffold.

- Read actual source and current repository interfaces before making decisions.
- Reuse existing tools and maintained peer implementations before adding helpers.
- Generate model-specific conversion or data-preparation scripts when executable
  work is needed; reasoning does not substitute for running them.
- Keep analysis and scratch artifacts in `.agents_workspace/`; put reusable
  model-specific code in its normal repository location.
- Add skill auxiliary files only for demonstrated repeated needs where a
  deterministic tool materially improves reliability. They are not forbidden,
  but no analyzer, YAML scaffold, or report template is required by this skill.

Follow `AGENTS.md` and its context-loading requirements. For substantial work,
track the phases and acceptance checks with the agent's plan mechanism. Keep
plans proportional to the task; a separate report is optional.

## Phase 1: Establish source and migration scope

### Inspect the implementation

Record the source repository/revision, local modifications, relevant licenses,
and the VeOmni base revision. When weights are used, record their revision or
checksum and identify which components they cover. Keep downloaded source and
weights outside committed deliverables.

Trace configuration, construction, base classes, forward, and the source training
path. A hosting URL, repository name, or imported library does not determine the
modeling route. For a mixed system, classify components independently.

For each component, determine:

- Trainable or frozen role, parameter namespace, and initialization requirements.
- Inputs/outputs, tensor shapes/layouts, dtype, masks, and positional encoding.
- Objective and ownership of targets, conditioning, and loss computation.
- Source checkpoint format, shared/tied parameters, and component boundaries.
- Required operations, framework abstractions, and backend dependencies.

Select the closest current text, VLM, Omni, or DiT integration. Model category
and implementation framework are separate decisions. Build a concise mapping
between source responsibilities and VeOmni model, data, runtime, and checkpoint
interfaces; do not import the source training framework wholesale.

### Fix the completion boundary

Use the user's target environment and topology. If no topology was specified,
start with the smallest feasible configuration; do not claim broader distributed
support. Ask about target hardware only when it cannot be derived from context
and changes the implementation or required validation.

Default completion includes pretrained initialization, real-data training, and
fresh-process resume. Long convergence experiments, performance tuning, and
support for additional hardware are not default requirements.

Identify missing prerequisites early: source code, weights, real-format examples,
runtime dependencies, or hardware. Continue independent structural work when
possible, but mark required checks blocked rather than silently reducing scope.
With configuration alone, construction and random-initialization checks are
possible; pretrained loading and numerical parity remain unverified.

**Exit evidence:** source/component mapping, intended behavior and backend,
implementation route, affected areas, and explicit acceptance checks or blockers.

## Phase 2: Implement modeling and registration

### Choose the route from source evidence

| Source implementation | VeOmni approach |
|---|---|
| Transformers modeling/configuration contracts | Use `/veomni-patchgen-model` under `veomni/models/transformers/`. |
| Diffusers components or pipelines | Follow a maintained integration; inspect the trainable components separately from the inference pipeline. |
| Independent implementation or another framework | Wrap compatible PyTorch modules, port the required subset, or translate unsupported operations. |

For Transformers, the patchgen skill owns generated modeling, registration,
model-level patch tests, and relevant ExtraParallel/converter details. Never
hand-edit generated modeling. Return here for weights, data, training, and
resume after the applicable modeling checks pass.

Place DiT integrations, including independent diffusion backbones and their
model-specific conditioning, under `veomni/models/diffusers/`. This layout does
not require Diffusers base classes. Reusable Transformers encoders still follow
the Transformers route. Other categories use their nearest maintained model.
For DiT contracts, read `docs/usage/support_new_models/dit_model_guide.md`.

### Preserve semantics and component boundaries

Choose the smallest compatible adaptation. Preserve defaults, normalization,
attention masks, positional encoding, tensor layouts, and the source objective.
For copied or translated code, retain source revision, path, and license
attribution; document deliberate semantic differences and their validation.

Define which component owns preprocessing/encoding, timestep and noise sampling,
target construction, and scalar loss. In video diffusion, distinguish the
trainable backbone from frozen text/image encoders and VAE. Follow the source's
actual training objective; an inference pipeline does not define it.

Use current registration and loading conventions. Verify registry/config lookup
in a fresh process so an earlier import cannot hide missing registration.
Extend a loader only when existing contracts cannot express model construction.
Keep backend-specific imports isolated so one backend does not break another's
import path.

First obtain a working eager/native path on representative small inputs.
Introduce fused kernels or distributed execution only after this reference works.
FSDP2 wrapping is generic in `build_parallelize_model()`; `parallel_plan.py`
describes ExtraParallel sharding, such as expert or embedding parallelism.
Do not add a parallel plan merely to select FSDP wrapping.

**Exit evidence:** clean import/registration, valid construction and forward
contracts, component placement, and applicable model-level checks.

## Phase 3: Load or convert pretrained weights

### Determine whether conversion is needed

Inspect the source checkpoint and obtain the target model's expected parameter
keys, shapes, dtypes, and tied-weight relationships. Use config/meta construction
when useful to avoid allocating a large randomly initialized model.
Distinguish backbone, encoder, VAE, EMA, and training-state namespaces.

Prefer the existing loader when names, layouts, and serialization are compatible.
Direct loading still needs coverage and behavior checks, but requires no new
conversion script. Do not convert solely to follow a standard sequence.

When source, checkpoint, and VeOmni keys conflict, show concrete old/new keys and
their loading, export, and optimizer/DCP-resume consequences. Apply an already
chosen compatibility direction. Otherwise obtain the user's choice before
implementing a rename, alias, or compatibility mapping. Do not silently preserve
an obsolete hierarchy or conceal mismatches with `strict=False`.

### Implement only the required transformation

Have the model derive mapping rules from the actual implementation. Reuse an
existing converter where suitable; otherwise write a model-specific script under
`scripts/model_conversion/`. Keep its interface focused on source, destination,
and the actual conversion choices rather than a generic conversion framework.

Account for the transformations that apply:

- Renaming prefixes or component namespaces.
- Transposing/permuting parameter layouts.
- Splitting fused tensors or concatenating multiple source tensors.
- Preserving tied weights or explicitly representing aliases in the target.
- Deliberate dtype changes, tensor exclusions, or initialized target parameters.

For every dropped tensor or initialized target parameter, state the reason.
Track all inputs to multi-tensor transformations; per-tensor mapping alone cannot
verify concatenation coverage. Reject duplicate destination keys, unexplained
missing/extra keys, and incompatible shapes or dtypes. Avoid processing the same
shard twice and respect the checkpoint's shard index when present.

Keep the original checkpoint recoverable and write converted output to a new
location. Choose streaming or sharded output when model size makes full in-memory
conversion impractical. Include necessary component configs for the intended
loader, without inventing unsupported configuration fields.

### Verify the result

Compare source accounting and target coverage against the expected manifest.
Record mapping decisions, exclusions/initializations, counts, dtype changes, and
source/output fingerprints in logs or a compact artifact; no fixed report format
is required.

Strictly load the target model using the intended VeOmni path, honoring explicit
loader contracts for tied or nonpersistent state. Then run a functional check and
the source comparison in Phase 5. A converter that successfully writes a file
has not proved target completeness or equivalent outputs.
If export is requested, also reload and execute in the intended consumer.

**Exit evidence:** direct-loading or conversion rationale, complete tensor
accounting, strict loading, and functional/numerical results or explicit blockers.
Weight import does not establish optimizer or training-state resume.

## Phase 4: Integrate data, configuration, and training

### Trace a real sample

Read the actual dataset schema and its correspondence between text, media, and
labels/targets. Trace decoding, preprocessing, transforms, batching/collation,
conditioning, model inputs, and loss. Random tensors check mechanics but cannot
establish correctness of the real data path.

For video tasks, verify frame sampling, frame count, spatial resizing, channel
layout, normalization, text/image conditions, and source-specific target/noise
semantics. Preserve alignment between each caption/condition and its video.
Reuse current transforms/collators when their semantics match; add only missing
behavior. For VLM/Omni metadata, follow
`.agents/knowledge/multimodal_metadata.md`.

If caching embeddings or latents, record source-example identity, encoder and
weight revision, preprocessing, dtype/shape, and cache schema. Reject incompatible
caches rather than silently using stale conditions.

### Derive configuration from the current code

Read the selected trainer's current argument classes and a working peer YAML.
Place configs in the existing model/category directories: model configs in
`configs/model_configs/` where applicable, and training YAML under
`configs/text/`, `configs/multimodal/`, or `configs/dit/`.

Set required model/component paths, dataset schema and primary text fields,
precision, seed, batch sizes, optimizer/scheduler, training steps, checkpoint
paths, and requested accelerator/parallel settings. Check parsing and field
ownership; do not copy a generic template or machine-specific paths as defaults.
Use a small validation workload before scaling the requested workload.

### Reuse the trainer/runtime lifecycle

Adapt the model to the current `TextTrainer`, `VLMTrainer`, or `DiTTrainer` path
as appropriate. Model construction, freezing/LoRA, parallelization, optimizer,
and model-bound state belong to the existing runtime abstractions; inspect
`veomni/models/model_runtime.py` and the selected trainer's runtime subclass.
Extend the appropriate hook only for a demonstrated gap.

Use the runtime-owned `ModelCheckpointManager` and existing trainer callbacks.
Do not introduce a source-framework training loop or independent checkpoint
system. Separate model state from job state such as data cursor, RNG, and step.
Initialize from external weights through the model loader; resume training
through the intended trainer/checkpoint path.

**Exit evidence:** parsed configuration, correctly collated real-format batch,
expected trainable/frozen parameters, and a working trainer entry point.

## Phase 5: Validate behavior and training-state resume

Read `.agents/knowledge/testing.md` for test placement and CI enumeration.
Prefer existing test locations and exercise the actual integration path.
Run checks in increasing cost order, omitting only those outside explicit scope.

### Import, forward, and numerical parity

Check clean-process import, config/registry lookup, output shapes, and expected
forward behavior. For generated modeling, run applicable patch tests and drift
checks required by the patchgen skill.

For ported modeling, layouts, or operations, compare source and VeOmni with
matched weights, inputs, masks, dtype, and execution mode. Supply identical
noise/timesteps for stochastic objectives; matching seeds alone may not match
random draws across frameworks. Compare component boundaries when full-model
execution is too costly, stating the narrower property proven.

Declare tolerances before running. Record compared tensors, shapes, maximum
absolute/relative errors, and the relative-error denominator convention.
Establish native correctness before evaluating a different backend kernel.
If the source cannot run, do not substitute shape checks for numerical parity.

### Real-data training

For the default training scope, verify finite loss, gradients on expected
trainable parameters, frozen-component behavior, and an actual optimizer update.
Run a short job through real preprocessing, collation, and the chosen trainer.
A forward-only smoke test does not complete training migration.

Use a controlled repeated fixture only when convergence evidence is requested.
Define its acceptance rule in advance, such as initial/final window medians;
do not require every stochastic step to decrease or claim production quality
from a tiny overfit experiment.

### Fresh-process resume

For default full migration, compare uninterrupted N+1-step training with an
N-step save followed by a new process resuming for one step. Keep workload,
seed, configuration, and topology matched. Exercise the real checkpoint path.

Check saved/restored global step, model parameter fingerprints, optimizer and
scheduler state, scaler when used, RNG, and rank-local data cursor. Compare the
next batch, next-step loss, and parameter update with declared tolerances.
Use `docs/usage/checkpoint.md` for the current checkpoint layout and interfaces.
A successful weight reload alone does not prove resume.

### Backend and completion checks

Run on the backend/topology actually claimed. CPU or single-device results do
not establish NPU/GPU kernel or multi-device correctness. Start with the smallest
feasible topology when the user did not specify one, and report that boundary.
Measure performance only when requested or affected, recording workload,
precision, topology, warmup, synchronization, throughput, and peak memory.

Run `make quality` and focused checks for the changes. Distinguish passed,
failed, blocked, and explicitly out-of-scope checks. Missing required weights or
hardware is a blocker for the affected claim, not a passing substitute test.

## Phase 6: Deliver a reproducible integration

Summarize changed areas, supported behavior, source revision, dependencies,
configuration, exact reproduction commands, and validation results. Commands
must use documented inputs and environment setup rather than hidden shell state.
Update relevant supported-model/example documentation.

Describe limitations and unfinished acceptance checks explicitly. For inference-
only work, omit training/resume requirements; for config-only work, state that
pretrained loading and numerical equivalence have not been established.

Use concise task output or existing project documentation for evidence. Create
an E2E or Ascend experience report only if requested or useful for substantial
work; include measured behavior and concrete diagnostics without a mandatory
report template. Do not add extra backends, long training, or performance work
merely to fill a checklist.

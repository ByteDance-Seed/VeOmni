---
name: veomni-new-model
description: "Integrate an existing external model into VeOmni, adapting modeling, data, training, and weight loading as needed. Use for new model support or migration from Transformers, Diffusers, or standalone implementations."
---

# Model migration into VeOmni

Adapt an existing model to the requested inference or training workflow. Keep
architecture and numerical semantics intact; implement only the requested scope.
For failures in an already supported model, use `/veomni-debug`.

## Work handled by the model

Use the language model's code-reading and implementation abilities to inspect
source, trace component boundaries, compare interfaces, plan changes, write
configs and tests, and summarize results. This skill needs no generic source
analyzer, converter scaffold, checklist, or report templates.

Read actual source and use tools to verify conclusions. Write model-specific
scripts only when conversion, data preparation, or reproducible validation
requires executable work; reuse existing repository utilities first. Derive
configs from current argument classes and a maintained peer configuration.
Create detailed plans or reports only when the task's complexity or user request
calls for them. Reasoning alone does not replace runtime or numerical evidence.

## Choose the integration route

Inspect model construction, config, forward, and the training path when relevant.
Record source revision and local modifications. Identify trainable/frozen
components, input/output shapes, loss, and checkpoint layout. Classify mixed
systems per component; a hosting URL or imported library does not establish the
implementation framework.

- **Transformers:** use `/veomni-patchgen-model` for modeling and registration in
  `veomni/models/transformers/`. Never edit generated modeling directly.
- **Diffusers:** follow a maintained integration and
  `docs/usage/support_new_models/dit_model_guide.md` for DiT interfaces.
- **Standalone or other framework:** wrap compatible PyTorch modules or port the
  required source subset. Preserve source revision, path, and license attribution.
  Keep the upstream launcher and training loop out of VeOmni's execution path.

Place DiT integrations, including independent diffusion backbones and their
conditioning components, under `veomni/models/diffusers/`. This location does not
require Diffusers base classes. Other model categories follow their nearest
VeOmni implementation.

## Connect the model

Reuse current registration, loaders, trainers, transforms, and collators. Add
extensions only for demonstrated interface gaps. For training, trace a real-format
sample through preprocessing, batching, conditioning, forward, and loss. For
VLM/Omni metadata, follow `.agents/knowledge/multimodal_metadata.md`.

Add model/training configs in the existing category-specific directories. First
establish a working native path; then add requested kernels or parallelism.
FSDP2 wrapping is generic; `parallel_plan.py` is for ExtraParallel sharding.
Keep backend-specific imports isolated.

Load external weights directly when compatible. If parameter names conflict,
show concrete keys and their loading/export/resume impact; obtain the user's
choice unless already specified. For necessary conversion, account for source
and target tensors, reject unexplained key/shape mismatches and collisions,
retain the source checkpoint, and verify strict target loading. Do not mask
incompatibility with `strict=False`.

Use the current model runtime's `ModelCheckpointManager` and trainer callbacks
for training checkpoints. Weight initialization and training-state resume are
separate capabilities.

## Verify the requested behavior

Read `.agents/knowledge/testing.md` for test placement and CI coverage. Select
checks from the actual changes and scope:

- Always check clean-process import, config/registry lookup, representative
  forward behavior, and `make quality` for code changes.
- For ported modeling or operations, compare with the source using matched
  weights, inputs, dtype, mode, and explicit noise/timesteps; declare tolerances.
- When using or converting weights, verify key/shape coverage, strict loading,
  and functional behavior. Config-only work cannot establish pretrained parity.
- For training, check real data flow, finite loss/gradients, optimizer updates,
  and a short trainer run. Add controlled overfit only when convergence evidence
  is requested.
- For requested resume or changed checkpoint handling, compare a fresh-process
  resumed next step against an uninterrupted run, including training/data state.
- Claim GPU/NPU or distributed support only after running that backend/topology.
  Measure performance only when requested or affected.

Summarize changed files, reproduction commands, supported behavior, test results,
and limitations. Update relevant model documentation. State missing prerequisites
and unverified checks explicitly; do not expand the task to fill every category.

# External implementation mapping

Read this for another framework or a standalone repository. Transformers
modeling stays with /veomni-patchgen-model; Diffusers integrations can reuse
the data, weight, or checkpoint guidance here when needed.

## Separate model semantics from framework machinery

Trace construction, forward, and the requested training behavior before porting.
Preserve architecture, defaults, masks, normalization, positional encoding,
tensor layouts, and objective. Separate launcher, framework trainer, logging,
and checkpoint utilities from reusable model code.

Prefer a thin wrapper around compatible PyTorch modules. If that cannot express
the model, port the required source subset. For a non-PyTorch source, explicitly
translate operations, parameter layouts, initialization, and dtype behavior.
Retain source URL, revision, path, and license attribution in source-derived
files. Record intentional differences and their validation.

Use current loader and model registration conventions. Registration must work
in a fresh process. Extend a loader only for a demonstrated construction gap;
do not carry the upstream training framework into the VeOmni training loop.

## Define component boundaries

Specify inputs, outputs, shapes, dtypes, and parameter ownership at each boundary.
For a DiT condition/trainable split, follow the current DiT guide and trainer:
identify encoding, noise/timestep sampling, target construction, and scalar loss
ownership. Do not impose a diffusion-specific split on other model categories.

Keep parameter namespaces stable for loading, DCP, and export. Isolate
backend-specific imports. First establish a small eager/native execution path,
then add the requested distributed and kernel adaptations.

## Weight loading and conversion, when required

Try the appropriate existing loader first. If conversion is necessary, the
weight-converter.py template supports PyTorch state dictionaries and safetensors,
name/drop rules, and per-tensor transforms. It is not a universal reader for
other frameworks. Adapt source reading for the actual format.

Its per-tensor hook supports splitting and layout changes. Concatenation across
source tensors or explicit initialization requires additional conversion logic
and accounting. Derive an expected key/shape/dtype manifest from the target
model and pass it with --expected-manifest. Without that manifest, the template
cannot verify target completeness.

Record source/destination fingerprints, source and target counts, mapping rules,
drops and initialization with reasons, dtype changes, and the strict target-load
result. The template's report is not itself a target-model load test. Perform
that load and a functional comparison separately. Write to new output/report
paths and retain the source checkpoint.

## Data and training, when in scope

Trace a real-format example through transformations and collation to the model.
A random tensor fixture can check model mechanics but cannot establish data-path
correctness. If cached conditions are introduced, record encoder/weight revision,
preprocessing, source-example identity, dtype/shape, and cache schema; reject
incompatible caches.

Use the closest current trainer and argument dataclasses. Configuration templates
are optional scaffolds, not universally valid model configs. Verify defaults and
expose data/model paths, precision, batch sizes, optimizer, seed, steps, and the
requested parallel dimensions. Establish a finite-loss forward/backward and
optimizer step before a short trainer run.

## Training checkpoints, when in scope

Use ModelCheckpointManager in `veomni/models/checkpoint_manager.py` and trainer
callbacks. Initial import of model weights does not restore optimizer, scheduler,
RNG, step, or data cursor. If resume is requested or state handling changes, use
the fresh-process comparison in the validation reference. Add a model-private
state path only when required state cannot be represented by current interfaces,
and document the gap.

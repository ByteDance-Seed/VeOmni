# Source analysis and planning

Read this for a substantial external repository, an unclear component boundary,
or a cross-framework port. Keep a concise plan for simpler integrations.

## Establish the source

Record the actual model or component entry point, repository and revision,
checkpoint revision or checksum when weights are used, and the VeOmni base.
Record relevant runtime dependencies and source/checkpoint licenses. Note local
source modifications alongside the commit; a revision alone does not pin a dirty
checkout. Keep downloaded source and weights outside the committed deliverable.

Inspect config classes, base classes, construction, forward, and the relevant
training path. A hosted checkpoint or an imported library does not establish a
model's framework. For mixed systems, list components with separate source,
framework, parameter namespace, input/output interface, and trainable/frozen role.

## Optional local inventory

From the VeOmni root:

```bash
python .agents/skills/veomni-new-model/scripts/analyze_upstream.py \
  --upstream <local-source-checkout> \
  --output .agents_workspace/migrations/<model-slug>/inventory
```

The script writes source-inventory.json and source-inventory.md. It scans local
Python AST and JSON configuration without importing source code or accessing the
network. It records imports with aliases, class bases and methods, configuration
signals, source revision, and unreadable or skipped files.

These are observations, not a framework classifier or a complete call graph.
Resolve aliases and indirect inheritance in the source. Non-Python code, dynamic
construction, and YAML configuration require manual inspection. Output files
are never overwritten. Use a new output directory for a refreshed inventory.
The script does not copy converters, training configs, or reports.

## Map the integration gaps

Record only applicable responsibilities:

| Responsibility | Questions to resolve |
|---|---|
| Model construction | Which config fields, defaults, components, and initialization are required? |
| Forward and objective | What are the shapes, dtypes, masks, outputs, and loss semantics? |
| Weights | Can existing loaders preserve names/layouts? What must be converted or excluded? |
| Data | What happens from raw example through decoding, processing, conditions, and collation? |
| Training | Which current trainer owns the step? What belongs in model versus condition/data code? |
| Checkpoint | Is this weight initialization, or must training state also resume? |
| Backend/parallelism | Which devices, kernels, and topology are actually requested? |

Choose file locations from current implementations and loaders, not from the
source framework name alone. A custom diffusion backbone may reuse DiT
conventions after inspection; an arbitrary custom model is not automatically a
Diffusers model. Transformers modeling goes through /veomni-patchgen-model.

## Make the plan reviewable

Include the chosen per-component route and evidence, source-to-VeOmni interface
mapping, proposed files with reasons, implementation sequence, and applicable
acceptance checks. Reuse direct loading when correct; add a converter only for
an actual format/layout gap. Define numerical tolerances before comparison.

For extensive work, adapt the migration checklist template. Use PASS, FAIL,
BLOCKED, or N/A with a reason. N/A means outside the agreed scope or genuinely
inapplicable; it must not hide an unavailable resource for a required check.

# Checkpoint layout

`train.checkpoint.output_dir` is the run root. Every per-step artifact lives
under `output_dir/checkpoints/global_step_{N}/` (`train.checkpoint.save_path`);
the model assets are written once to `output_dir/model_assets/`
(`output_dir/model_assets/<module>/` in a [multi-module job](#multi-module-jobs-seedomni)).

```
checkpoints/global_step_{N}/
├── checkpoint_manifest.json      # completion marker, rank 0 writes it last
├── model/[<module>/]
│   ├── ckpt/                     # weights, DCP
│   ├── optimizer/                # optimizer state, DCP
│   └── extra_state/
│       └── extra_state_rank_{R}.pt   # scheduler, condition-model RNG
├── loader/rank_{R}.pt            # dataloader / sampler cursor
├── extra_state/rank_{R}.pt       # global_step, RNG, meters
├── hf_ckpt/[<module>/]           # full-model export, not resume
└── lora_ckpt/[<module>/]         # LoRA adapter export, not resume
```

A step directory is split by *who owns the state*. `model/`, `loader/` and
`extra_state/` are the resume tree — everything needed to continue training and
nothing else. The two exports are inference artifacts that resume never reads.
`<module>/` appears only in a [multi-module job](#multi-module-jobs-seedomni).

## What each path holds

| Path | Holds | Written by |
|------|-------|------------|
| `model/**/ckpt/` | Weights. A LoRA run stores only the trainable adapter tensors; the frozen base is reloaded from `model.model_path`. | All ranks, via DCP |
| `model/**/optimizer/` | Optimizer state. | All ranks, via DCP |
| `model/**/extra_state/extra_state_rank_{R}.pt` | Model-bound extra state: `lr_scheduler.state_dict()` and whatever the runtime contributes (e.g. the DiT condition model's RNG). | Rank R |
| `loader/rank_{R}.pt` | Dataloader / sampler cursor. | Rank R |
| `extra_state/rank_{R}.pt` | `global_step`, `environ_meter`, `channel_loss_callback`, `offline_cache_writer` (SeedOmni `offline_embedding` only), `torch_rng_state`. | Rank R |
| `hf_ckpt/` | `model*.safetensors`, its index, and the model assets. Written when `train.checkpoint.save_hf_weights` is set and the run is not LoRA. | Rank 0, after a collective gather |
| `lora_ckpt/` | `adapter_config.json`, `adapter_model.safetensors`. Written instead of `hf_ckpt/` when `model.lora_config` is set. | Rank 0, after a collective gather |
| `checkpoint_manifest.json` | Format version, `global_step`, world size, and `model_state`: whether the step has a `model/` at all. Trainer-level completion only. | Rank 0, last |
| `model_assets/[<module>/]` | Each model's config / tokenizer / processor, once per run at train start. Nested like `hf_ckpt/`; a SeedOmni run loaded from an upstream HF checkpoint writes both flat, in that checkpoint's layout ([Upstream Checkpoint Layout](../seed_omni/usage/hf_layout.md)). | Rank 0 |

**Model assets** is whatever that model's runtime `model_assets` list carries, in type terms
`Union[PretrainedConfig, GenerationConfig, PreTrainedTokenizer, ProcessorMixin]`
(`veomni/models/checkpoint/weights.py`). Each is written by calling its own
`save_pretrained`, so the files that appear are whatever those objects emit —
`config.json`, `generation_config.json`, then tokenizer files, processor files,
or both. Do not read a fixed file list into these directories. Two modules
cannot share one directory: both would write `config.json`.

Three splits in the tree above are deliberate:

- **`loader/` and `extra_state/` are per rank** because their contents are.
  Iterable datasets are `split_dataset_by_node`-sharded on `dp_rank` and the
  multisource sampler filters on `dp_rank`, so restoring rank 0's cursor
  everywhere would replay rank 0's shard and skip the rest. RNG and meters are
  rank-local for the same reason. The scheduler is rank-local too: it lives in
  each rank's `model/<module>/extra_state/extra_state_rank_{R}.pt`, beside the
  runtime-contributed state, rather than in a replicated sidecar.
- **Weights and optimizer are two DCP directories** because they have different
  lifetimes. Optimizer state is roughly twice the size of the weights under Adam
  and is useless outside this run, so a checkpoint being shipped, archived or
  converted wants the weights alone. A single directory interleaves both into
  the same `__{i}_{rank}.distcp` files, which is what made them inseparable
  after the fact.
- **`hf_ckpt/` and `lora_ckpt/` are two directories** even though they are
  mutually exclusive today (`ModelCheckpointManager.save_hf_or_lora` routes on
  `model.lora_config`). That stops being true the moment LoRA merging lands: a
  merge writes the adapter *and* a merged full-model export for the same step.
  `lora_ckpt/` is a PEFT directory — point `PeftModel.from_pretrained` at it. It
  is not what a LoRA run resumes from; see
  [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving).

## Multi-module jobs (SeedOmni)

A SeedOmni job trains several modules side by side, each with its own weights,
optimizer, scheduler and accelerator config. Module names are the keys of
`model.model_config.modules`, declared in a `modules_train.yaml`, and they become
directory names verbatim. `model/`, the exports and `model_assets/` nest one
level deeper under that name; `loader/`, `extra_state/` and the manifest do
**not** — one dataloader feeds the job, so there is one cursor and one marker
per step.

```
output_dir/
├── checkpoints/global_step_{N}/
│   ├── checkpoint_manifest.json      # trainer cursor is down; does not list modules
│   ├── model/
│   │   ├── janus_siglip/             # each module directory holds the same
│   │   │   ├── ckpt/                 # things as a single-module model/
│   │   │   ├── optimizer/
│   │   │   └── extra_state/extra_state_rank_{R}.pt
│   │   └── janus_llama/
│   ├── loader/rank_{R}.pt            # one cursor for the whole job
│   ├── extra_state/rank_{R}.pt
│   └── hf_ckpt/                      # or lora_ckpt/, same nesting
│       ├── janus_siglip/
│       └── janus_llama/
└── model_assets/                     # run root, beside checkpoints/
    ├── janus_siglip/
    └── janus_llama/
```

Both cases go through the same writer and the module name is the only
difference: a single-module job passes an empty name and writes directly into
`model/`. Every path on both the save and the load side comes from
`veomni/checkpoint/layout.py`, so the two cannot disagree. On resume each module
resolves its own `model/<name>/`, which is why `train.checkpoint.load_path`
stays a single path for the whole job.

## Completion

A step is complete when two independent things are on disk, each written by
whoever owns it:

- **`checkpoint_manifest.json`** — written by rank 0 once every rank's `loader/`
  and `extra_state/` files are down. It covers the trainer-level state and
  nothing else.
- **`.metadata` in every DCP directory** — written by DCP itself, one per
  `model/<name>/ckpt/` and `model/<name>/optimizer/`, at the end of that
  directory's save. Resume discovery finds those directories under `model/`
  (`model/ckpt/` for a single model, `model/<name>/ckpt/` otherwise); the
  manifest does not list them.

Neither marker can stand in for the other. The manifest is written while the
async DCP saves may still be in flight, so it says nothing about the model; a
`.metadata` covers one directory and knows nothing about the loader, the job
cursor, or its sibling modules. Module markers are not interchangeable either —
in `configs/seed_omni/Janus/janus_1.3b/train/modules_train.yaml`, `janus_siglip`
runs DDP while `janus_llama` runs FSDP2, so each records a different topology.

Resume discovery accepts a `global_step_{N}/` directory only when both halves
are there, so a crash mid-save leaves a directory that is skipped rather than
half-loaded.

The one step with a single half is a run that saves no model state. SeedOmni's
`offline_embedding` freezes every module, so no module has a checkpoint
manager and the step has no `model/`. Its manifest records
[`model_state: false`](../../veomni/checkpoint/layout.py#L254), and the manifest
alone then marks the step complete. Without that field, the same tree would look
like a step whose shards never started. A manifest written before the field
existed counts as carrying model state.

Rewriting a step — a restarted run reaching the same step again — deletes the
markers first, so an interrupted rewrite cannot leave a stale one standing over
new, partial data. Each half clears its own, right before it writes it back:
`DistributedCheckpointer` drops the module's `.metadata` files and
`GlobalStateCallback` drops the manifest. Since completeness is the conjunction,
the step is correctly rejected throughout, including in the window where one
half has been invalidated and the other has not.

With `stage_dir` the model half waits: a staged save has overwritten nothing
until the copy to the destination, so it invalidates the destination there
instead of at the start. A staged save that fails leaves the previous checkpoint
complete and resumable. The copy empties the module's directory first rather
than writing over it file by file, so a save that drops something the previous
one wrote — weights without an optimizer, at train end — cannot leave the two
runs' state mixed in one step.

The cursor files follow the same cadences as the model state, including HF and
LoRA exports: with `save_steps=100` and an export at train end, step 150 gets
its `loader/`, `extra_state/` and manifest too, instead of being left invisible
to resume.

SeedOmni (`OmniTrainer`) writes both halves the same way: the model half through
`ModelCheckpointManager`, one `model/<name>/` per module, and the trainer half
through the shared `GlobalStateCallback`, so its steps carry the same per-rank
`loader/` and `extra_state/` files and the same manifest.

## Staged and asynchronous saves

`train.checkpoint.stage_dir` and `train.checkpoint.save_async` are two answers to
the same problem, a destination slow enough that writing to it blocks the train
loop. **They cannot be combined** — `DistributedCheckpointer.save` raises on the
pair, because an async write is still running when `save()` returns and drops the
staged copy, so it would write straight to the destination staging was meant to
avoid.

With `stage_dir` (synchronous), the whole `model/` subtree — both DCP
directories and the per-rank `extra_state/` pickles — is written to node-local
scratch and copied into the step directory afterwards, with every `.metadata`
copied last. The staging directory is never part of the checkpoint itself. The copy's collectives
run on a dedicated Gloo group rather than the training backend, bounded by
`train.checkpoint.save_timeout_seconds` (Gloo's 30-minute default when unset):
the ranks not copying wait on it for the whole copy. A multi-node staged save
therefore needs working Gloo networking (for example `GLOO_SOCKET_IFNAME`), and a
misconfiguration only shows at the first checkpoint.

With `save_async` (unstaged), `model/` is written in place and the call returns
while the DCP write is still in flight. The per-rank `extra_state/` pickles are
written first, before either DCP save starts. Weights and optimizer are two
independent saves, each holding its own future and its own Gloo process group
(also bounded by `save_timeout_seconds`) so they overlap rather than serialise.
Nothing downstream waits for them: the cursor files and the manifest claim only
what they cover, and the shards answer for themselves through the `.metadata`
DCP writes last. Pending saves are drained at
the next save of the same kind and at train end; a drain that finds a failed
save raises on every rank, not just the one that saw it, agreeing on that
save's own Gloo group so the ranks whose write finished first wait there
rather than on the training backend.

## Older checkpoints do not resume

Only the layout above is read. A step written by an older VeOmni — one whose
root was the DCP directory, whose cursor lived in `extra_state_rank_{R}.pt` or
`trainer_state_rank_{R}.pt`, or whose module directories were nested one level
too shallow — carries no `checkpoint_manifest.json` where this layout puts one,
so `load_path: auto` skips it. An explicit `train.checkpoint.load_path` at such
a step still attempts the load, and `DistributedCheckpointer.load` raises a
`FileNotFoundError` naming the missing `model/<module>/ckpt/.metadata`, pointing
here. The same error is what an interrupted save looks like: a directory that
was begun and never finished is indistinguishable from an old one on disk, so
both are reported the same way. Nothing translates either.

The compatibility loader that handled VeOmni 0.1.12 and 0.2.x
(`veomni/checkpoint/legacy_v0_1_12.py`) has been removed. To keep training from
such a run, restart it, or convert the weights out first with
[Checkpoint conversion](checkpoint_conversion.md) and resume from the converted
export as a fresh run.

## Versioned ExtraParallel layout

The separately namespaced EP-outer layout is opt-in through
`SpecInfo.checkpoint_layout_v2`. A plan enabling it must retain the original
`global_shape` and must use FSDP DTensor parameters. Every
non-persistent ExtraParallel shard in a checkpoint must use the canonical
layout: mixing legacy and canonical shards is rejected. For tensors whose
source layout has been proved, the v2 namespace represents canonical global
values independently of source geometry, so DCP can reshard a model checkpoint
across supported FSDP shard axes. Use
`DistributedCheckpointer` for save and resume: it rejects old/unversioned
layouts before copying model tensors. Migrating a legacy checkpoint requires
verified source layout information. Canonical v2 optimizer save and resume are
rejected until an owner-aware codec can preserve rank-specific optimizer
states; this opt-in therefore does not yet support resuming training. The
standard VeOmni HF export strips the namespace.

The namespace marker alone does not prove that an ordinary or shared expert's
replicas were synchronized. Older v2 checkpoints whose planner metadata omits
an expert's source layout are rejected on resume and export; verify and migrate
them from their original training layout before use.

The standard, non-v2 writer uses canonical EP-first coordinates for same-axis
EP/FSDP shards and records source expert layouts in DCP planner metadata.
Older readers may ignore that metadata and interpret these shards using the old
FSDP-first order; do not use them for these checkpoints without a verified
migration. Conversely, an old unversioned checkpoint with expert-only chunks
may not prove its source order even when resumed with the same topology. The
distributed reader rejects ambiguous sources for EP/FSDP as well as ordinary
FSDP or unsharded targets. An old expert-only FSDP checkpoint without layout
metadata also needs verified source-layout migration, since it cannot be
distinguished safely from an old EP checkpoint.
Offline export also rejects unversioned expert-only chunks for stacked tensors
under an `experts` module when their source order cannot be established. Other
legacy tensor naming schemes may still need explicit source-layout verification
or migration before export.

New planner metadata records supported named ExtraParallel/FSDP meshes rather
than only `ep`. Ordinary FSDP expert shards are marked canonical only when the
actual TP1 runtime proves that every ExtraParallel dimension has size one, or
the model's applied plan and parameter aliases prove that the expert is not
sharded by any active ExtraParallel dimension.
An expert on the EP-complementary FSDP mesh is accepted only for trainable
shared-expert LoRA A/B parameters whose wrapper installs cross-EP gradient
synchronization; other parameters on that mesh remain unproven. A normal
single-process save assumes the caller supplied the complete model, not a
manually extracted expert shard. The singleton case is not a source-layout
proof for external, manually sharded tensors.
For a plain model tensor sharded on an ExtraParallel axis and replicated across
its FSDP/HSDP axes, all replicas at the same ExtraParallel coordinate must have
identical values; checkpoint serialization does not verify or synchronize them.
Plain optimizer shards on multiple such ranks require a rank-local state codec.
Older version-1 metadata may omit non-`ep` shards: an unrecorded expert tensor
is rejected unless its source layout can be verified and migrated. This also
rejects an indistinguishable old ordinary-FSDP expert tensor. Historical non-expert tensors
with unrecorded ExtraParallel axes require separate source-layout verification
before loading into a different mesh or offline export; their provenance cannot
be inferred from chunks alone, and not all such cases can be rejected
automatically when the target has no ExtraParallel mesh.

There is currently no general, verified migration utility for these ambiguous
legacy expert layouts. Preserve the original checkpoint and its training code,
rank-to-expert mapping, mesh, and optimizer configuration. A model-specific
migration needs to reconstruct both the global expert tensors and their matching
optimizer states from that provenance, compare every expert against an
independent reference, and save to a new location with the current writer. A
fresh-process restore and next-update comparison are required before resuming.
Without enough provenance to establish the mapping, do not resume or export the
ambiguous state as if it were canonical; start from independently verified
weights and a fresh optimizer instead. This describes necessary checks, not a
provided or generally validated migration procedure.

For ordinary TP/FSDP-only models, the writer may certify a strided same-axis
layout when the active runtime mesh and a unique source-model parameter owner
prove its provenance. Parameters with unknown ownership, unexpected meshes,
partial placements, or any active ExtraParallel dimension remain unproven.

All ranks in a checkpoint operation must use the same logical model and
checkpoint configuration, including the layout version and corresponding
sharding specifications. Normal rank-local shard contents and mesh coordinates
may differ. When any rank enables the canonical v2 layout, a preflight checks
that every rank agrees on the logical layout and optimizer policy before any
checkpoint is modified. Legacy-only configurations still rely on this agreement
as an input contract. Rank-local layout validation errors are propagated to peers.

Raw `torch.distributed.checkpoint.load(..., planner=DefaultLoadPlanner(allow_partial_load=True))` is not a
supported migration API: it can return success without reading any tensor when
the namespaces differ. Replicated values must already agree across ranks.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export under `lora_ckpt/`.

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
│   └── lr_scheduler.pt           # replicated pickle
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
| `model/**/lr_scheduler.pt` | `lr_scheduler.state_dict()`. | Rank 0 |
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
  rank-local for the same reason. The scheduler is replicated instead: rank 0
  writes it and every rank reads that one file.
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
│   │   ├── janus_siglip/             # each module directory holds the same three
│   │   │   ├── ckpt/                 # things as a single-module model/
│   │   │   ├── optimizer/
│   │   │   └── lr_scheduler.pt
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
[`model_state: false`](../../veomni/checkpoint/layout.py#L259), and the manifest
alone then marks the step complete. Without that field, the same tree would look
like a step whose shards never started. A manifest written before the field
existed counts as carrying model state.

Rewriting a step — a restarted run reaching the same step again — deletes the
markers first, so an interrupted rewrite cannot leave a stale one standing over
new, partial data. Each half clears its own, right before it writes it back:
`DistributedCheckpointer` drops the module's `.metadata` files (including the
one at the step root that an older VeOmni's fused save left, which would
otherwise keep the step discoverable through the legacy fallback below), and
`GlobalStateCallback` drops the manifest. Since completeness is the conjunction,
the step is correctly rejected throughout, including in the window where one
half has been invalidated and the other has not.

For a `MultiOptimizer`, a completed checkpoint is not overwritten in place:
the save fails before changing its manifest or shards. When rewinding to an
earlier checkpoint, write the new run to a different output directory to avoid
colliding with checkpoints from the previous run. A same-step export can
reuse an existing checkpoint only when its model, optimizer and trainer state
have been proven to belong to the same completed save.
Trainer sidecars record the generation of each `MultiOptimizer` checkpoint.
An absent sidecar, a missing generation, or a trainer step/generation that
differs from the restored optimizer stops resume before training. An older
unversioned sidecar requires a verified migration or a fresh weights-only run;
it is not silently combined with a restored optimizer and scheduler. Matching
step and generation alone do not prove exact dataloader or RNG recovery.
If a restored run has not established a same-step save receipt, an immediate
export to the already completed step fails rather than overwriting that DCP.

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

Checkpoints from older layouts have no manifest; discovery falls back to a
`.metadata` at the step root, which is what marked those steps complete, but
only while the step is still entirely pre-split. See
[Resuming older checkpoints](#resuming-older-checkpoints).

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
directories and `lr_scheduler.pt` — is written to node-local scratch and copied
into the step directory afterwards, with every `.metadata` copied last. The
staging directory is never part of the checkpoint itself. The copy's collectives
run on a dedicated Gloo group rather than the training backend, bounded by
`train.checkpoint.save_timeout_seconds` (Gloo's 30-minute default when unset):
the ranks not copying wait on it for the whole copy. A multi-node staged save
therefore needs working Gloo networking (for example `GLOO_SOCKET_IFNAME`), and a
misconfiguration only shows at the first checkpoint.

With `save_async` (unstaged), `model/` is written in place and the call returns
while the DCP write is still in flight. `lr_scheduler.pt` is written first, by
rank 0, before either DCP save starts. Weights and optimizer are two independent
saves, each holding its own future and its own Gloo process group (also bounded
by `save_timeout_seconds`) so they overlap rather than serialise. Nothing downstream waits for them: the cursor
files and the manifest claim only what they cover, and the shards answer for
themselves through the `.metadata` DCP writes last. Pending saves are drained at
the next save of the same kind and at train end; a drain that finds a failed
save raises on every rank, not just the one that saw it, agreeing on that
save's own Gloo group so the ranks whose write finished first wait there
rather than on the training backend.

## Resuming older checkpoints

Temporary. New saves never produce the older shapes, and the compatibility
loader in `veomni/checkpoint/legacy_v0_1_12.py` will be removed. Current files
always win — a fallback is consulted only when the current path is absent.

| Missing in the current layout | Falls back to |
|-------------------------------|---------------|
| `checkpoint_manifest.json` (discovery, `load_path: auto`) | `.metadata` at the step root, which is what marked an older step complete — unless the step also has a `model/`, which means a current-layout rewrite landed on it and did not finish |
| `model/ckpt/.metadata` | `.metadata` at the step root; weights and optimizer are both read from that fused directory |
| `model/lr_scheduler.pt` | `lr_scheduler.pt` at the step root, then the 0.1.12 pickle's `lr_scheduler` key |
| `loader/rank_{R}.pt`, `extra_state/rank_{R}.pt` | `trainer_state_rank_{R}.pt`, then the 0.1.12 pickle's job cursor when `global_step` is present |

What those older shapes looked like:

| Era | Step directory |
|-----|----------------|
| 0.1.12 | Step root *was* the DCP directory (`.metadata` + fused `__{i}_{rank}.distcp`); `extra_state/extra_state_rank_{R}.pt` mixed the scheduler and the job cursor; the LoRA adapter went to `<output_dir>/global_step_{N}/`, a sibling of `checkpoints/` |
| 0.2.x flat | The same, with the cursor split out into `trainer_state_rank_{R}.pt` and `extra_state_rank_{R}.pt` left holding the scheduler alone |
| SeedOmni modules | Nested per module, but one level too shallow: `global_step_{N}/<module>/` was the DCP directory *and* the HF export directory, so safetensors landed on top of the shards |

The same fallbacks resolve a SeedOmni module checkpoint, because they are applied
*within* a module's directory: a module whose `model/<name>/` is absent falls
back to `<name>/` at the step root.

`extra_state/` therefore means something different in each era. The two are told
apart by structure, not by that name: a current checkpoint has `model/` and
`loader/` siblings and a manifest, and the file inside is `rank_{R}.pt`, not
`extra_state_rank_{R}.pt`.

To drop legacy resume, delete `veomni/checkpoint/legacy_v0_1_12.py` and the
imports that load it (search for `legacy_v0_1_12`). An old checkpoint then fails
resume instead of falling back.

## Versioned ExtraParallel layout

For SeedOmni, `extra_parallel_canonical_layout: true` is applied after each
module-scope wrap. Combined `fsdp_scope: model` and active ExtraParallel is not
supported with this option and fails before wrapping; use module scope. All
ranks participating in a module or model scope must use the same layout flag.

The versioned EP-outer tensor namespace is opt-in through
`SpecInfo.checkpoint_layout_v2`. Ordinary ExtraParallel saves also use canonical
EP-first shard coordinates and record per-tensor source layout. A versioned
plan must retain the original `global_shape` and use FSDP DTensor parameters. Every
non-persistent ExtraParallel shard in a checkpoint must use the canonical
layout: mixing legacy and canonical shards is rejected. The v2 tensor namespace
identifies canonical global values independently of source geometry, so DCP can
reshard a model checkpoint across supported FSDP shard axes. Use
`DistributedCheckpointer` for save and resume: it rejects ambiguous unversioned
expert layouts before copying model tensors unless the model-only source is
explicitly declared as described below. Migrating a legacy optimizer checkpoint
requires verified source layout information. Canonical v2 optimizer save and resume use
the rank-local codec described below. The standard VeOmni HF export strips the
namespace.

New TP1/EP1 checkpoints also record no-EP provenance for expert tensors on the
standard VeOmni FSDP mesh. Model and optimizer restoration can use this source
record without a legacy declaration. The record is not inferred for custom
meshes, tensor parallelism, active EP, or older checkpoints missing provenance;
those cases retain the existing source-layout checks.

### Known legacy VeOmni model-only sources

An unversioned checkpoint does not record its producing EP/FSDP geometry. If
the caller has independently verified that **every expert tensor** came from
the legacy standard VeOmni TP1 implementation with expert axis 0 and FSDP
axis 1, they may explicitly declare `legacy_expert_source_layout="veomni_tp1_ep0_fsdp1"`
on `DistributedCheckpointer.load` or `dcp_to_torch_state_dict`. All ranks must
provide the same declaration. A legacy VeOmni TP1 source that used no expert
parallelism at all can instead be declared as `"veomni_tp1_no_ep"`. A missing
declaration keeps ambiguous sources
fail-closed. This is a trusted assertion about the source, not a conclusion drawn
from target geometry or the DCP chunks. In particular, do not declare this for
same-axis Muon/EP checkpoints or external TP-produced shards. An incorrect
assertion can silently permute experts. The declaration supports model-only
restore/export and is rejected when restoring an optimizer; it does not migrate
optimizer groups, moments, master weights or scheduler state. It does not load
directly into the canonical v2 namespace: export the verified legacy weights
and load them as model weights into the v2 target instead.
This also applies to old non-EP optimizer checkpoints: an unversioned,
FSDP-sharded expert bias or moment without source provenance can be rejected.
Neither source declaration authorizes optimizer-state migration.
When per-tensor source layout metadata is present, a recorded noncanonical
expert is always rejected. Missing metadata for one expert is not inferred
from the presence of metadata for another expert.

For CLI export, pass `--legacy-expert-source-layout veomni_tp1_ep0_fsdp1` to
`scripts/merge_dcp_to_hf.py` only after verifying the source. Use
`--save-dtype preserve` when the checkpoint contains FP32 tensors that must not
be rounded to BF16; the CLI retains its historical BF16 export default. This
preserves tensor dtypes on disk; consumers must configure their loader to avoid
subsequently casting the FP32 tensors.

All ranks in a checkpoint operation must use the same logical model and
checkpoint configuration, including the layout version and corresponding
sharding specifications. Normal rank-local shard contents and mesh coordinates
may differ. Cross-rank agreement is an input contract, not a runtime preflight;
divergent configurations are unsupported and can hang or produce an invalid
checkpoint. Rank-local layout validation errors are still propagated to peers.

Raw `torch.distributed.checkpoint.load(..., planner=DefaultLoadPlanner(allow_partial_load=True))` is not a
supported migration API: it can return success without reading any tensor when
the namespaces differ. Replicated values must already agree across ranks.
## MultiOptimizer recovery identity

**Breaking change relative to earlier VeOmni checkpoint behavior:**

- MultiOptimizer DCP save now requires `save_async: false`. Existing async
  MultiOptimizer jobs must switch to synchronous save before adopting this
  codec; model-only and ordinary-optimizer async paths remain separate.
- A MultiOptimizer checkpoint can only resume with the recorded owner, mesh,
  world size and optimizer layout. It cannot be restored across an EP/FSDP
  shard-axis or world-size change, even when model weights can be resharded.
- Older MultiOptimizer checkpoints whose parameter-group and scheduler identity
  cannot be proven are rejected. The repository does not yet provide a
  validated general migration for these checkpoints. Continue with the
  original compatible runtime or use an explicitly accepted weights-only
  restart with new optimizer/scheduler state; that is not an exact continuation.

Supported CPU optimizer offload adapters are independent of asynchronous DCP
save and remain available with synchronous checkpointing. A future migration must preserve
owner-specific master weights, moments and step counters as well as optimizer
groups and scheduler state, and verify the next update against the source.

Direct `DistributedCheckpointer` calls containing a `MultiOptimizer` must include
`"_veomni_scheduler_object"` in the state, set to the supported live scheduler
or explicitly to `None` for runs without a scheduler. This declaration is a
trusted contract: a caller must not declare `None` when a scheduler is in use.
The framework checkpoint manager supplies the declaration and restores the
scheduler automatically. A direct `DistributedCheckpointer.load` caller must
apply the returned scheduler state to the live scheduler before continuing:

```python
state = {"model": model, "optimizer": optimizer, "extra_state": {},
         "_veomni_scheduler_object": scheduler}
DistributedCheckpointer.load(path, state)
scheduler.load_state_dict(state["extra_state"]["lr_scheduler"])
```

Unregistered schedulers are not supported by the proven resume codec.

Use `DistributedCheckpointer` for training checkpoints containing a
`MultiOptimizer`. The save is synchronous and requires a completed optimizer
step with no pending gradients. Setting `save_async: true` for a runtime whose
checkpoint uses `MultiOptimizer` is rejected before training begins. Its
identity records parameter ownership,
group structure, scheduler recipe and checkpoint generation. Loading rejects an
unversioned checkpoint, a changed optimizer layout, or a model/optimizer/sidecar
from different saves. An old checkpoint whose original group and scheduler
identity cannot be established needs a verified offline migration.

The generic codec supports AdamW, AnyPrecisionAdamW and DistributedMuon
without ExtraParallel when every checkpoint rank has the same logical parameter
inventory and replicated values already agree. For the ExtraParallel candidate,
optimizer state is stored under rank-specific keys and recovery requires the
identical rank-to-mesh mapping. This avoids merging distinct expert-owner steps
and uninitialized states, but increases checkpoint storage for replicated
optimizer shards and does not support elastic resharding. All ranks must use
the same trusted model and configuration, including optimizer leaf ordering.
Owner-specific freezing that removes a whole leaf from only some ranks is not
supported or checked and can cause checkpoint collectives to hang. Replica
values, trainability, dtype, shape and geometry
are trusted to be consistent across ranks. Source and target checkpoint layout
metadata are checked during restore; tensor bytes are not hashed for replica
consistency. Trainable-only MultiOptimizer checkpoints do hash each rank's local
frozen base parameters on save and load to verify the base identity. That may copy
device tensors to CPU and can add substantial time for a large frozen model.
Hooks, adapters and scheduler method wrappers are trusted: a wrapper must
preserve serializable scheduler state and implement the same learning-rate
behavior on both sides of a restore, and hooks must not mutate checkpoint
model, optimizer, or layout during a save/load call. A hook may transform the
serialized representation within the codec's supported inventory, dtypes, and
shapes, but exact continuation requires the restored optimizer state,
parameter-group bindings, and scheduler behavior to match; when necessary, a
matching load hook must undo the transformation. The persisted recipe does not
verify arbitrary hook or
wrapper code. MultiOptimizer asynchronous saves and detected unsupported layouts fail
closed; the cross-rank leaf mismatch above is not detected. Synchronous staged
saves are supported.
The runtime assumes ranks use a consistent model configuration and compatible
replica geometries; saved source layout is validated on restore.
Group/state inventory and generation mismatches are rejected before model
copying. If an I/O or optimizer-application error occurs after model copying,
the load fails and clears its receipt but does not roll back in-memory objects;
discard and rebuild that runtime before retrying.
`MultiOptimizer.state_dict()` returns a detached inspection snapshot for
supported leaves, but `load_state_dict()` alone cannot prove group/scheduler
identity and rejects restoration; use the proven DCP path instead.
The codec rejects tensor learning rates shared between groups. The saved alias
identity covers parameters and buffers, including nonpersistent buffer aliases.
Optimizer adapters and model serialization hooks are trusted to preserve their
state and semantics. Identity tracks repeated tensor objects, but does not detect
distinct views sharing storage; models that rely on mutable shared views are
outside this checkpoint contract. A reset dynamic RoPE cache can have a
different alias graph from a newly constructed model and is conservatively
rejected before writing until that cache state can be restored safely.
Dynamic RoPE frequencies and cached maximum lengths are not checkpointed. If
the cache grew beyond its original context and has not reset, a fresh process
can compute different later outputs. MultiOptimizer checkpoints therefore
reject a non-reconstructible dynamic or longrope cache at save. With a dynamic
or longrope target, resume requires a matching cache contract; older checkpoints
without that proof need a verified migration. Older checkpoints do not identify
the source's RoPE configuration, so the operator must also independently verify
that the source and target configurations match. Saving an expanded cache is not
yet supported. DCP training with MultiOptimizer rejects dynamic and longrope
modes before training until their transient history can be restored exactly;
ordinary static/default RoPE is unaffected.

For unversioned expert checkpoints whose chunks divide the expert axis, the
saved offsets alone do not prove source order, even if another axis is also
sharded. Such known expert tensors require a verified migration with the
original training layout. Offline export can recognize VeOmni's standard
`experts` names; custom names require the operator to verify source layout.

Mutable optimizer options must not be shared between groups or optimizer
defaults: ordinary YAML `betas` are normalized to a tuple; custom shared lists,
dictionaries and tensor-valued learning rates require explicitly independent
copies. This avoids changing aliasing semantics on resume.
The model implementation and training program must also be the same trusted
version on save and resume; custom `state_dict`/`load_state_dict` overrides or
other user code can change training semantics outside this identity check.
Do not use raw DCP `OptimizerState` or `MultiOptimizer.load_state_dict` to
resume: they cannot establish scheduler and group provenance. The standalone
`MultiOptimizer.state_dict()` returns a snapshot and copies tensor states on
their current devices, which can temporarily require as much extra device
memory as the optimizer tensors themselves. That raw snapshot supports the
built-in AdamW, AnyPrecisionAdamW and DistributedMuon leaves, as well as
registered adapters. Custom serialization hooks and adapters must preserve state faithfully.
It does not prove the identity needed for raw restoration. Use the high-level
`DistributedCheckpointer` for save and restore; its save path avoids this copy.

For the base and DPO policy ExtraParallel runtimes, a versioned checkpoint's rank inventory is
used to temporarily include its previously owned frozen parameters while building
the optimizer. Their frozen flags are restored before the first forward and the
complete source group and scheduler identity is still checked on load. Frozen
parameters can be intentionally unfrozen after recovery. Unproven legacy
checkpoints and changed freeze configurations require verified migration.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Rank-local AdamW checkpoint adapters](../design/rank_local_adamw_checkpoint_adapter.md) — registration, ownership and save/load contract.
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export under `lora_ckpt/`.

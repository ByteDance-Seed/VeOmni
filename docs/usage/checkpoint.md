# Checkpoint layout

`train.checkpoint.output_dir` is the run root. Every per-step artifact lives
under `output_dir/checkpoints/global_step_{N}/` (`train.checkpoint.save_path`);
the model assets are written once to `output_dir/model_assets/`.

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
`<module>/` appears only in a [multi-module job](#multi-module-jobs-seedomni-v2).

## What each path holds

| Path | Holds | Written by |
|------|-------|------------|
| `model/**/ckpt/` | Weights. A LoRA run stores only the trainable adapter tensors; the frozen base is reloaded from `model.model_path`. | All ranks, via DCP |
| `model/**/optimizer/` | Optimizer state. | All ranks, via DCP |
| `model/**/lr_scheduler.pt` | `lr_scheduler.state_dict()`. | Rank 0 |
| `loader/rank_{R}.pt` | Dataloader / sampler cursor. | Rank R |
| `extra_state/rank_{R}.pt` | `global_step`, `environ_meter`, `channel_loss_callback`, `torch_rng_state`. | Rank R |
| `hf_ckpt/` | `model*.safetensors`, its index, and the model assets. Written when `train.checkpoint.save_hf_weights` is set and the run is not LoRA. | Rank 0, after a collective gather |
| `lora_ckpt/` | `adapter_config.json`, `adapter_model.safetensors`. Written instead of `hf_ckpt/` when `model.lora_config` is set. | Rank 0, after a collective gather |
| `checkpoint_manifest.json` | Format version, `global_step`, world size, module names. | Rank 0, last |
| `model_assets/` | The model assets again, once per run at train start. | Rank 0 |

**Model assets** is whatever `trainer.model_assets` carries, in type terms
`Union[PretrainedConfig, GenerationConfig, PreTrainedTokenizer, ProcessorMixin]`
(`veomni/models/module_utils.py`). Each is written by calling its own
`save_pretrained`, so the files that appear are whatever those objects emit —
`config.json`, `generation_config.json`, then tokenizer files, processor files,
or both. Do not read a fixed file list into these directories.

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

## Multi-module jobs (SeedOmni V2)

A V2 job trains several modules side by side, each with its own weights,
optimizer, scheduler and accelerator config. Module names are the keys of
`model.model_config.modules`, declared in a `modules_train.yaml`, and they become
directory names verbatim. `model/` and the exports nest one level deeper under
that name; `loader/`, `extra_state/` and the manifest do **not** — one dataloader
feeds the job, so there is one cursor and one marker per step.

```
checkpoints/global_step_{N}/
├── checkpoint_manifest.json      # lists every module below
├── model/
│   ├── janus_siglip/             # each module directory holds the same three
│   │   ├── ckpt/                 # things as a single-module model/
│   │   ├── optimizer/
│   │   └── lr_scheduler.pt
│   └── janus_llama/
├── loader/rank_{R}.pt            # one cursor for the whole job
├── extra_state/rank_{R}.pt
└── hf_ckpt/                      # or lora_ckpt/, same nesting
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
  nothing else; it also names the modules the job saved, so a reader knows which
  markers to look for without walking the tree.
- **`.metadata` in every DCP directory** — written by DCP itself, one per
  `model/<name>/ckpt/` and `model/<name>/optimizer/`, at the end of that
  directory's save.

Neither marker can stand in for the other. The manifest is written while the
async DCP saves may still be in flight, so it says nothing about the model; a
`.metadata` covers one directory and knows nothing about the loader, the job
cursor, or its sibling modules. Module markers are not interchangeable either —
in `configs/seed_omni/Janus/janus_1.3b/train/modules_train.yaml`, `janus_siglip`
runs DDP while `janus_llama` runs FSDP2, so each records a different topology.

Resume discovery accepts a `global_step_{N}/` directory only when both halves
are there, so a crash mid-save leaves a directory that is skipped rather than
half-loaded.

Rewriting a step — a restarted run reaching the same step again — deletes the
markers first, so an interrupted rewrite cannot leave a stale one standing over
new, partial data. Each half clears its own, right before it writes it back:
`DistributedCheckpointer` drops the module's `.metadata` files (including the
one at the step root that an older VeOmni's fused save left, which would
otherwise keep the step discoverable through the legacy fallback below), and
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

Checkpoints from older layouts have no manifest; discovery falls back to a
`.metadata` at the step root, which is what marked those steps complete, but
only while the step is still entirely pre-split. See
[Resuming older checkpoints](#resuming-older-checkpoints).

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
staging directory is never part of the checkpoint itself.

With `save_async` (unstaged), `model/` is written in place and the call returns
while the DCP write is still in flight. `lr_scheduler.pt` is written first, by
rank 0, before either DCP save starts. Weights and optimizer are two independent
saves, each holding its own future and its own Gloo process group so they
overlap rather than serialise. Nothing downstream waits for them: the cursor
files and the manifest claim only what they cover, and the shards answer for
themselves through the `.metadata` DCP writes last. Pending saves are drained at
the next save of the same kind and at train end; a drain that finds a failed
save raises on every rank, not just the one that saw it.

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
| SeedOmni V2 modules | Nested per module, but one level too shallow: `global_step_{N}/<module>/` was the DCP directory *and* the HF export directory, so safetensors landed on top of the shards |

The same fallbacks resolve a V2 module checkpoint, because they are applied
*within* a module's directory: a module whose `model/<name>/` is absent falls
back to `<name>/` at the step root.

`extra_state/` therefore means something different in each era. The two are told
apart by structure, not by that name: a current checkpoint has `model/` and
`loader/` siblings and a manifest, and the file inside is `rank_{R}.pt`, not
`extra_state_rank_{R}.pt`.

To drop legacy resume, delete `veomni/checkpoint/legacy_v0_1_12.py` and the
imports that load it (search for `legacy_v0_1_12`). An old checkpoint then fails
resume instead of falling back.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export under `lora_ckpt/`.

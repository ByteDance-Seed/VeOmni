# Checkpoint layout

This page is the on-disk contract for a VeOmni training run: which directories
are written, what each file holds, and how that differs from earlier layouts.
New writes use **this** layout. Resume also reads the two older layouts through
a compatibility loader (see [Legacy resume](#legacy-resume)).

`train.checkpoint.output_dir` is the run root. Every per-step artifact — DCP
shards, scheduler, dataloader cursor, job state, HuggingFace export, and LoRA
adapter — lives under `output_dir/checkpoints/global_step_{N}/`
(`train.checkpoint.save_path`). The model assets — configs plus whichever of a
tokenizer and a processor the model has — are written once to `model_assets/`.

A step directory is split by *who owns the state*. The first three are the
resume tree — everything needed to continue training, and nothing else. The last
two are inference exports, which resume never reads:

| Directory | Owns | Sharded by |
|-----------|------|------------|
| `model/ckpt/` | Weights | DCP (collective) |
| `model/optimizer/` | Optimizer state | DCP (collective) |
| `model/lr_scheduler.pt` | Scheduler state | Replicated |
| `loader/` | Dataloader / sampler cursor | Rank |
| `extra_state/` | Step counter, RNG, meters | Rank |
| `hf_ckpt/` | Full-model safetensors export | Not sharded |
| `lora_ckpt/` | LoRA adapter export | Not sharded |

Plus one file: `checkpoint_manifest.json`, the completion marker.

`hf_ckpt/` and `lora_ckpt/` are normally mutually exclusive — a run either
trains a full model or trains an adapter, and `save_hf_or_lora` picks one. They
are separate directories rather than one because that stops being true the
moment LoRA merging lands: a merge writes the adapter *and* a merged full-model
export for the same step, and those are two different artifacts that a reader
must be able to tell apart by path.

## Current layout

```
<output_dir>/                              # train.checkpoint.output_dir
├── checkpoints/                           # train.checkpoint.save_path
│   └── global_step_{N}/
│       ├── checkpoint_manifest.json       # Completion marker, written last by
│       │                                  # rank 0 after every directory below
│       │                                  # is on disk. Resume considers a step
│       │                                  # only when this file is present.
│       │                                  # Records format version, world size,
│       │                                  # module names, and global_step.
│       ├── model/                         # Model-bound state. Two separate DCP
│       │   │                              # directories plus one sidecar.
│       │   ├── ckpt/                      #   Weights. A LoRA run stores only
│       │   │   ├── .metadata              #   trainable adapter tensors here; the
│       │   │   └── __{i}_{rank}.distcp    #   frozen base is reloaded from
│       │   │                              #   model.model_path.
│       │   ├── optimizer/                 #   Optimizer state, its own DCP
│       │   │   ├── .metadata              #   directory so it can be shipped or
│       │   │   └── __{i}_{rank}.distcp    #   dropped without touching weights.
│       │   └── lr_scheduler.pt            #   One pickle: lr_scheduler.state_dict().
│       │                                  #   Replicated; rank 0 writes it and
│       │                                  #   every rank reads this same file.
│       ├── loader/
│       │   └── rank_{R}.pt                # Dataloader / sampler cursor. Rank R
│       │                                  # writes and rank R restores.
│       ├── extra_state/
│       │   └── rank_{R}.pt                # Rank R's remaining job state:
│       │                                  #   global_step
│       │                                  #   environ_meter
│       │                                  #   channel_loss_callback
│       │                                  #   torch_rng_state
│       ├── hf_ckpt/                       # Full-model export, not resume.
│       │   ├── model*.safetensors         #   Written when
│       │   ├── model.safetensors.index.json  # train.checkpoint.save_hf_weights
│       │   └── <model assets>             #   is set and the run is not LoRA.
│       │                                  #   Not a fixed file list: the assets
│       │                                  #   are whatever trainer.model_assets
│       │                                  #   holds, each dumped with
│       │                                  #   save_pretrained. See below.
│       └── lora_ckpt/                     # Adapter export, not resume. Written
│           ├── adapter_config.json        #   instead of hf_ckpt/ when
│           └── adapter_model.safetensors  #   model.lora_config is set.
└── model_assets/                          # The same model assets again, written
                                           # once at train start (rank 0).
```

**Model assets** is the set `trainer.model_assets` carries — in type terms
`Union[PretrainedConfig, GenerationConfig, PreTrainedTokenizer, ProcessorMixin]`
(`veomni/models/module_utils.py`). Each one is written by calling its own
`save_pretrained`, so the files that appear are whatever those objects emit:
`config.json` and `generation_config.json` from the configs, and then tokenizer
files, processor files, or both — a text model carries a tokenizer, a VLM
carries a processor that wraps one. VeOmni does not enumerate or rename them, so
do not read a fixed file list into this directory.

Weights and optimizer are two DCP directories, not one, because they have
different lifetimes. Optimizer state is roughly twice the size of the weights
under Adam and is useless outside this run, so a checkpoint that is being
shipped, archived or converted wants the weights alone. With a single DCP
directory both live in the same `__{i}_{rank}.distcp` files and cannot be
separated after the fact. Note this is about moving *files*: reading only the
weights was always possible by omitting `optimizer` from the load state, since
`dcp.load` reads only the keys it is asked for.

Only one of `hf_ckpt/` and `lora_ckpt/` exists in a given step today;
`ModelCheckpointManager.save_hf_or_lora` routes on `model.lora_config`.

`lora_ckpt/` is a PEFT-format directory: point `PeftModel.from_pretrained` at it
directly. It is not what a LoRA run resumes from — the adapter's *training*
state (trainable tensors plus optimizer) is in `model/`, and the frozen base
comes from `model.model_path`. See
[LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving).

### Multi-module jobs (SeedOmni V2)

A V2 job trains several modules side by side, each with its own weights,
optimizer, scheduler — and its own accelerator config. Module names are the keys
of `model.model_config.modules`, declared in a `modules_train.yaml`, and they
become directory names verbatim:

```yaml
# configs/seed_omni/Qwen/qwen3vl_2b/packed/modules_train.yaml
qwen3vl_vision: {...}
qwen3vl_text_encoder: {...}
qwen3vl_llm: {...}
```

`model/` and the export directories nest one level deeper under that name.
`loader/`, `extra_state/` and the manifest do **not** nest: one dataloader feeds
the job, so there is one cursor and one completion marker per step, not one per
module.

```
checkpoints/global_step_{N}/
├── checkpoint_manifest.json               # Lists every module below.
├── model/
│   ├── janus_siglip/                      # Each module directory holds the same
│   │   ├── ckpt/                          # three things as a single-module
│   │   ├── optimizer/                     # model/: weights, optimizer, and a
│   │   └── lr_scheduler.pt                # replicated scheduler.
│   ├── janus_vqvae/
│   ├── janus_text_encoder/
│   └── janus_llama/
├── loader/
│   └── rank_{R}.pt                        # One cursor for the whole job.
├── extra_state/
│   └── rank_{R}.pt
└── hf_ckpt/                               # Or lora_ckpt/, same nesting.
    ├── janus_siglip/                      # Per-module export. This is what
    ├── janus_vqvae/                       # modules.<name>.model_path points at
    ├── janus_text_encoder/                # when starting from HF weights.
    └── janus_llama/
```

The per-module `.metadata` files are not interchangeable, and that is the reason
the step-level marker is a separate file rather than one of them. Modules are
deliberately heterogeneous: in `configs/seed_omni/Janus/janus_1.3b/train/modules_train.yaml`,
`janus_siglip` runs DDP with `init_device: cuda`, `janus_llama` runs FSDP2, and
`janus_text_encoder` adds a size-4 `emb` extra-parallel dim. Each module's DCP
directory therefore records a different sharding topology, and none of them says
anything about whether the *other* modules finished writing.

Single-module and multi-module jobs go through the same writer. The module name
is the only difference: a single-module job passes an empty name and writes
directly into `model/`, so `model/` and `model/<module>/` are the same code path
with an empty and a non-empty subfolder. On resume each module resolves its own
`model/<name>/` under the step directory, so `train.checkpoint.load_path` stays
a single path for the whole job.

### Who writes what

| Path | Writer | Why |
|------|--------|-----|
| `model/**/{ckpt,optimizer}/` | All ranks, via DCP | Collective; shards are rank-local. |
| `model/**/lr_scheduler.pt` | Rank 0 | Scheduler state is replicated. Peers still join the save reduction so a failed write cannot leave them inside `dcp.save`. |
| `loader/rank_{R}.pt` | Rank R | Iterable datasets are `split_dataset_by_node`-sharded on `dp_rank` and the multisource sampler filters on `dp_rank`, so the cursor is rank-local. Restoring rank 0's cursor everywhere would replay rank 0's shard and skip the rest. |
| `extra_state/rank_{R}.pt` | Rank R | RNG and meters are rank-local for the same reason. |
| `checkpoint_manifest.json` | Rank 0, last | Single completion marker for the whole step. |
| `hf_ckpt/`, `lora_ckpt/` | Rank 0 after a DCP gather | Export, not resume. All ranks must call the save — the gather is collective — but only rank 0 writes. |
| `model_assets/` | Rank 0 | `save_pretrained` of every model asset, once per run. |

### Completion and staging

`.metadata` marks one DCP directory complete, and there are now at least two per
module — `ckpt/` and `optimizer/`. It says nothing about the sibling directory,
the other modules, the dataloader cursor, or the job state.
`checkpoint_manifest.json` is the step-level marker: rank 0 writes it only after
every module's DCP saves have returned and every rank's `loader/` and
`extra_state/` files are on disk.
Resume discovery accepts a `global_step_{N}/` directory only when the manifest
is there, so a crash mid-save leaves a directory that is skipped rather than
half-loaded.

`train.checkpoint.stage_dir`, when set, is node-local scratch: `model/` is
written there first and copied into `global_step_{N}/` after DCP finishes. The
staging directory is not part of the published checkpoint.

Async saves (`train.checkpoint.save_async`) return while the DCP write is still
in flight. The manifest write drains the pending save first, which is what keeps
the marker from appearing before the shards it claims are complete.

## Previous layouts

Three earlier layouts exist on disk. All are read-only: new saves never produce
them.

**Flat layout (VeOmni 0.2.x, before this page).** Model, cursor and export all
sat at the step root, and the scheduler lived in a per-rank `extra_state`
pickle even though its contents were replicated.

```
<output_dir>/
├── checkpoints/
│   └── global_step_{N}/
│       ├── .metadata                      # Step root was also the DCP directory.
│       ├── __{i}_{rank}.distcp
│       ├── extra_state/
│       │   └── extra_state_rank_{R}.pt    # lr_scheduler.state_dict() only.
│       ├── trainer_state_rank_{R}.pt      # global_step, dataloader, rng, meters.
│       └── hf_ckpt/
├── model_assets/
└── global_step_{N}/                       # LoRA adapter, sibling of checkpoints/.
    ├── adapter_config.json
    └── adapter_model.bin
```

**VeOmni 0.1.12.** Same tree, but there was no `trainer_state_rank_{R}.pt`:
`extra_state_rank_{R}.pt` was a mixed bag holding both the scheduler and the
job cursor. That is why it was per-rank in the first place — the cursor is
rank-local — and why changing world size required a matching file per rank.

| Era | Contents of `extra_state_rank_{R}.pt` |
|-----|----------------------------------------|
| 0.1.12 | `lr_scheduler.state_dict()` **and** the job cursor (`global_step`, dataloader, RNG, meters). |
| 0.2.x flat | `lr_scheduler.state_dict()` only; the cursor moved to `trainer_state_rank_{R}.pt`. |

**SeedOmni V2 module layout.** V2 nested per module from the start, but one
level too shallow: the module directory *was* the DCP directory, so the HF
safetensors export landed on top of the shards, and the LoRA adapter went to the
0.1.12 sibling path. `OmniModuleCheckpointManager` computed those paths itself
(`_save_dir`, `_hf_export_dir`, `_output_dir`) instead of deriving them from the
base manager, which is why they drifted.

```
<output_dir>/
├── checkpoints/
│   └── global_step_{N}/
│       └── janus_llama/                   # Module dir == DCP dir == HF export dir.
│           ├── .metadata                  #   DCP shards and safetensors
│           ├── __{i}_{rank}.distcp        #   share one directory.
│           ├── extra_state/               #   lr_scheduler, per rank.
│           ├── model*.safetensors
│           └── config.json
└── global_step_{N}/
    └── janus_llama/                       # LoRA adapter, sibling of checkpoints/.
```

The current layout separates all three: `model/janus_llama/` for the resume
state (itself split into `ckpt/` and `optimizer/`), `hf_ckpt/janus_llama/` for
the full-model export, and `lora_ckpt/janus_llama/` for the adapter.

Note that `extra_state/` means something different in the current layout: it
holds `rank_{R}.pt` with the job state minus the dataloader, and never the
scheduler. The two are told apart by structure, not by directory name — a
current checkpoint has `model/` and `loader/` siblings and a manifest, and an
old one has neither.

## Legacy resume

Temporary. The compatibility loader exists so an older checkpoint can still
resume; it will be removed.

**Where:** `veomni/checkpoint/legacy_v0_1_12.py`.

| Missing in the new layout | Falls back to |
|---------------------------|---------------|
| `model/lr_scheduler.pt` | `lr_scheduler.pt` at the step root, then `extra_state/extra_state_rank_{R}.pt` (`lr_scheduler` key) |
| `loader/rank_{R}.pt`, `extra_state/rank_{R}.pt` | `trainer_state_rank_{R}.pt`, then the 0.1.12 pickle's job cursor when `global_step` is present |
| `model/ckpt/.metadata` | `.metadata` at the step root — an old checkpoint has weights and optimizer fused in one directory, and both are read from there |

**How to drop it:** delete `veomni/checkpoint/legacy_v0_1_12.py` and the imports
that load it (search for `legacy_v0_1_12`). After that, an old checkpoint fails
resume instead of falling back.

Current files always win; the fallbacks are consulted only when the current path
is absent. The same three fallbacks resolve a V2 module checkpoint, because they
are applied *within* a module's directory: a module whose `model/<name>/` is
absent falls back to `<name>/` at the step root, which is where V2 put it.

A 0.1.12 LoRA export lived at `<output_dir>/global_step_{N}/`, a sibling of
`checkpoints/`. That path is an inference artifact and DCP resume never read it.
Point `PeftModel.from_pretrained` at the old directory if you need the adapter.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export under `lora_ckpt/`.

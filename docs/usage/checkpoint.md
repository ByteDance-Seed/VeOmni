# Checkpoint layout

The on-disk contract for a training run: what is written, who writes it, and how
resume reads the older layouts.

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

`loader/` and `extra_state/` are per rank because their contents are: iterable
datasets are `split_dataset_by_node`-sharded on `dp_rank` and the multisource
sampler filters on `dp_rank`, so restoring rank 0's cursor everywhere would
replay rank 0's shard and skip the rest. RNG and meters are rank-local for the
same reason. The scheduler is replicated instead — rank 0 writes it and every
rank reads that one file, though the peers still join the save reduction so a
failed write cannot leave them inside `dcp.save`.

**Model assets** is whatever `trainer.model_assets` carries, in type terms
`Union[PretrainedConfig, GenerationConfig, PreTrainedTokenizer, ProcessorMixin]`
(`veomni/models/module_utils.py`). Each is written by calling its own
`save_pretrained`, so the files that appear are whatever those objects emit:
`config.json` and `generation_config.json` from the configs, then tokenizer
files, processor files, or both — a text model carries a tokenizer, a VLM
carries a processor that wraps one. Do not read a fixed file list into these
directories.

### Why weights and optimizer are two DCP directories

They have different lifetimes. Optimizer state is roughly twice the size of the
weights under Adam and is useless outside this run, so a checkpoint being
shipped, archived or converted wants the weights alone. Note this is about
moving *files*: reading only the weights was always possible by omitting
`optimizer` from the load state, since `dcp.load` reads only the keys it is
asked for. What a single directory made impossible was separating them
afterwards, because both land in the same `__{i}_{rank}.distcp` files.

### Why the exports are two directories

`hf_ckpt/` and `lora_ckpt/` are mutually exclusive today —
`ModelCheckpointManager.save_hf_or_lora` routes on `model.lora_config`. They are
separate paths because that stops being true the moment LoRA merging lands: a
merge writes the adapter *and* a merged full-model export for the same step, and
a reader must be able to tell them apart by path.

`lora_ckpt/` is a PEFT-format directory — point `PeftModel.from_pretrained` at
it. It is not what a LoRA run resumes from: the adapter's *training* state
(trainable tensors plus optimizer) is in `model/`, and the frozen base comes
from `model.model_path`. See
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
    ├── janus_siglip/             # what modules.<name>.model_path points at
    └── janus_llama/              # when starting from HF weights
```

Single-module and multi-module jobs go through the same writer, and the module
name is the only difference: a single-module job passes an empty name and writes
directly into `model/`, so `model/` and `model/<module>/` are one code path with
an empty and a non-empty subfolder. Every path on both the save and the load
side comes from `veomni/checkpoint/layout.py`, so the two cannot disagree. On
resume each module resolves its own `model/<name>/`, which is why
`train.checkpoint.load_path` stays a single path for the whole job.

## Completion and staging

`.metadata` marks one DCP directory complete, and there are at least two per
module. It says nothing about the sibling directory, the other modules, the
cursor, or the job state. Module `.metadata` files are not interchangeable
either, because modules are deliberately heterogeneous — in
`configs/seed_omni/Janus/janus_1.3b/train/modules_train.yaml`, `janus_siglip`
runs DDP with `init_device: cuda`, `janus_llama` runs FSDP2, and
`janus_text_encoder` adds a size-4 `emb` extra-parallel dim — so each records a
different sharding topology. That is why the step-level marker is its own file.

`checkpoint_manifest.json` is that marker. Rank 0 writes it only after every
module's DCP save has returned and every rank's `loader/` and `extra_state/`
files are on disk. For a checkpoint in this layout, resume discovery accepts a
`global_step_{N}/` directory only when the manifest is present, so a crash
mid-save leaves a directory that is skipped rather than half-loaded. A
checkpoint from an older layout has no manifest to find; discovery falls back to
a `.metadata` at the step root for those, which is the marker they were
published by. See [Legacy resume](#legacy-resume).

### Staged and asynchronous saves

`train.checkpoint.stage_dir` and `train.checkpoint.save_async` are two answers to
the same problem, a destination slow enough that writing to it blocks the train
loop. **They cannot be combined** — `DistributedCheckpointer.save` raises on the
pair, because an async write is still running when `save()` returns and drops the
staged copy, so it would write straight to the destination staging was meant to
avoid.

With `stage_dir` (synchronous), the whole `model/` subtree — both DCP
directories and `lr_scheduler.pt` — is written to node-local scratch and copied
into the step directory afterwards, with every `.metadata` copied last. The
staging directory is not part of the published checkpoint.

With `save_async` (unstaged), `model/` is written in place and the call returns
while the DCP write is still in flight. `lr_scheduler.pt` is written first, by
rank 0, before either DCP save starts. Weights and optimizer are two independent
saves, each holding its own future and its own Gloo process group so they
overlap rather than serialise. The manifest write drains both first, which is
what keeps the marker from appearing before the shards it claims are complete;
a drain that finds a failed save raises on every rank, not just the one that
saw it.

## Previous layouts

Three earlier shapes exist on disk. All are read-only: new saves never produce
them.

**Flat (VeOmni 0.2.x).** Model, cursor and export sat at the step root, and the
scheduler lived in a per-rank `extra_state` pickle even though its contents were
replicated.

```
checkpoints/global_step_{N}/
├── .metadata                             # step root was also the DCP directory
├── __{i}_{rank}.distcp                   # weights and optimizer fused
├── extra_state/extra_state_rank_{R}.pt   # lr_scheduler only
├── trainer_state_rank_{R}.pt             # global_step, dataloader, rng, meters
└── hf_ckpt/
<output_dir>/global_step_{N}/             # LoRA adapter, sibling of checkpoints/
```

**VeOmni 0.1.12.** The same tree without `trainer_state_rank_{R}.pt`:
`extra_state_rank_{R}.pt` was a mixed bag holding both the scheduler and the job
cursor. That is why it was per-rank in the first place — the cursor is — and why
changing world size required a matching file per rank.

| Era | Contents of `extra_state_rank_{R}.pt` |
|-----|---------------------------------------|
| 0.1.12 | `lr_scheduler.state_dict()` **and** the job cursor (`global_step`, dataloader, RNG, meters) |
| 0.2.x flat | `lr_scheduler.state_dict()` only |

**SeedOmni V2 modules.** V2 nested per module from the start, but one level too
shallow: the module directory *was* the DCP directory, so the safetensors export
landed on top of the shards and the adapter went to the 0.1.12 sibling path.
`OmniModuleCheckpointManager` computed those paths itself instead of deriving
them from the base manager, which is how they drifted.

```
checkpoints/global_step_{N}/janus_llama/  # module dir == DCP dir == HF export dir
├── .metadata
├── __{i}_{rank}.distcp
├── extra_state/                          # lr_scheduler, per rank
├── model*.safetensors
└── config.json
```

The current layout separates all three: `model/janus_llama/` for the resume
state, `hf_ckpt/janus_llama/` for the export, `lora_ckpt/janus_llama/` for the
adapter.

`extra_state/` therefore means something different in each era, and the two are
told apart by structure rather than by that name: a current checkpoint has
`model/` and `loader/` siblings and a manifest, and the file inside is
`rank_{R}.pt`, not `extra_state_rank_{R}.pt`.

## Legacy resume

Temporary. The compatibility loader in `veomni/checkpoint/legacy_v0_1_12.py`
exists so an older checkpoint can still resume; it will be removed. Current
files always win — a fallback is consulted only when the current path is absent.

| Missing in the current layout | Falls back to |
|-------------------------------|---------------|
| `checkpoint_manifest.json` (discovery, `load_path: auto`) | `.metadata` at the step root, which is how an older step was published |
| `model/ckpt/.metadata` | `.metadata` at the step root; weights and optimizer are both read from that fused directory |
| `model/lr_scheduler.pt` | `lr_scheduler.pt` at the step root, then the 0.1.12 pickle's `lr_scheduler` key |
| `loader/rank_{R}.pt`, `extra_state/rank_{R}.pt` | `trainer_state_rank_{R}.pt`, then the 0.1.12 pickle's job cursor when `global_step` is present |

The same three fallbacks resolve a V2 module checkpoint, because they are applied
*within* a module's directory: a module whose `model/<name>/` is absent falls
back to `<name>/` at the step root, which is where V2 put it.

To drop legacy resume, delete `veomni/checkpoint/legacy_v0_1_12.py` and the
imports that load it (search for `legacy_v0_1_12`). An old checkpoint then fails
resume instead of falling back.

A 0.1.12 LoRA export lived at `<output_dir>/global_step_{N}/`, a sibling of
`checkpoints/`. That path is an inference artifact and DCP resume never read it;
point `PeftModel.from_pretrained` at the old directory if you need the adapter.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export under `lora_ckpt/`.

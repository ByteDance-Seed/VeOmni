# Checkpoint layout

This page is the on-disk contract for a VeOmni training run: which directories
are written, what each file holds, and how that differs from the previous
layout. Resume reads **this** layout only. Older `extra_state/` pickles are
not loaded.

`train.checkpoint.output_dir` is the run root. Every per-step artifact — DCP
shards, scheduler, job cursor, HuggingFace export, and LoRA adapter — lives
under `output_dir/checkpoints/global_step_{N}/` (`train.checkpoint.save_path`).
Tokenizer / processor / config sidecars are written once to `model_assets/`.

## Current layout

```
<output_dir>/                          # train.checkpoint.output_dir
├── checkpoints/                       # train.checkpoint.save_path
│   └── global_step_{N}/
│       ├── .metadata                  # DCP completion marker. A reader treats
│       │                              # this directory as a complete checkpoint
│       │                              # only when this file is present.
│       ├── __{i}_{rank}.distcp        # PyTorch DCP shards. Together they hold
│       │                              # the model weights and the optimizer
│       │                              # state. LoRA / PEFT runs store only
│       │                              # trainable adapter tensors here; the
│       │                              # frozen base is reloaded from
│       │                              # model.model_path.
│       ├── lr_scheduler.pt            # One pickle: lr_scheduler.state_dict().
│       │                              # Replicated across ranks; rank 0 writes
│       │                              # it. Every rank reads this same file
│       │                              # on resume.
│       ├── trainer_state_rank_{R}.pt  # Per-rank job cursor. Rank R writes and
│       │                              # rank R restores. Contents:
│       │                              #   global_step
│       │                              #   train_dataloader   (sampler / iterator)
│       │                              #   environ_meter
│       │                              #   channel_loss_callback
│       │                              #   torch_rng_state
│       ├── adapter_config.json        # LoRA / PEFT export (adapter runs only).
│       ├── adapter_model.safetensors  # Inference artifact; not used to resume
│       │                              # the optimizer. Same directory as DCP.
│       └── hf_ckpt/                   # Optional full-model HuggingFace export
│           ├── model*.safetensors     #   (train.checkpoint.save_hf_weights).
│           ├── model.safetensors.index.json
│           ├── config.json
│           └── tokenizer files
└── model_assets/                      # Tokenizer / processor / config, written
                                       # once at train start (rank 0).
```

A multi-module job (Omni) nests the DCP shards, `lr_scheduler.pt`, LoRA adapter
files, and `hf_ckpt/` one level deeper under the module name
(`ModelCheckpointManager.checkpoint_subfolder`). `trainer_state_rank_{R}.pt`
stays at `checkpoints/global_step_{N}/` — there is one job cursor, not one per
module.

`train.checkpoint.stage_dir`, when set, is scratch: the shards and
`lr_scheduler.pt` are written there first and copied to `global_step_{N}/`
only after DCP finishes. The staging directory is not part of the published
checkpoint.

### Who writes what

| File | Writer | Why |
|------|--------|-----|
| `__*.distcp`, `.metadata` | All ranks, via DCP | Collective; shards are rank-local. |
| `lr_scheduler.pt` | Rank 0 | Scheduler state is replicated. Peers still join the save reduction so a failed write cannot leave them inside `dcp.save`. |
| `trainer_state_rank_{R}.pt` | Rank R | Dataloader cursor, RNG, and meters are rank-local. Restoring rank 0's cursor on every rank would replay rank 0's shard. |
| `model_assets/` | Rank 0 | Config / tokenizer dump. |
| `hf_ckpt/`, LoRA adapter | Rank 0 after a DCP gather | Export, not resume. |

## Previous layout

Checkpoints written before this layout used a per-rank `extra_state` pickle
instead of `lr_scheduler.pt`. That path is **not** read on resume.

```
<output_dir>/
├── checkpoints/
│   └── global_step_{N}/
│       ├── .metadata
│       ├── __{i}_{rank}.distcp        # Model + optimizer (same role as today).
│       ├── extra_state/
│       │   └── extra_state_rank_{R}.pt
│       └── hf_ckpt/                   # Optional HF export (same role as today).
├── model_assets/
└── global_step_{N}/                   # LoRA adapter used to live here, as a
    ├── adapter_config.json            # sibling of checkpoints/. It now sits in
    └── adapter_model.bin              # checkpoints/global_step_{N}/ instead.
```

`extra_state/extra_state_rank_{R}.pt` was a per-rank `torch.save` pickle. What
it contained changed once:

| Era | Contents of `extra_state_rank_{R}.pt` |
|-----|----------------------------------------|
| `CheckpointerCallback` (before the manager / global_state split) | Mixed bag: `lr_scheduler.state_dict()`, and the job cursor (`global_step`, dataloader, RNG, meters). Rank-local cursor is why the file was per-rank. Changing world size required a matching `extra_state_rank_{R}.pt`. |
| After the split, before this layout | **Only** `lr_scheduler.state_dict()`. The job cursor moved to `trainer_state_rank_{R}.pt` beside the shards. The per-rank `extra_state` files were leftover: the scheduler is replicated, so every rank wrote the same pickle. |

There is no loader for `extra_state/`. A directory that only has
`extra_state_rank_*.pt` will skip the scheduler (and, for the mixed-bag era,
will not restore the job cursor from that pickle either). Re-run from the
HuggingFace / adapter export, or convert the pickle yourself.

## Related pages

- [Checkpoint conversion](checkpoint_conversion.md) — DCP shards → HuggingFace safetensors (`scripts/merge_dcp_to_hf.py`).
- [Trainer callbacks](trainer.md#callbacks) — `CheckpointCallback` (when) vs `GlobalStateCallback` (job cursor).
- [LoRA checkpoints](../key_features/lora.md#4-checkpoint-saving) — adapter export next to DCP.

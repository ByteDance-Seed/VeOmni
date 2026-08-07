# DeepSpec Draft-Model Integration via the VeOmni Engine

This document describes how DeepSpec speculative-decoding **draft models**
(DSpark / DFlash / Eagle3) train through VeOmni's distributed engine, so they
gain VeOmni's performance features — **FSDP2 parameter/gradient/optimizer
sharding, meta-device init, gradient checkpointing, activation offloading, and
Torch Distributed Checkpoint (DCP)** — instead of DeepSpec's original
single-node `torch.multiprocessing.spawn` + `FSDP NO_SHARD` (fully-replicated)
training loop.

The integration reuses DeepSpec's own modeling / loss / dataset code verbatim;
no algorithm is re-implemented. Only the thin seams VeOmni expects are adapted.

## Why the integration is clean

DeepSpec training is **decoupled from the target model**. A target model is run
offline exactly once to dump per-token hidden states to disk (the *target
cache*); training then only reads that cache. The target model is **never in the
training loop** — it is used only at init to copy *frozen* `embed_tokens` /
`lm_head` weights into the draft model.

Consequently, from VeOmni's point of view a draft model is just a small
`transformers.PreTrainedModel` trained on a tensor dataset. That is exactly the
shape VeOmni already scales, so the bridge is a set of adapters rather than a
rewrite.

## Correctness: why FSDP2 sharding yields identical gradients

This is the load-bearing property of the integration.

DeepSpec's loss functions (`compute_dspark_loss`, `compute_eagle3_loss`) compute
a **globally-correct mean** by all-reducing the loss *denominators* across the
whole process group and then multiplying the backward loss by `world_size`:

```
backward_loss = (sum_local_numerator / sum_global_denominator) * world_size
```

FSDP (both the original `NO_SHARD` and FSDP2) reduce-scatters gradients with an
**average** over the shard group. When the **only** parallelism is FSDP data
parallelism — the FSDP shard group spans the entire world and every other
parallel dim is size 1 — the shard group equals the world, so:

```
grad = AVG_over_world( d(backward_loss)/dθ )
     = (1/world_size) * Σ_ranks d( local_loss * world_size )/dθ
     = Σ_ranks d(local_loss)/dθ                      # ×world_size cancels ÷world_size
```

This is exactly the gradient DeepSpec's original `NO_SHARD` (replicated) setup
produced — but now with real parameter/gradient/optimizer-state sharding for the
memory and throughput win. `DeepSpec` did all-reduce + `×world_size` precisely so
that DDP/FSDP averaging would cancel; VeOmni's FSDP2 averaging is the same
operation, so the math carries over unchanged.

The equivalence **breaks** under sequence / tensor / expert / pipeline / context
parallel or HSDP replication, because then the FSDP shard group is a strict
subset of the world and the loss would need to reduce over the fsdp/dp group (and
for SP, account for the sequence split). Therefore `DraftModelTrainer`
**enforces** `ulysses_size == tp_size == pp_size == cp_size == ep_size ==
dp_replicate_size == 1` and fails loudly otherwise, so gradients are never
silently mis-scaled. Broadening to SP/EP is a well-scoped follow-up (see
[Limitations](#limitations-and-follow-ups)).

## Architecture

```
                DeepSpec repo (imported, not vendored)
                ┌───────────────────────────────────────────┐
                │ modeling.dspark.* / modeling.eagle3.*       │
                │ modeling.*.loss.compute_{dspark,eagle3}_loss│
                │ data.CacheDataset / data.CacheCollator      │
                │ utils.metrics (distributed accumulator)     │
                └───────────────────────────────────────────┘
                        ▲            ▲              ▲
   register / lazy-import│  reuse     │ reuse        │ flush
                        │            │              │
VeOmni bridge ──────────┴────────────┴──────────────┴──────────────
  veomni/integrations/deepspec/         locate + lazily import deepspec
  veomni/models/transformers/deepspec_draft/   MODELING_REGISTRY + config
  veomni/data/deepspec/                 CacheDataset → VeOmni mapping dataset
  veomni/trainer/deepspec/              DraftModelTrainer(BaseTrainer)
  tasks/train_deepspec_draft.py         task entry (parse_args → trainer.train)
  scripts/deepspec/prepare_draft_init.py  build init ckpt + config.json (offline)
  scripts/deepspec/train_draft.sh       torchrun launcher wrapper
  configs/deepspec/*.yaml               DSpark / Eagle3 Qwen3-4B recipes
```

### 1. DeepSpec path bootstrap — `veomni/integrations/deepspec/`

DeepSpec is normally installed via the `deepspec` extra (a git-pinned dependency;
see Prerequisites). For local DeepSpec development it can instead be resolved from
a checkout: `ensure_deepspec_importable()` first honours an already-importable
`deepspec`, and only if that fails resolves a checkout (via `$DEEPSPEC_PATH`, or a
sibling `DeepSpec/` next to the VeOmni repo) and puts it on `sys.path`. The actual
`import deepspec` is **lazy** (inside the function), so importing the VeOmni model
package at startup does *not* require DeepSpec to be present — other models are
unaffected.

### 2. Model + config registration — `veomni/models/transformers/deepspec_draft/`

VeOmni looks up a model class by `config.model_type` via `MODELING_REGISTRY`
(when `MODELING_BACKEND != "hf"`, the default). A DeepSpec draft config is a deep
copy of the *target* config, so it carries the target's `model_type` (`"qwen3"` /
`"gemma4"`) — which would shadow VeOmni's own Qwen3/Gemma. The prep step
(§ prepare) rewrites `model_type` to a dedicated **`deepspec_draft`** key and
records the concrete draft class in `architectures` (e.g. `["Qwen3DSparkModel"]`).

We register:

* `DeepSpecDraftConfig` in `MODEL_CONFIG_REGISTRY` — required because HF
  `AutoConfig` does not know `deepspec_draft`. It reconstructs a faithful,
  normalized target config from `base_model_type` (so DeepSpec modeling code can
  read `rope_parameters`, `rms_norm_eps`, `layer_types`, … as usual) and layers
  the draft fields on top.
* A modeling factory in `MODELING_REGISTRY` keyed by `deepspec_draft` that
  dispatches on `config.architectures[0]` to one of the four draft classes
  (DSpark/DFlash × Qwen3/Gemma4 share `*DSparkModel`; Eagle3 × Qwen3/Gemma4).

The four DeepSpec classes are imported lazily inside the factory, so the
registry decorators can run at VeOmni import time without pulling in DeepSpec.

### 3. Data adapter — `veomni/data/deepspec/`

`TargetCacheMappingDataset` wraps DeepSpec's `CacheDataset` for VeOmni's
`build_dataloader(dyn_bsz=False)` path. It returns each sample as `[sample_dict]`
to match the 1-to-N contract that VeOmni's `MakeMicroBatchCollator` inverts
(`features[i][0]`). DeepSpec's `CacheCollator` is reused verbatim as the
per-micro-batch collate function; it pads `input_ids` / `loss_mask` /
`target_hidden_states` / `target_last_hidden_states` and builds `attention_mask`.

Gradient accumulation is handled by VeOmni's dataloader:
`MakeMicroBatchCollator` splits each `dataloader_batch_size` block into
`num_micro_batch = global_batch_size / (micro_batch_size · dp_size)` micro
batches, so the trainer receives `list[dict]` per optimizer step.

### 4. Trainer — `veomni/trainer/deepspec/draft_trainer.py`

`DraftModelTrainer` subclasses `BaseTrainer` and reuses everything (distributed
init, FSDP2 parallelization, meta-init weight loading, DCP checkpointing,
callbacks, optimizer / LR scheduler). It overrides exactly five seams:

| Seam | Why it differs |
|------|----------------|
| `_build_model` | Build with `attn_implementation="flex_attention"` (DeepSpec requires it; VeOmni's `OpsImplementationConfig` can't express that literal and rewrites `flash_attention_2`→SP kernel). We pre-install the ops singleton via `apply_ops_config(...)` and pass `ops_implementation=None` + explicit `attn_implementation`, so `build_foundation_model` honours our value. Weights (incl. frozen target embeds) meta-load from the init checkpoint. |
| `_freeze_model_module` | Mark `embed_tokens` / `lm_head` frozen. VeOmni's `build_optimizer` only optimizes `requires_grad` params and `veomni_clip_grad_norm` only clips grads that exist, so flipping `requires_grad` is sufficient. Runs before parallelization; FSDP2 preserves `requires_grad` on the sharded DTensors. |
| `_build_dataset` / `_build_collate_fn` | Use the target-cache dataset + `CacheCollator` instead of the text pipeline; validate cache `target_layer_ids` / `hidden_size` against the model (mirrors DeepSpec's `validate_train_cache`). |
| `forward_backward_step` | Run DeepSpec's own loss (`compute_dspark_loss` / `compute_eagle3_loss`). DeepSpec already scales by `world_size`, so we divide only by `num_micro_steps` for grad accumulation. |
| `train_step` | DeepSpec-style loop; after `optimizer.step()`, flush DeepSpec's distributed metric buffer into VeOmni's step metrics / wandb. |

A guard (`_validate_parallelism`) enforces the pure-FSDP constraint described in
[Correctness](#correctness-why-fsdp2-sharding-yields-identical-gradients).

Eagle3's loss calls the (FSDP2-sharded) model multiple times per step with
different kwarg sets (`target_logits_only=True`, then a KV-cached TTT unroll).
`fully_shard` registers forward pre/post hooks on the module's `__call__`, so
each call correctly all-gathers parameters — the multi-call pattern works
unchanged.

### 5. Metric bridging

DeepSpec's loss functions emit rich metrics (`ce_loss`, `l1_loss`,
`accept_rate@k`, `confidence_*`, `tau_probabilistic`, Eagle3 `ploss_k`) through
`deepspec.utils.metrics`, whose `flush()` performs collective all-reduces and
asserts a consistent metric schema across ranks. `train_step` calls `flush()`
once per step on **every** rank and merges the result into VeOmni's `loss_dict`.
The emitted metric keys are gated by config (not by per-rank data), so the
cross-rank schema stays consistent.

## Meta-init details

VeOmni requires `init_device="meta"` for FSDP2. `init_empty_weights()` only
patches `register_parameter` (not buffers), so the rotary `inv_freq` buffers are
computed with real values and preserved through `buffer_dict` during weight
load. All trainable + frozen params are streamed from the init checkpoint's
`model.safetensors`, so nothing is left uninitialized. The prep step writes the
full state dict (frozen target embeds included) precisely so meta-load is
deterministic.

## Workflow

### Prerequisites

1. The `deepspec` library importable as `deepspec`. Two options:
   - **Recommended — install the extra** (pins DeepSpec by git commit in
     `pyproject.toml`, so it is reproducible and needs no env vars):
     ```bash
     uv sync --extra gpu --extra deepspec
     ```
     DeepSpec has no PyPI release, so the extra installs it from git. The pin is
     defined in `[tool.uv.sources]`. It currently points at a fork commit that
     carries DeepSpec's packaging (`pyproject.toml`), proposed upstream in
     [`deepseek-ai/DeepSpec#54`](https://github.com/deepseek-ai/DeepSpec/pull/54);
     repoint the `rev` to the upstream commit once that PR merges. VeOmni imports
     only `deepspec.modeling.*`, `deepspec.data`, and
     `deepspec.utils.{config,sampling,metrics}`.
   - **Dev fallback — a local checkout on `sys.path`**: set `DEEPSPEC_PATH`, or
     place a `DeepSpec/` checkout next to the VeOmni repo (the launcher
     auto-detects it). Useful when iterating on DeepSpec itself without
     reinstalling; `ensure_deepspec_importable()` prefers an already-installed
     `deepspec` and only falls back to this path.
2. A target cache built with DeepSpec's `scripts/data/prepare_target_cache.py`
   (unchanged DeepSpec tooling; can be large — ~38 TB for Qwen3-4B at defaults).

### 1. Build the draft init checkpoint (offline; uses the target once)

```bash
python scripts/deepspec/prepare_draft_init.py \
    --algorithm dspark --arch qwen3 \
    --target_model_name_or_path Qwen/Qwen3-4B \
    --output_dir ~/deepspec_init/dspark_qwen3_4b \
    --block_size 7 --num_draft_layers 5 \
    --target_layer_ids 1 9 17 25 33 \
    --mask_token_id 151669 --num_anchors 512 \
    --markov_rank 256 --markov_head_type vanilla \
    --confidence_head_alpha 1.0 --confidence_head_with_markov \
    --loss_decay_gamma 4.0 --ce_loss_alpha 0.1 --l1_loss_alpha 0.9
```

`--algorithm ∈ {dspark, dflash, eagle3}`, `--arch ∈ {qwen3, gemma4}`. This writes
`config.json` (with `model_type=deepspec_draft`, `base_model_type`, and the
concrete `architectures`) and `model.safetensors` (full draft state dict with
frozen target embeds/lm_head copied in). DSpark loss weights are baked into
`config.json` as the single source of truth; the YAML `model.model_config` can
override them per run.

### 2. Point the config at the init checkpoint + target cache

Edit `configs/deepspec/dspark_qwen3_4b.yaml`: `model.config_path` /
`model.model_path` → the prep `--output_dir`; `data.train_path` → the target
cache directory.

### 3. Launch

```bash
# 8-GPU single node (auto-detects a sibling DeepSpec/ checkout)
bash scripts/deepspec/train_draft.sh configs/deepspec/dspark_qwen3_4b.yaml

# or directly via VeOmni's launcher
DEEPSPEC_PATH=/path/to/DeepSpec \
  bash train.sh tasks/train_deepspec_draft.py configs/deepspec/dspark_qwen3_4b.yaml
```

FSDP2 now shards the draft parameters/gradients/optimizer state across all GPUs.
For a fixed global batch size this cuts per-GPU memory (enabling larger drafts /
longer sequences) and improves throughput versus the original replicated
`NO_SHARD` setup.

## Checkpoints

VeOmni saves a DCP checkpoint (model + optimizer + step) and, at train end / every
`hf_save_steps`, an HF-format `hf_ckpt/`. The HF config carries
`model_type=deepspec_draft`, `base_model_type`, and the concrete `architectures`.
To evaluate with DeepSpec's `eval.py`, load the weights into the matching DeepSpec
draft class (named by the `architectures` field).

## Limitations and follow-ups

* **Parallelism**: only pure FSDP2 data parallelism today (see Correctness).
  Sequence / expert / tensor / pipeline parallel require reducing the DeepSpec
  loss over the fsdp/dp group and (for SP) accounting for the sequence split;
  additionally `flex_attention` is incompatible with VeOmni's SP-aware flash-attn
  kernel, so SP support needs an attention path change.
* **MFU**: `VeomniFlopsCounter` has no `deepspec_draft` estimator yet, so MFU
  reports 0 (safe fallback). A DSpark/Eagle3 FLOPs estimator is a follow-up.
* **Validation status**: the integration has been validated statically (syntax,
  config parse, registration/override logic, import-graph safety). A live GPU
  smoke test (prep → a few training steps) is the recommended next step.

## File map

| Piece | Path |
|-------|------|
| DeepSpec path bootstrap | `veomni/integrations/deepspec/` |
| Model + config registration | `veomni/models/transformers/deepspec_draft/` |
| Target-cache dataset adapter | `veomni/data/deepspec/` |
| Trainer | `veomni/trainer/deepspec/draft_trainer.py` |
| Task entry | `tasks/train_deepspec_draft.py` |
| Init-checkpoint prep | `scripts/deepspec/prepare_draft_init.py` |
| Launcher | `scripts/deepspec/train_draft.sh` |
| Configs + usage README | `configs/deepspec/` |

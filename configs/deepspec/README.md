# DeepSpec draft-model training on VeOmni

This directory documents the **DeepSpec ↔ VeOmni** integration: it lets DeepSpec
speculative-decoding *draft* models (DSpark / DFlash / Eagle3) train through
VeOmni's distributed engine, so they inherit VeOmni's performance features —
**FSDP2 parameter/gradient/optimizer sharding, meta-device init, gradient
checkpointing, activation offloading, and Torch Distributed Checkpoint (DCP)** —
instead of DeepSpec's original single-node `mp.spawn` + `FSDP NO_SHARD` loop.

## Why this works (and is correct)

DeepSpec training is **decoupled from the target model**: a target model is run
offline once to dump per-token hidden states to disk (the *target cache*), and
training then only reads that cache. The target is never in the training loop —
it is used only at init to copy *frozen* `embed_tokens` / `lm_head` weights.

So from VeOmni's point of view, a draft model is just a small
`transformers.PreTrainedModel` trained on a tensor dataset. The bridge reuses
DeepSpec's own modeling / loss / dataset code (no algorithm is re-implemented)
and adapts the thin seams VeOmni expects.

**Gradient-scaling equivalence.** DeepSpec computes a globally-correct mean loss
by all-reducing the loss *denominators* over the whole process group and then
multiplying the backward loss by `world_size` to cancel the gradient averaging
that DDP/FSDP applies. FSDP2 reduce-scatters gradients with `AVG` over the shard
group. When the **only** parallelism is FSDP data parallelism — the FSDP shard
group spans the entire world, every other parallel dim is size 1 — the shard
group equals the world, so `×world_size` exactly cancels FSDP's `÷fsdp_size`.
The result is the same gradient as DeepSpec's original setup, now with real
sharding. `DraftModelTrainer._validate_parallelism` **enforces** this (SP / TP /
EP / PP / CP / HSDP-replicate must all be 1) so gradients are never silently
mis-scaled.

## Files

| Piece | Path |
| --- | --- |
| DeepSpec path bootstrap | `veomni/integrations/deepspec/` |
| Model + config registration | `veomni/models/transformers/deepspec_draft/` |
| Target-cache dataset adapter | `veomni/data/deepspec/` |
| Trainer | `veomni/trainer/deepspec/draft_trainer.py` |
| Task entry | `tasks/train_deepspec_draft.py` |
| Init-checkpoint prep | `scripts/deepspec/prepare_draft_init.py` |
| Launcher | `scripts/deepspec/train_draft.sh` |
| Configs | `configs/deepspec/{dspark,eagle3}_qwen3_4b.yaml` |

## Prerequisites

1. **A DeepSpec checkout** importable as `deepspec`. Set `DEEPSPEC_PATH` to its
   repo root, or place a `DeepSpec/` checkout next to the VeOmni repo (the
   launcher auto-detects a sibling checkout).
2. **A target cache** built with DeepSpec's
   `scripts/data/prepare_target_cache.py` (this is unchanged DeepSpec tooling;
   it can be large — ~38 TB for Qwen3-4B at the default settings).

## Workflow

### 1. Build the draft init checkpoint

VeOmni meta-loads weights from a HuggingFace-format checkpoint. This step builds
the draft model, copies the frozen target embeddings / lm_head into it, and
writes `config.json` (with `model_type=deepspec_draft`) + `model.safetensors`:

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

`--algorithm` ∈ `{dspark, dflash, eagle3}`, `--arch` ∈ `{qwen3, gemma4}`. Use the
same hyper-parameters as the corresponding DeepSpec config under
`DeepSpec/config/`. The DSpark loss weights are baked into `config.json` so the
trainer reads them from the model; you can override them per run via
`model.model_config` in the YAML.

### 2. Point the config at the init checkpoint + target cache

Edit `configs/deepspec/dspark_qwen3_4b.yaml`:

* `model.config_path` / `model.model_path` → the `--output_dir` from step 1.
* `data.train_path` → the target-cache directory.

### 3. Launch

```bash
# 8-GPU single node (torchrun); auto-detects a sibling DeepSpec/ checkout.
bash scripts/deepspec/train_draft.sh configs/deepspec/dspark_qwen3_4b.yaml

# or directly via VeOmni's launcher:
DEEPSPEC_PATH=/path/to/DeepSpec \
  bash train.sh tasks/train_deepspec_draft.py configs/deepspec/dspark_qwen3_4b.yaml
```

FSDP2 now shards the draft parameters/gradients/optimizer state across all GPUs.
For the same global batch size this cuts per-GPU memory (enabling larger draft
models / longer sequences) and improves throughput versus the original
replicated `NO_SHARD` setup.

## What the trainer overrides

`DraftModelTrainer` subclasses VeOmni's `BaseTrainer` and reuses everything
except five seams:

1. `_build_model` — builds the draft model with `flex_attention` (DeepSpec
   requires it) and meta-loads all weights (frozen target embeds included).
2. `_freeze_model_module` — freezes `embed_tokens` / `lm_head` so the optimizer
   and grad-norm clip skip them.
3. `_build_dataset` / `_build_collate_fn` — DeepSpec target-cache dataset +
   `CacheCollator` instead of the text pipeline.
4. `forward_backward_step` — runs DeepSpec's own loss (`compute_dspark_loss` /
   `compute_eagle3_loss`); divides by the micro-batch count for grad accum.
5. `train_step` — flushes DeepSpec's metric buffer (`ce_loss`, `l1_loss`,
   `accept_rate@k`, `confidence_*`, `tau_probabilistic`, Eagle3 `ploss_k`) into
   VeOmni's step metrics / wandb.

## Notes & limitations

* **Parallelism**: only pure FSDP2 data parallelism is supported today (see the
  correctness note above). Sequence/expert/tensor/pipeline parallel are a
  follow-up — they require reducing the DeepSpec loss over the fsdp/dp group and
  (for SP) accounting for the sequence split; `flex_attention` is also
  incompatible with VeOmni's SP-aware flash-attn kernel.
* **Checkpoints**: VeOmni saves a DCP checkpoint (model + optimizer + step) and,
  at the end / every `hf_save_steps`, an HF-format `hf_ckpt/`. The HF config
  carries `model_type=deepspec_draft` + `base_model_type` + the concrete
  `architectures`. To evaluate with DeepSpec's `eval.py`, load the weights into
  the matching DeepSpec draft class (the `architectures` field names it).
* **`local_batch_size`**: DeepSpec used `local_batch_size=1`; the equivalent
  here is `train.micro_batch_size: 1` with grad accumulation derived from
  `global_batch_size / dp_size`.

# Qwen3-0.6B (SeedOmni)

End-to-end recipe for training and inferring **Qwen3-0.6B** as a SeedOmni
graph model. This is the minimal **text-only** omni model: the monolithic
`Qwen3ForCausalLM` is split into two OmniModules wired as
`token_encode → qwen3_llm → token_decode`.

| Module | Holds | Role |
|--------|-------|------|
| `qwen3_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE loss head, ChatML template |
| `qwen3_llm` | decoder backbone (no embed / no head) | `inputs_embeds → hidden_states` |

All paths below assume the upstream HuggingFace checkpoint (the post-trained
chat model) lives at `/mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B`. Adjust to your own
storage.

Config dir: `configs/seed_omni/Qwen/qwen3_0.6b/`.

The layout follows [the shared config layout](../usage/training_and_inference.md#1-config-layout).
Both training and inference take the **same** `base.yaml`.

| File | Role |
|------|------|
| `qwen3_0.6b/train/base.yaml` | Launcher: model paths, `model.accelerator`, data, train, and the `infer` block. |
| `qwen3_0.6b/train/modules_train.yaml` | Per-module training overrides. Add `--model.accelerator.ulysses_size N` for Ulysses SP (see [§3.1](#31-sequence-parallelism-ulysses)). |
| `qwen3_0.6b/train/graph_train.yaml` | Training DAG (`qwen3_text_encoder → qwen3_llm → qwen3_text_encoder.decode`). |
| `qwen3_0.6b/train/data.yaml` | Weighted multisource data list (Tulu-3 SFT mixture). |
| `qwen3_0.6b/train/graph_infer.yaml` | Text chat generation graph (mapped under `model.model_config.infer_graph.infer_text`). |

---

## 1. Checkpoint

Qwen3 registers an HF layout (`modules/qwen3/hf_layout.py`), so `model.model_path`
can be the **upstream checkpoint itself** — `base.yaml` already points at
`/mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B`. The layout routes `model.embed_tokens.*`
(and `lm_head.*`) plus the tokenizer to `qwen3_text_encoder`, and the rest of
`model.*` to `qwen3_llm`; the graphs default to the YAML in this config dir. HF
exports are then written in the upstream layout (see
[Load an upstream checkpoint directly](../usage/training_and_inference.md#21-load-an-upstream-checkpoint-directly)).

The offline split still works, through the same layout, and is what a
cross-family composed model needs:

```bash
python scripts/seed_omni/convert_model.py \
  --model_path /mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B \
  --output_dir /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B
```

Either root works as `model.model_path`.
(Qwen3-0.6B has `tie_word_embeddings=True`, so `qwen3_text_encoder` stores only
`embed_tokens` and the decode head reuses that table via `embed_tokens.project`.
The upstream file still stores a copy as `lm_head.weight`; the load skips it and
the export rewrites it from the trained embedding.)

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `tulu-3-sft-mixture` maps each row's
`messages` list to a `[role, ("text", content)]` conversation.

```yaml
# configs/seed_omni/Qwen/qwen3_0.6b/train/data.yaml
sources:
  - /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/mini_data
names:
  - tulu-3-sft-mixture
schedule:
  - { schedule_type: const, weights: [1.0] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

The on-disk row schema is documented in
[Data Format](../usage/data_format.md).

---

## 3. Train

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/train/base.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — the upstream Qwen3 root, or a split root from step 1.
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length.
- `--model.optimizer.lr` — learning rate (global default; override per module in `train/modules_train.yaml`).
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.

Quick 1-node smoke run (no wandb, tiny step budget):

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/train/base.yaml \
  --model.model_path /mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B \
  --train.max_steps 15 \
  --train.global_batch_size 8 \
  --train.micro_batch_size 1 \
  --data.max_seq_len 2048 \
  --train.wandb.enable false
```

### 3.1 Sequence parallelism (Ulysses)

The `qwen3_text_encoder` (wte) and the `qwen3_llm` backbone both shard the packed
token sequence at the shared SP size. SP has **no dedicated config**: add
`--model.accelerator.ulysses_size N` (design in
[Sequence Parallelism](../distributed/sequence_parallel.md)). The dataloader replicates each DP shard across the
SP group; each module slices to its `1/sp` chunk, runs one forward, and all-gathers
the output back:

```bash
NPROC_PER_NODE=4 bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/train/base.yaml \
  --model.accelerator.ulysses_size 4 \
  --train.global_batch_size 16 --train.micro_batch_size 4
```

- Every module inherits the global `ulysses_size`. The wte is a per-token lookup, so its sequence shards exactly like
  the LLM backbone.
- `world % sp_size == 0`; a larger `micro_batch_size` gives longer packed
  sequences to slice across SP ranks.
- If your environment exports `TORCH_DISTRIBUTED_DEBUG=DETAIL`, **unset it** — its
  `_ProcessGroupWrapper` lacks the coalesced all-gather that FSDP2's tied-head
  `full_tensor()` needs, which crashes any FSDP2 run (SP or not).

---

## 4. Resume

Each save writes per-module DCP shards plus the per-rank job state (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/train/base.yaml \
  --train.checkpoint.load_path outputs/qwen3_0.6b_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

`tasks/omni/infer_omni.py` runs the `infer_text` generation graph (the default
and only `model.model_config.infer_type` for Qwen3). `--model.model_path` is the
upstream Qwen3 root (what `base.yaml` points at) or a split root holding
`qwen3_text_encoder/` and `qwen3_llm/`:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/train/base.yaml \
  --model.model_config.infer_type infer_text \
  --model.model_path /mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B \
  --infer.prompt "What is 2+2?" \
  --infer.output_dir qwen3_out \
  --infer.generation_kwargs.max_new_tokens 1024
# -> "<think> ... the sum of 2 plus 2, which is 4 ..." (coherent chat output)
```

A run trained from the upstream root exports `hf_ckpt/` as one Qwen3 checkpoint,
so infer from it directly with `--model.model_path <step_dir>/hf_ckpt` (it loads
in `transformers` too). A run trained from a split root exports one `hf_ckpt/`
subfolder per module; assemble a flat root from them:

```bash
STEP=outputs/qwen3_0.6b_omni_sft/checkpoints/global_step_500
ASM=outputs/qwen3_0.6b_omni_sft/infer_ckpt/global_step_500
mkdir -p "$ASM"
for m in qwen3_text_encoder qwen3_llm; do
  ln -sfn "$(realpath "$STEP/hf_ckpt/$m")" "$ASM/$m"
done
# then: --model.model_path "$ASM"
```

---

## 6. Alignment with the original text trainer

Because Qwen3 is text-only, the omni path must compute the **same loss** as the
classic text trainer (`tasks/train_text.py`). Run the classic text trainer on
the same model + data and compare the loss curve:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path /mnt/hdfs/veomni/models/Qwen/Qwen3-0.6B \
  --data.train_path /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/mini_data \
  --data.max_seq_len 2048 --train.global_batch_size 8 --train.micro_batch_size 1 \
  --train.max_steps 15 --train.wandb.enable false
```

The classic `chatml` template labels the whole assistant block (role prefix +
`<|im_end|>`); the omni `qwen3_text_encoder` template masks the
`<|im_start|>assistant\n` prefix, which accounts for the small residual gap.

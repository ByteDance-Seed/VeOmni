# Qwen3-0.6B (SeedOmni V2)

End-to-end recipe for training and inferring **Qwen3-0.6B** as a SeedOmni V2
graph model. This is the minimal **text-only** omni model: the monolithic
`Qwen3ForCausalLM` is split into two OmniModules wired as
`token_encode → qwen3_llm → token_decode`.

| Module | Holds | Role |
|--------|-------|------|
| `qwen3_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE loss head, ChatML template |
| `qwen3_llm` | decoder backbone (no embed / no head) | `inputs_embeds → hidden_states` |

All paths below assume the upstream HuggingFace checkpoint (the post-trained
chat model) lives at `/mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B`. Adjust to your own
storage.

Config dir: `configs/seed_omni/Qwen/qwen3_0.6b/` (the `data.yaml` lives one level
up at `configs/seed_omni/Qwen/data.yaml`, shared across Qwen launchers).

The omni config layout splits the old monolithic launcher into a `base.yaml` plus
per-purpose module/graph files. Both training and inference take the **same**
`base.yaml`.

| File | Role |
|------|------|
| `qwen3_0.6b/base.yaml` | Top-level omni launcher: model paths, top-level `accelerator`, data, train, and the `infer` block. |
| `qwen3_0.6b/modules_train.yaml` | Per-module training overrides. |
| `qwen3_0.6b/graph_train.yaml` | Training DAG (`qwen3_text_encoder → qwen3_llm → qwen3_text_encoder.decode`). |
| `data.yaml` | Weighted multisource data list (Tulu-3 SFT mixture). |
| `qwen3_0.6b/graph_infer.yaml` | Text chat generation graph (mapped under `infer.infer_graph.infer_text`). |

---

## 1. Convert the checkpoint

The converter reads `model_type` from the HF `config.json` and dispatches to the
Qwen3 family converter (`modules/qwen3/convert_model.py`), splitting the weights
into `qwen3_text_encoder/` (embeddings + tokenizer) and `qwen3_llm/` (backbone).

```bash
python scripts/convert_model.py \
  --model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B \
  --output_dir /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B
```

The `output_dir` becomes `model.model_path` in `base.yaml`.
(Qwen3-0.6B has `tie_word_embeddings=True`, so `qwen3_text_encoder` stores only
`embed_tokens` and the decode head reuses that weight via `F.linear`.)

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `tulu-3-sft-mixture` maps each row's
`messages` list to a `[role, ("text", content)]` conversation.

```yaml
# configs/seed_omni/Qwen/data.yaml
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
[`docs/seed_omni/data_format.md`](../data_format.md).

---

## 3. Train

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/base.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — split-checkpoint root from step 1.
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length.
- `--train.optimizer.lr` — learning rate.
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.

Quick 1-node smoke run (no wandb, tiny step budget):

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/base.yaml \
  --model.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B \
  --train.max_steps 15 \
  --train.global_batch_size 8 \
  --train.micro_batch_size 1 \
  --data.max_seq_len 2048 \
  --train.wandb.enable false
```

---

## 4. Resume

Each save writes per-module DCP shards plus a `trainer_state.pt` (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/base.yaml \
  --train.checkpoint.load_path outputs/qwen3_0.6b_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

`tasks/omni/infer_omni.py` runs the `infer_text` generation graph (the default
and only `infer.infer_type` for Qwen3). Point `--infer.model_path` at a
**split-checkpoint root** holding `qwen3_text_encoder/` and `qwen3_llm/`, or omit
it to fall back to `model.model_path`. The step-1 converter output already has
this layout, so you can infer directly:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/base.yaml \
  --infer.infer_type infer_text \
  --infer.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B \
  --infer.prompt "What is 2+2?" \
  --infer.output_dir qwen3_out \
  --infer.generation_kwargs.max_new_tokens 1024
# -> "<think> ... the sum of 2 plus 2, which is 4 ..." (coherent chat output)
```

To infer from a **trained** checkpoint, assemble a flat root from the per-module
`hf_ckpt/` subfolders (same pattern as Janus):

```bash
STEP=outputs/qwen3_0.6b_omni_sft/checkpoints/global_step_500
ASM=outputs/qwen3_0.6b_omni_sft/infer_ckpt/global_step_500
mkdir -p "$ASM"
for m in qwen3_text_encoder qwen3_llm; do
  ln -sfn "$(realpath "$STEP/$m/hf_ckpt")" "$ASM/$m"
done
# then: --infer.model_path "$ASM"
```

---

## 6. Alignment with the original text trainer

Because Qwen3 is text-only, the omni path must compute the **same loss** as the
classic text trainer (`tasks/train_text.py`). Two checks:

**(a) Numeric equivalence** — feed one identical token sequence through the
monolithic `Qwen3ForCausalLM` and through the split path
(`qwen3_text_encoder` embed → `qwen3_llm` → tied projection):

```bash
python scripts/seed_omni/check_qwen3_alignment.py \
  --base /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B \
  --split /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B
# -> max|logit diff| ~8e-5, CE loss identical to 6 d.p. (RESULT: ALIGNED)
```

**(b) Trainer comparison** — run the classic text trainer on the same model +
data and compare the loss curve:

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B \
  --data.train_path /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/mini_data \
  --data.max_seq_len 2048 --train.global_batch_size 8 --train.micro_batch_size 1 \
  --train.max_steps 15 --train.wandb.enable false
```

The classic `chatml` template labels the whole assistant block (role prefix +
`<|im_end|>`); the omni `qwen3_text_encoder` template masks the
`<|im_start|>assistant\n` prefix, which accounts for the small residual gap.

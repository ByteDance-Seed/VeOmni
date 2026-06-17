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

Config dir: `configs/seed_omni/Qwen/qwen3_0.6b/`.

The omni config layout splits the old monolithic launcher into a `base.yaml` plus
per-purpose module/graph files. Both training and inference take the **same**
`base.yaml`.

| File | Role |
|------|------|
| `qwen3_0.6b/base.yaml` | Top-level omni launcher: model paths, top-level `accelerator`, data, train, and the `infer` block. |
| `qwen3_0.6b/modules_train.yaml` | Per-module training overrides. |
| `qwen3_0.6b/graph_train.yaml` | Training DAG (`qwen3_text_encoder → qwen3_llm → qwen3_text_encoder.decode`). |
| `qwen3_0.6b/data.yaml` | Weighted multisource data list (Tulu-3 SFT mixture). |
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
# configs/seed_omni/Qwen/qwen3_0.6b/data.yaml
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

---

## 7. Visual instruction tuning: Qwen3-0.6B into image understanding

This recipe turns the **text-only** Qwen3-0.6B into an image-understanding model
by bolting on the **Qwen3-VL ViT** and training as little as possible: only the
ViT's **patch merger** and the vision **special-token** embedding rows — the ViT
blocks and the whole LLM stay frozen.

**No bespoke modeling or build script** — it reuses the standard `qwen3_text_encoder`,
`qwen3_llm` and `qwen3vl_vision` modules; the image-understanding behaviour is
switched on entirely through per-module `model_config:` overrides in the modules YAML.

| Module | Source | `model_config` override | Trainable? |
|--------|--------|--------|-----------|
| `qwen3vl_vision` (ViT + **patch merger**) | Qwen3-VL-2B ViT | `out_hidden_size: 1024`, `disable_deepstack: true`, `freeze: true` | **patch merger only** (ViT blocks frozen) |
| `qwen3_text_encoder` (tied wte/lm_head + tokenizer) | Qwen3-0.6B embed | `enable_image: true` | **only the vision special-token rows** |
| `qwen3_llm` (decoder backbone) | Qwen3-0.6B | `freeze: true` | frozen |

### 7.1 Retarget the patch merger (no separate projector)

Qwen3-VL's ViT outputs **2048-d** tokens (it was pretrained for the 2B LLM), but
Qwen3-0.6B's hidden size is **1024**. Rather than bolt a separate `Linear` on top,
we **retarget the ViT's own patch merger** to the LLM hidden size: the merger
already ends in `linear_fc2: Linear(merger_hidden → out_hidden_size)`, so setting
`out_hidden_size: 1024` via `model_config:` makes that the bridge into the LLM
embedding space. `freeze: true` then freezes the ViT **blocks** but keeps the
**merger** trainable (`freeze_model` re-enables `visual.merger`).

Because a stock Qwen3-VL checkpoint's `merger.linear_fc2` is `→ 2048`, its shape
no longer matches after retargeting. `Qwen3VLVisionEncoder`'s checkpoint converter
(`_MergerProjectionConverter`) **drops the mismatched `linear_fc2`** at load so the
weight loader re-initialises it (via the ViT's `_init_weights`); `merger.norm` and
`merger.linear_fc1` still load from the checkpoint. No build-time surgery needed.

The plain `qwen3_llm` backbone (flat `arange` positions) uses **neither M-RoPE nor
DeepStack**, so `disable_deepstack: true` zeroes `deepstack_visual_indexes` — the
ViT produces only the merged image tokens, which are packed into the sequence like
any other embedding segment.

### 7.2 Special tokens already exist — train only their rows

Qwen3's tokenizer **already reserves** the vision special tokens
(`<|vision_start|>`=151652, `<|vision_end|>`=151653, `<|image_pad|>`=151655, all
below the 151936 embedding rows). So there is **no vocab/embedding expansion** to
do (adding duplicates would be wrong — the tokenizer still maps to the reserved
ids). The standard `qwen3_text_encoder` gains a single config knob (off by
default → plain text-only Qwen3):

- `enable_image: true` — use the **Qwen3-VL image ChatML template** (image →
  `<|vision_start|>` · the vision-embed segment · `<|vision_end|>`) and handle
  image/video parts in decode. With it off, the original text-only template runs
  verbatim (`if self._enable_image:` branches in `modulemixin.py`). It also makes
  `freeze_model()` keep the tied embedding's `requires_grad=True` but register a
  **per-row gradient mask** that zeroes every row except the vision special tokens
  (`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`), so only those rows
  update. **You don't pass the token ids** — the module resolves them from its own
  tokenizer (`convert_tokens_to_ids`), since the user can't know them but the
  module can. (With `enable_image: false` the embedding is fully trainable.)

The grad mask uses **global** row indices, so the module is loaded **`ddp`**
(replicated, not FSDP-sharded) and with **`weight_decay: 0`** (otherwise AdamW's
decoupled decay would erode the frozen rows). Both are set in `modules_train.yaml`.

> The special-token rows load verbatim from Qwen3-0.6B (an untrained reserved
> stub). They start training from there; if you want a better starting point,
> mean-init them before training (HF `mean_resizing` trick).

### 7.3 Assemble the split checkpoint (two standard converts + combine)

No dual-source script: run the two per-model converters, then copy the vision
tower in:

```bash
# text LLM -> qwen3_llm/ + qwen3_text_encoder/
python scripts/convert_model.py --model_type qwen3 \
  --model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-0.6B \
  --output_dir /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B-visual-instruction-tuning

# Qwen3-VL -> qwen3vl_vision/ (+ others); keep only the vision tower
python scripts/convert_model.py --model_type qwen3_vl \
  --model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-VL-2B-Instruct \
  --output_dir /tmp/qwen3vl_split
cp -r /tmp/qwen3vl_split/qwen3vl_vision \
  /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B-visual-instruction-tuning/
```

The combined dir holds `qwen3_llm/`, `qwen3_text_encoder/` (+ tokenizer) and
`qwen3vl_vision/` (ViT + image/video processors). At train time the
`out_hidden_size` override retargets the merger and its mismatched `linear_fc2` is
re-initialised; `disable_deepstack` drops the unused DeepStack mergers.

### 7.4 Config

The configs live alongside the text-only Qwen3-0.6B ones in
`configs/seed_omni/Qwen/qwen3_0.6b/`, distinguished by a `visual_instruction_tuning`
suffix:

| File | Role |
|------|------|
| `visual_instruction_tuning.yaml` | Launcher (model paths, accelerator, data, train, infer). |
| `modules_train_visual_instruction_tuning.yaml` | All overrides: `qwen3vl_vision` merger retarget (`out_hidden_size`) + `disable_deepstack` + `freeze`; `qwen3_text_encoder` image mode + special-token freeze (`ddp` + `weight_decay: 0`); `qwen3_llm` freeze. |
| `graph_train_visual_instruction_tuning.yaml` | `{qwen3vl_vision, qwen3_text_encoder.encode} → qwen3_llm → qwen3_text_encoder.decode → end`. |
| `data_visual_instruction_tuning.yaml` | ShareGPT4V captions (image + text). |
| `graph_infer_visual_instruction_tuning.yaml` | I2T generation FSM. |

### 7.5 Train on ShareGPT4V

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/visual_instruction_tuning.yaml
```

Trainable params are exactly `qwen3vl_vision.visual.merger.*` plus the masked
text-encoder embedding (only the vision special-token rows receive gradient; the
ViT and LLM are frozen).

### 7.6 Inference

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/visual_instruction_tuning.yaml \
  --infer.infer_type understanding \
  --infer.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-0.6B-visual-instruction-tuning \
  --infer.image /path/to/image.jpg \
  --infer.prompt "What is in this image?" \
  --infer.output_dir qwen3_vit_out
```

To infer from a **trained** checkpoint (per-module weights live under
`<step>/<module>/hf_ckpt/`), point `--infer.model_path` at the checkpoint step
dir and override each module's `model_path` **relative to that root** — do NOT
repeat the `--infer.model_path` prefix. Per-module override paths are joined
under `--infer.model_path` unless they are absolute (start with `/`); passing a
cwd-relative full path double-joins it and fails with a cryptic
`HFValidationError: Repo id must be in the form ...`.

```bash
STEP=outputs/qwen3_0.6b_visual_instruction_tuning/checkpoints/global_step_2000
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3_0.6b/visual_instruction_tuning.yaml \
  --infer.infer_type understanding \
  --infer.model_path "$STEP" \
  --infer.image /path/to/image.jpg \
  --infer.prompt "What is in this image?" \
  --infer.output_dir qwen3_vit_out \
  --infer.modules.qwen3vl_vision.model.model_path qwen3vl_vision/hf_ckpt \
  --infer.modules.qwen3_text_encoder.model.model_path qwen3_text_encoder/hf_ckpt \
  --infer.modules.qwen3_llm.model.model_path qwen3_llm/hf_ckpt
```

> **Scope**: this is a deliberately minimal setup (frozen ViT blocks + frozen LLM
> + a retargeted patch merger + the vision special-token rows). It exercises the
> full image pipeline and trains, but real image-understanding quality needs more
> capacity (unfreeze the ViT blocks / LLM, or use an aligned ViT). DeepStack is
> disabled because the plain `qwen3_llm` backbone can't consume those features.

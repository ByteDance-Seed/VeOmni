# Qwen3-VL-2B-Instruct (SeedOmni V2)

End-to-end recipe for training and inferring **Qwen3-VL-2B-Instruct** as a
SeedOmni V2 graph model. This is a **vision-language** omni model: the monolithic
`Qwen3VLForConditionalGeneration` is split into three OmniModules wired as
image-understanding (I2T):

```
image_encode ─┐
              ├─→ qwen3vl_llm → token_decode → end
token_encode ─┘
```

| Module | Holds | Role |
|--------|-------|------|
| `qwen3vl_vision` | ViT + patch merger + DeepStack mergers + image processor | image → merged patch tokens + per-layer DeepStack features |
| `qwen3vl_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE loss head, ChatML (text + image) template |
| `qwen3vl_llm` | text backbone (no embed / no head) | `inputs_embeds → hidden_states`, M-RoPE + DeepStack injection |

Two Qwen3-VL specifics the backbone reconstructs from `conversation_list`:

- **M-RoPE** — 3-row (t/h/w) position ids; image segments use grid positions
  derived from each image item's `grid_thw` (mirrors
  `Qwen3VLModel.get_vision_position_ids`). Text segments use a 1-D run broadcast
  to all 3 rows.
- **DeepStack** — vision features from layers `deepstack_visual_indexes`
  (`[5, 11, 17]`) are added into the matching *interior* decoder layers. The
  vision module emits them on `item.meta["deepstack"]`; the backbone threads them
  into `Qwen3VLTextModel` as `deepstack_visual_embeds` + `visual_pos_masks`.

All paths below assume the upstream HuggingFace checkpoint lives at
`/mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-VL-2B-Instruct`. Adjust to your
own storage.

Config dir: `configs/seed_omni/Qwen/qwen3vl_2b/`

| File | Role |
|------|------|
| `base.yaml` | Launcher: `model` / top-level `accelerator` / `data` (incl. `mm_configs`) / `train` + `infer` block. |
| `modules_train.yaml` | Per-module training overrides (`qwen3vl_vision` / `qwen3vl_text_encoder` / `qwen3vl_llm`). |
| `graph_train.yaml` | Training DAG — flat edge list (`{qwen3vl_vision, qwen3vl_text_encoder.encode} → qwen3vl_llm → qwen3vl_text_encoder.decode → end`). |
| `data.yaml` | Weighted multisource data list (ShareGPT4V images + LLaVA-Video). |
| `graph_infer.yaml` | Image/video-understanding (I2T / VQA) generation graph (`infer.infer_type: vision_understanding`). |

---

## 1. Convert the checkpoint

The converter reads `model_type=qwen3_vl` from the HF `config.json` and dispatches
to the Qwen3-VL family converter (`modules/qwen3vl/convert_model.py`), splitting
the weights into `qwen3vl_vision/`, `qwen3vl_text_encoder/` (embeddings +
tokenizer) and `qwen3vl_llm/` (backbone).

```bash
python scripts/convert_model.py \
  --model_path /mnt/hdfs/veomni/models/transformers/Qwen/Qwen3-VL-2B-Instruct \
  --output_dir /mnt/hdfs/veomni/models/seed_omni/Qwen3-VL-2B-Instruct
```

The `output_dir` becomes `model.model_path` in `base.yaml`.
(Qwen3-VL-2B has `tie_word_embeddings=True`, so `qwen3vl_text_encoder` stores only
`embed_tokens` and the decode head reuses that weight via `F.linear`.)

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `sharegpt4v_cap_100k` maps each row to an
`image` + `text` user turn plus an assistant caption, `llava_video` maps to a
`video` + `text` turn (the video is decoded into a `VideoInputs` carrier), and
`tulu-3-sft-mixture` adds pure-text turns so text ability isn't lost.

```yaml
# configs/seed_omni/Qwen/qwen3vl_2b/data.yaml
sources:
  - /mnt/hdfs/veomni/datasets/sharegpt4v_cap_100k
  - /mnt/hdfs/veomni/datasets/llava_video
  - /mnt/hdfs/veomni/datasets/tulu-3-sft-mixture/data
names:
  - sharegpt4v_cap_100k
  - llava_video
  - tulu-3-sft-mixture
schedule:
  - { schedule_type: const, weights: [0.4, 0.4, 0.2] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

Video decoding is driven by `data.mm_configs` in `base.yaml`
(`use_audio_in_video: false`, `fps`, `min_frames` / `max_frames`) — Qwen3-VL has
no audio modality. The on-disk row schema is documented in
[`docs/seed_omni/data_format.md`](../data_format.md).

---

## 3. Train

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3vl_2b/base.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — split-checkpoint root from step 1.
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length.
- `--train.optimizer.lr` — learning rate.
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.

---

## 4. Resume

Each save writes per-module DCP shards plus a `trainer_state.pt` (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3vl_2b/base.yaml \
  --train.checkpoint.load_path outputs/qwen3vl_2b_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

`tasks/omni/infer_omni.py` runs the `vision_understanding` generation graph. Point
`--infer.model_path` at a **split-checkpoint root** holding `qwen3vl_vision/`,
`qwen3vl_text_encoder/` and `qwen3vl_llm/`. The step-1 converter output already
has this layout, so you can infer directly:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3vl_2b/base.yaml \
  --infer.infer_type vision_understanding \
  --infer.model_path /mnt/hdfs/veomni/models/seed_omni/Qwen3-VL-2B-Instruct \
  --infer.image /path/to/image.jpg \
  --infer.prompt "What is in this image?" \
  --infer.output_dir qwen3vl_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

To infer from a **trained** checkpoint, assemble a flat root from the per-module
`hf_ckpt/` subfolders (same pattern as Janus):

```bash
STEP=outputs/qwen3vl_2b_omni_sft/checkpoints/global_step_500
ASM=outputs/qwen3vl_2b_omni_sft/infer_ckpt/global_step_500
mkdir -p "$ASM"
for m in qwen3vl_vision qwen3vl_text_encoder qwen3vl_llm; do
  ln -sfn "$(realpath "$STEP/$m/hf_ckpt")" "$ASM/$m"
done
# then: --infer.model_path "$ASM"
```

---

## 6. Alignment with the original HF model

The split graph must reproduce the monolithic `Qwen3VLForConditionalGeneration`
forward. For one image + text prompt, the backbone hidden states match the
reference model **exactly** (max abs diff `0.0`) — confirming the weight split,
image processing + vision encode, DeepStack features, ChatML tokenization, M-RoPE
position ids, DeepStack injection, and the backbone splice are all consistent.

A self-contained check builds the same sequence through both paths:

```text
ref:  Qwen3VLForConditionalGeneration.model(input_ids, pixel_values, image_grid_thw)
v2:   qwen3vl_vision → qwen3vl_text_encoder.encode → qwen3vl_llm
-> max abs diff 0.0 over last_hidden_state  (RESULT: ALIGNED)
```

> **Scope**: this recipe covers image understanding (I2T). Video inputs
> (`<|video_pad|>`) and Ulysses sequence parallelism are not wired into the V2
> path yet — the backbone raises on `sp_enabled`.

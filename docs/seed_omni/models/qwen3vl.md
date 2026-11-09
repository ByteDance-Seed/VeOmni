# Qwen3-VL-2B-Instruct (SeedOmni)

End-to-end recipe for training and inferring **Qwen3-VL-2B-Instruct** as a
SeedOmni graph model. This is a **vision-language** omni model: the monolithic
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
- **DeepStack** — `deepstack_visual_indexes` (`[5, 11, 17]`) selects *ViT* layers;
  the extra merger per selected layer produces one feature set each, and those
  are added into the **first `len(deepstack_visual_indexes)` decoder layers**
  (0, 1, 2 here), not into decoder layers with the same numbers. The vision
  module emits them on `item.meta["deepstack"]`; the backbone threads them into
  `Qwen3VLTextModel` as `deepstack_visual_embeds` + `visual_pos_masks`.

`model.model_path` and the `sources` entries in `data.yaml` are **placeholders**:
replace each with an absolute path (or an `hdfs://` URL) this job can read before
launching. Both are resolved by plain filesystem checks — `model_path` by looking
for `config.json`, a dataset source by `os.path.isdir` / `isfile` — so a bare
name fails instead of being expanded against some storage root. The checkpoint
may be the upstream HF root or a split one (step 1); datasets are your own copies.

Config dir: `configs/seed_omni/Qwen/qwen3vl/`

Layout follows [the shared config layout](../usage/training_and_inference.md#1-config-layout):
`data.yaml` and `infer/` sit at the model root because both tasks use them.

| File | Role |
|------|------|
| `train/base.yaml` | Launcher: `model` (incl. `model.accelerator`) / `data` (incl. `mm_configs`) / `train` + `infer` block. |
| `train/modules_train.yaml` | Per-module training overrides (`qwen3vl_vision` / `qwen3vl_text_encoder` / `qwen3vl_llm`). Add `--model.accelerator.ulysses_size N` for Ulysses SP (see [§3.1](#31-sequence-parallelism-ulysses)). |
| `train/graph_train.yaml` | Training DAG — flat edge list (`{qwen3vl_vision, qwen3vl_text_encoder.encode} → qwen3vl_llm → qwen3vl_text_encoder.decode → end`). |
| `data.yaml` | Weighted multisource data list (ShareGPT4V images + LLaVA-Video). Shared by both tasks. |
| `infer/graph_infer.yaml` | Image/video-understanding (I2T / VQA) generation graph (`model.model_config.infer_type: vision_understanding`). Shared by both tasks. |
| `packed/base.yaml` | Launcher for the packed training graph (optional, see [§3.2](#32-packed-graph-optional)). |
| `packed/modules_train.yaml` | Same module overrides plus `processor_config: {packed_preprocess: true}` on the text encoder. |
| `packed/graph_train.yaml` | Packed DAG (`pack_encode → pack_encode → pack_forward → pack_decode`). |

The directory name carries no size: one `qwen3vl/` recipe serves every Qwen3-VL
checkpoint, and `model.model_path` alone picks which one.

---

## 1. Checkpoint

Qwen3-VL registers an HF layout (`modules/qwen3vl/hf_layout.py`), so
`model.model_path` can be the **upstream checkpoint itself** — `base.yaml` already
points at one, and no offline split is needed. The layout routes
`model.visual.*` to `qwen3vl_vision` (with both its image and video processors),
`model.language_model.embed_tokens.*` (and `lm_head.*`) plus the tokenizer to
`qwen3vl_text_encoder`, and the rest of `model.language_model.*` to
`qwen3vl_llm`. HF exports are then written in the upstream layout (see
[Load an upstream checkpoint directly](../usage/training_and_inference.md#21-load-an-upstream-checkpoint-directly)).

The offline split still works, through the same key map, and is what a
cross-family composed model needs:

```bash
python scripts/seed_omni/convert_model.py \
  --model_path /abs/path/to/Qwen3-VL-2B-Instruct \
  --output_dir /abs/path/to/Qwen3-VL-2B-Instruct-split
```

Either root works as `model.model_path`; a split root holds `qwen3vl_vision/`,
`qwen3vl_text_encoder/` and `qwen3vl_llm/`.
(Qwen3-VL-2B has `tie_word_embeddings=True`, so `qwen3vl_text_encoder` stores only
`embed_tokens` and the decode head reuses that table via `embed_tokens.project`.
The upstream file still stores a copy as `lm_head.weight`; the load skips it and
the export rewrites it from the trained embedding.)

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`); `sharegpt4v_cap_100k` maps each row to an
`image` + `text` user turn plus an assistant caption, `llava_video` maps to a
`video` + `text` turn (the video is decoded into a `VideoInputs` carrier), and
`tulu-3-sft-mixture` adds pure-text turns so text ability isn't lost.

```yaml
# configs/seed_omni/Qwen/qwen3vl/data.yaml
sources:
  - sharegpt4v_cap_100k
  - LLaVA-Video-178K_0_30_s_academic_v0_1
  - tulu-3-sft-mixture
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

The shipped `sources` are placeholders. `get_data_files` accepts a local
directory, a local file, or an `hdfs://` URL — nothing else — so replace every
entry with an absolute path your job can read; a bare name fails with
`FileNotFoundError` before the first batch. `names` are preprocessor keys, not
paths, so leave them alone.

Video decoding is driven by `data.mm_configs` in `base.yaml`
(`use_audio_in_video: false`, `fps`, `min_frames` / `max_frames`) — Qwen3-VL has
no audio modality. The on-disk row schema is documented in
[Data Format](../usage/data_format.md).

### Video prompts and timestamps

Qwen3-VL does not just drop a clip's frames into one vision block: upstream emits
`<t seconds><|vision_start|>…<|vision_end|>` **per temporal patch**, and the
backbone's M-RoPE restarts at each one. `Qwen3VLChatTemplate` does the same,
taking each stamp from the item's `VideoMetadata` via `frame_timestamps()` — frame
index over source fps, then the patch's mid-time.

> **Limitation: a real clip only matches the pretrained format when its decode
> metadata reaches the template.** The stamps are real only if the item still
> carries `meta["video_metadata"]` at templating time. The decoder
> (`fetch_videos` → `VideoMetadata`) always writes it, and the vision preprocessor
> preserves it, so the shipped `data.yaml` path is fine. A source that supplies
> pre-sampled frames with no `fps` — or any caller that builds
> `ConversationItem(type="video", …)` without that metadata — falls back to
> synthetic stamps (`0, patch, 2·patch, …`). The token *layout* is then still
> correct, so training runs and the M-RoPE positions stay consistent, but the
> prompt describes a different clip than the model was trained on. Check
> `item.meta["video_metadata"]` when a video sample's loss looks off.

---

## 3. Train

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3vl/train/base.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — split-checkpoint root from step 1.
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length (the `dyn_bsz` token budget unit when `train.dyn_bsz` is on).
- `--model.optimizer.lr` — learning rate (global default; override per module in `train/modules_train.yaml`).
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.

### 3.1 Sequence parallelism (Ulysses)

Set the SP size once with `--model.accelerator.ulysses_size N`; every module (vision
tower, text encoder and LLM backbone) inherits it, and there is no dedicated SP
config (design in [Sequence Parallelism](../distributed/sequence_parallel.md)). The dataloader replicates each DP shard
across the SP group; each module slices to its `1/sp` chunk, runs one forward, and
all-gathers the output back — the in-model backbone `qwen3vl_llm` shards its
DeepStack visual embeds, `visual_pos_masks` and 3-row M-RoPE `position_ids` too:

```bash
NPROC_PER_NODE=4 bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3vl/train/base.yaml \
  --model.accelerator.ulysses_size 4 \
  --train.global_batch_size 16 --train.micro_batch_size 4
```

- Every module inherits the global `ulysses_size`; do not set a per-module
  `ulysses_size` (nothing validates that modules agree).
- `world % ulysses_size == 0`; a larger `micro_batch_size` gives longer packed
  sequences to slice across SP ranks.
- If your environment exports `TORCH_DISTRIBUTED_DEBUG=DETAIL`, **unset it** — its
  `_ProcessGroupWrapper` lacks the coalesced all-gather that FSDP2's tied-head
  `full_tensor()` needs, which crashes any FSDP2 run (SP or not).

### 3.2 Packed graph (optional)

`packed/base.yaml` is a second training entry point, not a replacement for
`train/base.yaml`. It builds one flat token sequence per sample on the CPU — tokens,
M-RoPE positions and the visual mask — so each node only mask-selects its own
embeddings and `masked_scatter`s them back, instead of walking the conversation
carrier item by item.

It works because these three modules are one family over one sequence: the text
encoder owns the text rows, the vision tower owns the visual rows, and the
backbone reads all of them at one shared width — the tower's own 1024-wide
features are projected to it by its merger before the scatter. A graph whose
modules write rows of a different width into that tensor, or that need a different
sequence length or tokenisation per module, cannot be packed — see
[Adding a model §4.1](../usage/adding_a_model.md#41-packing-optional) for the
conditions.

```bash
bash train.sh tasks/omni/train_omni.py configs/seed_omni/Qwen/qwen3vl/packed/base.yaml
```

The conversation graph stays the reference path: it is what inference uses, it is
what the packed graph is meant to reproduce, and it is what you should ship first.
The parity gate is `test_qwen3vl_accel_align.py`: it runs one step of each graph
and compares the loss, every parameter gradient and the AdamW update that
follows. `test_qwen3vl_packing.py` covers only the packer's own outputs — masks,
cu_seqlens, visual placement.

What packing buys here is **not** padding removal — the conversation path does not
pad either; `pack_llm_conversations_for_forward` already concatenates every
sample's embed segments into one varlen sequence. The difference is *where the
per-item work happens*:

- **Packed.** The CPU preprocessor does template / tokenize / patchify in the
  dataloader worker and emits one `packed_input_ids` / `packed_position_ids` /
  `packed_cu_seqlens` set. `batch_to_device` moves those top-level tensors in one
  transfer; `pre_forward` then only builds the varlen args. Padding is absent by
  construction, so no attention mask travels with the batch — the varlen
  boundaries are the mask.
- **Conversation.** The carrier's tensors are built per item in the worker, and
  `batch_to_device` deliberately leaves them there — "tensors a module keeps
  inside its conversation items stay where the preprocessor built them; placing
  those is the module's own `pre_forward` job". So the step pays one
  `to(device)` per `ConversationItem` (18–48 of them per micro-batch) plus a
  `torch.cat` over all of them, on the critical path.

Measured on one A100-SXM4-80GB against the real
`/mnt/hdfs/veomni/models/Qwen/Qwen3-VL-2B-Instruct` (28 layers, hidden 2048), each
sample 4 images of 144 merged rows plus two ~400-token text turns — timing the two
`batch_to_device` shapes directly, not a full training run:

| Samples / micro-batch | items | rows | per-item placement (conversation) | one-shot (packed) | backbone forward | saved |
|---|---|---|---|---|---|---|
| 3 | 18 | 4 128 | 1.26 ms | 0.05 ms | 223 ms | **0.54%** |
| 8 | 48 | 11 008 | 4.55 ms | 0.08 ms | 511 ms | **0.88%** |

So packing is worth **under 1% of a step** at this size, and it is the only thing
it removes. The saving scales with **item count, not token count**: a micro-batch
of many short samples (more items per row) gains proportionally more than a few
long ones.

On Qwen3-Omni-30B-A3B the same fixed cost was −1.01% step time at a
VRAM-saturating batch and −0.44% at mbs=2, against `fsdp_scope: model` alone
recovering 0.47% / 0.21% of that gap, and 0.18% (inside its own jitter) in the
speech round. See [Adding a model §4.1](../usage/adding_a_model.md#41-packing-optional)
for those numbers.

Both training graphs report to the same metric meter, so per-module token counts
and FLOPs are available on the packed path as well as the per-sample one. Each
module reads its own boundaries there: vision recovers merged tokens per real
visual from `image_grid_thw[visual_is_real]`, while the text encoder and the
backbone take the row lengths from `packed_cu_seqlens` — so the counts align with
`ds_idx` and multi-source token counts are attributed per source. The sample
count also comes from `packed_cu_seqlens`, which `OmniEnvironMeter.add` counts
rather than assuming one sample per row, and `get_length_fn` knows the packed keys
so `dyn_bsz` can size packed samples too.

---

## 4. Resume

Each save writes per-module DCP shards plus the per-rank job state (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3vl/train/base.yaml \
  --train.checkpoint.load_path outputs/qwen3vl_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

SeedOmni has **two inference entry points**:

| Script | Checkpoint | Runtime | When to use |
|--------|------------|---------|-------------|
| `tasks/omni/infer_omni.py` | Upstream HF root or split SeedOmni root | VeOmni runtime — `OmniModelRuntimeArguments` + generation FSM; optional per-module FSDP | Distributed inference or launcher-YAML overrides |
| `tasks/omni/infer_omni_native.py` | Upstream HF root or split SeedOmni root | ``OmniModel.from_pretrained`` — all modules eager, simple process + generate | Single-process smoke tests and quick experiments |

### 5.1 VeOmni runtime (`infer_omni.py`)

`tasks/omni/infer_omni.py` runs the `vision_understanding` generation graph.
`model.model_path` is either the **upstream Qwen3-VL root** (what `base.yaml`
points at, loaded through the HF layout) or a split root holding
`qwen3vl_vision/`, `qwen3vl_text_encoder/` and `qwen3vl_llm/`. `base.yaml` already
points at an upstream root, so you can infer directly:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3vl/train/base.yaml \
  --model.model_config.infer_type vision_understanding \
  --infer.prompt "What is in this image?" \
  --infer.images /path/to/image.jpg \
  --infer.output_dir qwen3vl_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

Video works the same way through `--infer.videos`; the same `data.mm_configs`
decode knobs apply, and this launcher is the only one of the two entry points
that accepts video (see §5.2):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3vl/train/base.yaml \
  --model.model_config.infer_type vision_understanding \
  --infer.prompt "Describe this clip." \
  --infer.videos /path/to/clip.mp4 \
  --infer.mm_configs.fps 2.0 \
  --infer.mm_configs.max_frames 8
```

Qwen3-VL has no audio tower — `vision/processing.py` decodes only a clip's frames
and skips its waveform — so keep `--infer.mm_configs.use_audio_in_video false`
(the default) and never also pass `--infer.audios`.

To infer from a **trained** checkpoint, there are two shapes depending on what
`model.model_path` was. Trained from the **upstream root**, the export is one
Qwen3-VL checkpoint under `<step_dir>/hf_ckpt/`, so infer from it directly with
`--model.model_path <step_dir>/hf_ckpt` (it loads in `transformers` too). Trained
from a **split root**, each module exports its own `hf_ckpt/` subfolder; assemble
a flat root from them:

```bash
STEP=outputs/qwen3vl_omni_sft/checkpoints/global_step_500
ASM=outputs/qwen3vl_omni_sft/infer_ckpt/global_step_500
mkdir -p "$ASM"
for m in qwen3vl_vision qwen3vl_text_encoder qwen3vl_llm; do
  ln -sfn "$(realpath "$STEP/hf_ckpt/$m")" "$ASM/$m"
done
# then: --model.model_path "$ASM"
```

### 5.2 Native eager (`infer_omni_native.py`)

This is the native HF launch described in
[Training and Inference](../usage/training_and_inference.md#5-inference)
(`OmniModel.from_pretrained`, no VeOmni runtime); §5.1 is the VeOmni Inferencer.

Single-process load — preprocess the request, then ``model.generate``:

```python
from veomni.models.seed_omni import OmniModel, OmniProcessor

model = OmniModel.from_pretrained(checkpoint_root, device_map="auto")
processor = OmniProcessor.from_pretrained(checkpoint_root)
inputs = processor(text="Describe this image.", images=["/path/to/image.jpg"])
model.reset()
model.generate(inputs, generation_kwargs={"max_new_tokens": 128})
```

CLI equivalent:

```bash
python tasks/omni/infer_omni_native.py \
  --model_path /abs/path/to/Qwen3-VL-2B-Instruct \
  --infer_type vision_understanding \
  --prompt "Describe this image." \
  --image /path/to/image.jpg \
  --max_new_tokens 128
```

This entry point exposes `--image` only, so a **video** request has to go through
§5.1 (`--infer.videos`).

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
split: qwen3vl_vision → qwen3vl_text_encoder.encode → qwen3vl_llm
-> max abs diff 0.0 over last_hidden_state  (RESULT: ALIGNED)
```

> **Scope**: this check covers image understanding (I2T). Sequence parallelism
> runs at one shared SP size across the vision tower, text encoder and backbone
> (see [§3.1](#31-sequence-parallelism-ulysses)).

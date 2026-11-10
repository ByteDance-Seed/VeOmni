# Qwen3-0.6B (SeedOmni)

End-to-end recipe for training and inferring **Qwen3-0.6B** as a SeedOmni
graph model. This is the minimal **text-only** omni model: the monolithic
`Qwen3ForCausalLM` is split into two OmniModules wired as
`token_encode → qwen3_llm → token_decode`.

| Module | Holds | Role |
|--------|-------|------|
| `qwen3_text_encoder` | `embed_tokens` (+ `lm_head` if untied) + tokenizer | token ↔ embedding, CE loss head, ChatML template |
| `qwen3_llm` | decoder backbone (no embed / no head) | `inputs_embeds → hidden_states` |

`model.model_path` and the `sources` entry in `data.yaml` are **placeholders**:
replace each with an absolute path (or an `hdfs://` URL) this job can read before
launching. `data.train_path` in `base.yaml` also still resolves as a file path —
the loader checks `os.path.isdir` / `isfile`, so a bare name fails rather than
being expanded against a storage root.

Config dir: `configs/seed_omni/Qwen/qwen3/`.

The layout follows [the shared config layout](../usage/training_and_inference.md#1-config-layout).
Both training and inference take the **same** `base.yaml`.

| File | Role |
|------|------|
| `qwen3/train/base.yaml` | Launcher: model paths, `model.accelerator`, data, train, and the `infer` block. |
| `qwen3/train/modules_train.yaml` | Per-module training overrides. Add `--model.accelerator.ulysses_size N` for Ulysses SP (see [§3.1](#31-sequence-parallelism-ulysses)). |
| `qwen3/train/graph_train.yaml` | Training DAG (`qwen3_text_encoder → qwen3_llm → qwen3_text_encoder.decode`). |
| `qwen3/data.yaml` | Weighted multisource data list (Tulu-3 SFT mixture). Shared by every task. |
| `qwen3/infer/graph_infer.yaml` | Text chat generation graph (mapped under `model.model_config.infer_graph.infer_text`). |
| `qwen3/visual_instruction_tuning/` | Image-understanding recipe that composes the Qwen3 modules with the Qwen3-VL ViT — see [§7](#7-visual-instruction-tuning-composing-the-qwen3-vl-vit). |

The directory name carries no size: the modules, graphs and kernels are the same
at every Qwen3 size, and `model.model_path` alone picks the checkpoint. A recipe
that genuinely needs its own plumbing (Qwen3-MoE, visual instruction tuning) gets
its own task folder beside `train/`.

---

## 1. Checkpoint

Qwen3 registers an HF layout (`modules/qwen3/hf_layout.py`), so `model.model_path`
can be the **upstream checkpoint itself** — `base.yaml` already points at one. The
layout routes `model.embed_tokens.*`
(and `lm_head.*`) plus the tokenizer to `qwen3_text_encoder`, and the rest of
`model.*` to `qwen3_llm`; the graphs default to the YAML in this config dir. HF
exports are then written in the upstream layout (see
[Load an upstream checkpoint directly](../usage/training_and_inference.md#21-load-an-upstream-checkpoint-directly)).

The offline split still works, through the same layout, and is what a
cross-family composed model needs:

```bash
python scripts/seed_omni/convert_model.py \
  --model_path /abs/path/to/Qwen3-0.6B \
  --output_dir /abs/path/to/seed_omni/Qwen3-0.6B
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
# configs/seed_omni/Qwen/qwen3/data.yaml
sources:
  - tulu-3-sft-mixture   # placeholder: replace with an absolute path
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
  configs/seed_omni/Qwen/qwen3/train/base.yaml
```

Key knobs (override on the CLI):

- `--model.model_path` — the upstream Qwen3 root, or a split root from step 1.
- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length (`dyn_bsz` token budget unit: `micro_batch_size * max_seq_len`).
- `--model.optimizer.lr` — learning rate (global default; override per module in `train/modules_train.yaml`).
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.

Quick 1-node smoke run (no wandb, tiny step budget):

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3/train/base.yaml \
  --model.model_path /abs/path/to/Qwen3-0.6B \
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
  configs/seed_omni/Qwen/qwen3/train/base.yaml \
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
  configs/seed_omni/Qwen/qwen3/train/base.yaml \
  --train.checkpoint.load_path outputs/qwen3_omni_sft/checkpoints/global_step_500
```

---

## 5. Inference

`tasks/omni/infer_omni.py` runs the `infer_text` generation graph (the default
and only `model.model_config.infer_type` for Qwen3). `--model.model_path` is the
upstream Qwen3 root (what `base.yaml` points at) or a split root holding
`qwen3_text_encoder/` and `qwen3_llm/`:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3/train/base.yaml \
  --model.model_config.infer_type infer_text \
  --model.model_path /abs/path/to/Qwen3-0.6B \
  --infer.prompt "What is 2+2?" \
  --infer.output_dir qwen3_out \
  --infer.generation_kwargs.max_new_tokens 1024
# -> "<think> ... the sum of 2 plus 2, which is 4 ..." (coherent chat output)
```

`tasks/omni/infer_omni_native.py` is the other entry point — a single-process
`OmniModel.from_pretrained` + `generate` load with no VeOmni runtime. It takes
`--model_path` / `--prompt` / `--max_new_tokens` and writes `reply.txt`:

```bash
python tasks/omni/infer_omni_native.py \
  --model_path /abs/path/to/Qwen3-0.6B \
  --prompt "What is 2+2?" \
  --output_dir qwen3_out \
  --max_new_tokens 1024
```

See [Training and Inference §5](../usage/training_and_inference.md#5-inference)
for how the two entry points differ; `--infer_type` defaults to the checkpoint's
own `infer_text`, so Qwen3 needs no flag for it.

A run trained from the upstream root exports `hf_ckpt/` as one Qwen3 checkpoint,
so infer from it directly with `--model.model_path <step_dir>/hf_ckpt` (it loads
in `transformers` too). A run trained from a split root exports one `hf_ckpt/`
subfolder per module; assemble a flat root from them:

```bash
STEP=outputs/qwen3_omni_sft/checkpoints/global_step_500
ASM=outputs/qwen3_omni_sft/infer_ckpt/global_step_500
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
  --model.model_path /abs/path/to/Qwen3-0.6B \
  --data.train_path /abs/path/to/tulu-3-sft-mixture \
  --data.max_seq_len 2048 --train.global_batch_size 8 --train.micro_batch_size 1 \
  --train.max_steps 15 --train.wandb.enable false
```

Two intentional input differences keep a small residual gap even on matched
data:

1. **Assistant-prefix mask.** The classic `chatml` template labels the whole
   assistant block (role prefix + `<|im_end|>`); the omni
   `qwen3_text_encoder` template masks the `<|im_start|>assistant\n` prefix.
2. **Tulu system turns.** For Tulu rows that open with a system message,
   SeedOmni's `tulu_3_sft_mixture_preprocess` folds that system text into the
   first user turn (`<system>…</system>\n…`), while the classic ChatML
   template keeps a separate system turn. The two trainers therefore see
   different token sequences on those rows.

For a parity comparison, filter to Tulu rows **without** a system message (or
otherwise feed both trainers the same already-normalized conversations) and
compare losses on those matched inputs and labels.

---

## 7. Visual instruction tuning: composing the Qwen3-VL ViT

This recipe turns the **text-only** Qwen3-0.6B into an image-understanding model by
composing it with the **Qwen3-VL vision tower**. It is the cross-family worked
example for [Composing modules across families](../usage/adding_a_model.md#42-composing-modules-across-families):
three modules from two checkpoints, wired by YAML with **no new Python class**.

| Module | Source family | `model_config` override | Trainable |
|--------|---------------|------------------------|-----------|
| `qwen3vl_vision` (ViT + patch merger) | Qwen3-VL-2B-Instruct | `out_hidden_size: 1024`, `disable_deepstack: true`, `freeze: true` | **patch merger only** |
| `qwen3_text_encoder` (tied wte + lm_head + tokenizer) | Qwen3-0.6B | `enable_image: true` | **vision special-token rows only** |
| `qwen3_llm` (decoder backbone) | Qwen3-0.6B | — | **yes** |

Config dir: `configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/`.

| File | Role |
|------|------|
| `base.yaml` | Launcher: the Qwen3-0.6B split root, accelerator, data, train, `infer`. |
| `modules_train.yaml` | The three overrides above plus the ViT's own `model_path`. |
| `graph_train.yaml` | `{qwen3vl_vision, qwen3_text_encoder.encode} → qwen3_llm → qwen3_text_encoder.decode → end`. |
| `graph_infer.yaml` | Image-understanding FSM (prompt encode with the ViT, then the ChatML AR loop). |
| `data.yaml` | ShareGPT4V captions (image + text). |

### 7.1 Retarget the patch merger instead of adding a projector

Qwen3-VL's ViT emits **2048-d** tokens (it was pretrained for the 2B LLM), while
Qwen3-0.6B's hidden size is **1024**. Rather than bolt a separate `Linear` on top,
`out_hidden_size: 1024` retargets the ViT's own patch merger: it already ends in
`linear_fc2: Linear(merger_hidden → out_hidden_size)`, so that layer *is* the bridge
into the LLM embedding space.

A stock checkpoint's `linear_fc2` then has the wrong shape, so the vision module's
checkpoint converter (`_MergerProjectionConverter`) drops it at load and the weight
loader re-initialises it. `merger.norm` and `merger.linear_fc1` still load from the
checkpoint — no build-time surgery.

`freeze: true` then freezes the ViT blocks and keeps only the merger trainable, and
`disable_deepstack: true` zeroes `deepstack_visual_indexes` because the plain
`qwen3_llm` backbone has no DeepStack injection path. The ViT consequently produces
only merged image tokens, packed into the sequence like any other embed segment.

### 7.2 Train only the vision special-token rows

Qwen3's tokenizer already **reserves** the vision special tokens
(`<|vision_start|>`, `<|vision_end|>`, `<|image_pad|>`, all below the 151936
embedding rows), so there is no vocabulary expansion to do — adding duplicates would
be wrong, because the tokenizer still maps to the reserved ids.

`enable_image: true` on `qwen3_text_encoder` switches two things:

- the CPU preprocessor uses the **Qwen3-VL chat template** (media wrapped in
  `<|vision_start|> … <|vision_end|>`, one block per video frame) instead of the
  text-only ChatML template;
- `freeze_model()` keeps the tied embedding's `requires_grad=True` but registers a
  **per-row gradient mask** that zeroes every row except the vision special tokens.
  The ids come from the module's own tokenizer (`convert_tokens_to_ids`) — a user
  cannot know them, the module can. With `enable_image: false` the table is fully
  trainable, as in the text-only recipe.

The mask is attached from a forward pre-hook on `embed_tokens`, after FSDP2 has
unsharded it, sliced to this rank's vocab rows when an `emb` plan splits the table.
The tied `embed_tokens.project` reads the same parameter, so its gradient is masked
too, provided `encode` runs before the first `decode` (it does in this graph). It
therefore holds under `ddp`, `fsdp2` and `emb` alike.

AdamW's decoupled weight decay (`p *= 1 - lr * wd`) does not go through the gradient,
so the mask alone would still let it shrink the frozen rows — and with them the tied
output head. The module's `configure_optimizer` takes the table's weight decay over:
it sets that param group's `weight_decay` to 0 and applies the module's configured
value (the model-level `optimizer.weight_decay` unless the module overrides it) to the
vision rows alone, from an optimizer step pre-hook, before the update as AdamW would.
`ModuleRuntime` calls `configure_optimizer(optimizer)` right after building a module's
optimizer, before the lr scheduler and any checkpoint load.

> The special-token rows load verbatim from Qwen3-0.6B (an untrained reserved stub).
> They start training from there; for a better starting point, mean-init them before
> training.

### 7.3 Checkpoints: two standard converts, no combine step

There is no dual-source script and no combined directory. Convert each upstream model
once, then let the launcher compose them. `model.model_path` is the Qwen3-0.6B split
root (`qwen3_text_encoder/`, `qwen3_llm/` + tokenizer), and `modules_train.yaml`
points `qwen3vl_vision` at the Qwen3-VL root's `qwen3vl_vision/` subfolder (ViT +
image/video processors):

```bash
python scripts/seed_omni/convert_model.py \
  --model_path Qwen3-0.6B \
  --output_dir Qwen3-0.6B-split

python scripts/seed_omni/convert_model.py \
  --model_path Qwen3-VL-2B-Instruct \
  --output_dir Qwen3-VL-2B-Instruct-split
```

At train time the `out_hidden_size` override retargets the merger and its mismatched
`linear_fc2` is re-initialised; `disable_deepstack` drops the unused DeepStack
mergers.

Per-module `model_path`s are resolved **relative to `model.model_path`** unless they
are absolute. Making `qwen3vl_vision` point at a sibling split root is why its
`model_path` is written as an absolute path in the recipe.

A composed run cannot export one flat HF checkpoint — the two sources have different
layouts — so `save_hf_weights` writes one `hf_ckpt/<module>/` per module.

### 7.4 Train on ShareGPT4V

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/base.yaml
```

Trainable parameters are `qwen3vl_vision.visual.merger.*`, the whole `qwen3_llm`
backbone, and the masked text-encoder embedding (only the vision special-token rows
receive gradient; the ViT blocks are frozen).

**Sequence parallelism**: the ViT, the text encoder and the LLM backbone all run at
the shared SP size. Add `--model.accelerator.ulysses_size N` to the launcher; the
dataloader replicates each DP shard across the SP group, and each module slices to
its `1/sp` chunk, runs one forward, and all-gathers the output back:

```bash
NPROC_PER_NODE=4 bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/base.yaml \
  --model.accelerator.ulysses_size 4 \
  --train.global_batch_size 16 --train.micro_batch_size 4
```

Same caveats as [§3.1](#31-sequence-parallelism-ulysses): one shared `ulysses_size`
for the whole graph, `world % sp_size == 0`, and `TORCH_DISTRIBUTED_DEBUG` unset on
FSDP2 runs.

### 7.5 Inference

Run straight from `base.yaml`, this pairs the base Qwen3-0.6B root with the raw
Qwen3-VL ViT whose merger `linear_fc2` is re-initialised for `out_hidden_size: 1024`,
so it only checks the wiring — the reply is meaningless until trained. Use the
trained-checkpoint command below for real outputs.

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/base.yaml \
  --infer.images /path/to/image.jpg \
  --infer.prompt "What is in this image?" \
  --infer.output_dir qwen3_vit_out
```

To infer from a **trained** checkpoint (per-module weights live under
`<step>/hf_ckpt/<module>/`), point `--model.model_path` at the checkpoint step
directory and override each module's `model_path` **relative to that root** — do NOT
repeat the `--model.model_path` prefix. Per-module override paths are joined under
`--model.model_path` unless they are absolute (start with `/`); passing a
cwd-relative full path double-joins it and fails with a cryptic
`HFValidationError: Repo id must be in the form ...`.

```bash
STEP=outputs/qwen3_visual_instruction_tuning/checkpoints/global_step_2000
python tasks/omni/infer_omni.py \
  configs/seed_omni/Qwen/qwen3/visual_instruction_tuning/base.yaml \
  --model.model_path "$STEP" \
  --infer.images /path/to/image.jpg \
  --infer.prompt "What is in this image?" \
  --infer.output_dir qwen3_vit_out \
  --model.model_config.modules.qwen3vl_vision.model_path hf_ckpt/qwen3vl_vision \
  --model.model_config.modules.qwen3_text_encoder.model_path hf_ckpt/qwen3_text_encoder \
  --model.model_config.modules.qwen3_llm.model_path hf_ckpt/qwen3_llm
```

> **Scope**: this is a deliberately minimal setup — frozen ViT blocks, a retargeted
> patch merger, the vision special-token rows and the LLM. It exercises the full
> image pipeline and trains, but real image-understanding quality needs more
> capacity (unfreeze the ViT blocks, or use an aligned ViT). DeepStack is disabled
> because the plain `qwen3_llm` backbone cannot consume those features.

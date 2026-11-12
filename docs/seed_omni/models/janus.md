# Janus-1.3B (SeedOmni)

End-to-end recipe for training and inferring **Janus-1.3B** as a SeedOmni
graph model: a unified understanding (image→text) + generation (text→image)
model whose SigLIP / VQVAE / LLaMA backbone are wired as separate OmniModules.

Paths below are written as the placeholder `Janus-1.3B` for the upstream
HuggingFace checkpoint — replace it with an absolute path (or `hdfs://` URL) this
job can read. The same applies to `data.yaml`'s `sources`.

## Modules

| Module (`model_type`) | HF base | Role |
|-----------------------|---------|------|
| `janus_siglip` | `SiglipVisionModel` | encode **understanding** images to patch embeddings |
| `janus_vqvae` | `JanusVQVAE` + generation heads | encode **generation** images / decode the VQ grid to pixels |
| `janus_text_encoder` | LLaMA `wte` + `lm_head` | chat template, token embedding, LM head, `<boi>` / `<eoi>` emission |
| `janus_llama` | patched `LlamaModel` | backbone (no `wte`, no `lm_head`) |

The LLM is split into `text_encoder` + `llama` because the word-token embedding
and the LM head are vocabulary-dependent: they mirror the discrete-image VQ codec
on the text side. The split lets the graph treat text and image symmetrically
(both have `encode` / `decode` nodes) and lets Janus own the boundary-token logic
without the framework knowing about it. The training and generation graphs are
walked through in [Architecture](../design/architecture.md#3-training-flow).

## Config files

Config dir: `configs/seed_omni/Janus/janus_1.3b/` (layout explained in
[Training and Inference](../usage/training_and_inference.md#1-config-layout)).
Training and inference take the **same** `base.yaml`; its `model.*` block drives
the trainer and its `infer.*` block drives the inferencer.

| File | Role |
|------|------|
| `train/base.yaml` | Launcher: model paths, `model.accelerator`, data, train, and the `infer` block. References the module/graph files below. |
| `train/modules_train.yaml` | Per-module **training** overrides (`model` / `train` / `accelerator` per module). `janus_text_encoder` carries an embedding-parallel `emb` extra-parallel block (see below). Add `--model.accelerator.ulysses_size N` for Ulysses SP (see [Sequence parallelism](#sequence-parallelism-ulysses)). |
| `train/graph_train.yaml` | Training DAG — the file *is* the flat edge list. |
| `train/base_model_fsdp.yaml` | Conversation DAG with `fsdp_scope: model` (one FSDP tree over OmniModel). |
| `../data.yaml` | Weighted multisource data list (ImageNet + ShareGPT4V). |
| `infer/modules_infer_fsdp.yaml` | Per-module **inference** overrides — distributed: `janus_text_encoder` vocab-parallel `emb` (`fsdp2` + `emb`), `janus_llama` `fsdp2`, `janus_vqvae` `ddp`, `janus_siglip` eager. Select it with `--model.model_config.modules`. |
| `infer/modules_infer_eager.yaml` | Per-module **inference** overrides — every module `eager`; the file pins `janus_siglip` explicitly and `for_inference` supplies the `eager` default to the rest, so the whole run is a single-process replica. |
| `infer/graph_infer_und.yaml` / `infer/graph_infer_gen.yaml` / `infer/graph_infer_interleave.yaml` | Per-scenario generation graphs (mapped under `model.model_config.infer_graph`). |

---

## 1. Point at the checkpoint

Janus registers an [`OmniHFLayout`](../usage/hf_layout.md) under `model_type:
janus`, so `model.model_path` can name the upstream HF checkpoint directly — no
offline convert, and `save_hf_weights` writes back one checkpoint in that
upstream layout, which loads with `transformers` or with `model.model_path`
again.

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.model_path Janus-1.3B
```

The layout cuts the checkpoint into one sub-checkpoint per OmniModule (SigLIP
encoder, VQVAE codec + generation head, LLaMA backbone) at the upstream key
prefixes, so every source tensor lands on exactly one module.

To materialise the modules on disk instead, run the family converter — the
layout gives both paths, and on a checkpoint whose tie flags agree they hold
identical module weights:

```bash
python scripts/seed_omni/convert_model.py \
  --model_path Janus-1.3B \
  --output_dir Janus-1.3B-split
```

The `output_dir` is then a split-checkpoint root (a folder of per-module
subfolders, not a single HF checkpoint). The converter is also the only path for
a DeepSeek ``multi_modality`` source, which it rewrites to the HF layout first,
and the only one that records the family's generation defaults (`guidance_scale`
among them) and the tokenizer-derived `gen_image_token_id`.

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`).

```yaml
# configs/seed_omni/Janus/data.yaml
sources:
  - imagenet1k_train      # text -> image (T2I)
  - tulu-3-sft-mixture    # text-only SFT
  - sharegpt4v_cap_100k   # image -> text (I2T)
names:
  - imagenet1k
  - tulu-3-sft-mixture
  - sharegpt4v_cap_100k
schedule:
  - { schedule_type: const, weights: [0.5, 0.2, 0.3] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

Sources have heterogeneous schemas, so we use the VeOmni weighted multisource
sampler (`multisource_datasets_type: veomni_weighted_multisource`) instead of an
HF interleave. The on-disk row schema is documented in
[Data Format](../usage/data_format.md).

---

## 3. Train

`train.sh` is the thin `torchrun` launcher (auto-detects GPU/NPU count, single-
or multi-node). Pass the task and config after it.

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml
```

The conversation DAG runs every module on the `conversation_list` carrier. A single
FSDP2 tree over the composed OmniModel uses `train/base_model_fsdp.yaml`, or add
`--model.accelerator.fsdp_config.fsdp_scope model` to any launcher:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.accelerator.fsdp_config.fsdp_scope model
```

Key knobs (override on the CLI, e.g. `--train.global_batch_size 32`):

- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — sequence length budget. It does **not** resize images: the
  Janus image processor emits its configured size, and the transform's own
  `image_max_pixels` bounds the initial downscale.
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.checkpoint.save_steps` / `--train.checkpoint.hf_save_steps` — DCP / HF save cadence.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.
- `--model.accelerator.fsdp_config.fsdp_mode` — global FSDP mode, inherited by every module.
- `--model.model_config.modules.janus_llama.accelerator.fsdp_config.fsdp_mode eager` — per-module override (arbitrary nested keys deep-merge into the referenced module file).

### Per-module parallelism

Each module can carry its own `accelerator` block in `train/modules_train.yaml`; when a
module's topology differs from the top-level one, the trainer builds it its **own**
`ParallelState` (device mesh + process groups) on the full world, while modules that
match the global topology reuse it. `janus_text_encoder` ships with an embedding-parallel
**embedding** (`emb`) extra-parallel group; its `embed_tokens` is a `ShardedEmbedding`
(see [Sharded Embedding](../../key_features/sharded_embedding.md)):

```yaml
# train/modules_train.yaml — janus_text_encoder
accelerator:
  extra_parallel_sizes: [4]            # shard embed_tokens.weight dim-0 (vocab) across 4 ranks
  extra_parallel_names: ["emb"]
  extra_parallel_placement_innermost: [false]
```

Quick 1-node smoke run (no wandb, tiny step budget). The launcher's
`modules_train.yaml` shards `embed_tokens.weight` across 4 ranks (`emb=4`), so
this needs 4 GPUs — on a smaller node `emb` cannot divide the mesh; run
`--model.model_config.modules configs/seed_omni/Janus/janus_1.3b/train/modules_train.yaml`
with the `emb` block removed, or use `train/base_model_fsdp.yaml`:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --train.max_steps 20 \
  --train.checkpoint.save_steps 10 \
  --train.wandb.enable false
```

### Sequence parallelism (Ulysses)

Add `--model.accelerator.ulysses_size N`; every module inherits it, and there is no
separate SP config. The dataloader replicates each DP shard across the SP group,
each module slices to its `1/N` chunk, runs one forward, and all-gathers the output
(see [Sequence Parallelism](../distributed/sequence_parallel.md)). SigLIP and VQVAE
slice the image batch; the text encoder and LLaMA slice the token sequence.

On **4 GPUs** with `ulysses_size 4` this gives `dp=1`. The `janus_text_encoder`
`emb=4` extra-parallel group composes with it: on a 4-GPU box the `dp_shard_sp`
mesh dim (`dp_shard=1 × ulysses=4`) is the SP group, so the `emb` group and the
`ulysses` group coincide. That is harmless, because the `ShardedEmbedding` lookup
is a sequence-preserving all-to-all (each rank still gets embeddings for exactly
its own `1/sp` token shard):

```bash
NPROC_PER_NODE=4 bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.accelerator.ulysses_size 4 \
  --train.global_batch_size 4 --train.micro_batch_size 1
```

If your environment exports `TORCH_DISTRIBUTED_DEBUG=DETAIL`, **unset it** — its
`_ProcessGroupWrapper` lacks the coalesced all-gather that FSDP2's tied-head
`full_tensor()` needs, which crashes any FSDP2 run (SP or not).

---

## 4. Resume

Each save writes per-module DCP shards plus the per-rank job state (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --train.checkpoint.load_path outputs/janus_1.3b_omni_sft/checkpoints/global_step_500
```

Training continues from step 500 with the dataloader and RNG state restored.

---

## 5. Inference

The two inference launches are described in
[Training and Inference](../usage/training_and_inference.md#5-inference):

* **Native HF** — `python tasks/omni/infer_omni_native.py --model_path <split-ckpt> …`
  (`OmniModel.from_pretrained`, `modeling.py`, no runtime).
* **VeOmni Inferencer** — `python tasks/omni/infer_omni.py <base.yaml> …`
  (all-eager still loads native `OmniModel`; any FSDP2 / DDP / ExtraParallel
  module builds `OmniModelRuntime` + `accelerated/accelerated.py`).

`tasks/omni/infer_omni.py` runs a generation graph selected by
`--model.model_config.infer_type` (a key into the `model.model_config.infer_graph`
map in `base.yaml`; defaults to `infer_interleave`). `--model.model_path` takes
either the upstream HF checkpoint (loaded through the layout, as in step 1) or a
split-checkpoint root holding one subfolder per module (`janus_siglip/`,
`janus_vqvae/`, `janus_text_encoder/`, `janus_llama/`), each with its own
`config.json` + weights. `base.yaml` points it at the split root the step-1
converter writes, so you can infer directly with the converted base model.

Each module opts into FSDP / extra-parallel via its inference module YAML's
`accelerator` block; `OmniInferencer` auto-detects whether any module needs a
distributed run. Two ready-made inference module files ship with the config:

| `--model.model_config.modules` file | Layout | Launch |
|-------------------------------------|--------|--------|
| `infer/modules_infer_fsdp.yaml` | `janus_text_encoder` → distributed **vocab-parallel `emb`** (`fsdp2` + `emb`), `janus_llama` → `fsdp2`, `janus_vqvae` → `ddp`, `janus_siglip` → eager | **torchrun** (`bash train.sh …`) |
| `infer/modules_infer_eager.yaml` | every module `eager` — plain per-rank replica (the file only pins `janus_siglip`; `for_inference` supplies `fsdp_mode: eager` to the rest) | single-process (`python …`) |

### 5.1 Inferring from a trained checkpoint

Under an HF `model_path`, `save_hf_weights` writes one merged export in the
source's own layout at
`<output_dir>/checkpoints/global_step_N/hf_ckpt/` — the same keys, dtypes and
shard files as the upstream root, plus its non-weight sidecars. Pass it straight
to `--model.model_path` (the layout loads it again) or to `transformers`:

```bash
--model.model_path outputs/janus_1.3b_omni_sft/checkpoints/global_step_20/hf_ckpt
```

Two cases do **not** produce that merged export:

- A **LoRA** run saves adapters per module instead. With only LoRA-trained
  modules there is no `hf_ckpt/` at all; with mixed full/LoRA modules the merged
  export carries the original bases, not the adapter updates. Load the adapters
  on top of the base (`model.model_path=<HF root>` plus the per-module adapter
  path) rather than pointing at `hf_ckpt/`.
- A module that ended up **fully frozen** (for example one whose LoRA targets
  matched nothing) has no per-module export, so the assembled root below would
  lack its weights. Copy those modules from the original checkpoint.

A split-checkpoint run (no HF layout to write back to) instead writes per-module
folders under `hf_ckpt/<module>/`; assemble those into a flat root by linking
each module's folder to `<root>/<module>`. A module with no `hf_ckpt/` folder was
fully frozen, so it comes from `$ORIG` instead:

```bash
STEP=outputs/janus_1.3b_omni_sft/checkpoints/global_step_20
ASM=outputs/janus_1.3b_omni_sft/infer_ckpt/global_step_20
ORIG=/path/to/split/checkpoint          # the step-1 convert output
mkdir -p "$ASM"
for m in janus_siglip janus_vqvae janus_text_encoder janus_llama; do
  src="$STEP/hf_ckpt/$m"
  [ -d "$src" ] || src="$ORIG/$m"
  [ -d "$src" ] || { echo "missing module: $m" >&2; exit 1; }
  ln -sfn "$(realpath "$src")" "$ASM/$m"
done
```

Then pass `--model.model_path "$ASM"`.

**Image understanding (I2T / VQA)** — `infer/graph_infer_und.yaml`.

Distributed — `infer/modules_infer_fsdp.yaml`, torchrun:

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.model_config.infer_type infer_und \
  --model.model_config.modules configs/seed_omni/Janus/janus_1.3b/infer/modules_infer_fsdp.yaml \
  --model.model_path Janus-1.3B-split \
  --infer.prompt "What do you see in this image?" \
  --infer.images /path/to/image.png \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

Single-process — the all-eager modules file, so every module loads as a plain
replica (no torchrun):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.model_config.infer_type infer_und \
  --model.model_config.modules configs/seed_omni/Janus/janus_1.3b/infer/modules_infer_eager.yaml \
  --model.model_path Janus-1.3B-split \
  --infer.prompt "What do you see in this image?" \
  --infer.images /path/to/image.png \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

**Text-to-image (T2I)** — `infer/graph_infer_gen.yaml` (`guidance_scale` enables CFG).

Distributed — `infer/modules_infer_fsdp.yaml`, torchrun:

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.model_config.infer_type infer_gen \
  --model.model_config.modules configs/seed_omni/Janus/janus_1.3b/infer/modules_infer_fsdp.yaml \
  --model.model_path Janus-1.3B-split \
  --infer.prompt "A photo of the Sydney Opera House under a starry night sky." \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 2048 \
  --infer.generation_kwargs.guidance_scale 5.0
```

Single-process — the all-eager modules file, so every module loads as a plain
replica (no torchrun):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/train/base.yaml \
  --model.model_config.infer_type infer_gen \
  --model.model_config.modules configs/seed_omni/Janus/janus_1.3b/infer/modules_infer_eager.yaml \
  --model.model_path Janus-1.3B-split \
  --infer.prompt "A photo of the Sydney Opera House under a starry night sky." \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 2048 \
  --infer.generation_kwargs.guidance_scale 5.0
```

**Interleaved** — `infer/graph_infer_interleave.yaml` (default `model.model_config.infer_type`)
mixes text and image generation in one graph.

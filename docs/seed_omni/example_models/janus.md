# Janus-1.3B (SeedOmni V2)

End-to-end recipe for training and inferring **Janus-1.3B** as a SeedOmni V2
graph model: a unified understanding (image→text) + generation (text→image)
model whose SigLIP / VQVAE / LLaMA backbone are wired as separate OmniModules.

All paths below assume the upstream HuggingFace checkpoint lives at
`/mnt/hdfs/user_dir/veomni_omni/models/transformers/Janus-1.3B`. Adjust to your
own storage.

Config dir: `configs/seed_omni/Janus/janus_1.3b/`

The omni config layout splits the old monolithic launcher into a `base.yaml` plus
per-purpose module/graph files. Both training and inference take the **same**
`base.yaml`; its `model.*` block drives the trainer and its `infer.*` block
drives the inferencer.

| File | Role |
|------|------|
| `base.yaml` | Top-level omni launcher: model paths, top-level `accelerator`, data, train, and the `infer` block. References the module/graph files below. |
| `modules_train.yaml` | Per-module **training** overrides (`model` / `train` / `accelerator` per module). `janus_text_encoder` carries a embed-parallel `emb` extra-parallel block (see below). |
| `graph_train.yaml` | Training DAG (`training_graph:` flat edge list). |
| `data.yaml` | Weighted multisource data list (ImageNet + ShareGPT4V). |
| `modules_infer_fsdp.yaml` | Per-module **inference** overrides — distributed: `janus_text_encoder` vocab-parallel `emb` + `janus_llama` `ddp`, vision modules eager (base.yaml's default `infer.modules`). |
| `modules_infer_eager.yaml` | Per-module **inference** overrides — every module `eager` (single-process replica). |
| `graph_infer_und.yaml` / `graph_infer_gen.yaml` / `graph_infer_interleave.yaml` | Per-scenario generation graphs (mapped under `infer.infer_graph`). |

---

## 1. Convert the checkpoint

The monolithic HF checkpoint is split into one sub-checkpoint per OmniModule
(SigLIP encoder, VQVAE codec + generation head, LLaMA backbone). The converter
reads `model_type` from the HF `config.json` and dispatches to the Janus family
converter.

```bash
python scripts/convert_model.py \
  --model_path /mnt/hdfs/user_dir/veomni_omni/models/transformers/Janus-1.3B \
  --output_dir /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B
```

The `output_dir` becomes `model.model_path` in `base.yaml` (a split-checkpoint
root folder, not a single HF checkpoint).

---

## 2. Prepare data

`data.yaml` lists a weighted multisource mix. Each `names` entry must match a
preprocessor key in `veomni/data/seed_omni/preprocess.py`
(`SEED_OMNI_PREPROCESSOR_REGISTRY`).

```yaml
# configs/seed_omni/Janus/janus_1.3b/data.yaml
sources:
  - /mnt/hdfs/user_dir/dataset/imagenet1k_train      # text -> image (T2I)
  - /mnt/hdfs/veomni/datasets/sharegpt4v_cap_100k    # image -> text (I2T)
names:
  - imagenet1k
  - sharegpt4v_cap_100k
schedule:
  - { schedule_type: const, weights: [0.5, 0.5] }
level: token
stopping_strategy: all_exhausted
upstream_sharded: true
```

Sources have heterogeneous schemas, so we use the VeOmni weighted multisource
sampler (`multisource_datasets_type: veomni_weighted_multisource`) instead of an
HF interleave. The on-disk row schema is documented in
[`docs/seed_omni/data_format.md`](../data_format.md).

---

## 3. Train

`train.sh` is the thin `torchrun` launcher (auto-detects GPU/NPU count, single-
or multi-node). Pass the task and config after it.

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml
```

Key knobs (override on the CLI, e.g. `--train.global_batch_size 32`):

- `--train.global_batch_size` / `--train.micro_batch_size` — global vs. per-step micro batch.
- `--data.max_seq_len` — packed sequence length (large images are resized to fit).
- `--train.checkpoint.output_dir` — run root; DCP checkpoints land in `<output_dir>/checkpoints/`.
- `--train.checkpoint.save_steps` / `--train.checkpoint.hf_save_steps` — DCP / HF save cadence.
- `--train.wandb.enable false` — disable wandb for quick smoke runs.
- `--accelerator.fsdp_config.fsdp_mode` — top-level FSDP mode (the omni schema lifts `accelerator` out of `train`).
- `--model.modules.janus_llama.accelerator.fsdp_config.fsdp_mode eager` — per-module override (arbitrary nested keys deep-merge into the referenced module file).

### Per-module parallelism

Each module can carry its own `accelerator` block in `modules_train.yaml`; when a
module's topology differs from the top-level one, the trainer builds it its **own**
`ParallelState` (device mesh + process groups) on the full world, while modules that
match the global topology reuse it. `janus_text_encoder` ships with a embed-parallel
**embedding** (`emb`) extra-parallel group:

```yaml
# modules_train.yaml — janus_text_encoder
accelerator:
  extra_parallel_sizes: [4]            # shard embed_tokens.weight dim-0 (vocab) across 4 ranks
  extra_parallel_names: ["emb"]
  extra_parallel_placement_innermost: [false]
```

Quick 1-node smoke run (no wandb, tiny step budget):

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --train.max_steps 20 \
  --train.checkpoint.save_steps 10 \
  --train.wandb.enable false
```

---

## 4. Resume

Each save writes per-module DCP shards plus a `trainer_state.pt` (global step,
dataloader position, RNG state) under `<output_dir>/checkpoints/global_step_N/`.
Resume by pointing `load_path` at that directory:

```bash
bash train.sh tasks/omni/train_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --train.checkpoint.load_path outputs/janus_1.3b_omni_sft/checkpoints/global_step_500
```

Training continues from step 500 with the dataloader and RNG state restored.

---

## 5. Inference

`tasks/omni/infer_omni.py` runs a generation graph selected by
`--infer.infer_type` (a key into the `infer.infer_graph` map in `base.yaml`;
defaults to `infer_interleave`). Point `--infer.model_path` at a
**split-checkpoint root** that holds one subfolder per module (`janus_siglip/`,
`janus_vqvae/`, `janus_text_encoder/`, `janus_llama/`), each with its own
`config.json` + weights — or omit it to fall back to `model.model_path`. The
step-1 converter output already has this layout, so you can infer directly with
the converted base model.

Each module opts into FSDP / extra-parallel via its inference module YAML's
`accelerator` block; `OmniInferencer` auto-detects whether any module needs a
distributed run. Two ready-made inference module files ship with the config:

| `infer.modules` file | Layout | Launch |
|----------------------|--------|--------|
| `modules_infer_fsdp.yaml` (base.yaml default) | `janus_text_encoder` → distributed **vocab-parallel `emb`** (`fsdp2` + `emb`), `janus_llama` → `ddp`, vision modules eager | **torchrun** (`bash train.sh …`) |
| `modules_infer_eager.yaml` | every module `eager` — plain per-rank replica | single-process (`python …`) |

Four launcher scripts at the repo root wrap the two paths for both scenarios:

| Script | Modules | Launcher |
|--------|---------|----------|
| `infer_fsdp_i2t.sh` / `infer_fsdp_t2i.sh` | base default (`modules_infer_fsdp.yaml`) | `bash train.sh` (torchrun) |
| `infer_eager_i2t.sh` / `infer_eager_t2i.sh` | `--infer.modules …/modules_infer_eager.yaml` | `python` (single process) |

The commands below mirror those scripts.

### 5.1 Inferring from a trained checkpoint

Training writes each module's HF weights one level deeper, under
`<output_dir>/checkpoints/global_step_N/<module>/hf_ckpt/`. Assemble a flat root
the loader understands by linking each module's `hf_ckpt/` to `<root>/<module>`:

```bash
STEP=outputs/janus_1.3b_omni_sft/checkpoints/global_step_20
ASM=outputs/janus_1.3b_omni_sft/infer_ckpt/global_step_20
mkdir -p "$ASM"
for m in janus_siglip janus_vqvae janus_text_encoder janus_llama; do
  ln -sfn "$(realpath "$STEP/$m/hf_ckpt")" "$ASM/$m"
done
```

Then pass `--infer.model_path "$ASM"` to any of the commands below. (Verified:
the `global_step_20` checkpoint loads all four modules and runs both the I2T and
T2I graphs end-to-end.)

**Image understanding (I2T / VQA)** — `graph_infer_und.yaml`.

Distributed (`infer_fsdp_i2t.sh`) — base default `modules_infer_fsdp.yaml`, torchrun:

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_und \
  --infer.modules configs/seed_omni/Janus/janus_1.3b/modules_infer_fsdp.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "What do you see in this image?" \
  --infer.image /path/to/image.png \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

Single-process (`infer_eager_i2t.sh`) — swap `infer.modules` to the all-eager
file so every module loads as a plain replica (no torchrun):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_und \
  --infer.modules configs/seed_omni/Janus/janus_1.3b/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "What do you see in this image?" \
  --infer.image /path/to/image.png \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

**Text-to-image (T2I)** — `graph_infer_gen.yaml` (`guidance_scale` enables CFG).

Distributed (`infer_fsdp_t2i.sh`) — base default `modules_infer_fsdp.yaml`, torchrun:

```bash
bash train.sh tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_gen \
  --infer.modules configs/seed_omni/Janus/janus_1.3b/modules_infer_fsdp.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "A photo of the Sydney Opera House under a starry night sky." \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 2048 \
  --infer.generation_kwargs.guidance_scale 5.0
```

Single-process (`infer_eager_t2i.sh`) — swap `infer.modules` to the all-eager
file so every module loads as a plain replica (no torchrun):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_gen \
  --infer.modules configs/seed_omni/Janus/janus_1.3b/modules_infer_eager.yaml \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "A photo of the Sydney Opera House under a starry night sky." \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 2048 \
  --infer.generation_kwargs.guidance_scale 5.0
```

**Interleaved** — `graph_infer_interleave.yaml` (default `infer.infer_type`)
mixes text and image generation in one graph.

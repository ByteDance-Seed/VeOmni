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
| `modules_train.yaml` | Per-module **training** overrides (`model` / `train` / `accelerator` per module). |
| `graph_train.yaml` | Training DAG (`training_graph:` flat edge list). |
| `data.yaml` | Weighted multisource data list (ImageNet + ShareGPT4V). |
| `modules_infer.yaml` | Per-module **inference** overrides (deep-merged onto the training modules; modules default to eager load). |
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
the converted base model. Each module loads eager (`fsdp_mode: eager`) by
default; opt a module into FSDP via `modules_infer.yaml`.

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

**Image understanding (I2T / VQA)** — `graph_infer_und.yaml`:

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_und \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "What do you see in this image?" \
  --infer.image /path/to/image.png \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 1024
```

**Text-to-image (T2I)** — `graph_infer_gen.yaml` (`guidance_scale` enables CFG):

```bash
python tasks/omni/infer_omni.py \
  configs/seed_omni/Janus/janus_1.3b/base.yaml \
  --infer.infer_type infer_gen \
  --infer.model_path /mnt/hdfs/user_dir/veomni_omni/models/seed_omni/Janus-1.3B \
  --infer.prompt "A photo of the Sydney Opera House under a starry night sky." \
  --infer.output_dir janus_out \
  --infer.generation_kwargs.max_new_tokens 2048 \
  --infer.generation_kwargs.guidance_scale 5.0
```

**Interleaved** — `graph_infer_interleave.yaml` (default `infer.infer_type`)
mixes text and image generation in one graph.

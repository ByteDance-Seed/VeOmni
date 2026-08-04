# LTX-2.3 training guide

## Download model

Download the LTX-2.3 transformer weights and the Gemma3 text encoder:

```shell
# LTX-2.3 transformer weights
python3 scripts/download_hf_model.py \
    --repo_id Lightricks/LTX-2.3 \
    --local_dir /path/to/models

# Gemma3 text encoder (required for conditioning)
python3 scripts/download_hf_model.py \
    --repo_id google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local_dir /path/to/models
```

The download helper appends each Hugging Face repository name to
`--local_dir`, producing `/path/to/models/LTX-2.3` and
`/path/to/models/gemma-3-12b-it-qat-q4_0-unquantized` in this example.

## Prepare Dataset

Use the built-in preprocessing pipeline to split videos, generate captions, and compute latents/embeddings:

### Step 1: Split scenes (optional)

Split raw videos into scene clips using PySceneDetect:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py split-scenes \
    --video_dir /path/to/raw/videos \
    --output_dir /path/to/output/clips
```

### Step 2: Generate captions

Auto-caption video clips using a multimodal model (Qwen2.5-Omni by default):

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py caption \
    --input_dir /path/to/output/clips \
    --output /path/to/output/clips/dataset.json
```

The captioner writes each `media_path` as a filename relative to the dataset
file's directory. Keep `dataset.json` beside the clips so later stages resolve
those paths correctly.

### Step 3: Generate reference videos (optional, for IC-LoRA)

Generate Canny edge reference videos before preprocessing so their paths are
written to the dataset file:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py compute-reference \
    --input_dir /path/to/output/clips \
    --dataset_file /path/to/output/clips/dataset.json
```

### Step 4: Compute text embeddings and VAE latents

Compute text embeddings + VAE latents from the dataset file:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py preprocess \
    --dataset_file /path/to/output/clips/dataset.json \
    --gemma_model_path /path/to/models/gemma-3-12b-it-qat-q4_0-unquantized \
    --checkpoint_path /path/to/models/LTX-2.3 \
    --resolution_buckets "960x544x49" \
    --with_audio
```

The shipped AV configs set `condition_model_cfg.with_audio: true`, so their
preprocessing command must include `--with_audio`. For video-only training,
set that config field to `false` and omit the flag. For IC-LoRA, append
`--reference_column reference_path` to the command above; this encodes the
reference videos generated in Step 3.

### Step 5: Pack precomputed files

Pack precomputed `.pt` files into parquet shards for offline training:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py save-parquet \
    --precomputed_dir /path/to/output/clips/.precomputed \
    --output_dir /path/to/output/parquet_output \
    --pad_to_multiple_of 8 \
    --with_audio
```

Use `--with_audio` whenever the training config has audio enabled. Reference
latents are included automatically when `reference_latents/` exists.

Output directory structure:

```
output/
├── clips/
│   ├── dataset.json          # Captions + clip-relative media paths
│   ├── *.mp4                 # Scene-split video clips
│   ├── *_reference.mp4       # Optional IC-LoRA reference videos
│   └── .precomputed/
│       ├── audio_latents/    # Optional, for AV training
│       ├── conditions/       # Gemma text embeddings
│       ├── latents/          # VAE-encoded video latents
│       └── reference_latents/# Optional, for IC-LoRA training
└── parquet_output/           # Parquet shards (from save-parquet)
    ├── shard_0000.parquet
    ├── shard_0001.parquet
    └── ...
```

## Update config paths

Before training, update the model and data paths in the config file:

```yaml
# configs/dit/ltx2_av_lora.yaml
model:
  model_path: "/path/to/models/LTX-2.3"
  condition_model_path: "/path/to/models/LTX-2.3"

data:
  train_path: "/path/to/output/parquet_output"
```

The Gemma path is used by the preprocessing command; the shipped offline
training configs consume the precomputed embeddings and do not reload Gemma.

## Start training

### Audio-Video LoRA (default)

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_av_lora.yaml
```

### Audio-Video LoRA (Low VRAM)

For GPUs with limited VRAM, use the low-memory configuration with reduced LoRA rank (16 vs 32):

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_av_lora_low_vram.yaml
```

### Video-to-Video (IC-LoRA)

For video-to-video transformations (e.g., depth-to-video, style transfer), use the IC-LoRA configuration. This requires reference videos in your dataset:

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_v2v_ic_lora.yaml
```

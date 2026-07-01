# LTX-2.3 training guide

## Download model

Download the LTX-2.3 transformer weights and the Gemma3 text encoder:

```shell
# LTX-2.3 transformer weights
python3 scripts/download_hf_model.py \
    --repo_id Lightricks/LTX-2.3 \
    --local_dir /path/to/ltx-2.3-model.safetensors

# Gemma3 text encoder (required for conditioning)
python3 scripts/download_hf_model.py \
    --repo_id google/gemma-3-12b-it-qat-q4_0-unquantized \
    --local_dir /path/to/gemma3-model
```

## Prepare Dataset

Use the built-in preprocessing pipeline to split videos, generate captions, and compute latents/embeddings:

```shell
# Full pipeline: split scenes → caption → preprocess
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py all \
    --video_dir /path/to/raw/videos \
    --data_dir /path/to/output \
    --gemma_model_path /path/to/gemma3-model \
    --checkpoint_path /path/to/ltx-2.3-model.safetensors \
    --resolution_buckets "960x544x49"

# Or only preprocess (if you already have dataset.json with captions)
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py preprocess \
    --dataset_file /path/to/dataset.json \
    --gemma_model_path /path/to/gemma3-model \
    --checkpoint_path /path/to/ltx-2.3-model.safetensors \
    --resolution_buckets "960x544x49"
```

Pack precomputed `.pt` files into parquet shards for offline training:

```shell
python veomni/models/diffusers/ltx2_3/ltx_condition/preprocess_dataset.py save-parquet \
    --precomputed_dir /path/to/output/.precomputed \
    --output_dir /path/to/parquet_output \
    --pad_to_multiple_of 8
```

Output directory structure:

```
data_dir/
├── .precomputed/
│   ├── latents/              # VAE-encoded video latents
│   ├── conditions/           # Gemma text embeddings
│   └── audio_latents/        # Audio latents (optional, for AV training)
├── clips/                    # Scene-split video clips (from split-scenes or all)
├── parquet_output/           # Parquet shards (from save-parquet)
│   ├── shard_0000.parquet
│   ├── shard_0001.parquet
│   └── ...
└── dataset.json              # Captions + video paths
```

## Update config paths

Before training, update the model and data paths in the config file:

```yaml
# configs/dit/ltx2_av_lora.yaml
model:
  model_path: "/path/to/ltx-2.3-model.safetensors"
  condition_model_path: "/path/to/ltx-2.3-checkpoint"
  condition_model_cfg:
    gemma_model_path: "/path/to/gemma3-model"

data:
  train_path: "/path/to/preprocessed/data"
```

## Start training on GPU

### Audio-Video LoRA (default)

```shell
bash train.sh tasks/train_dit.py configs/dit/ltx2_av_lora.yaml
```

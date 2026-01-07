# Checkpoint Conversion

This guide explains how to convert VeOmni's Distributed Checkpoint (DCP) format to HuggingFace format using the `merge_dcp_to_hf.py` script.

## Overview

The `merge_dcp_to_hf.py` script provides memory-efficient conversion from PyTorch Distributed Checkpoint (DCP) format to HuggingFace format. It processes checkpoints shard-by-shard to minimize memory usage, making it suitable for large models.

## Usage

### Basic Usage

```bash
python scripts/merge_dcp_to_hf.py --load-dir <DCP_CHECKPOINT_PATH>
```

This will create a HuggingFace format checkpoint in `<DCP_CHECKPOINT_PATH>/hf_ckpt`.

### Advanced Usage

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir <DCP_CHECKPOINT_PATH> \
    --save-dir <OUTPUT_PATH> \
    --model-assets-dir <MODEL_CONFIG_PATH> \
    --shard-size 2000000000
```

## Command-Line Arguments

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--load-dir` | str | Yes | - | Directory containing the DCP checkpoint |
| `--save-dir` | str | No | `<load-dir>/hf_ckpt` | Output directory for HuggingFace format checkpoint |
| `--model-assets-dir` | str | No | None | Directory containing model config and processor (e.g., tokenizer) |
| `--shard-size` | int | No | 2000000000 | Maximum shard size in bytes (default: 2GB) |

## Examples

### Convert DCP Checkpoint to HuggingFace Format

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/dcp_checkpoint
```

Output will be saved to: `checkpoints/my_model/dcp_checkpoint/hf_ckpt`

### Convert with Custom Output Directory

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/dcp_checkpoint \
    --save-dir hf_models/my_model
```

### Include Model Assets (Config & Tokenizer)

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/dcp_checkpoint \
    --save-dir hf_models/my_model \
    --model-assets-dir pretrained_models/qwen3-8b
```

This will copy the model configuration and tokenizer from `pretrained_models/qwen3-8b` to the output directory.

### Customize Shard Size

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/dcp_checkpoint \
    --shard-size 5000000000
```

This sets the maximum shard size to 5GB instead of the default 2GB.

## Output Format

The script generates a HuggingFace-compatible checkpoint with the following structure:

```
output_directory/
├── model.safetensors                    # Single file (if total size < shard_size)
└── config.json                          # Model config (if --model-assets-dir provided)
└── tokenizer.json                       # Tokenizer (if --model-assets-dir provided)
```

Or for sharded checkpoints:

```
output_directory/
├── model-00001-of-00005.safetensors
├── model-00002-of-00005.safetensors
├── model-00003-of-00005.safetensors
├── model-00004-of-00005.safetensors
├── model-00005-of-00005.safetensors
├── model.safetensors.index.json         # Weight mapping index
├── config.json                          # Model config (if --model-assets-dir provided)
└── tokenizer.json                       # Tokenizer (if --model-assets-dir provided)
```

## Key Conversion Details

### Weight Name Mapping

The script automatically converts DCP key names to HuggingFace format:

- `model.model.*` → `model.*` (removes first "model." prefix)
- `model.lm_head.weight` → `lm_head.weight`
- Other `model.*` keys → strips "model." prefix with warning

Non-model weights (keys not starting with `model.`) are filtered out.

### Data Type Conversion

By default, all weights are converted to `bfloat16` format. This can be customized in the code by modifying the `save_dtype` parameter in `save_model_weights()`.

### Memory Efficiency

The script uses a shard-by-shard processing approach:

1. Analyzes checkpoint metadata to plan sharding
2. Loads only one shard's worth of weights at a time
3. Converts and saves the shard
4. Frees memory before processing the next shard

This approach allows conversion of very large models without requiring all weights to fit in memory simultaneously.

## Troubleshooting

### No Model Weights Found

If you see the warning "No model weights found!", check:

- The checkpoint path is correct
- The checkpoint contains keys starting with `model.`
- The checkpoint was saved in DCP format

### Out of Memory

If you encounter OOM errors:

- Reduce the `--shard-size` to a smaller value
- Ensure no other processes are consuming GPU/CPU memory
- Consider using a machine with more RAM

### Missing Model Assets

If model config or tokenizer is missing from the output:

- Ensure `--model-assets-dir` points to a valid HuggingFace model directory
- Check that the directory contains `config.json` and tokenizer files
- Verify the model type is supported by HuggingFace's `AutoConfig` and `AutoProcessor`

## See Also

- [Basic Modules](basic_modules.md) - Understanding VeOmni's checkpoint saving
- [Arguments](arguments.md) - Checkpoint-related training arguments

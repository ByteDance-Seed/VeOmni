# Checkpoint Conversion

This guide explains how to convert VeOmni's Distributed Checkpoint (DCP) format to HuggingFace format using the `scripts/merge_dcp_to_hf.py` script.

`--load-dir` is a `global_step_{N}` directory. The script reads the weights from its `model/ckpt/` subdirectory; the optimizer state beside them, the scheduler sidecar and the job cursor are not part of a HuggingFace export and are never opened. You can also point `--load-dir` straight at a DCP directory, which is what a checkpoint written before weights and optimizer were split requires. For the full training-run tree, see [Checkpoint layout](checkpoint.md).

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
| `--load-dir` | str | Yes | - | A `global_step_{N}` directory, or a DCP directory directly |
| `--save-dir` | str | No | `<load-dir>/hf_ckpt` | Output directory for HuggingFace format checkpoint |
| `--model-assets-dir` | str | No | None | Directory containing model config and processor (e.g., tokenizer). For a multi-module run, point this at `output_dir/model_assets/<module>/`. Also used to identify tied-weight aliases (see below). |
| `--shard-size` | int | No | 2000000000 | Maximum shard size in bytes (default: 2GB) |
| `--num-workers` | int | No | 1 | Worker processes that load and write independent HF shards in parallel (spawned, CPU-only) |
| `--local-staging-dir` | str | No | None | Write each shard to this local directory first, then stream it to `--save-dir`. Use it when the destination filesystem does not support safetensors mmap writes. |

## Examples

### Convert DCP Checkpoint to HuggingFace Format

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/global_step_200
```

Output will be saved to: `checkpoints/my_model/global_step_200/hf_ckpt`

### Convert with Custom Output Directory

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/global_step_200 \
    --save-dir hf_models/my_model
```

### Include Model Assets (Config & Tokenizer)

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/global_step_200 \
    --save-dir hf_models/my_model \
    --model-assets-dir pretrained_models/qwen3-8b
```

This will copy the model configuration and tokenizer from `pretrained_models/qwen3-8b` to the output directory.
When the directory has no processor, the tokenizer is exported on its own, and a
`generation_config.json` is carried over. Processor, tokenizer or generation-config
files that are present but fail to load stop the conversion before any weight is written.

### Customize Shard Size

```bash
python scripts/merge_dcp_to_hf.py \
    --load-dir checkpoints/my_model/global_step_200 \
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

### Tied Weights

A tensor shared under several names (e.g. `lm_head.weight` with `tie_word_embeddings`,
or whole submodules a model shares between two stacks) appears once per name in the DCP.
With `--model-assets-dir`, the script builds the model on the meta device and drops every
state-dict name that `named_parameters()` / `named_buffers()` deduplicate away, so each
tensor is exported once under its canonical name; `from_pretrained` re-establishes the ties.
If the model class cannot be resolved, every DCP tensor is exported as before.

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

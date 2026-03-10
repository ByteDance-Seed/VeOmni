# Qwen3.5 training guide

> **Note:** Qwen3.5 requires transformers v5. Vision input is not supported yet; only text-only training is available.

## Install dependencies

Qwen3.5 depends on transformers v5 (experimental). Sync the correct dependency group:

```shell
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu
```

## Download dataset

Since vision input is not yet supported, use a text-only dataset for now. Download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

```python
import pyarrow.parquet as pq
input_path = "tulu-3-sft-mixture/data/train-00000-of-00006.parquet"
output_path = "tulu-first2000.parquet"
# Read parquet file and extract the first 2000 rows
table = pq.read_table(input_path)
table_first_2000 = table.slice(0, 2000)
pq.write_table(table_first_2000, output_path)
```

## Download Qwen3.5 model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-27B \
    --local_dir ${HOME}/Qwen3.5-27B
```

## Start training on GPU

Testing in 8x80GB GPUs.
```shell
# Note: max_seq_len is set to 128 to avoid OOM with 8x80GB GPUs since the only currently available
# Qwen3.5 model size is 27B.
# We recommend that you use more GPUs to train Qwen3.5 27B so that you can get a proper seq len.
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-27B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --data.max_seq_len 128 \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.max_steps 20 \
    --train.checkpoint.output_dir /mnt/local/localcache00
```

## Ulysses Sequence Parallelism

Qwen3.5 supports Ulysses sequence parallelism for both its softmax attention layers and
linear attention (GatedDeltaNet) layers. This enables training with longer sequences by
distributing the sequence across multiple GPUs.

To enable Ulysses SP, set `ulysses_parallel_size` in your config. The total GPU count must
equal `data_parallel_size * ulysses_parallel_size`.

```shell
# Example: 8 GPUs, dp=4, sp=2
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-27B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.data_parallel_size 4 \
    --train.ulysses_parallel_size 2 \
    --train.attn_implementation flash_attention_3
```

### Requirements

- `flash_attention_2` or `flash_attention_3` attention implementation (softmax layers use
  VeOmni's flash attention with built-in SP support).
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) installed
  (for GatedDeltaNet triton kernels).
- `num_k_heads` and `num_v_heads` (linear attention head counts) must be divisible by
  `ulysses_parallel_size`.

### How It Works

Qwen3.5 is a hybrid model alternating between softmax and linear attention layers:

- **Softmax attention layers** — SP is handled transparently by VeOmni's `flash_attention_forward`,
  which performs all-to-all gather/scatter around the flash attention kernel.
- **Linear attention layers (GatedDeltaNet)** — SP is handled explicitly in the patched
  `Qwen3_5GatedDeltaNet.forward`. Q/K/V/b/a projections are all-to-all'd to gather the full
  sequence with local heads, the causal conv1d runs with sharded weights, the recurrent attention
  kernel runs on local heads, and the output is all-to-all'd back.

For detailed implementation notes, see the
[Ulysses documentation](../key_features/ulysses.md#-linear-attention-ulysses-gateddeltanet).

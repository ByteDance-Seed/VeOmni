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
bash train.sh tasks/train_torch.py configs/sft/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-27B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --data.max_seq_len 128 \
    --train.data_parallel_mode fsdp2 \
    --train.init_device meta \
    --train.max_steps 20 \
    --train.output_dir /mnt/local/localcache00
```

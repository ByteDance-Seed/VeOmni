# Qwen3.5 training guide

> **Note:** Qwen3.5 requires transformers v5.

## Install dependencies

Qwen3.5 depends on transformers v5 (experimental). Sync the correct dependency group:

```shell
uv sync --no-group transformers-stable --extra transformers5-exp --extra gpu
```

## Download dataset

### Vision-Language Dataset

Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset and download the data annotation JSON file [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the sharegpt4v_instruct_gpt4-vision_cap100k.json:

```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        new_item['images'] = [image_path]
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

### Text-only Dataset (Optional)

If you want to train on text-only data, download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

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

Dense 9B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-9B \
    --local_dir ${HOME}/Qwen3.5-9B
```

MoE 35B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-35B-A3B \
    --local_dir ${HOME}/Qwen3.5-35B-A3B
```

## Start training on GPU

### Qwen3.5-9B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Qwen3.5 MoE 35B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-35B-A3B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Text-only training on GPU
Testing in 8x80GB GPUs.

Qwen3.5 Dense 9B:
```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.max_steps 20 \
    --train.checkpoint.output_dir /mnt/local/localcache00
```

Qwen3.5-35B-A3B. 8X80GB GPU will likely OOM due to the model size. Use 8X192GB GPU or more GPUs.

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --model.ops_implementation.moe_implementation fused \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.global_batch_size 16 \
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
    --model.model_path ${HOME}/Qwen3.5-9B \
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

# Qwen3.5 training guide

> **Note:** Qwen3.5 requires transformers v5 (now the project default).

## Install dependencies

Qwen3.5 depends on transformers v5, which is now the default install:

```shell
uv sync --extra gpu
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
    --train.checkpoint.output_dir ./exp/qwen3_5_9b_sft
```

Qwen3.5-35B-A3B. 8X80GB GPU will likely OOM due to the model size. Use 8X192GB GPU or more GPUs.

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --model.ops_implementation.moe_implementation fused_triton \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.global_batch_size 16 \
    --train.checkpoint.output_dir ./exp/qwen3_5_35b_a3b_sft
```

## ChunkMBS

Dense Qwen3.5 packed SFT supports decoder-layer ChunkMBS. It splits the packed sequence only at sample boundaries,
then runs each `Qwen3_5DecoderLayer` chunk sequentially while keeping the outer FSDP2 layer invocation intact. The
same token ranges are used by both full-attention and GatedDeltaNet layers, so every ChunkMBS cut must be present in
both cumulative-length tensors; their internal boundaries may differ.

The example config exposes the feature but keeps it disabled by default:

```yaml
train:
  chunk_mbs_config:
    enable: true
    chunk_mbs: 2
```

`chunk_mbs` is the number of packed samples per layer chunk, not a token count or `train.micro_batch_size`.
ChunkMBS may be combined with non-reentrant gradient checkpointing. It currently does not support Qwen3.5-MoE,
Ulysses SP, TP/PP, `torch.compile`, `pad_to_length`, DPO, or RL training. See
[ChunkMBSConfig](../usage/arguments.md#chunkmbsconfig) for the complete support boundary.
The model-level automated tests cover decoder routing, metadata slicing, outputs, and gradients on CPU. Real
FlashAttention, FLA, and NPU fused-kernel execution requires separate accelerator validation.

## Ulysses Sequence Parallelism

Qwen3.5 supports Ulysses sequence parallelism for both its softmax attention layers and
linear attention (GatedDeltaNet) layers. This enables training with longer sequences by
distributing the sequence across multiple GPUs.

To enable Ulysses SP, set `train.accelerator.ulysses_size`. VeOmni derives the effective
data-parallel size from the world size and the other parallel dimensions; set
`train.accelerator.dp_shard_size` only when you need to pin the FSDP shard degree explicitly.
For the example below, the total GPU count is `dp_shard_size * ulysses_size = 4 * 2 = 8`.

```shell
# Example: 8 GPUs, dp=4, sp=2
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.dp_shard_size 4 \
    --train.accelerator.ulysses_size 2 \
    --model.ops_implementation.attn_implementation flash_attention_3
```

### Requirements

- `flash_attention_2` or `flash_attention_3` attention implementation (softmax layers use
  VeOmni's flash attention with built-in SP support).
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) installed
  (for GatedDeltaNet triton kernels).
- `num_k_heads` and `num_v_heads` (linear attention head counts) must be divisible by
  `ulysses_size`.

### Selecting linear-attention kernels

GatedDeltaNet has three OpSlot-driven kernels: `rms_norm_gated`, `causal_conv1d`, and
`chunk_gated_delta_rule`. All three default to `fla`. Recommended value per platform:

- **GPU** — `fla` (the FLA Triton kernels shipped under the `gpu` extra; required for
  varlen training).
- **NPU** — `npu` (vendored MindSpeed-MM Triton kernels for all three ops; needs the
  `triton-ascend` package — see below). Not auto-selected — the default `fla`
  requires a GPU and raises on NPU, so set the fields explicitly:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: npu
    causal_conv1d_implementation: npu
    chunk_gated_delta_rule_implementation: npu
```

Install `triton-ascend` on the NPU host. VeOmni main pins PyTorch and
`torch_npu` 2.10.0; the corresponding CANN 9.0.0 stack uses v3.2.1:

```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

See the [triton-ascend quick-start](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/quick_start.md)
for the compatibility matrix and troubleshooting. Keep CANN, `torch_npu`,
and `triton-ascend` on a mutually compatible release set.

> **arch35 limitation:** the vendored `causal_conv1d` refuses to run on arch35 NPUs
> (`Ascend910_95` / `Ascend950`) — its `is_arch35()` guard raises `NotImplementedError`
> rather than mis-computing. Validated on `Ascend910B2C`. arch35 support would need to come
> from upstream MindSpeed-MM.

To switch `chunk_gated_delta_rule` to QwenLM's [`flash-qla`](https://github.com/QwenLM/FlashQLA)
kernel (already shipped under the `gpu` extra), set the field explicitly:

```yaml
model:
  ops_implementation:
    chunk_gated_delta_rule_implementation: flash_qla
```

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

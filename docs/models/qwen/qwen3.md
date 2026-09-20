<span id="qwen3-training-guide"></span>

# Qwen3

## 1. Model introduction

### Scope and prerequisites

Text SFT for dense Qwen3 and the shown MoE variant.

Configuration: [training YAML](../../../configs/text/qwen3.yaml). Read the
[catalog prerequisites and validation scope](../index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## 2. Variants and recipes

| Variant / component | Model repository | Recipe scope |
| --- | --- | --- |
| Qwen3-0.6B | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | Five-step Quick Start |
| Qwen3-8B | [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B) | Dense SFT walkthrough |
| Qwen3-30B-A3B-Instruct-2507 | [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | MoE example; see the dedicated MoE recipe |

This table identifies checkpoints used by the recipe. Full-size hardware validation
is separate from configuration availability; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md)
and run the commands from the repository root. Replace example filesystem paths
with your own model, dataset, and output locations.

### Download dataset

Download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

```python
import pyarrow.parquet as pq
input_path = "tulu-3-sft-mixture/data/train-00000-of-00006.parquet"
output_path = "tulu-first2000.parquet"
# Read parquet file and extract the first 2000 rows
table = pq.read_table(input_path)
table_first_2000 = table.slice(0, 2000)
pq.write_table(table_first_2000, output_path)
```

### Download Qwen3 model

#### Qwen3-8B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-8B \
    --local_dir .
```

#### Qwen3-30B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --local_dir .
```

> **Note.** VeOmni's runtime `CheckpointTensorConverter` folds the per-expert
> HF safetensor keys into VeOmni's fused expert layout at load time, so the
> stock HF checkpoint can be passed directly to training — no offline merge
> step is required. `scripts/moe_ckpt_merge/moe_merge.py` is deprecated but
> may still be useful as a one-time optimization for very large checkpoints
> (e.g. Qwen3-235B) to amortize per-load stacking cost. See
> `docs/transformers_v5/transformers_v5_moe_weight_loading.md` for details.

## 4. Launch training

### Start training on GPU/NPU

#### Qwen3-8B

```shell
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
    --model.model_path ./Qwen3-8B \
    --data.train_path ./tulu-first2000.parquet \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta
```

#### Qwen3-30B

```shell
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
    --model.model_path ./Qwen3-30B-A3B-Instruct-2507 \
    --model.ops_implementation.moe_implementation fused_triton \
    --data.train_path ./tulu-first2000.parquet \
    --model.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --model.accelerator.init_device meta \
    --train.global_batch_size 16
```

## 5. Recipe configuration

| Configuration | Data | Sequence length | SP / EP | Global batch |
| --- | --- | --- | --- | --- |
| [qwen3.yaml](../../../configs/text/qwen3.yaml) | conversation | 2048 | 1 / default | 8 |
| [qwen3_lora.yaml](../../../configs/text/qwen3_lora.yaml) | conversation | 2048 | 1 / default | 8 |

These are checked-in defaults, not minimum hardware requirements. Match the
world size, batch size, and parallel topology to your checkpoint and memory budget.

## 6. Validation and next steps

### Check outputs and continue

Use the [run checks](../index.md#choose-hardware-and-check-a-run) and
[checkpoint completion contract](../../usage/checkpoint.md#completion) for training.

### Related guides

- [Checkpoint save, resume, and export](../../usage/checkpoint.md)
- [LoRA](../../key_features/lora.md)
- [Model families](../index.md)

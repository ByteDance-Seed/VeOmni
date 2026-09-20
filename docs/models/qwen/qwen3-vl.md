<span id="qwen3-vl-training-guide"></span>

# Qwen3 VL

## 1. Model introduction

### Scope and prerequisites

Vision-language SFT for Qwen3 VL dense and MoE models, with image/video data preparation.

Configurations: [dense](../../../configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml) and
[MoE](../../../configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml). Read the
[catalog prerequisites and validation scope](../index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## 2. Variants and recipes

| Variant / component | Model repository | Recipe scope |
| --- | --- | --- |
| Qwen3-VL-8B-Instruct | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | Dense image/video SFT |
| Qwen3-VL-30B-A3B-Instruct | [Qwen/Qwen3-VL-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-30B-A3B-Instruct) | MoE image/video SFT |

This table identifies checkpoints used by the recipe. Full-size hardware validation
is separate from configuration availability; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md)
and run the commands from the repository root. Replace example filesystem paths
with your own model, dataset, and output locations.

### Download dataset

Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset and download the data annotation JSON file [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the sharegpt4v_instruct_gpt4-vision_cap100k.json

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

### Download Qwen3 VL model

#### Qwen3-VL-8B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir .
```

#### Qwen3-VL-30B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --local_dir .
```

## 4. Launch training

### Start training on GPU/NPU

#### Qwen3-VL-8B

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.dataloader.num_workers 8 \
    --train.micro_batch_size 3
```

#### Qwen3-VL-30B

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml \
    --model.model_path ./Qwen3-VL-30B-A3B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.dataloader.num_workers 8 \
    --train.micro_batch_size 2
```

## 5. Recipe configuration

| Configuration | Data | Sequence length | SP / EP | Global batch |
| --- | --- | --- | --- | --- |
| [qwen3_vl_dense.yaml](../../../configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml) | conversation | 4096 | 1 / default | 16 |
| [qwen3_vl_moe.yaml](../../../configs/multimodal/qwen3_vl/qwen3_vl_moe.yaml) | conversation | 4096 | default / 1 | 32 |
| [qwen3_vl_moe_lora.yaml](../../../configs/multimodal/qwen3_vl/qwen3_vl_moe_lora.yaml) | conversation | 4096 | default / 1 | 32 |

These are checked-in defaults, not minimum hardware requirements. Match the
world size, batch size, and parallel topology to your checkpoint and memory budget.

### Optional per-block torch.compile on CUDA

Dense Qwen3-VL can compile each text decoder block before FSDP2 sharding while keeping the vision tower, DeepStack injection, and language-model head eager. Enable fixed-length packed inputs together with compile:

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --train.dyn_bsz true \
    --train.pad_to_length true \
    --model.accelerator.torch_compile.enable true
```

This path currently requires CUDA, FSDP2, the default `model.accelerator.torch_compile.backend=inductor` and `model.accelerator.torch_compile.mode=None`, `model.accelerator.torch_compile.dynamic=false`, `model.accelerator.ulysses_size=1`, `model.accelerator.cp_size=1`, and `model.accelerator.enable_async=false`. Padding fixes the token-tensor shapes, while different packed sequence boundaries can still produce separate Inductor specializations. CUDA Graph replay, Qwen3-VL-MoE, ExtraParallel, and NPU execution are not yet supported.

## 6. Validation and next steps

### Check outputs and continue

Use the [run checks](../index.md#choose-hardware-and-check-a-run) and
[checkpoint completion contract](../../usage/checkpoint.md#completion) for training.

### Related guides

- [Checkpoint save, resume, and export](../../usage/checkpoint.md)
- [LoRA](../../key_features/lora.md)
- [Model families](../index.md)

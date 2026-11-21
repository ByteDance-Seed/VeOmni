# Qwen3 VL training guide

## Download dataset

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

## Qwen3-VL-2B-Instruct training example

Prepare the dataset above in the repository root, then download the checkpoint.
For ShareGPT4V, each entry in `images` must resolve to an image file from the training working directory.

```shell
python scripts/download_hf_model.py --repo_id Qwen/Qwen3-VL-2B-Instruct --local_dir .
```

Run the command for your backend from the repository root.

```shell
# NVIDIA CUDA
bash train.sh tasks/train_vlm.py configs/models/qwen3_vl/cuda.yaml

# Ascend NPU
bash train.sh tasks/train_vlm.py configs/models/qwen3_vl/ascend.yaml
```

The Qwen3-VL-2B-Instruct example completed three FSDP2 optimizer steps on a single node with eight H20 GPUs and on a single node with eight Ascend 910B2 NPUs with the same ShareGPT4V image conversations and a global batch size of 8. To run this short check, append `--train.max_steps 3 --train.checkpoint.save_steps 3 --data.dataloader.num_workers 0` to the command. The step-3 checkpoint includes model and optimizer shards.

## Download Qwen3 VL model

### Qwen3-VL-8B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir .
```

### Qwen3-VL-30B

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-30B-A3B-Instruct \
    --local_dir .
```

## Start training on GPU/NPU

### Qwen3-VL-8B

```shell
bash train.sh tasks/train_vlm.py configs/models/qwen3_vl/recipes/multisource_8b.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader.type native \
    --data.datasets_type iterable \
    --data.source_name sharegpt4v_sft \
    --data.dataloader.num_workers 8 \
    --train.micro_batch_size 3
```

### Qwen3-VL-30B

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

## Optional per-block torch.compile on CUDA

Dense Qwen3-VL can compile each text decoder block before FSDP2 sharding while keeping the vision tower, DeepStack injection, and language-model head eager. Enable fixed-length packed inputs together with compile:

```shell
bash train.sh tasks/train_vlm.py configs/models/qwen3_vl/recipes/multisource_8b.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --train.dyn_bsz true \
    --train.pad_to_length true \
    --model.accelerator.torch_compile.enable true
```

This path currently requires CUDA, FSDP2, the default `model.accelerator.torch_compile.backend=inductor` and `model.accelerator.torch_compile.mode=None`, `model.accelerator.torch_compile.dynamic=false`, `model.accelerator.ulysses_size=1`, `model.accelerator.cp_size=1`, and `model.accelerator.enable_async=false`. Padding fixes the token-tensor shapes, while different packed sequence boundaries can still produce separate Inductor specializations. CUDA Graph replay, Qwen3-VL-MoE, ExtraParallel, and NPU execution are not yet supported.

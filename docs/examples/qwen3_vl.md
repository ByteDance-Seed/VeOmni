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
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
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

Dense Qwen3-VL can compile its blocks before FSDP2 sharding. Enable fixed-length packed inputs together with compile:

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --train.dyn_bsz true \
    --train.pad_to_length true \
    --model.accelerator.torch_compile.enable true
```

Every entry in `_no_split_modules` is compiled, which for this model is both the text decoder layers and the vision blocks. This path currently requires CUDA, FSDP2, the default `model.accelerator.torch_compile.backend=inductor` and `model.accelerator.torch_compile.mode=None`, `model.accelerator.ulysses_size=1`, `model.accelerator.cp_size=1`, and `model.accelerator.enable_async=false` — the last three are refused for every trainer, text included, not just here. Leave `model.accelerator.torch_compile.dynamic` unset: padding fixes the text token count, but a vision tower's patch count follows the images in the batch, and a static graph exhausts the recompile cache on it. CUDA Graph replay, ExtraParallel, and NPU execution are not yet supported.

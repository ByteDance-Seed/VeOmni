# Qwen3 VL training guide

## Download dataset

1. Download the ShareGPT4V and place it in the root directory of VeOmni:  [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingfacce.co/datasets/Lin-Chen/ShareGPT4V/blob/main/sharegpt4v_instruct_gpt4-vision_cap100k.json)

2. Use the following Python script to filter the data file `sharegpt4v_instruct_gpt4-vision_cap100k.json` and retain only the content from the COCO dataset.

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
After executing the Python script, a file named `sharegpt4v_instruct_gpt4-vision_cap100k_coco.json` will be generated in the root directory of VeOmni.

3. Download the COCO 2017 training images:  [COCO train2017 dataset](https://images.cocodataset.org/zips/train2017.zip)

4. Final directory structure should be like this:
> ```
> VeOmni
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k.json
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k_coco.json
> └—— coco/
>     └—— train2017/
>         ├—— 000000000009.jpg
>         ├—— 000000000026.jpg
>         └—— ... (more images)
> ```

## Download qwen3vl model
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-VL-8B-Instruct \
    --local_dir .
```

## Start training on NPU

```shell
bash train.sh tasks/omni/train_qwen_vl.py configs/multimodal/qwen3_vl/qwen3_vl_dense.yaml \
    --model.model_path ./Qwen3-VL-8B-Instruct \
    --data.train_path ./sharegpt4v_instruct_gpt4-vision_cap100k_coco.json \
    --data.dataloader_type native \
    --data.dataset_type iterable \
    --data.sourcename sharegpt4v_sft \
    --train.use_wandb false
```
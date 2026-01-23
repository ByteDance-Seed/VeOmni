# Qwen3 Omni MoE training guide

## Download multisource dataset

### sharegpt4v_cap_100k + COCO2017
Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset and download the data annotation JSON file [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the `sharegpt4v_instruct_gpt4-vision_cap100k.json` and genrate `sharegpt4v_instruct_gpt4-vision_cap100k_coco.json`.

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

The directory structure should be like this:
> ```
> VeOmni
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k.json
> ├—— sharegpt4v_instruct_gpt4-vision_cap100k_coco.json
> ├—— coco/
> ├   └—— train2017/
> ├       ├—— 000000000009.jpg
> ├       ├—— 000000000026.jpg
> ├       └—— ... (more images)
> └—— ...(code files)
> ```

### tulu-3-sft-mixture

Download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

The directory structure should be like this:
> ```
> VeOmni
> ├—— tulu-3-sft-mixture/
> ├   └—— data/
> ├       ├—— train-00000-of-00006.parquet
> ├       ├—— train-00001-of-00006.parquet
> ├       ├—— train-00002-of-00006.parquet
> ├       ├—— train-00003-of-00006.parquet
> ├       ├—— train-00004-of-00006.parquet
> ├       └—— train-00005-of-00006.parquet
> └—— ...(code files)
> ```

### LLaVA-Video-178K

Download the [LLaVA-Video-178K](https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K/tree/main/0_30_s_academic_v0_1) dataset.
Extract all tar.gz files to the root directory of VeOmni.

Modify the `0_30_s_academic_mc_v0_1_qa_processed.json` and generate `video.json`.

```python
import json
with open('0_30_s_academic_mc_v0_1_qa_processed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
new_data = []
for item in data:
    new_item = item.copy()
    image_path = new_item.pop('video')
    new_item['videos'] = [image_path]
    new_data.append(new_item)  
with open('video.json', 'w', encoding='utf-8') as f:
    json.dump(new_data, f, ensure_ascii=False, indent=4)
```

The directory structure should be like this:
> ```
> VeOmni
> ├—— 0_30_s_academic_mc_v0_1_qa_processed.json
> ├—— video.json
> ├—— academic_source/
> ├   ├—— activitynet/
> ├   ├   └—— ...
> ├   ├—— Charades/
> ├   ├   └—— ...
> ├   ├—— ego4d/
> ├   ├   └—— ...
> ├   ├—— NextQA/
> ├   ├   └—— ...
> ├   └—— youcook2/
> ├       └—— ...
> └—— ...(code files)
> ```

### modify multisource yaml

Modify `configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml`:

```yaml
sources:
- sharegpt4v_instruct_gpt4-vision_cap100k_coco.json
- tulu-3-sft-mixture/data
- video.json
names:
- sharegpt4v_captioner_sft
- tulu-3-sft-mixture
- LLaVA-Video-178K
schedule:
- schedule_type: const
  weights: [0.4, 0.2, 0.4]
```

## Download Qwen3-Omni-MoE model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --local_dir .
```

## Start training on GPU

```shell
bash train.sh tasks/omni/train_qwen3_omni.py configs/multimodal/qwen3_omni/qwen3_omni.yaml
```

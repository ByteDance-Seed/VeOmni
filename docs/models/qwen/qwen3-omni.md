<span id="qwen3-omni-moe-training-guide"></span>

# Qwen3 Omni

## 1. Model introduction

This recipe trains the **thinker** with text, images, video, and audio. It does
not train the talker or code2wav components.


### Scope and prerequisites

Multisource Qwen3 Omni training with text, image, video, and audio inputs.

Configuration: [training YAML](../../../configs/multimodal/qwen3_omni/qwen3_omni.yaml). Read the
[catalog prerequisites and validation scope](../index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## 2. Variants and recipes

| Variant / component | Model repository | Recipe scope |
| --- | --- | --- |
| Qwen3-Omni-30B-A3B-Instruct | [Qwen/Qwen3-Omni-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) | Thinker training; no talker or code2wav training |

This table identifies checkpoints used by the recipe. Full-size hardware validation
is separate from configuration availability; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md)
and run the commands from the repository root. Replace example filesystem paths
with your own model, dataset, and output locations.

### Download multisource dataset

#### sharegpt4v_cap_100k + COCO2017
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

#### tulu-3-sft-mixture

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

#### LLaVA-Video-178K

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

#### modify multisource yaml

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

### Download Qwen3-Omni-MoE model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --local_dir .
```

## 4. Launch training

### Start training on GPU/NPU

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_omni/qwen3_omni.yaml \
    --model.model_path Qwen3-Omni-30B-A3B-Instruct
```

> **Note.** VeOmni's runtime `CheckpointTensorConverter` folds per-expert HF
> safetensor keys into VeOmni's fused `gate_up_proj` / `down_proj` layout at
> load time, so the stock HF checkpoint can be passed directly to training —
> no offline merge step is required. See
> `docs/transformers_v5/transformers_v5_moe_weight_loading.md` for the full
> format matrix and how to convert a VeOmni-format training checkpoint back
> to per-expert HF keys for inference engines.
>
> `scripts/moe_ckpt_merge/moe_merge.py` is deprecated. It still works and may
> be useful as a one-time optimization for very large checkpoints (e.g.
> Qwen3-235B) where you want to amortize the per-load stacking cost across
> many runs, but it is no longer a prerequisite.

## 5. Recipe configuration

| Configuration | Data | Sequence length | SP / EP | Global batch |
| --- | --- | --- | --- | --- |
| [qwen3_omni.yaml](../../../configs/multimodal/qwen3_omni/qwen3_omni.yaml) | conversation | 8196 | 1 / default | 32 |
| [qwen3_omni_lora.yaml](../../../configs/multimodal/qwen3_omni/qwen3_omni_lora.yaml) | conversation | 8196 | 1 / default | 32 |

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

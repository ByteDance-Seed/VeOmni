<span id="wan2-1-i2v-training-guide"></span>

# Wan2.1 I2V

## 1. Model introduction

### Scope and prerequisites

Wan diffusion training with prepared video data.

Configuration: [training YAML](../../../configs/dit/wan_sft.yaml). Read the
[catalog prerequisites and validation scope](../index.md#choose-hardware-and-check-a-run)
before following the model-specific steps below.

## 2. Variants and recipes

| Variant / component | Model repository | Recipe scope |
| --- | --- | --- |
| Wan2.1-I2V-14B-480P | [Wan-AI/Wan2.1-I2V-14B-480P-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers) | Image-to-video diffusion |

This table identifies checkpoints used by the recipe. Full-size hardware validation
is separate from configuration availability; see the [validation scope](../index.md#choose-hardware-and-check-a-run).

## 3. Environment and data

Install the [environment for your accelerator](../../hardware_support/index.md)
and run the commands from the repository root. Replace example filesystem paths
with your own model, dataset, and output locations.

### Download model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --local_dir .
```

### Prepare Dataset

End-to-end training for the **wan2.1 i2v** model is not yet supported, so real-world datasets are not being used at this time.
We are constructing random tensors to conduct test training.

Ensure the current working directory is the **project root**.

```shell
python docs/examples/generate_wan_dataset.py
```
You can adjust parameter **num_files and video specifications (T, H, W)** in the script to control the scale of the test dataset.

## 4. Launch training

### Start training on GPU

```shell
bash train.sh tasks/train_dit.py configs/dit/wan_sft.yaml \
    --model.model_path Wan2.1-I2V-14B-480P-Diffusers/transformer
```

### Start training on NPU

```shell
bash train.sh tasks/train_dit.py configs/dit/wan_sft.yaml \
    --model.model_path Wan2.1-I2V-14B-480P-Diffusers/transformer \
    --model.accelerator.init_device npu
```

## 5. Recipe configuration

| Configuration | Data | Sequence length | SP / EP | Global batch |
| --- | --- | --- | --- | --- |
| [wan_sft.yaml](../../../configs/dit/wan_sft.yaml) | diffusion | 8192 | 4 / default | 8 |
| [wan_lora.yaml](../../../configs/dit/wan_lora.yaml) | diffusion | 8192 | 4 / default | 8 |

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

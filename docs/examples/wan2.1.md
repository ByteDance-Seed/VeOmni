# Wan2.1-I2V training guide

## Download model

```shell
python3 scripts/download_hf_model.py \
    --repo_id Wan-AI/Wan2.1-I2V-14B-480P-Diffusers \
    --local_dir .
```

## Prepare Dataset

End-to-end training for the **wan2.1 i2v** model is not yet supported, so real-world datasets are not being used at this time.
We are constructing random tensors to conduct test training.

Ensure the current working directory is the **project root**.

```shell
python docs/examples/generate_wan_dataset.py
```
You can adjust parameter **num_files and video specifications (T, H, W)** in the script to control the scale of the test dataset.

## Start training on GPU

```shell
bash train.sh tasks/omni/train_wan.py configs/dit/wan_sft.yaml \
    --model.model_path Wan2.1-I2V-14B-480P-Diffusers/transformer
```

## Start training on NPU

```shell
bash train.sh tasks/omni/train_wan.py configs/dit/wan_sft.yaml \
    --model.model_path Wan2.1-I2V-14B-480P-Diffusers/transformer \
    --train.init_device npu
```

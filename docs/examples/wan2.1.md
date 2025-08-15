Flux training guide

1. Download wan2.1 i2v model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Wan-AI/Wan2.1-I2V-14B-480P \
  --local_dir .
```

2. Prepare dataset

Your dataset should contain a metadata.csv file and a train folder, the train folder should contain all the training videos tneosors.

```shell
metadata.csv
train
├── video01.tensors.pth
├── video01.tensors.pth
└── ...
```

The format of metadata.csv is as follows:
```shell
file_name,text
video01.tensors.pth,"an aerial view of a large, ancient stone structure surrounded by lush greenery ..."
video02.tensors.pth,"a view of a river flowing through a forested area. the water appears calm and reflects the blue of the sky ..."
```

3. Train wan2.1 i2v model

SFT:

```
bash train.sh tasks/omni/train_wan.py configs/dit/wan_sft.yaml
```

LoRa:
```
bash train.sh tasks/omni/train_wan.py configs/dit/wan_lora.yaml
```

You can configure training parameters by modifying configs/dit/wan_sft.yaml and configs/dit/wan_lora.yaml.

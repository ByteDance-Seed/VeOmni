Flux training guide

1. Download wan2.2 i2v model

```shell
python3 scripts/download_hf_model.py \
  --repo_id Wan-AI/Wan2.2-I2V-A14B \
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

3. Train wan2.2 i2v model

SFT:

- high_noise:
```
bash train.sh tasks/omni/train_wan2_2.py configs/dit/wan2_2_sft.yaml --model.model_path Wan-AI/Wan2.2-I2V-A14B/high_noise_model --train.max_timestep_boundary 1.0 --train.min_timestep_boundary 0.875
```

- low_noise:
```
bash train.sh tasks/omni/train_wan2_2.py configs/dit/wan2_2_sft.yaml --model.model_path Wan-AI/Wan2.2-I2V-A14B/low_noise_model --train.max_timestep_boundary 0.875 --train.min_timestep_boundary 0.0
```

LoRa:

- high_noise:
```
bash train.sh tasks/omni/train_wan2_2.py configs/dit/wan2_2_lora.yaml --model.model_path Wan-AI/Wan2.2-I2V-A14B/high_noise_model --train.max_timestep_boundary 1.0 --train.min_timestep_boundary 0.875
```

- low_noise:
```
bash train.sh tasks/omni/train_wan2_2.py configs/dit/wan2_2_lora.yaml --model.model_path Wan-AI/Wan2.2-I2V-A14B/low_noise_model --train.max_timestep_boundary 0.875 --train.min_timestep_boundary 0.0
```


You can configure training parameters by modifying configs/dit/wan2_2_sft.yaml and configs/dit/wan2_2_lora.yaml.

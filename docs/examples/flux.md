Flux training guide

1. Download flux model

```shell
python3 scripts/download_hf_model.py \
  --repo_id black-forest-labs/FLUX.1-dev \
  --local_dir .
```

2. Prepare dataset

Your dataset should contain a metadata.csv file and a train folder, the train folder should contain all the training images.

```shell
metadata.csv
train
├── image01.png
├── image02.png
└── ...
```

The format of metadata.csv is as follows:
```shell
file_name,text
image01.png,"A serene autumn forest with golden leaves falling."
image02.png,"A futuristic space station floating in a galaxy."
```

3. Train flux model

SFT:
```
bash train.sh tasks/omni/train_flux.py configs/dit/flux_sft.yaml
```

LoRa:
```
bash train.sh tasks/omni/train_flux.py configs/dit/flux_lora.yaml
```

You can configure training parameters by modifying configs/dit/flux_sft.yaml and configs/dit/flux_lora.yaml.

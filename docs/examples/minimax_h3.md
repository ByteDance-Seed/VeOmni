# MiniMax H3 FL2VA 快速开始

本指南帮助你快速开始，在 Ascend NPU 机器上按步骤完成 MiniMax H3 FL2VA（首末帧 + 文本 → 视频 + 音频）的**训练**和**推理**。每条命令可直接复制执行。

- 验证环境：4 张 Ascend NPU，torch_npu + torchrun
- 已验证流程：离线两阶段训练（embedding → offline）30 步 + 单卡推理

---

## 目录

1. [模型](#1-准备环境模型数据)
2. [数据格式说明](#2-数据格式说明)
3. [训练（两阶段，逐步操作）](#3-训练两阶段逐步操作)
4. [训练关键配置说明](#4-训练关键配置说明)
5. [推理](#5-推理逐步操作)
6. [推理关键配置说明](#6-推理关键配置说明)

---

## 1. 模型



```shell
modelscope download --model MiniMax/MiniMax-H3 \
    --local_dir pretrained_models/MiniMax-H3
```

---

## 2. 数据格式说明

### 2.1 目录结构

stage1 离线生成训练数据阶段，文件内无图片时，首尾帧是从视频首尾帧抽取。
```
dataset/my_data/
├── metadata.csv          # 索引文件（必须叫 metadata.csv，或由配置的 train_path 指定）
├── video.mp4             # 训练视频
├── first.png             # （可选）推理用首帧关键帧
└── last.png              # （可选）推理用末帧关键帧
```

### 2.2 metadata.csv 格式

**必须**包含以下 4 列，列名固定：

```csv
video,prompt,input_audio,frame_rate
video.mp4,"A girl is very happy, she is speaking in english.",video.mp4,24
```

| 列 | 必填 | 含义 |
|:---|:-----|:-----|
| `video` | 是 | 视频文件名（相对 CSV 所在目录） |
| `prompt` | 是 | 文本描述（对应视频内容） |
| `input_audio` | 是 | 音频来源，填 `video.mp4` 表示用视频自带的音轨 |
| `frame_rate` | 是 | 帧率，填 `24` |

### 2.3 视频硬性约束（不满足会直接报错）

| 约束 | 值 | 说明 |
|:-----|:---|:-----|
| 帧数 | **124**（必须满足 `(帧数-5) % 17 == 0`） | Video VAE 按 17 帧分组，124 是演示配置的值；改成 73、107、141 等也合法（`(N-5) % 17 == 0`） |
| 分辨率 | **480×832**（高×宽） | 必须能被 VAE 下采样倍数整除 |
| 帧率 | 24 | 配置里 `fps: 24`，音频 latent 长度按 `帧数/24×40` 计算 |
| 音频 | 32kHz 立体声 | 音频自动 resample 到 32kHz；无声的视频训练会报错 |



---

## 3. 训练（两阶段，逐步操作）


### Step 1：先跑 Stage 1（离线 embedding）

```shell
# MiniMax H3 FL2VA
# 离线训练
bash train.sh tasks/train_dit.py configs/dit/minimax_h3_fl2va_embedding.yaml
```

Stage 1 干什么：

- 加载 Video VAE / Audio VAE / Text Encoder（**不加载 DiT**）
- 编码视频/音频/文本 → VAE latents + prompt embedding + packed 信息
- 每卡写一份 parquet：`output/minimax_h3_fl2va_embedding/rank_<卡序号>_shard_0.parquet`
- 正常结束标志：进程退出、无 Traceback



### Step 2：切到 Stage 2（离线训练）

```shell
bash train.sh tasks/train_dit.py configs/dit/minimax_h3_fl2va_offline.yaml
```

Stage 2 干什么：

- 加载 DiT（FSDP2 + 梯度检查点 + bf16），**跳过** VAE/Text Encoder（`skip_encoder_load: true`）
- 读 parquet → 加噪 → DiT 前向/反向 → AdamW 更新


## 4. 训练关键配置说明

两个配置文件：

| 阶段 | 配置 | 训练任务 |
|:-----|:-----|:---------|
| Stage 1 | `configs/dit/minimax_h3_fl2va_embedding.yaml` | `training_task: offline_embedding` |
| Stage 2 | `configs/dit/minimax_h3_fl2va_offline.yaml` | `training_task: offline_training` |

### Stage 1 配置（embedding）

```yaml
model:
  condition_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
  condition_model_cfg:
    base_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
    video_vae_subfolder: video_vae/source   # VAE 权重子目录
    skip_encoder_load: false                # Stage 1 必须 false（要加载编码器）
    use_keyframe_condition: true
    keyframe_indices: [0, -1]               # 首帧 + 末帧
    video_max_frames: 73                    # 视频最大 latent 帧组
    video_max_resolution: 848
    sigma_shift_video: 12.0
    sigma_shift_audio: 3.0

data:
  train_path: dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/metadata.csv
  data_transform: minimax_h3_online         # Stage 1 从原始视频在线编码
  datasets_type: minimax_h3_online
  dataloader:
    num_workers: 0                          # Stage 1 编码重，worker 放 0 防显存争抢
    drop_last: false
  mm_configs:
    data_dir: dataset/minimax-h3-demo/minimax_h3/MiniMax-H3-FL2VA/
    fps: 24
    min_frames: 124
    max_frames: 124
    height: 480
    width: 832
  offline_embedding_save_dir: output/minimax_h3_fl2va_embedding   # Stage 2 读这里
```

**特别需要注意**：

- `train_path` 必须是 **metadata.csv 文件**，不是目录，否则报 `png files are not supported`
- 换数据时 `fps/min_frames/max_frames/height/width` 必须与视频实际参数一致；帧数必须满足 `(N-5) % 17 == 0`
- `offline_embedding_save_dir` 必须与 Stage 2 的 `data.train_path` 一致

### Stage 2 配置（offline training）

```yaml
model:
  model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer
  condition_model_path: pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA
  condition_model_cfg:
    skip_encoder_load: true                 # 必须 true：不加载 VAE/TextEncoder
    video_max_frames: 120
    video_max_resolution: 832

data:
  train_path: output/minimax_h3_fl2va_embedding    # Stage 1 的输出目录
  data_transform: minimax_h3_offline
  datasets_type: minimax_h3_offline
  shuffle: false
  mm_configs:
    repeat: 100                             # 数据集重复次数（小数据集必须 > 1）

train:
  training_task: offline_training
  global_batch_size: 8  
  micro_batch_size: 1
  init_device: meta
  max_steps: 30  
  gradient_checkpointing:
    enable: true                            # 显存不够时关闭会 OOM
  optimizer:
    type: adamw
    lr: 1.0e-5
    max_grad_norm: 1.0e9  
  accelerator:
    fsdp_config:
      fsdp_mode: fsdp2
      mixed_precision:
        enable: true
        param_dtype: bfloat16
        reduce_dtype: float32
  checkpoint:
    output_dir: output/minimax_h3_fl2va_offline
    save_steps: 10  
    save_hf_weights: false
```


---

## 5. 推理

```shell
python tasks/infer/infer_minimax_h3.py
```

脚本做两件事（顺序执行）：

1. **t2va**：纯文本 → 视频 + 音频（480×832，124 帧，50 步）
2. **fl2va**：首帧 + 末帧 + 文本 → 视频 + 音频（832×480 竖屏，124 帧，50 步）

输出文件（仓库根目录）：

- `t2va.mp4`
- `fl2va.mp4`


---

## 6. 推理关键配置说明

推理配置全部写在 `tasks/infer/infer_minimax_h3.py` 里：

```python
pipe = MiniMaxH3Pipeline.from_pretrained(
    torch_dtype=torch.bfloat16,  
    device=device,
    condition_model_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
    condition_model_cfg={
        "base_model_path": "pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA",
        "use_keyframe_condition": True,
        "keyframe_indices": [0, -1],       # 首末帧
    },
    transformer_config_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer/config.json",
    transformer_weights_path="pretrained_models/MiniMax-H3/MiniMax/MiniMax-H3/FL2VA/transformer",
    ops_implementation=OpsImplementationConfig(
        attn_implementation="eager",
        rotary_pos_emb_implementation="eager",
        rms_norm_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    ),
)
```

调用参数：

```python
# t2va
video, audio = pipe(
    prompt=prompt,
    height=480, width=832, num_frames=124,   # 帧数必须满足 (N-5) % 17 == 0
    num_inference_steps=50, seed=0,          # 步数改小更快，seed 固定可复现
)

# fl2va
video, audio = pipe(
    prompt=prompt,
    height=832, width=480, num_frames=124,
    num_inference_steps=50, seed=0,
    keyframes=[first_frame, last_frame],     # 图片必须存在，否则 FileNotFoundError
    keyframe_indices=[0, -1],
)
```

**特别需要注意**：

- `num_frames` 必须满足 `(N-5) % 17 == 0`，否则 Video VAE 报错

---

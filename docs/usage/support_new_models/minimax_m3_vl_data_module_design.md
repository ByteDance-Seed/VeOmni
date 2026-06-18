# MiniMax M3 VL 数据模块适配报告

日期：2026-06-18

## 结论

当前 PR 已新增 `minimax_m3_vl` 数据 transform 和 `minimax_m3_vl` multimodal chat template。

正式示例配置使用：

- 数据配置：`configs/multimodal/minimax_m3_vl/minimax_m3_vl.yaml`
- 训练数据清单：`configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml`
- 数据类型：`conversation`
- chat template：`minimax_m3_vl`

MiniMax transform 复用 VeOmni 现有 image/video fetch、packing 和 collate 规则，但图像/视频张量化使用 MiniMax Hugging Face processor：

- `processor.image_processor(images=..., return_tensors="pt")`
- `processor.video_processor(videos=..., video_metadata=..., return_tensors="pt", return_metadata=True)`

也就是说，正式路径不是旧 smoke 的 token-id JSONL，也不是直接复用 Qwen 的 template；它是 VeOmni VLM 数据流 + MiniMax processor 输出的组合。

## 是否使用 Hugging Face video processor

是。MiniMax M3 VL 在 transformers v5.12.0 中提供：

- `processing_minimax_m3_vl.py`
- `image_processing_minimax_m3_vl.py`
- `video_processing_minimax_m3_vl.py`

当前 `process_sample_minimax_m3_vl()` 调用 MiniMax `video_processor` 生成：

| key | 来源 | 用途 |
|---|---|---|
| `pixel_values_videos` | MiniMax HF video processor | 输入 vision tower |
| `video_grid_thw` | MiniMax HF video processor | vision 3D RoPE/grid shape |
| `video_metadata` | MiniMax HF video processor | 展开 `]<]{seconds} seconds[>[` timestamp token |

`video_metadata` 只用于 template 展开 timestamp，不进入最终 batch，避免 collator 误打包非 tensor 元数据。

## 是否使用 VeOmni 默认数据处理流程

部分使用。MiniMax 走的是 VeOmni 的正式 VLM 管线入口：

1. `conv_preprocess()` 处理 ShareGPT/LLaVA 风格 conversation；
2. `fetch_images()` / `fetch_videos_metadata()` 读取图片和视频；
3. MiniMax HF image/video processor 产生 pixel/grid tensor；
4. `MiniMaxM3VLChatTemplate` 展开 MiniMax 专属视觉占位 token；
5. `MainCollator` 用已有规则 pack `pixel_values`、`pixel_values_videos`、`image_grid_thw`、`video_grid_thw`；
6. `MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func()` 在 collator 侧把 `image_grid_thw` / `video_grid_thw` 转成 Python list，并写入 `multimodal_metadata`；
7. labels 中 image/video placeholder token 位置置为 `IGNORE_INDEX=-100`。

没有复用 Qwen 的 mRoPE position-id 逻辑。MiniMax generated model 的 `get_position_id_func()` 返回 `None`，数据侧使用默认 1-D packed position ids。

MiniMax 3D RoPE 只需要 image/video grid list，不需要 Qwen/Qwen2.5 系列的 `cu_seqlens`、window index 或 mRoPE position-id 预计算。因此 MiniMax 的 metadata fast path 比 Qwen 简单：collator 在 CPU 侧生成 `image_grid_thw_list` / `video_grid_thw_list`，generated `MiniMaxM3VLVisionModel.forward()` 再把它传给 `MiniMaxM3VL3DRotaryEmbedding.forward()`，避免训练热路径里对 CUDA `grid_thw` 调用 `.tolist()`。

视频容器文件路径仍由 VeOmni 默认 `video_utils.fetch_videos_metadata()` 负责：当输入是 str/bytes 容器时优先使用 torchcodec；如果本机 torchcodec extension 无法加载，则回退到 PyAV 解码本地 path/bytes 容器。输入是预解码 `List[PIL.Image]` / `List[bytes]` / dict frames 时仍走 VeOmni 的无 ffmpeg 路径，先采样/resize 成 video tensor，再交给 MiniMax HF video processor。

## MiniMax token 处理差异

MiniMax 官方 processor 使用这些视觉 token：

| token | 说明 |
|---|---|
| `]<]image[>[` | image placeholder |
| `]<]video[>[` | video placeholder |
| `]<]start of image[>[` | vision span start |
| `]<]end of image[>[` | vision span end |

MiniMax upstream forward 通过 `config.image_token_id` 和 `config.video_token_id` 在 `input_ids` 里寻找占位符。因此 MiniMax transform 保留真实 placeholder token id，不像部分 Qwen/Seed-Omni 路径那样把视觉 token 替换为内部 sentinel 0/negative ids。

## Smoke 数据集

减层 loss smoke 仍保留一个本地 token-id JSONL fixture：

- `tests/fixtures/minimax_m3_vl_sft/tiny_sft.jsonl`

它只用于 generated-model/text-only backward 证据，不代表正式多模态数据集。正式 example 指向 `configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml`，覆盖 ShareGPT4V caption、Tulu SFT mixture 和 LLaVA-Video 清单。

## 数据 transform 验证

新增单测：

```text
pytest tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q

6 passed in 17.42s
```

该测试文件使用 fake MiniMax processor、真实 `MiniMaxM3VLChatTemplate`、以及 monkeypatch 的 media fetch 函数，验证：

- `process_sample_minimax_m3_vl()` 调用 `processor.image_processor(..., return_tensors="pt")`；
- `process_sample_minimax_m3_vl()` 调用 `processor.video_processor(..., video_metadata=..., return_tensors="pt", return_metadata=True)`；
- `pixel_values`、`pixel_values_videos`、`image_grid_thw`、`video_grid_thw` 回填到样本；
- `video_metadata` 只传给 chat template / `processor.replace_video_token()` 用于展开 timestamp，不进入最终 tensor batch；
- 真实 MiniMax chat template 路径会调用 `processor.replace_image_token(image_inputs, image_idx)` 和 `processor.replace_video_token(video_inputs, video_idx)`；
- image/video placeholder label 被置为 `IGNORE_INDEX=-100`；
- 默认 1-D `position_ids` 与 input length 对齐。

其中 `test_minimax_m3_vl_transform_uses_veomni_video_fetch_with_real_processor` 不 monkeypatch `fetch_videos_metadata()`，而是把合成的 `List[PIL.Image]` 作为视频输入交给 VeOmni 默认 video fetch 路径，再用 transformers v5.12.0 `MiniMaxM3VLVideoProcessor` 生成真实 `pixel_values_videos` / `video_grid_thw`：

```text
pixel_values_videos: (16, 1176)
video_grid_thw: [[1, 4, 4]]
video placeholder labels: IGNORE_INDEX
```

其中 `test_minimax_m3_vl_transform_uses_video_container_with_real_processor` 进一步参数化覆盖本地 MP4 `path` 和 `bytes` 两种输入，不 monkeypatch `fetch_videos_metadata()`；测试先用 VeOmni 写出 4 帧 tiny MP4，再通过 PyAV fallback + transformers v5.12.0 `MiniMaxM3VLVideoProcessor` 生成真实 `pixel_values_videos` / `video_grid_thw`：

```text
container_kind: path, bytes
pixel_values_videos: (16, 1176)
video_grid_thw: [[1, 4, 4]]
video placeholder labels: IGNORE_INDEX
```

其中 `test_minimax_m3_vl_transform_to_collator_to_generated_model_backward` 进一步把 transform 输出送入真实 `MainCollator(metadata_collate_func=model.get_metadata_collate_func())`，再送入 toy generated MiniMax model 做一次 loss/backward，验证：

- MiniMax transform 产出的 `pixel_values` / `pixel_values_videos` / `image_grid_thw` / `video_grid_thw` 能被 `MainCollator` 正确 pack；
- collator 生成的 `multimodal_metadata` 与 generated model 的 3D RoPE fast path 对齐；
- image/video mask 能把视觉特征 scatter 回 text embedding；
- vision patch embedding、multimodal projector、LM head 在同一次 backward 中均产生非零梯度。

这些测试已经覆盖 fake processor、真实 chat template replacement、VeOmni 预解码帧 video fetch、str/bytes 视频容器 path/bytes decode、真实 MiniMax video processor、collator metadata 和 generated model backward；它们仍不替代真实 public checkpoint 或多卡 torchrun e2e。

## Collator metadata 验证

新增 MiniMax 专属 collator hook 单测：

```text
pytest tests/data/test_mm_metadata.py -q

5 passed in 5.99s
```

其中 MiniMax case 使用 generated `collate_multimodal_metadata` hook 和真实 `MainCollator`，验证：

- hook 是 module-level callable，可被 pickle，适合 DataLoader worker；
- `MainCollator` 会在 pack 后调用 MiniMax hook；
- packed `image_grid_thw` / `video_grid_thw` 保留 tensor batch 形态；
- `multimodal_metadata` 只包含 CPU 构造的 `image_grid_thw_list` / `video_grid_thw_list`，供 generated vision tower 的 3D RoPE fast path 使用。

数据 transform 与 metadata hook 组合验证：

```text
pytest tests/data/test_mm_metadata.py tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q

11 passed in 17.13s
```

## Mixed image/video backward 验证

新增 generated-model mixed image/video loss/backward 单测：

```text
pytest tests/models/test_models_patch.py -k minimax_m3_vl -q

2 passed, 12 deselected in 6.22s
```

其中 `test_minimax_m3_vl_mixed_image_video_forward_backward` 使用 toy MiniMax generated model、显式 image/video masks、`pixel_values` / `pixel_values_videos`、`image_grid_thw` / `video_grid_thw` 和 generated metadata hook，验证：

- image 与 video vision tower 均参与同一次 forward；
- projected image/video features scatter 到 text embeddings；
- language loss 可计算并 backward；
- vision patch embed、multimodal projector、LM head 都产生非零梯度。

## VLMTrainer glue 验证

新增 MiniMax trainer-glue 单进程 forward/backward 测试：

```text
pytest tests/models/test_vlm_trainer.py -k minimax_m3_vl -q

6 passed, 12 deselected, 3 warnings in 20.39s
```

其中 `test_minimax_m3_vl_vlm_trainer_transform_collate_forward_backward` 使用 `VLMTrainer._build_data_transform()` 和 `VLMTrainer._build_collate_fn()` 生成真实 trainer transform/collator，然后调用 `BaseTrainer.forward_backward_step()` 跑 image+video micro-batch。该测试验证：

- trainer 侧会为 `minimax_m3_vl` 构建 MiniMax 数据 transform；
- trainer collator 会读取模型的 `get_metadata_collate_func()`；
- transformers v5.12.0 的 `MiniMaxM3VLImageProcessor` / `MiniMaxM3VLVideoProcessor` 会把合成 PIL image 和 frame list 处理成真实 `pixel_values` / `pixel_values_videos` / `grid_thw` tensor；
- MiniMax placeholder id 200025/200026 可直接进入 generated model，不再通过 `inputs_embeds` 绕过 embedding；
- VeOmni ops binding 后 MiniMax top-level forward 能把 CE kernel 返回的 loss tuple 解包为 trainer 可消费的 tensor loss；
- image/video vision tower、projector 和 LM head 都产生非零梯度。

其中 `test_minimax_m3_vl_vlm_trainer_init_dataloader_optimizer_smoke` 参数化覆盖 image-only、video-only、mixed image+video 三种 fixture，并通过 `VLMTrainer.__init__()` 建立 model、model assets、MiniMax data transform、dataset、dataloader、collator、optimizer、LR scheduler、training context 和 callbacks；测试随后从真实 dataloader 取出 micro-batch，验证 `pixel_values` / `pixel_values_videos`、image/video masks 和 `multimodal_metadata`，再执行 `BaseTrainer.forward_backward_step()`、`optimizer.step()`、`lr_scheduler.step()`。该 init smoke 不再 monkeypatch media fetch：image-only 使用本地 JPEG path，video-only 使用本地 MP4 path，mixed 使用本地 JPEG path + raw MP4 bytes。

这些 trainer-glue 测试仍使用 toy generated model、真实 transformers MiniMax image/video processors、以及一个只负责 fake tokenizer 和 replacement glue 的 processor shim，并临时放大内存中的 toy config vocab 以容纳真实 MiniMax placeholder ids；它们仍不等同于真实 public checkpoint 或多卡 torchrun e2e。trainer init-smoke 已覆盖本地 image path、video path 和 raw video bytes 容器，但尚未扩展成 public-checkpoint full trainer smoke。

## 当前未完成的数据门

当前 PR 已提供真实 transformers MiniMax image/video processor 输出驱动的 trainer transform/collator glue 单进程 forward/backward 证据，以及 CPU 单进程 `VLMTrainer.__init__()` dataloader/optimizer smoke。仍需要后续补：

- 真实 public checkpoint full weight load；
- public-checkpoint full trainer smoke，覆盖真实 checkpoint 和真实数据清单；
- 多卡硬件上的 FSDP2 asymmetric multimodal rank 回归证据。

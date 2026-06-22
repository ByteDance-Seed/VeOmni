# MiniMax M3 VL 模型结构与迁移报告

日期：2026-06-18

## 当前完成范围

本 PR 已从 intake 前进到 patchgen-generated 模型切片：

- `veomni/models/transformers/minimax_m3_vl/configuration_minimax_m3_vl.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_gpu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/minimax_m3_vl_npu_patch_gen_config.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_gpu.py`
- `veomni/models/transformers/minimax_m3_vl/generated/patched_modeling_minimax_m3_vl_npu.py`
- `veomni/models/transformers/minimax_m3_vl/parallel_plan.py`
- `veomni/models/transformers/minimax_m3_vl/checkpoint_tensor_converter.py`

模型代码来源是 Hugging Face transformers v5.12.0：

- source module：`transformers.models.minimax_m3_vl.modeling_minimax_m3_vl`
- generated header：`Based on: transformers==5.12.0`
- 旧版本状态：transformers v5.9.0/v5.10.0/v5.11.0 不包含 `modeling_minimax_m3_vl.py`

精度来源是 MiniMax 官方 Hugging Face `MiniMaxAI/MiniMax-M3` 的 config、processor 和 public checkpoint。Transformers v5.12.0 在本 PR 中是加载该官方源的 reference loader；不能把 Transformers 仓本身当作 MiniMax 原始精度真值。

VeOmni 全局 `transformers-stable` pin 不改；MiniMax example 文档要求局部 `transformers>=5.12.0`。

## 超参数加载报告

详细超参加载证据见 [minimax_m3_vl_hyperparams_loading_report.md](./minimax_m3_vl_hyperparams_loading_report.md)。本报告保留结构迁移相关摘要。

已验证 Hugging Face 官方配置：

| 字段 | 值 |
|---|---|
| repo | `MiniMaxAI/MiniMax-M3` |
| HF sha | `b1c79b9c07578aeebf33c2aeff0f6de8a96b02b1` |
| last modified | `2026-06-22T06:35:16.000Z` |
| MiniMax GitHub release repo | `MiniMax-AI/MiniMax-M3`, main `a4ee8150dd58920c6f87c6aba26e3b6cdea56d75`; README/figures only |
| `model_type` | `minimax_m3_vl` |
| `architectures` | `["MiniMaxM3SparseForConditionalGeneration"]` |
| processor class | `MiniMaxVLProcessor` |
| text hidden size | `6144` |
| text layers | `60` |
| attention heads / KV heads | `64` / `4` |
| experts / top-k | `128` / `4` |
| max positions | `1048576` |
| vision hidden size | `1280` |
| vision layers | `32` |
| patch size | `14` |
| spatial/temporal merge | `2` / `2` |

ModelScope 页面 `https://modelscope.cn/models/MiniMax/MiniMax-M3` 返回 HTTP 200，可作为国内镜像入口。ModelScope raw `config.json` / `preprocessor_config.json` 已验证与 Hugging Face 字节级一致；当前 checkpoint-index audit 使用 Hugging Face raw `model.safetensors.index.json`，因为它只读取小型索引文件，不下载真实权重 shard。

本地 toy config 验证：

- `tests/toy_config/minimax_m3_vl_toy/config.json`
- `tests/models/test_model_registry.py -k minimax_m3_vl`
- `transformers==5.12.0` 下 generated model meta build 通过；
- text/vision `rope_parameters` 默认值已修复，避免 upstream modeling 读取 `config.rope_parameters["rope_type"]` 时失败。
- nested `text_config` / `vision_config` 为 `PretrainedConfig` 实例时可安全转换为 dict 后解析，避免 review 指出的 `_drop_nested_model_type().pop()` 崩溃路径。

## 模型结构

transformers v5.12.0 中 MiniMax M3 VL 的主要结构如下：

| 模块 | 说明 |
|---|---|
| `MiniMaxM3SparseForConditionalGeneration` | 顶层 VLM Causal LM wrapper，持有 `model` 和 `lm_head` |
| `MiniMaxM3VLModel` | 多模态 wrapper，负责 vision features scatter 到 language embeddings |
| `MiniMaxM3VLVisionModel` | CLIP-like vision tower，输入 `pixel_values` + `image_grid_thw/video_grid_thw` |
| `MiniMaxM3VLMultiModalProjector` | vision hidden 到 text hidden 的 projector 和 spatial merge projector |
| `MiniMaxM3VLTextModel` | decoder-only language model |
| `MiniMaxM3VLAttention` | GQA attention，稀疏层配合 `MiniMaxM3VLIndexer` |
| `MiniMaxM3VLSparseMoeBlock` | router + experts + shared experts |
| `MiniMaxM3VLExperts` | generated modeling 期望 fused `gate_up_proj` / `down_proj` expert 参数 |
| `MiniMaxM3VLDenseMLP` | dense MLP 层，`gate_up_proj` + `down_proj` |

VeOmni patchgen 当前注入：

- `MiniMaxM3VL3DRotaryEmbedding.forward()`：接收 collator 预计算的 `grid_thw_list`，避免训练热路径在 CUDA/NPU tensor 上调用 `.tolist()`。
- `MiniMaxM3VLVisionModel.forward()`：消费 `vit_metadata`，将 image/video grid list 传给 3D RoPE。
- `MiniMaxM3VLVisionModel.dummy_forward()`：为 FSDP2 asymmetric multimodal batch 的 text-only rank 构造 dummy vision path。
- `MiniMaxM3VLModel.forward()`：接入 `multimodal_metadata`、剥离 VeOmni data-pipeline helper masks/ids，并在 FSDP text-only rank 跑零贡献 dummy vision/projector。
- `MiniMaxM3SparseForConditionalGeneration.get_parallel_plan()`
- `MiniMaxM3SparseForConditionalGeneration.get_position_id_func()`
- `MiniMaxM3SparseForConditionalGeneration.get_metadata_collate_func()`
- `MiniMaxM3SparseForConditionalGeneration.forward()`：解包 VeOmni CE kernel 的 `(loss, logits, aux)` 返回，保证 trainer 看到 tensor loss；裸 transformers loss tensor 返回仍兼容。
- `MiniMaxM3VLForCausalLM.get_parallel_plan()`

`get_position_id_func()` 返回 `None`，表示 MiniMax 当前走 VeOmni 默认 1-D packed position id，而不是 Qwen mRoPE 预计算。

## 前向迁移边界

MiniMax generated 前向代码不是手写模型实现；它来自 Hugging Face transformers v5.12.0 的 `transformers.models.minimax_m3_vl.modeling_minimax_m3_vl`，再通过 VeOmni patchgen 增量改写。迁移边界如下：

| 前向路径 | 来源 | VeOmni 改动 |
|---|---|---|
| text decoder / sparse attention / router | transformers v5.12.0 | 保留 upstream 语义，注册 parallel plan |
| vision tower patch embedding / CLIP blocks / 3D RoPE | transformers v5.12.0 | 3D RoPE grid list 改为 collator metadata fast path |
| multimodal scatter | transformers v5.12.0 | 消费 VeOmni `image_mask` / `video_mask`，避免把 helper kwargs 泄漏给 language model |
| FSDP asymmetric rank | VeOmni patch | text-only rank 运行 dummy vision/projector，输出乘 `0.0` 接入 embeddings，保证共享参数参与 collectives |
| data position ids | VeOmni default | `get_position_id_func()` 返回 `None`，使用默认 1-D packed position ids |
| checkpoint key layout | VeOmni converter | language/MoE/vision prefix rename、expert merge、projector/patch-merge index mapping 已覆盖；full shard load 待跑 |

## Checkpoint 适配

真实 HF safetensors index 已验证包含 `23416` 个权重 key。它与 transformers v5.12 generated modeling 之间存在命名/layout 差异：

| public checkpoint key | generated target |
|---|---|
| `language_model.lm_head.weight` | `lm_head.weight` |
| `language_model.model.embed_tokens.weight` | `model.language_model.embed_tokens.weight` |
| `language_model.model.layers.*.mlp.gate_proj/up_proj.weight` | `model.language_model.layers.*.mlp.gate_up_proj.weight` |
| `language_model.model.layers.*.block_sparse_moe.gate.weight` | `model.language_model.layers.*.mlp.gate.weight` |
| `language_model.model.layers.*.block_sparse_moe.experts.*.w1/w3.weight` | `model.language_model.layers.*.mlp.experts.gate_up_proj` |
| `language_model.model.layers.*.block_sparse_moe.experts.*.w2.weight` | `model.language_model.layers.*.mlp.experts.down_proj` |
| `vision_tower.vision_model.encoder.layers.*` | `model.vision_tower.layers.*` |
| `vision_tower.vision_model.embeddings.patch_embedding.weight` | `model.vision_tower.embeddings.proj.weight` |
| `multi_modal_projector.linear_{1,2}.*` | `model.multi_modal_projector.linear_{1,2}.*` |
| `patch_merge_mlp.linear_{1,2}.*` | `model.multi_modal_projector.merge_linear_{1,2}.*` |

因此本 PR 新增 `checkpoint_tensor_converter.py`，并在 MiniMax generated model class 上注册：

- `_create_checkpoint_tensor_converter`
- `_convert_fqn_to_index_mapping`

### 真实权重加载边界

public checkpoint 的 projector / patch-merge index 暴露：

- `multi_modal_projector.linear_1.*`
- `multi_modal_projector.linear_2.*`
- `patch_merge_mlp.linear_1.*`
- `patch_merge_mlp.linear_2.*`

而 transformers v5.12 generated projector 需要：

- `model.multi_modal_projector.linear_1.*`
- `model.multi_modal_projector.linear_2.*`
- `model.multi_modal_projector.merge_linear_1.*`
- `model.multi_modal_projector.merge_linear_2.*`

vLLM 的 MiniMax M3 实现把 `multi_modal_projector` 和 `patch_merge_mlp` 拆成两个模块；transformers v5.12 generated modeling 则把二者折叠在同一个 `MiniMaxM3VLMultiModalProjector` 中，其中 `patch_merge_mlp.linear_{1,2}` 对应 `merge_linear_{1,2}`。因此当前 converter 已能在 index level 闭合 projector keys。

2026-06-18 checkpoint-index audit 只读取 `model.safetensors.index.json`，不下载权重 shard。结果确认 HF public index 的 `weight_map` 有 `23416` 个 key，其中 projector / patch-merge 相关 key 为：

```text
multi_modal_projector.linear_1.bias
multi_modal_projector.linear_1.weight
multi_modal_projector.linear_2.bias
multi_modal_projector.linear_2.weight
patch_merge_mlp.linear_1.bias
patch_merge_mlp.linear_1.weight
patch_merge_mlp.linear_2.bias
patch_merge_mlp.linear_2.weight
```

本地 generated toy MiniMax state dict 需要额外：

```text
model.multi_modal_projector.merge_linear_1.bias
model.multi_modal_projector.merge_linear_1.weight
model.multi_modal_projector.merge_linear_2.bias
model.multi_modal_projector.merge_linear_2.weight
```

`test_public_projector_index_mapping_covers_merge_weights_from_patch_merge_mlp` 固化了该边界，保证 converter 使用 public `patch_merge_mlp` 映射 generated `merge_linear_{1,2}`，而不是静默合成没有源权重证明的 projector 参数。

transformers v5.12.0 官方 `MiniMaxM3VLMultiModalProjector` 明确包含 `linear_{1,2}` 和 `merge_linear_{1,2}` 两组 MLP；VeOmni generated 文件与该源码一致。当前已读取 59 个 public safetensors shard 的 header metadata 并验证转换后 `shape_mismatch_count=0`。剩余未完成项是 869 GB tensor payload 的真实加载验证，而不是 projector index mapping 或 shard-header shape coverage。

真实 HF index 与 safetensors shard-header 覆盖检查：

```text
public_weight_map_keys 23416
converted_keys 1582
public_safetensors_metadata_keys 23416
converted_metadata_keys 1582
safetensors_shards_read 59
safetensors_header_bytes_read 3440088
missing_metadata_key_count 0
unexpected_metadata_key_count 0
shape_mismatch_count 0
dtype_mismatch_groups {"F32->BF16": 115}
projector_missing []
checkpoint_values_downloaded false
full_checkpoint_load_executed false
```

## 验证

已通过：

```text
pytest tests/models/test_model_registry.py tests/models/test_models_patch.py \
  tests/models/test_model_forward_no_implicit_sync.py -k minimax_m3_vl -q

4 passed, 2 skipped, 34 deselected in 6.47s
```

skip 原因：no-sync runtime / metadata equivalence 是 CUDA-only；静态 generated hook 和 mixed image/video generated-model loss/backward 测试已通过。

```text
pytest tests/models/test_models_patch.py -k minimax_m3_vl -q

2 passed, 12 deselected in 6.22s
```

mixed image/video case 使用 `inputs_embeds` 加显式 image/video masks 绕开 toy vocab 与真实 MiniMax placeholder id 的范围差异，覆盖 image/video vision tower、multimodal projector、scatter、language loss 和 backward 梯度。

最新 patchgen/docstring 回归验证：

```text
build_foundation_model(
    config_path="./tests/toy_config/minimax_m3_vl_toy/config.json",
    weights_path=None,
    torch_dtype="float32",
    init_device="cpu",
    ops_implementation=make_eager_ops_config(),
)

model_class MiniMaxM3SparseForConditionalGeneration
docstring_error_present False
```

这验证重新生成 GPU/NPU patchgen 输出后，HF `auto_docstring` 不再报告顶层 `forward()` 的 `image_grid_thw` / `video_grid_thw` 参数缺文档。

```text
pytest tests/models/test_vlm_trainer.py -k minimax_m3_vl -q

6 passed, 12 deselected, 3 warnings in 20.39s
```

新增 MiniMax trainer-glue case 通过 `VLMTrainer._build_data_transform()`、`VLMTrainer._build_collate_fn()` 和 `BaseTrainer.forward_backward_step()` 跑 image+video micro-batch。该测试覆盖 trainer transform/collator 入口、transformers v5.12.0 `MiniMaxM3VLImageProcessor` / `MiniMaxM3VLVideoProcessor` 输出、真实 MiniMax placeholder ids、generated metadata hook、VeOmni ops binding 后的 loss tuple 解包、以及 vision/projector/LM head backward 梯度。

新增 MiniMax `VLMTrainer.__init__()` smoke 参数化覆盖 image-only、video-only、mixed image+video 三种 fixture，构建 model assets、MiniMax transform、dataset、dataloader、collator、optimizer、LR scheduler、training context 和 callbacks 后，从 dataloader 取 micro-batch 并执行 backward + optimizer/scheduler step。该 init smoke 使用本地 JPEG path、本地 MP4 path 和 raw MP4 bytes 进入真实 VeOmni media fetch；它仍使用 fake tokenizer shim 和 toy generated model，不声明真实 public checkpoint 或多卡训练完成。

```text
pytest tests/models/test_checkpoint_tensor_converter.py -k MiniMax -q

10 passed, 51 deselected in 11.56s
```

converter 单测覆盖 public checkpoint prefix 识别、dense/shared expert `gate_proj+up_proj` 合并、sparse expert `w1/w3` 合并、`w2` stack、sparse router gate 映射、public projector + `patch_merge_mlp` 到 generated `merge_linear` 的 index mapping、nested/flat config factory，以及 incomplete checkpoint fail-fast。

```text
pytest tests/data/multimodal/test_minimax_m3_vl_data_transform.py -q

6 passed in 17.42s
```

新增的 transform-to-collator-to-generated-model case 覆盖 `process_sample_minimax_m3_vl()`、`MainCollator`、generated MiniMax toy model 的 image+video loss/backward。该用例证明数据侧视觉 tensor、grid metadata、image/video masks 和 generated model 前向接口能在同一条链路中对齐。该测试文件还包含 VeOmni 预解码帧 video fetch + transformers v5.12.0 真实 `MiniMaxM3VLVideoProcessor` 的输出验证，并参数化覆盖本地 MP4 path/bytes 容器经 torchcodec/PyAV decode path 进入真实 MiniMax video processor；它仍不声明真实 public checkpoint 或多卡 `VLMTrainer` 完成。

```text
pytest tests/data/test_mm_metadata.py -q

5 passed in 5.99s
```

MiniMax metadata case 使用 generated `collate_multimodal_metadata` hook 和 `MainCollator`，证明 image/video `grid_thw` tensor pack 后会生成 CPU `image_grid_thw_list` / `video_grid_thw_list`，并进入模型 forward 的 `multimodal_metadata` fast path。

```text
pytest tests/distributed/test_dummy_forward.py -k minimax_m3_vl -q -rs

1 skipped, 5 deselected in 5.81s
```

skip 原因：当前本机没有测试要求的两个分布式设备；MiniMax case 已不再 xfail。

MiniMax e2e alignment case 现在使用 `DummyMiniMaxM3VLDataset`，不再复用 Qwen3-VL 的 3-D mRoPE dummy `position_ids`。该 dummy dataset 生成：

- 1-D `position_ids`；
- toy vocab 范围内的 `input_ids`，避免 MiniMax 真实 placeholder id 超出 toy embedding；
- 显式 `image_mask` / `video_mask`，并把视觉 token label 置为 `IGNORE_INDEX`；
- `pixel_values` / `pixel_values_videos` 行宽 `1176`，匹配 MiniMax `patch_size=14`、`temporal_patch_size=2`、RGB channel；
- `image_grid_thw` / `video_grid_thw` 为 `[[1, 2, 2]]`，每种 modality 对应 1 个 projected visual token。

未通过/未声明完成：

- full real checkpoint load；
- CUDA no-implicit-sync forward gate；
- public-checkpoint `VLMTrainer` smoke with real media；
- multi-device FSDP2 asymmetric dummy-forward runtime evidence；
- SP/EP e2e alignment；
- Ascend NPU kernel evidence。

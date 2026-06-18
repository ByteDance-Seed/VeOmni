# MiniMax M3 VL 超参加载报告

日期：2026-06-18

## 结论

MiniMax M3 VL 的模型超参数以 Hugging Face `MiniMaxAI/MiniMax-M3` 的 `config.json` 为当前自动验证来源，ModelScope `MiniMax/MiniMax-M3` 页面作为国内镜像入口已确认可访问。VeOmni 本 PR 不改全局 `transformers-stable` pin；默认 `transformers==5.9.0` 环境只保证 MiniMax config registry 和 fail-fast modeling gate 可用，实际 generated modeling 需要局部 `transformers>=5.12.0`。

本地 `MiniMaxM3VLConfig` 能解析官方 config 的 top-level / text / vision 三段结构，并把 legacy 字段转换为 transformers v5.12 generated modeling 需要的字段：

- `text_config.hidden_act="swigluoai"` 归一为 generated modeling 可用的 `hidden_act="silu"`，同时保留 `swiglu_alpha` / `swiglu_limit`。
- `text_config.sparse_attention_config` 展开为 `layer_types`、`index_*` 稀疏注意力字段。
- `text_config.moe_layer_freq` 展开为 dense/sparse MLP 分层。
- 缺省 `rope_parameters` 被补为 `{"rope_theta": ..., "rope_type": "default"}`，避免 generated modeling 读取 RoPE 参数失败。
- `vision_config.img_token_compression_config` 展开为 `spatial_merge_size` / `temporal_patch_size`。

## 官方源验证

执行过的源验证命令：

```bash
curl -fsSL https://huggingface.co/api/models/MiniMaxAI/MiniMax-M3
curl -fsSL https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/config.json
curl -fsSL https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/preprocessor_config.json
curl -sS -o /tmp/minimax_m3_vl_v590.py -w "%{http_code}" \
  https://raw.githubusercontent.com/huggingface/transformers/v5.9.0/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py
curl -sS -o /tmp/minimax_m3_vl_v512.py -w "%{http_code}" \
  https://raw.githubusercontent.com/huggingface/transformers/v5.12.0/src/transformers/models/minimax_m3_vl/modeling_minimax_m3_vl.py
curl -sS -o /tmp/modelscope_minimax_m3.html -w "%{http_code}" \
  https://modelscope.cn/models/MiniMax/MiniMax-M3
curl -sS -L -o /tmp/modelscope_minimax_config.json -w "%{http_code}" \
  https://modelscope.cn/models/MiniMax/MiniMax-M3/resolve/master/config.json
curl -sS -L -o /tmp/modelscope_minimax_preprocessor_config.json -w "%{http_code}" \
  https://modelscope.cn/models/MiniMax/MiniMax-M3/resolve/master/preprocessor_config.json
curl -fsSL -o /tmp/minimax_m3_model_index.json -w "%{http_code}" \
  https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/model.safetensors.index.json
```

结果：

| 项目 | 值 |
|---|---|
| Hugging Face repo | `MiniMaxAI/MiniMax-M3` |
| Hugging Face sha | `051e8f961274fb4e18ac3b57991f13bffedde212` |
| Last modified | `2026-06-16T05:18:24.000Z` |
| ModelScope page | `https://modelscope.cn/models/MiniMax/MiniMax-M3`, HTTP `200` |
| ModelScope API/files | HTTP `200`; `config.json`、`preprocessor_config.json`、`model.safetensors.index.json` present |
| ModelScope `config.json` | HTTP `200`, SHA256 `c9c97ce1e4eece60012d5a10ea87717458bfb1f19c2c7a615a3dbff83d090c6b`, byte-identical to HF |
| ModelScope `preprocessor_config.json` | HTTP `200`, SHA256 `36785da1dc7a00324048bd7c537c24247fca97c626884818257fd0b92287de31`, byte-identical to HF |
| HF `model.safetensors.index.json` | HTTP `200`; `weight_map` has `23416` keys |
| `model_type` | `minimax_m3_vl` |
| `architectures` | `["MiniMaxM3SparseForConditionalGeneration"]` |
| processor class | `MiniMaxVLProcessor` |
| AutoImageProcessor | `image_processor.MiniMaxM3VLImageProcessor` |
| AutoVideoProcessor | `video_processor.MiniMaxM3VLVideoProcessor` |
| AutoProcessor | `processing_minimax.MiniMaxVLProcessor` |
| transformers v5.9.0 MiniMax modeling | HTTP `404` |
| transformers v5.12.0 MiniMax modeling | HTTP `200` |

本轮再次复查官方源：HF sha 仍为 `051e8f961274fb4e18ac3b57991f13bffedde212`，`lastModified=2026-06-16T05:18:24.000Z`，`model.safetensors.index.json` 仍为 HTTP `200` 且 `weight_map` 为 `23416` 个 key。projector 相关 key 分为两组：`multi_modal_projector.linear_{1,2}.{weight,bias}` 对应 transformers generated `linear_{1,2}`，`patch_merge_mlp.linear_{1,2}.{weight,bias}` 对应 transformers generated `merge_linear_{1,2}`。

Public processor load evidence:

```text
AutoProcessor.from_pretrained("MiniMaxAI/MiniMax-M3", trust_remote_code=True)

processor_class MiniMaxVLProcessor
tokenizer_class TokenizersBackend
image_processor_class MiniMaxM3VLImageProcessor
video_processor_class MiniMaxM3VLVideoProcessor
image_token_id 200025
video_token_id 200026
image_merge_size 2
video_merge_size 2
```

因此配置与 processor/tokenizer metadata 已验证可从 HF public repo 加载；ModelScope raw config/preprocessor 与 HF 字节级一致。当前未完成的是 full public weight checkpoint 加载，而不是超参或 processor 元数据加载。

## 官方关键超参数

| 模块 | 字段 | 官方值 |
|---|---|---|
| top-level | image token | `image_token_index=200025` |
| top-level | video token | `video_token_index=200026` |
| top-level | projector hidden | `projector_hidden_size=6144` |
| text | hidden size | `6144` |
| text | layers | `60` |
| text | attention heads / KV heads | `64` / `4` |
| text | head dim | `128` |
| text | vocab size | `200064` |
| text | max positions | `1048576` |
| text | dense intermediate | `12288` |
| text | routed intermediate | `3072` |
| text | shared intermediate | `3072` |
| text | local experts / top-k | `128` / `4` |
| text | sparse attention | enabled, block size `128`, top-k blocks `16`, local blocks `1` |
| text | RoPE theta / rotary dim | `5000000` / `64` |
| vision | hidden size | `1280` |
| vision | layers | `32` |
| vision | attention heads | `16` |
| vision | intermediate size | `5120` |
| vision | patch size | `14` |
| vision | image size | `2016` |
| vision | channels | `3` |
| vision | RoPE | 3D RoPE, theta `10000.0` |
| vision | compression | `spatial_merge_size=2`, `temporal_patch_size=2` |

## VeOmni 解析验证

本地 toy config 覆盖了官方字段映射的关键分支：

- `tests/toy_config/minimax_m3_vl_toy/config.json`
- `tests/models/test_model_registry.py::test_minimax_m3_vl_config_registry_and_modeling_gate`

验证点包括：

- `config.model_type == "minimax_m3_vl"`；
- `config.text_config.model_type == "minimax_m3_vl_text"`；
- `config.vision_config.model_type == "minimax_m3_vl_vision"`；
- image/video token index 保持为 `200025` / `200026`；
- `hidden_act` 从官方 `swigluoai` 映射为 generated modeling 使用的 `silu`；
- sparse attention freq 映射为 `layer_types`；
- `moe_layer_freq` 映射为 `mlp_layer_types`；
- text/vision `rope_parameters` 被补齐并可 round-trip 保存/重载。
- nested `text_config` / `vision_config` 已实例化为 `PretrainedConfig` 时也能正确转换为 dict 后解析，避免 `.pop("model_type")` 崩溃。

最新 registry 兼容性验证：

```text
transformers==5.9.0: pytest tests/models/test_model_registry.py -k minimax_m3_vl -q
2 passed, 4 deselected in 11.57s

transformers==5.12.0: pytest tests/models/test_model_registry.py -k minimax_m3_vl -q
2 passed, 4 deselected in 5.06s
```

`test_minimax_m3_vl_config_accepts_pretrained_nested_configs` 固化了 review 反馈中的边界：如果上游或缓存路径先把 nested config 实例化为 `PretrainedConfig`，MiniMax top-level config 仍能解析 text/vision 字段，并保留 sparse attention / MoE layer 映射。

## 权重加载边界

当前 checkpoint converter 只声明和测试了可由源字段确定的 rename / merge：

- language tower prefix rename；
- dense MLP `gate_proj` + `up_proj` 合并为 `gate_up_proj`；
- MoE expert `w1` + `w3` 合并为 fused `gate_up_proj`，`w2` stack 为 `down_proj`；
- shared expert `gate_proj` + `up_proj` 合并为 `gate_up_proj`，router gate 映射；
- sparse-attention `index_{q,k}_{proj,norm}` 映射为 generated `self_attn.indexer.{q,k}_{proj,norm}`；
- `block_sparse_moe.e_score_correction_bias` 映射为 generated persistent buffer `mlp.gate.e_score_correction_bias`；
- vision tower prefix rename；
- public projector `multi_modal_projector.linear_{1,2}` 映射为 generated `model.multi_modal_projector.linear_{1,2}`；
- public `patch_merge_mlp.linear_{1,2}` 映射为 generated `model.multi_modal_projector.merge_linear_{1,2}`。

验证命令：

```text
pytest tests/models/test_checkpoint_tensor_converter.py -k MiniMax -q

10 passed, 51 deselected in 4.99s
```

真实 full checkpoint payload 加载仍未完成。原因不是 config 解析、processor/tokenizer 加载，也不再是 projector index mapping 缺口；当前剩余门槛是没有下载并实际加载 869 GB public safetensors tensor payload 做端到端权重加载。2026-06-18 重新 audit HF public index：

```text
public projector / patch-merge keys:
multi_modal_projector.linear_1.bias
multi_modal_projector.linear_1.weight
multi_modal_projector.linear_2.bias
multi_modal_projector.linear_2.weight
patch_merge_mlp.linear_1.bias
patch_merge_mlp.linear_1.weight
patch_merge_mlp.linear_2.bias
patch_merge_mlp.linear_2.weight

generated model projector keys:
model.multi_modal_projector.linear_1.bias
model.multi_modal_projector.linear_1.weight
model.multi_modal_projector.linear_2.bias
model.multi_modal_projector.linear_2.weight
model.multi_modal_projector.merge_linear_1.bias
model.multi_modal_projector.merge_linear_1.weight
model.multi_modal_projector.merge_linear_2.bias
model.multi_modal_projector.merge_linear_2.weight

missing after index conversion:
<none for projector keys>
```

`tests/models/test_checkpoint_tensor_converter.py::TestMiniMaxM3VLCheckpointTensorConverter::test_public_projector_index_mapping_covers_merge_weights_from_patch_merge_mlp` 固化了 projector 边界：converter 只做源字段可证明的 rename/merge，并证明 public `patch_merge_mlp` 可以闭合 generated `merge_linear_{1,2}` projector keys。`test_sparse_attention_indexer_and_e_score_bias_mapping` 固化了 public sparse-attention indexer 和 router correction buffer 映射。仍不能宣称真实 public checkpoint full load 完成，直到完成 869 GB tensor payload 的实际加载验证。

真实 HF index 与 safetensors shard-header 覆盖检查：

```text
curl -fsSL --retry 5 --retry-all-errors --connect-timeout 20 --max-time 180 \
  -o /tmp/minimax_m3_model.safetensors.index.json \
  https://huggingface.co/MiniMaxAI/MiniMax-M3/raw/main/model.safetensors.index.json

PYTHONPATH=$PWD uv run --no-project --python 3.11 \
  --with torch==2.7.1 --with transformers==5.12.0 \
  --with packaging --with psutil --with rich --with numpy --with pillow \
  --with requests --with safetensors --with tqdm \
  scripts/multimodal/verify_minimax_m3_vl_checkpoint_index.py \
  --config-path MiniMaxAI/MiniMax-M3 \
  --index-json /tmp/minimax_m3_model.safetensors.index.json \
  --verify-shard-metadata \
  --metadata-cache-dir /tmp/minimax_m3_safetensors_headers \
  --fail-on-unexpected \
  --json-output /tmp/minimax_m3_vl_checkpoint_metadata_report.json

public_weight_map_keys: 23416
converted_index_keys: 1582
public_safetensors_metadata_keys: 23416
converted_metadata_keys: 1582
safetensors_shards_read: 59
safetensors_header_bytes_read: 3440088
model_parameter_keys: 1525
model_persistent_buffer_keys: 57
model_state_keys: 1582
missing_state_key_count: 0
unexpected_index_key_count: 0
missing_metadata_key_count: 0
unexpected_metadata_key_count: 0
shape_mismatch_count: 0
dtype_mismatch_groups: {"F32->BF16": 115}
missing_projector_keys: []
checkpoint_values_downloaded: false
full_checkpoint_load_executed: false
```

Artifact:

- [minimax_m3_vl_checkpoint_index_report.json](./artifacts/minimax_m3_vl_checkpoint_index_report.json)
- [minimax_m3_vl_checkpoint_metadata_report.json](./artifacts/minimax_m3_vl_checkpoint_metadata_report.json)

`converted_index_keys` 少于 public `weight_map` keys 是预期行为：MoE expert shard 和 dense/shared `gate_proj` / `up_proj` 会在 converter 中 stack 或 concat 成 fused generated 参数。该验证器比较的是 generated model `state_dict()` keys，并排除 `persistent=False` 的 RoPE runtime buffers。开启 `--verify-shard-metadata` 后还会读取 59 个 public safetensors shard 的 header metadata，证明转换后的 key 和 shape 覆盖；但它仍不下载 tensor payload，也不等于已经完成 full checkpoint load。115 个 dtype 差异全部为 public router/gate 类权重 `F32` 到目标 BF16 model state 的差异，VeOmni checkpoint dispatch 会在真实加载时 cast 到目标 parameter/buffer dtype。

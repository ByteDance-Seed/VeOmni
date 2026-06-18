# MiniMax M3 VL 减层 SFT Loss 报告

日期：2026-06-18

## 结论

本轮补跑了两条减层 SFT 证据：

1. 旧的手写 tiny smoke：证明本地 JSONL、loss mask、optimizer 链路可用，80 step loss 从 `5.774437427520752` 降到 `0.0010620863176882267`。
2. 新的 generated-model smoke：直接通过 VeOmni registry 构建 `veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration`，在 `transformers==5.12.0` 临时环境中跑 8 个真实 backward/AdamW step，loss 从 `5.57344913482666` 降到 `4.035123348236084`。

第二条是本 PR 更关键的证据：它覆盖 patchgen 生成模型、MiniMax toy config、默认 position id、forward、loss、backward 和参数更新。它仍不代表真实 428B checkpoint 已完成加载，也不代表 GPU/NPU 性能门已通过。

## Generated Model Smoke

| 项目 | 值 |
|---|---|
| generated model | `patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration` |
| config | `tests/toy_config/minimax_m3_vl_toy/config.json` |
| 数据集 | `tests/fixtures/minimax_m3_vl_sft/tiny_sft.jsonl` |
| 权重 | 随机初始化，`actual_weights_loaded=false` |
| step 数 | `8` |
| batch size | `2` |
| learning rate | `0.005` |
| seed | `20260618` |
| transformers | `5.12.0` |

执行命令摘要：

```bash
env PYTHONPATH=$PWD \
  uv run --no-project --python 3.11 \
  --with transformers==5.12.0 --with torch==2.7.1 \
  --with packaging --with psutil --with einops --with numpy \
  --with safetensors --with tqdm --with rich \
  python - <<'PY'
# Builds MiniMaxM3SparseForConditionalGeneration through VeOmni registry,
# trains 8 AdamW steps on tests/fixtures/minimax_m3_vl_sft/tiny_sft.jsonl.
PY
```

关键输出：

```json
{
  "first_loss": 5.57344913482666,
  "last_loss": 4.035123348236084,
  "losses": [
    5.57344913482666,
    5.1658034324646,
    4.941288948059082,
    4.761152744293213,
    4.579178810119629,
    4.394667625427246,
    4.216017246246338,
    4.035123348236084
  ]
}
```

完整 JSON 证据：

- [generated_model_loss_log.json](./artifacts/minimax_m3_vl_sft_smoke/generated_model_loss_log.json)

## Legacy Tiny Smoke

保留旧 smoke 作为数据/loss-mask 回归证据：

- 脚本：`tests/train_scripts/train_minimax_m3_vl_sft_smoke.py`
- 输出：`docs/usage/support_new_models/artifacts/minimax_m3_vl_sft_smoke/loss_log.json`
- loss curve：`docs/usage/support_new_models/artifacts/minimax_m3_vl_sft_smoke/loss_curve.png`

该脚本使用 test-scope 手写 tiny 模型，不是 production generated modeling。它的价值是快速证明 JSONL fixture、`IGNORE_INDEX=-100` 和优化器链路。

![MiniMax M3 VL tiny SFT smoke loss curve](./artifacts/minimax_m3_vl_sft_smoke/loss_curve.png)

## 验收解释

本报告证明：

- `minimax_m3_vl` toy config 可通过 VeOmni registry 加载。
- patchgen 生成的 MiniMax modeling 可在局部 `transformers==5.12.0` 环境中实例化并训练。
- `get_position_id_func()` 返回 `None` 后，默认 1-D packed position ids 能支撑文本减层训练。
- sparse MoE/dense MLP 分支在 toy config 中被覆盖。

本报告不证明：

- 真实 MiniMaxAI/MiniMax-M3 safetensors 已完整加载；
- 真实 public checkpoint full trainer smoke 已完成；
- 多卡 FSDP2 回归、SP metadata 或 NPU kernel 优化已完成；
- MSA 长上下文性能门已通过。

多模态 trainer glue 的 synthetic image/video 证据已在数据模块报告和迁移报告中记录：它通过 `VLMTrainer` transform/collator 入口、真实 transformers MiniMax image/video processor、toy generated model 和单进程 backward/optimizer smoke；本减层报告只声明 text-style toy SFT loss。

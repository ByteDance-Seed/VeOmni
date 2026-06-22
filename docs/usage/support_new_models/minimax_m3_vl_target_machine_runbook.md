# MiniMax M3 VL Target-Machine Precision Runbook

日期：2026-06-22

## 目标

本手册给目标机器操作者使用，用来产出 MiniMax M3 VL 真实 checkpoint 和 NPU/多卡精度证据。完成标准不是 SFT loss 下降，而是最终 artifact audit 同时证明：

- full public-checkpoint preflight passed；
- full public-checkpoint forward parity passed；
- multi-card SP/EP/FSDP2 parity passed；
- final strict audit passed。

官方精度来源固定为 Hugging Face `MiniMaxAI/MiniMax-M3` config、processor 和 checkpoint；Transformers MiniMax model class 只是 reference loader。当前建议 pin 到 HF revision：

```text
b1c79b9c07578aeebf33c2aeff0f6de8a96b02b1
```

## 0. 准备 PR 代码

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
cd VeOmni
git fetch https://github.com/Kirrito-k423/VeOmni.git codex/minimax-m3-vl-slice
git checkout -B minimax-m3-vl-precision FETCH_HEAD
```

记录本次代码版本：

```bash
git rev-parse HEAD | tee /tmp/minimax_m3_vl_veomni_commit.txt
```

## 1. 下载并固定官方 checkpoint

目标目录示例：

```bash
export MINIMAX_M3_REVISION=b1c79b9c07578aeebf33c2aeff0f6de8a96b02b1
export MINIMAX_M3_CHECKPOINT=/data/checkpoints/MiniMax-M3

python3 -m pip install -U "huggingface_hub[cli]"
hf download MiniMaxAI/MiniMax-M3 \
  --revision "$MINIMAX_M3_REVISION" \
  --local-dir "$MINIMAX_M3_CHECKPOINT"
```

下载后先确认本地文件规模，不进入模型加载：

```bash
find "$MINIMAX_M3_CHECKPOINT" -maxdepth 1 -name 'model-*.safetensors' | wc -l
du -sh "$MINIMAX_M3_CHECKPOINT"
test -f "$MINIMAX_M3_CHECKPOINT/config.json"
test -f "$MINIMAX_M3_CHECKPOINT/model.safetensors.index.json"
```

通过预期：

- safetensors shard 数量为 `59`；
- checkpoint 目录约 `854G`；
- `config.json` 和 `model.safetensors.index.json` 存在。

## 2. 准备 Python 与 NPU runtime

目标环境必须能在同一个 Python 里 import：

- `torch`
- `torch_npu`，如果 candidate 或 multi-card 是 NPU；
- `transformers>=5.12.0`
- `safetensors`
- VeOmni 当前工作树。

先做只读 runtime 探测：

```bash
which python3
python3 --version

if [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/set_env.sh
elif [ -f /usr/local/Ascend/ascend-toolkit/latest/set_env.sh ]; then
  . /usr/local/Ascend/ascend-toolkit/latest/set_env.sh
fi

/usr/local/sbin/npu-smi info || /usr/local/bin/npu-smi info || npu-smi info
```

如果普通用户不能跑 `npu-smi info`，但 root 可以，后续命令统一使用：

```bash
export MINIMAX_NPU_SMI_CMD='sudo -n /usr/local/sbin/npu-smi info'
```

如果 `sudo -n` 不可用，需要先按平台批准方式进入 root shell，再在 root shell 中运行本手册命令。不要把 root-only 失败误判为驱动坏。

Python runtime gate：

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}

python3 - <<'PY'
import json
import os
import torch
import transformers

report = {
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "cuda_available": bool(torch.cuda.is_available()) if hasattr(torch, "cuda") else False,
    "cuda_device_count": int(torch.cuda.device_count()) if hasattr(torch, "cuda") and torch.cuda.is_available() else 0,
    "ascend_env": {name: os.environ.get(name) for name in (
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
    )},
}
try:
    import torch_npu
    report["torch_npu"] = getattr(torch_npu, "__version__", None)
    report["npu_available"] = bool(torch.npu.is_available())
    report["npu_device_count"] = int(torch.npu.device_count()) if torch.npu.is_available() else 0
    if torch.npu.is_available():
        x = torch.ones((2, 2), device="npu")
        report["npu_tensor_sum"] = float(x.sum().cpu())
except Exception as exc:
    report["torch_npu_error"] = repr(exc)
print(json.dumps(report, indent=2, sort_keys=True))
PY
```

NPU 目标机通过预期：

- `torch_npu` import 成功；
- `npu_available=true`；
- `npu_device_count>=1`，多卡要求 `>=8`；
- tiny NPU tensor op 成功；
- 记录了 `ASCEND_RT_VISIBLE_DEVICES` 或 `ASCEND_VISIBLE_DEVICES`。

## 3. 单卡 toy parity 快速烟测

在跑 854G checkpoint 之前，先确认代码路径和 NPU tolerance 可用：

```bash
export OUTPUT_ROOT=docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite
mkdir -p "$OUTPUT_ROOT/toy"

python3 scripts/multimodal/verify_minimax_m3_vl_precision_parity.py \
  --reference-device cpu \
  --candidate-device npu \
  --lr 1e-5 \
  --atol 5e-4 --rtol 5e-4 \
  --grad-atol 1e-3 --grad-rtol 1e-3 \
  --param-atol 1e-4 --param-rtol 1e-4 \
  --output-json "$OUTPUT_ROOT/toy/toy_hf_cpu_veomni_npu_parity_target.json"
```

继续条件：

- JSON 顶层 `passed=true`；
- `failed=[]`；
- runtime 中 NPU 信息完整。

## 4. Full checkpoint preflight

先跑 preflight，不加载权重：

```bash
scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh \
  --checkpoint-dir "$MINIMAX_M3_CHECKPOINT" \
  --config-path "$MINIMAX_M3_CHECKPOINT" \
  --reference-device cpu \
  --candidate-device npu \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd "${MINIMAX_NPU_SMI_CMD:-}" \
  --official-reference-revision "$MINIMAX_M3_REVISION" \
  --preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_preflight.json \
  --preflight-only
```

如果没有 root-only `npu-smi`，去掉 `--npu-smi-cmd "${MINIMAX_NPU_SMI_CMD:-}"` 这一行。

强制审计 preflight：

```bash
python3 scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py \
  --require-full-checkpoint-preflight \
  --full-preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_preflight.json \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_preflight_audit.json
```

继续条件：

- preflight JSON `passed=true`、`issues=[]`；
- audit JSON `passed=true`、`full_checkpoint_preflight_passed=true`；
- `official_reference.revision` 等于 `$MINIMAX_M3_REVISION`；
- `model_entrypoints.transformers_reference_loader_class.import_ok=true`；
- `model_entrypoints.checkpoint_converter_class.import_ok=true`。

## 5. Full checkpoint forward parity

确认 preflight 通过后再加载完整 checkpoint：

```bash
scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh \
  --checkpoint-dir "$MINIMAX_M3_CHECKPOINT" \
  --config-path "$MINIMAX_M3_CHECKPOINT" \
  --reference-device cpu \
  --candidate-device npu \
  --torch-dtype bfloat16 \
  --prompt-kind multimodal \
  --seq-len 10 \
  --top-k 8 \
  --max-new-tokens 8 \
  --atol 5e-4 --rtol 5e-4 \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd "${MINIMAX_NPU_SMI_CMD:-}" \
  --official-reference-revision "$MINIMAX_M3_REVISION" \
  --preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_preflight.json \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_forward.json \
  --audit-json docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/full_checkpoint/full_checkpoint_audit.json
```

如果目标机有 CUDA 且可以作为 reference，也可以把 `--reference-device cpu` 改为 `--reference-device cuda`。NPU candidate 不要求 bitwise，需要保留 tolerance。

继续条件：

- `full_checkpoint_forward.json` 顶层 `passed=true`；
- `full_checkpoint_load_executed=true`；
- `payload.payload_bytes_read > 800000000000`；
- `payload.public_keys_read >= 20000`；
- `metadata_comparison.shape_mismatch_count=0`；
- `forward.state_dict_load.strict=true`；
- `forward.failed=[]`；
- `forward.checks` 中 logits、image/video hidden states、top-k ids、greedy ids 均通过；
- `full_checkpoint_audit.json` 中 `passed=true`、`full_checkpoint_forward_passed=true`。

## 6. 多卡 SP/EP/FSDP2 parity

目标机器至少 8 张设备可见，且每张设备满足 free HBM 门槛后再跑：

```bash
scripts/multimodal/run_minimax_m3_vl_multicard_parity.sh \
  --min-devices 8 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd "${MINIMAX_NPU_SMI_CMD:-}" \
  --output-dir docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/multicard
```

继续条件：

- `multicard_parity_summary.json` 顶层 `passed=true`；
- `preflight.returncode=0`；
- `dummy_forward.returncode=0`；
- `e2e_align.returncode=0`；
- preflight log 中 `device_count>=8`、`errors=[]`；
- NPU preflight log 中 `torch_npu_version` 非空，且 visible device env 非空。

## 7. 最终一键套件

如果希望一次性串联 full checkpoint、multi-card 和最终 audit，用套件入口：

```bash
scripts/multimodal/run_minimax_m3_vl_precision_suite.sh \
  --checkpoint-dir "$MINIMAX_M3_CHECKPOINT" \
  --config-path "$MINIMAX_M3_CHECKPOINT" \
  --reference-device cpu \
  --candidate-device npu \
  --torch-dtype bfloat16 \
  --prompt-kind multimodal \
  --seq-len 10 \
  --top-k 8 \
  --max-new-tokens 8 \
  --atol 5e-4 --rtol 5e-4 \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd "${MINIMAX_NPU_SMI_CMD:-}" \
  --official-reference-revision "$MINIMAX_M3_REVISION" \
  --min-devices 8 \
  --output-root docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite
```

最终完成标准：

```bash
python3 - <<'PY'
import json
from pathlib import Path

path = Path("docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite/final_precision_audit.json")
data = json.loads(path.read_text())
required = {
    "passed": True,
    "full_checkpoint_preflight_passed": True,
    "full_checkpoint_forward_passed": True,
    "multicard_passed": True,
}
for key, value in required.items():
    actual = data.get(key)
    print(f"{key}={actual}")
    if actual is not value:
        raise SystemExit(f"{path} does not prove completion: {key}={actual!r}")
PY
```

## 8. 打包回传 evidence

跑完后生成 manifest 并打包，不要把 854G checkpoint 打进包：

```bash
export OUTPUT_ROOT=docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite

{
  echo "veomni_commit $(git rev-parse HEAD)"
  echo "minimax_revision ${MINIMAX_M3_REVISION:-unset}"
  echo "checkpoint_dir ${MINIMAX_M3_CHECKPOINT:-unset}"
  date -u +"generated_at %Y-%m-%dT%H:%M:%SZ"
} > "$OUTPUT_ROOT/run_metadata.txt"

tmp_manifest="$(mktemp)"
find "$OUTPUT_ROOT" -type f ! -name artifact_manifest.sha256 -print0 | sort -z | xargs -0 sha256sum \
  > "$tmp_manifest"
mv "$tmp_manifest" "$OUTPUT_ROOT/artifact_manifest.sha256"

tar -czf /tmp/minimax_m3_vl_target_precision_suite_artifacts.tgz \
  "$OUTPUT_ROOT"
```

回传前可以在目标机自审一次包内容：

```bash
python3 scripts/multimodal/audit_minimax_m3_vl_target_artifact_bundle.py \
  --artifact-tar /tmp/minimax_m3_vl_target_precision_suite_artifacts.tgz \
  --expected-revision "$MINIMAX_M3_REVISION" \
  --expected-veomni-commit "$(git rev-parse HEAD)" \
  --require-target-toy \
  --output-json /tmp/minimax_m3_vl_target_precision_suite_artifacts_audit.json

python3 scripts/multimodal/audit_minimax_m3_vl_precision_objective.py \
  --target-bundle-audit-json /tmp/minimax_m3_vl_target_precision_suite_artifacts_audit.json \
  --require-complete \
  --output-json /tmp/minimax_m3_vl_target_precision_objective_audit.json
```

回传文件：

- `/tmp/minimax_m3_vl_target_precision_suite_artifacts.tgz`
- `/tmp/minimax_m3_vl_target_precision_suite_artifacts_audit.json`
- `/tmp/minimax_m3_vl_target_precision_objective_audit.json`
- 如果 full forward 或 multi-card 失败，也回传同一个 tar 包；失败日志本身是定位依据。

## 9. 失败时停止点

- Gate 0 / NPU runtime 失败：先修驱动、CANN、torch/torch_npu 或权限，不要跑模型。
- Toy parity 失败：先修 generated model/backend tolerance，不要加载 854G checkpoint。
- Full preflight 失败：先修 checkpoint 完整性、source revision、runtime import 或 HBM/disk，不要跑 full forward。
- Full forward 失败：检查 strict state load、input contract、vision hidden states、router selected experts、logits、greedy ids；不要直接扩大 tolerance。
- Multi-card 失败：保留 preflight/dummy/e2e logs，先定位设备数、rank hang、SP/EP/FSDP2 collectives，再判断是否模型精度问题。

只有最终 strict audit 通过后，才能声明 MiniMax M3 VL 真实 checkpoint 与多卡精度闭环完成。

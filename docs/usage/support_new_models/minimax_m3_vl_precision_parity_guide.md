# MiniMax M3 VL 精度对齐操作手册

日期：2026-06-22

## 目标

MiniMax M3 VL 迁移的精度结论必须来自 reference parity，而不是只看 SFT loss 是否下降。本手册把 `transformers==5.12.0` 原始 MiniMax M3 VL modeling 作为 reference，把 VeOmni patchgen generated modeling 作为 candidate，要求在同 config、同权重、同输入下逐层验证 forward、backward 和 optimizer update。

Transformers 原仓提供 MiniMax M3 VL reference modeling 和通用训练组件，但没有 MiniMax M3 专用 SFT recipe。因此：

- `Trainer` / `TRL SFTTrainer` 能作为训练工具参考；
- 精度保证必须由本手册的 parity gates 给出；
- toy SFT loss 下降只能证明训练链路可用，不能替代 reference parity。

## 已落地的 Toy Parity Gate

脚本：

- `scripts/multimodal/verify_minimax_m3_vl_precision_parity.py`
- `scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py`
- `scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh`
- `scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py`
- `scripts/multimodal/run_minimax_m3_vl_precision_suite.sh`

本地 CPU/NPU 证据：

- [toy_hf_veomni_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity.json)
- [toy_hf_veomni_parity_npu.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_npu.json)
- [toy_hf_cpu_veomni_npu_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_cpu_veomni_npu_parity.json)
- [parity_artifact_audit.json](./artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit.json)

结果摘要：

```json
{
  "passed": true,
  "reference_device": "cpu / npu:0",
  "candidate_device": "cpu / npu:0",
  "num_checks": 38,
  "failed": []
}
```

该 gate 覆盖：

- HF reference class: `transformers.models.minimax_m3_vl.modeling_minimax_m3_vl.MiniMaxM3SparseForConditionalGeneration`
- VeOmni candidate class: `veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration`
- 同一份随机初始化 `state_dict`，`strict=True` 加载到 candidate；
- 同一份 mixed image+video toy batch；
- input contract: `input_ids`、`attention_mask`、`position_ids`、VeOmni `multimodal_metadata` collate contract；
- forward: `loss`、`logits`、`image_hidden_states`、`video_hidden_states`；
- routing: `MiniMaxM3VLTopKRouter` 的 `router_logits`、`top_k_weights`、`selected_experts`；
- backward: embedding、attention q/k/v/o、MoE gate/experts、projector、`lm_head` 梯度；
- optimizer: 同一 AdamW one-step 后关键参数 delta。

## Artifact 审计

每次更新 parity evidence 后，先跑默认 artifact 审计，确认当前 PR 中已声明的证据包自洽：

```bash
cd /path/to/VeOmni

python scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit.json
```

当前 PR 的审计结果应为：

```json
{
  "passed": true,
  "full_checkpoint_preflight_passed": false,
  "full_checkpoint_forward_passed": false,
  "multicard_passed": false,
  "require_full_checkpoint_preflight": false,
  "require_full_checkpoint_forward": false,
  "require_multicard": false
}
```

这表示 toy precision、真实 tensor payload sample、toy checkpoint forward smoke 和 CPU reference vs NPU candidate checkpoint-forward smoke 都已通过，但完整 869 GB checkpoint preflight 和 forward parity 尚未作为审计输入提供。
多卡 SP/EP/FSDP2 summary 也尚未作为审计输入提供，所以 `multicard_passed=false`。

在目标机器只完成 full-checkpoint preflight 后，可以先用强制 preflight 模式确认目标机具备开始真实 forward 的条件：

```bash
cd /path/to/VeOmni

python scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py \
  --require-full-checkpoint-preflight \
  --full-preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu_preflight.json \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit_preflight.json
```

注意：`full_checkpoint_preflight_passed=true` 只能证明本地 59 shard、payload bytes、runtime 和资源门槛满足要求，不能替代 logits/top-k/greedy forward parity。

在目标机器跑完完整 public checkpoint forward 和多卡 SP/EP/FSDP2 alignment 后，必须改用最终强制模式，把 full-checkpoint preflight、full-checkpoint forward artifact 与 multicard artifact 都作为输入：

```bash
cd /path/to/VeOmni

python scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py \
  --require-full-checkpoint-preflight \
  --full-preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu_preflight.json \
  --require-full-checkpoint-forward \
  --full-forward-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu.json \
  --require-multicard \
  --multicard-json docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity/multicard_parity_summary.json \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit_full.json
```

只有强制模式也 `passed=true`，才可以把真实 checkpoint forward parity 和多卡 SP/EP/FSDP2 parity 视为完成。

目标机器也可以使用最终套件入口一次性串联 full-checkpoint forward、多卡 SP/EP/FSDP2 和最终 strict audit：

```bash
cd /path/to/VeOmni

scripts/multimodal/run_minimax_m3_vl_precision_suite.sh \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --reference-device cpu \
  --candidate-device npu \
  --torch-dtype bfloat16 \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd 'sudo -n /usr/local/sbin/npu-smi info' \
  --min-devices 8 \
  --output-root docs/usage/support_new_models/artifacts/minimax_m3_vl_target_precision_suite
```

套件会产出：

- `full_checkpoint/full_checkpoint_preflight.json`
- `full_checkpoint/full_checkpoint_forward.json`
- `full_checkpoint/full_checkpoint_audit.json`
- `multicard/multicard_parity_summary.json`
- `final_precision_audit.json`

只有 `final_precision_audit.json` 中 `passed=true`、`full_checkpoint_preflight_passed=true`、`full_checkpoint_forward_passed=true`、`multicard_passed=true` 同时成立，才可以声明 MiniMax M3 VL 的真实 checkpoint 和多卡精度闭环完成。

## CPU 复跑

在任意没有 GPU/NPU 的机器上，先跑 CPU parity：

```bash
cd /path/to/VeOmni

PYTHONPATH=$PWD uv run --no-project --python 3.11 \
  --with torch==2.7.1 \
  --with transformers==5.12.0 \
  --with packaging --with psutil --with safetensors \
  --with numpy --with einops --with regex \
  python scripts/multimodal/verify_minimax_m3_vl_precision_parity.py \
    --device cpu \
    --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_cpu.json
```

通过标准：

- 命令 exit code 为 `0`；
- JSON 中 `passed=true`；
- `state_dict_load.missing_keys=[]`；
- `state_dict_load.unexpected_keys=[]`；
- 所有 `checks[*].allclose` 或 `checks[*].equal` 为 true。

## GPU 复跑

当前开发机没有 CUDA，所以 GPU 需要在有 CUDA 的机器上复跑。建议使用项目 GPU 环境或本地 CUDA torch 环境，不要改变 VeOmni 全局 `transformers-stable` pin；只在本命令的局部环境中使用 `transformers==5.12.0`。

```bash
cd /path/to/VeOmni

# 示例：如果机器已有兼容 CUDA 的 torch/transformers 环境，直接激活后运行：
export PYTHONPATH=$PWD
python scripts/multimodal/verify_minimax_m3_vl_precision_parity.py \
  --device cuda \
  --atol 2e-5 --rtol 2e-5 \
  --grad-atol 5e-5 --grad-rtol 5e-5 \
  --param-atol 5e-5 --param-rtol 5e-5 \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_cuda.json
```

如果用 `uv` 临时环境，请把 `torch` 改成该 GPU 机器匹配的 CUDA wheel 来源。不要在 PR 里提交机器本地环境锁文件。

通过标准同 CPU。若 GPU 由于 kernel 或 dtype 产生微小差异，优先调查具体 failed check；只有确认差异来自预期 backend 数值误差时，才调整 tolerance，并把调整原因写进 JSON 对应报告。

## NPU 复跑

NPU parity 分两层：

1. HF reference 原始 modeling 与 VeOmni NPU generated modeling 的 toy parity；
2. VeOmni GPU/CPU reference 与 NPU candidate 的 backend tolerance parity。

本 PR 已在单卡 Ascend 910B3 容器环境中完成第 1 层 toy parity：

```json
{
  "passed": true,
  "reference_device": "npu:0",
  "candidate_device": "npu:0",
  "num_checks": 38,
  "torch_version": "2.10.0+cpu",
  "torch_npu_version": "2.10.0",
  "transformers_version": "5.12.0"
}
```

同一台容器环境也完成了第 2 层 CPU reference vs NPU candidate toy parity：

```json
{
  "passed": true,
  "reference_device": "cpu",
  "candidate_device": "npu:0",
  "num_checks": 38,
  "optimizer": {"name": "AdamW", "lr": 1e-5},
  "tolerances": {
    "forward": {"atol": 5e-4, "rtol": 5e-4},
    "grad": {"atol": 1e-3, "rtol": 1e-3},
    "param": {"atol": 1e-4, "rtol": 1e-4}
  }
}
```

单卡 NPU toy parity 示例：

```bash
cd /path/to/VeOmni

docker run --rm --shm-size=8g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD":/workspace/VeOmni \
  -w /workspace/VeOmni \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e MODELING_BACKEND=veomni \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 \
  bash -lc '
    export PYTHONPATH=/workspace/VeOmni
    python3 -m pip install --quiet --target /tmp/transformers512 --no-deps transformers==5.12.0
    export PYTHONPATH=/tmp/transformers512:/workspace/VeOmni:$PYTHONPATH
    python3 scripts/multimodal/verify_minimax_m3_vl_precision_parity.py \
      --device npu \
      --atol 5e-4 --rtol 5e-4 \
      --grad-atol 1e-3 --grad-rtol 1e-3 \
      --param-atol 1e-3 --param-rtol 1e-3 \
      --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_npu.json
  '
```

单卡 CPU reference vs NPU candidate parity 示例：

```bash
cd /path/to/VeOmni

docker run --rm --shm-size=8g \
  --device=/dev/davinci0 \
  --device=/dev/davinci_manager \
  --device=/dev/devmm_svm \
  --device=/dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$PWD":/workspace/VeOmni \
  -w /workspace/VeOmni \
  -e ASCEND_RT_VISIBLE_DEVICES=0 \
  -e MODELING_BACKEND=veomni \
  quay.io/ascend/vllm-ascend:v0.20.2rc1 \
  bash -lc '
    export PYTHONPATH=/workspace/VeOmni
    python3 -m pip install --quiet --target /tmp/transformers512 --no-deps transformers==5.12.0
    export PYTHONPATH=/tmp/transformers512:/workspace/VeOmni:$PYTHONPATH
    python3 scripts/multimodal/verify_minimax_m3_vl_precision_parity.py \
      --reference-device cpu \
      --candidate-device npu \
      --lr 1e-5 \
      --atol 5e-4 --rtol 5e-4 \
      --grad-atol 1e-3 --grad-rtol 1e-3 \
      --param-atol 1e-4 --param-rtol 1e-4 \
      --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/toy_hf_cpu_veomni_npu_parity.json
  '
```

跨 backend AdamW 首步对极小梯度符号差很敏感，`lr=1e-5` 用于避免把 backend 浮点噪声放大成近 `2 * lr` 的假失败；它仍然比较同一 AdamW 配置下的一步参数 delta。

如果使用已缓存的 transformers 5.12.0 目录，也可以像 NPU loss smoke 那样把缓存目录挂载到容器，只要 `transformers.__version__ == "5.12.0"` 且包含 `transformers.models.minimax_m3_vl`。

NPU 通过标准：

- `npu-smi info` 能看到目标设备；
- `torch.npu.is_available()` 为 true；
- parity JSON `passed=true`；
- tolerance 必须记录在 JSON；
- 不允许用 CPU loss 下降替代 NPU parity。

## 真实 Checkpoint Parity

现有 PR 已完成 public checkpoint index 和 safetensors header 级验证，但还没有下载并加载 869 GB tensor payload。因此真实 checkpoint parity 仍是生产完成前的必需门禁。

脚本：

- `scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py`
- `scripts/multimodal/preflight_minimax_m3_vl_full_checkpoint_parity.py`

已落地的远程抽样 payload 证据：

- [real_checkpoint_payload_remote_sample.json](./artifacts/minimax_m3_vl_precision_parity/real_checkpoint_payload_remote_sample.json)
- [toy_checkpoint_forward_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_checkpoint_forward_parity.json)
- [toy_checkpoint_cpu_npu_forward_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_checkpoint_cpu_npu_forward_parity.json)

该证据没有下载完整 shard，而是通过 HTTP Range 从 Hugging Face `model-00001/00003/00026/00059-of-00059.safetensors` 读取 11 个真实 tensor payload，共 `55808` bytes，经 converter 映射到 VeOmni generated state keys 后检查 shape/dtype/value fingerprint。样本覆盖 language dense/sparse attention norm、MoE router correction bias、vision tower、multi-modal projector 和 patch-merge projector；随后按 generated model state metadata 执行 sampled state-load cast/copy 校验，`sampled_state_load.passed=true`、`loaded_tensor_count=11`、`value_mismatch_count=0`。它证明真实 checkpoint payload 字节、converter 路径和目标 runtime state dtype cast/copy 路径已被执行，但仍不是 full-checkpoint logits parity。

`toy_checkpoint_forward_parity.json` 使用完整 toy safetensors checkpoint 和 `torch_dtype=float32` 验证了 `--mode forward --prompt-kind multimodal` 的执行路径：先加载 HF reference 生成 logits/top-k/greedy 和 image/video hidden-state baseline，释放 HF 模型，再把 public checkpoint tensors 边读、边转换、边写入 VeOmni generated model。该 smoke 中顶层 `full_checkpoint_load_executed=true`、`num_checks=12`、`failed=[]`，`streaming_model_load=true`，strict missing/unexpected key count 都为 `0`，input ids/mask/position/grid/pixels、`forward.logits`、`forward.image_hidden_states`、`forward.video_hidden_states`、top-k 和 greedy ids 全部一致；它只证明 runner 逻辑，不替代真实 869 GB checkpoint parity。`toy_checkpoint_cpu_npu_forward_parity.json` 使用同一 toy checkpoint 验证了 `--reference-device cpu --candidate-device npu` 路径，在单卡 Ascend 910B3 上同样 `num_checks=12`、`failed=[]`，并记录 `runtime.torch_npu_version=2.10.0`、`runtime.torch_npu_available=true`、`runtime.torch_npu_device_count=1`、`runtime.ascend_env.ASCEND_RT_VISIBLE_DEVICES=0` 和 forward tolerance。真实 CUDA/NPU 目标机可使用 `torch_dtype=bfloat16`，但必须在 artifact 中保留对应 tolerance、runtime 和 `failed=[]` 证据。

推荐分阶段执行。

1. **抽样 shard payload parity**
   - 下载覆盖 language embedding、至少一层 attention、至少一个 sparse MoE 层、projector、`lm_head` 的 safetensors shard；
   - 用 `scripts/multimodal/verify_minimax_m3_vl_checkpoint_index.py --verify-shard-metadata` 先确认 metadata；
   - 通过 payload parity 脚本读取真实 tensor payload，经 `MiniMaxM3VLCheckpointTensorConverter` 转成 VeOmni generated key；
   - 检查 converted tensor key、shape、dtype group、SHA256/value stats，并输出 JSON 证据；
   - 若抽样只覆盖部分 expert/gate-up group，可先加 `--allow-incomplete-groups` 产出诊断 JSON；正式通过证据必须覆盖完整 group，不应依赖该开关。

远程 range 抽样示例命令：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --config-path MiniMaxAI/MiniMax-M3 \
  --index-json /path/to/model.safetensors.index.json \
  --shard-base-url https://huggingface.co/MiniMaxAI/MiniMax-M3/resolve/main/ \
  --include-key-regex '^language_model\.model\.layers\.0\.self_attn\.[qk]_norm\.weight$|^language_model\.model\.layers\.3\.self_attn\.index_[qk]_norm\.weight$|^language_model\.model\.layers\.3\.block_sparse_moe\.e_score_correction_bias$|^vision_tower\.vision_model\.encoder\.layers\.0\.(layer_norm1\.weight|self_attn\.q_proj\.bias)$|^multi_modal_projector\.linear_[12]\.bias$|^patch_merge_mlp\.linear_[12]\.bias$' \
  --torch-dtype bfloat16 \
  --metadata-cache-dir /tmp/minimax_m3_safetensors_headers \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_payload_remote_sample.json
```

本地 shard 抽样示例命令：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --config-path /data/checkpoints/MiniMax-M3 \
  --include-key-regex 'language_model\.model\.embed_tokens\.weight|language_model\.lm_head\.weight|multi_modal_projector\.|patch_merge_mlp\.' \
  --include-key-regex 'language_model\.model\.layers\.0\.(self_attn|mlp|block_sparse_moe)\.' \
  --torch-dtype bfloat16 \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_payload_sample.json
```

抽样 payload 通过标准：

- 命令 exit code 为 `0`；
- JSON 中 `passed=true`；
- `payload.converter_finalize_error=null`；
- `metadata_comparison.shape_mismatch_count=0`；
- `metadata_comparison.missing_model_key_count=0`；
- `sampled_state_load.passed=true`，且 `sampled_state_load.value_mismatch_count=0`；
- dtype mismatch 仅允许为已解释的 checkpoint-to-runtime cast，例如 router/gate `F32 -> BF16`。

2. **全量 payload load parity**
   - 下载完整 `59` 个 public safetensors shard；
   - HF reference 通过原始 `from_pretrained` 或等价 loader 加载；
   - VeOmni candidate 通过 checkpoint converter 加载；
   - 固定 prompt 集比较：
     - logits max abs / max rel；
     - top-k token ids；
     - greedy decode 首 N token；
     - image/video prompt 的 projector output 和 final logits。

`--mode forward` 会按顺序执行 HF baseline 和 VeOmni candidate：HF 结果写入 CPU baseline 后释放 HF 模型，再读取/转换 checkpoint payload，并以 streaming assign 方式写入 VeOmni model。这样避免同时保留两个模型和完整 converted tensor dict，更适合大 checkpoint 目标机，但完整 869 GB payload 仍需要足够磁盘、CPU 内存和设备内存。

推荐在目标机器使用 launcher。第一步先只跑 preflight，不加载权重，确认本地 checkpoint、Python/NPU runtime 和资源条件足够：

```bash
cd /path/to/VeOmni

scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --reference-device cpu \
  --candidate-device npu \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd 'sudo -n /usr/local/sbin/npu-smi info' \
  --preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu_preflight.json \
  --preflight-only
```

Preflight 通过标准：

- `passed=true`，`issues=[]`；
- `checkpoint.index_exists=true`；
- `checkpoint.weight_map_keys>=20000`；
- `checkpoint.selected_shard_count>=59`；
- `checkpoint.missing_shards=[]`；
- `checkpoint.payload_bytes_present>=800000000000`；
- `config.config_json_exists=true`；
- `runtime.transformers_version>=5.12.0`；
- 如果 reference 或 candidate 是 NPU，`runtime.torch_npu_version` 不能为空，`runtime.npu_available=true`，并记录可见 Ascend 设备环境变量；
- 如果设置 `--require-free-hbm-mb`，`runtime.npu_smi.returncode=0`，且至少一张 NPU 满足 free HBM 门槛；
- 如果设置 `--require-free-disk-gb`，checkpoint 和 output filesystem 都满足对应 free disk 门槛。

第二步再运行完整 forward parity；launcher 会先重复 preflight，再运行完整 forward parity，最后用 artifact auditor 的强制模式确认 full checkpoint evidence 真的满足门禁：

```bash
cd /path/to/VeOmni

scripts/multimodal/run_minimax_m3_vl_full_checkpoint_parity.sh \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --reference-device cpu \
  --candidate-device npu \
  --torch-dtype bfloat16 \
  --prompt-kind multimodal \
  --top-k 8 \
  --max-new-tokens 8 \
  --require-free-disk-gb 50 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd 'sudo -n /usr/local/sbin/npu-smi info' \
  --preflight-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu_preflight.json \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu.json \
  --audit-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit_full.json
```

如需在目标机器只确认命令而不做 preflight 或加载权重，可加 `--dry-run`。

完整 text-prompt forward 示例：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --config-path /data/checkpoints/MiniMax-M3 \
  --mode forward \
  --reference-device cuda \
  --candidate-device cuda \
  --torch-dtype bfloat16 \
  --prompt-ids 1,1209,318,257,1332 \
  --top-k 8 \
  --max-new-tokens 8 \
  --atol 2e-4 --rtol 2e-4 \
  --confirm-full-load \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cuda.json
```

完整 image+video-prompt forward 示例：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --config-path /data/checkpoints/MiniMax-M3 \
  --mode forward \
  --reference-device cuda \
  --candidate-device cuda \
  --torch-dtype bfloat16 \
  --prompt-kind multimodal \
  --seq-len 10 \
  --top-k 8 \
  --max-new-tokens 8 \
  --atol 2e-4 --rtol 2e-4 \
  --confirm-full-load \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_multimodal_cuda.json
```

如果在 toy checkpoint 上复跑该路径，使用 `--image-token-id 250 --video-token-id 251` 覆盖 toy vocab 内的占位符；真实 MiniMax checkpoint 默认使用 config 中的 `200025/200026`，不需要覆盖。

完整 forward 通过标准：

- payload 部分同样 `passed=true`；
- JSON 顶层 `full_checkpoint_load_executed=true`；
- JSON 顶层 `reference_device` 和 `candidate_device` 等于本次目标设备组合，例如 `cuda:0`/`cuda:0` 或 `cpu`/`npu:0`；
- JSON 顶层 `device` 等于 candidate 设备，例如 `cuda:0` 或 `npu:0`；
- JSON 顶层 `tolerances.forward` 记录本次 `atol`/`rtol`，`tolerances.input` 为 exact input contract；
- JSON 顶层 `num_checks` 等于 `forward.checks` 数量，且 `failed=[]`；
- `forward.state_dict_load.missing_keys=[]`；
- `forward.state_dict_load.unexpected_keys=[]`；
- `forward.checks[name=input.input_ids].equal=true`；
- `forward.checks[name=input.attention_mask].equal=true`；
- `forward.checks[name=input.position_ids].equal=true`；
- `forward.checks[name=forward.logits].allclose=true`；
- image/video prompt 还要求 `forward.checks[name=forward.image_hidden_states].allclose=true` 和 `forward.checks[name=forward.video_hidden_states].allclose=true`；
- `forward.checks[name=forward.last_token_topk_ids].equal=true`；
- `forward.checks[name=generate.greedy_ids].equal=true`。

NPU 上复跑完整 forward 时优先使用 `--reference-device cpu` 或 `--reference-device cuda` 搭配 `--candidate-device npu`，先完成本手册 NPU runtime gates，并根据 backend 数值误差使用 NPU tolerance。NPU JSON 必须保留 `reference_device`、`candidate_device`、`tolerances`、prompt ids、top-k、greedy ids 和 runtime 证据；其中 `runtime.torch_npu_version` 不能为空，`runtime.torch_npu_available=true`，`runtime.torch_npu_device_count>=1`，并记录 `runtime.ascend_env.ASCEND_RT_VISIBLE_DEVICES`。

完整 checkpoint CPU reference vs NPU candidate 示例：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --config-path /data/checkpoints/MiniMax-M3 \
  --mode forward \
  --reference-device cpu \
  --candidate-device npu \
  --torch-dtype bfloat16 \
  --prompt-kind multimodal \
  --seq-len 10 \
  --top-k 8 \
  --max-new-tokens 8 \
  --atol 5e-4 --rtol 5e-4 \
  --confirm-full-load \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/real_checkpoint_forward_cpu_npu.json
```

3. **训练步 parity**
   - 选择小 batch SFT fixture；
   - 关 dropout，固定 seed；
   - 比较 loss、关键梯度和 one-step AdamW delta；
   - 记录 dtype policy，特别是 public checkpoint 中 `F32 -> BF16` router/gate tensor cast。

真实 checkpoint parity 通过前，不应声明 MiniMax M3 生产精度完成。

## 多卡 SP/EP/FSDP2 Parity

多卡门禁需要有对应硬件环境后运行：

- 单进程 CPU/GPU/NPU toy parity 先通过；
- 单卡真实或抽样 checkpoint parity 通过；
- `tests/e2e/test_e2e_parallel.py::test_minimax_m3_vl_parallel_align` 的 MiniMax case 从 xfail 提升为实际运行；
- 比较 SP/EP/FSDP2 下：
  - loss；
  - logits；
  - multimodal metadata scatter；
  - router selected experts；
  - grad norm；
  - rank 间 dummy visual path 是否无 hang。

目标机器推荐使用 launcher，它会先做 runtime/device-count preflight，再跑 FSDP2 asymmetric dummy-forward gate 和带 `--runxfail` 的 SP/EP/FSDP2 e2e alignment gate：

```bash
cd /path/to/VeOmni

scripts/multimodal/run_minimax_m3_vl_multicard_parity.sh \
  --min-devices 8 \
  --require-free-hbm-mb 4096 \
  --output-dir docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity
```

如果目标机的 `npu-smi info` 只能通过 root 运行，可以显式传入 root-only preflight 命令：

```bash
cd /path/to/VeOmni

scripts/multimodal/run_minimax_m3_vl_multicard_parity.sh \
  --min-devices 8 \
  --require-free-hbm-mb 4096 \
  --npu-smi-cmd 'sudo -n /usr/local/sbin/npu-smi info' \
  --output-dir docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity
```

通过标准：

- `multicard_parity_summary.json` 中 `passed=true`；
- `preflight.returncode=0`，且 log 中 `device_count>=8`、`transformers_version>=5.12.0`、`errors=[]`；
- 如设置 `--require-free-hbm-mb`，preflight log 中 `npu_smi.returncode=0`，且 `npu_smi.devices_with_required_free_hbm>=8`；
- `dummy_forward.returncode=0`；
- `e2e_align.returncode=0`，并且该命令必须使用 `--runxfail`，不能让 xfail marker 吞掉真实失败；
- 保存 `preflight.log`、`dummy_forward.log`、`e2e_align.log`，并把硬件拓扑、torchrun world size、JSON/日志路径补进 `minimax_m3_vl_migration_report.md`。

多卡 summary 必须进入 artifact 审计：

```bash
cd /path/to/VeOmni

python scripts/multimodal/audit_minimax_m3_vl_parity_artifacts.py \
  --multicard-json docs/usage/support_new_models/artifacts/minimax_m3_vl_multicard_parity/multicard_parity_summary.json \
  --require-multicard \
  --output-json docs/usage/support_new_models/artifacts/minimax_m3_vl_precision_parity/parity_artifact_audit_multicard.json
```

审计器会检查 summary 中的 `passed=true`、preflight/dummy/e2e return code、`--runxfail` 记录、log 路径、preflight log 中的 `device_count>=min_devices`、`transformers_version>=5.12.0` 和 `errors=[]`。如果 preflight 设备类型是 NPU，还会要求 `torch_npu_version` 和可见 Ascend 设备环境变量；如果开启了 free-HBM 门禁，还会要求 `npu_smi.devices_with_required_free_hbm>=min_devices`。

多卡通过前，不应声明 SP/EP/FSDP2 完成。

## 故障定位顺序

当 parity 失败时，按这个顺序排查：

1. `state_dict_load` 是否 strict clean；
2. `input.attention_mask`、`input.position_ids`、`input.multimodal_metadata_contract` 是否一致；
3. `forward.image_hidden_states` / `forward.video_hidden_states` 是否先漂；
4. `router.*.selected_experts` 是否发生 expert 选择分叉；
5. `forward.logits` 是否只在后几位浮点误差；
6. `grad.*` 是否定位到 attention、MoE、projector 或 `lm_head`；
7. `optimizer_delta.*` 是否说明 optimizer 配置或 dtype policy 不一致。

只有定位并解释每个 failed check 后，才能扩大 tolerance；不能直接把 failure 当作 backend noise。

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

本地 CPU/NPU 证据：

- [toy_hf_veomni_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity.json)
- [toy_hf_veomni_parity_npu.json](./artifacts/minimax_m3_vl_precision_parity/toy_hf_veomni_parity_npu.json)

结果摘要：

```json
{
  "passed": true,
  "device": "cpu / npu:0",
  "num_checks": 37,
  "failed": []
}
```

该 gate 覆盖：

- HF reference class: `transformers.models.minimax_m3_vl.modeling_minimax_m3_vl.MiniMaxM3SparseForConditionalGeneration`
- VeOmni candidate class: `veomni.models.transformers.minimax_m3_vl.generated.patched_modeling_minimax_m3_vl_gpu.MiniMaxM3SparseForConditionalGeneration`
- 同一份随机初始化 `state_dict`，`strict=True` 加载到 candidate；
- 同一份 mixed image+video toy batch；
- forward: `loss`、`logits`、`image_hidden_states`、`video_hidden_states`；
- routing: `MiniMaxM3VLTopKRouter` 的 `router_logits`、`top_k_weights`、`selected_experts`；
- input contract: `attention_mask`、`position_ids`、VeOmni `multimodal_metadata` collate contract；
- backward: embedding、attention q/k/v/o、MoE gate/experts、projector、`lm_head` 梯度；
- optimizer: 同一 AdamW one-step 后关键参数 delta。

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
  "device": "npu:0",
  "num_checks": 37,
  "torch_version": "2.10.0+cpu",
  "torch_npu_version": "2.10.0",
  "transformers_version": "5.12.0"
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

已落地的远程抽样 payload 证据：

- [real_checkpoint_payload_remote_sample.json](./artifacts/minimax_m3_vl_precision_parity/real_checkpoint_payload_remote_sample.json)
- [toy_checkpoint_forward_parity.json](./artifacts/minimax_m3_vl_precision_parity/toy_checkpoint_forward_parity.json)

该证据没有下载完整 shard，而是通过 HTTP Range 从 Hugging Face `model-00003-of-00059.safetensors` 读取 3 个真实 tensor payload，共 `1024` bytes，经 converter 映射到 VeOmni generated state keys 后检查 shape/dtype/value fingerprint。它证明真实 checkpoint payload 字节和 converter 路径已被执行，但仍不是 full-checkpoint logits parity。

`toy_checkpoint_forward_parity.json` 使用完整 toy safetensors checkpoint 验证了 `--mode forward` 的执行路径：先加载 HF reference 生成 logits/top-k/greedy baseline，释放 HF 模型，再把 public checkpoint tensors 边读、边转换、边写入 VeOmni generated model。该 smoke 中 `streaming_model_load=true`，strict missing/unexpected key count 都为 `0`，`forward.logits` max diff 为 `0.0`，top-k 和 greedy ids 完全一致；它只证明 runner 逻辑，不替代真实 869 GB checkpoint parity。

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
  --include-key-regex '^language_model\.model\.layers\.3\.self_attn\.index_[qk]_norm\.weight$|^language_model\.model\.layers\.3\.block_sparse_moe\.e_score_correction_bias$' \
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

完整 text-prompt forward 示例：

```bash
cd /path/to/VeOmni
export PYTHONPATH=$PWD

python scripts/multimodal/verify_minimax_m3_vl_checkpoint_payload_parity.py \
  --checkpoint-dir /data/checkpoints/MiniMax-M3 \
  --config-path /data/checkpoints/MiniMax-M3 \
  --mode forward \
  --device cuda \
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
  --device cuda \
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
- `forward.state_dict_load.missing_keys=[]`；
- `forward.state_dict_load.unexpected_keys=[]`；
- `forward.checks[name=forward.logits].allclose=true`；
- image/video prompt 还要求 `forward.checks[name=forward.image_hidden_states].allclose=true` 和 `forward.checks[name=forward.video_hidden_states].allclose=true`；
- `forward.checks[name=forward.last_token_topk_ids].equal=true`；
- `forward.checks[name=generate.greedy_ids].equal=true`。

NPU 上复跑完整 forward 时把 `--device cuda` 改为 `--device npu`，先完成本手册 NPU runtime gates，并根据 backend 数值误差使用 NPU tolerance。NPU JSON 必须保留 `device`、tolerance、prompt ids、top-k、greedy ids 和 runtime 证据路径。

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

多卡通过后，再把对应命令、硬件拓扑、torchrun world size、JSON/日志路径补进 `minimax_m3_vl_migration_report.md`。

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

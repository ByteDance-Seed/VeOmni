# Qwen3.5 training guide

> **Note:** Qwen3.5 requires transformers v5 (now the project default).

## Install dependencies

Qwen3.5 depends on transformers v5, which is now the default install:

```shell
uv sync --extra gpu
```

## Download dataset

### Vision-Language Dataset

Download the [COCO2017](https://images.cocodataset.org/zips/train2017.zip) dataset and download the data annotation JSON file [sharegpt4v_instruct_gpt4-vision_cap100k.json](https://huggingface.co/datasets/Lin-Chen/ShareGPT4V/tree/main).

Modify the sharegpt4v_instruct_gpt4-vision_cap100k.json:

```python
import json
with open('sharegpt4v_instruct_gpt4-vision_cap100k.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
filtered_data = []
for item in data:
    if item.get('image', '').startswith('coco'):
        new_item = item.copy()
        image_path = new_item.pop('image')
        new_item['images'] = [image_path]
        filtered_data.append(new_item)
with open('sharegpt4v_instruct_gpt4-vision_cap100k_coco.json', 'w', encoding='utf-8') as f:
    json.dump(filtered_data, f, ensure_ascii=False, indent=4)
```

### Text-only Dataset (Optional)

If you want to train on text-only data, download the [tulu-3-sft-mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture) dataset.

```python
import pyarrow.parquet as pq
input_path = "tulu-3-sft-mixture/data/train-00000-of-00006.parquet"
output_path = "tulu-first2000.parquet"
# Read parquet file and extract the first 2000 rows
table = pq.read_table(input_path)
table_first_2000 = table.slice(0, 2000)
pq.write_table(table_first_2000, output_path)
```

## Download Qwen3.5 model

Dense 9B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-9B \
    --local_dir ${HOME}/Qwen3.5-9B
```

MoE 35B:
```shell
python3 scripts/download_hf_model.py \
    --repo_id Qwen/Qwen3.5-35B-A3B \
    --local_dir ${HOME}/Qwen3.5-35B-A3B
```

## Start training on GPU

### Qwen3.5-9B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Qwen3.5 MoE 35B VL Training

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl.yaml \
    --model.model_path ./Qwen/Qwen3.5-35B-A3B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

### Text-only training on GPU
Testing in 8x80GB GPUs.

Qwen3.5 Dense 9B:
```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.max_steps 20 \
    --train.checkpoint.output_dir ./exp/qwen3_5_9b_sft
```

Qwen3.5-35B-A3B. 8X80GB GPU will likely OOM due to the model size. Use 8X192GB GPU or more GPUs.

```shell
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --model.ops_implementation.moe_implementation fused_triton \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta \
    --train.global_batch_size 16 \
    --train.checkpoint.output_dir ./exp/qwen3_5_35b_a3b_sft
```

## Start training on NPU

Qwen3.5 runs on Ascend NPUs, but its GatedDeltaNet kernels are **not** auto-selected: all three
OpSlot-driven ops (`rms_norm_gated`, `causal_conv1d`, `chunk_gated_delta_rule`) default to `fla`,
which requires a GPU and raises on NPU. They must be set explicitly.

### Kernel backends

`npu` provides vendored MindSpeed-MM Triton kernels for all three ops:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: npu
    causal_conv1d_implementation: npu
    chunk_gated_delta_rule_implementation: npu
```

These kernels need the `triton-ascend` package, which is not a declared VeOmni dependency —
install it on the NPU host. VeOmni main pins PyTorch and `torch_npu` 2.10.0; the corresponding
CANN 9.0.0 stack uses v3.2.1:

```bash
pip install triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple
```

See the [triton-ascend quick-start](https://github.com/triton-lang/triton-ascend/blob/main/docs/zh/quick_start.md)
for the compatibility matrix and troubleshooting. Keep CANN, `torch_npu`,
and `triton-ascend` on a mutually compatible release set.

> **arch35 limitation:** the vendored `causal_conv1d` refuses to run on arch35 NPUs
> (`Ascend910_95` / `Ascend950`) — its `is_arch35()` guard raises `NotImplementedError`
> rather than mis-computing. Validated on `Ascend910B2C`. arch35 support would need to come
> from upstream MindSpeed-MM.

### `npu_ascendc` — AscendC fused backend (`chunk_gated_delta_rule` only)

`npu_ascendc` is a second NPU backend for `chunk_gated_delta_rule` that delegates the heavy GDN
compute to the external [`fla_npu`](https://github.com/flashserve/flash-linear-attention-npu)
package (registered as `torch.ops.npu.*` fused ops); only the Triton glue stays vendored under
`_ascend/triton_core`. It coexists with `npu` (pure vendored Triton), which remains the fallback.
Set only `chunk_gated_delta_rule` to `npu_ascendc`; keep `rms_norm_gated` / `causal_conv1d` on `npu`:

```yaml
model:
  ops_implementation:
    rms_norm_gated_implementation: npu
    causal_conv1d_implementation: npu
    chunk_gated_delta_rule_implementation: npu_ascendc
```

`fla_npu` is not a declared VeOmni dependency (same as `triton-ascend`). It supports Ascend
910B (A2) / 910_93 (A3), both non-arch35 chips where the `solve_tril` step runs the vendored
Triton kernel (the arch35 native path is gated behind `is_arch35()`). arch35 (`Ascend910_95` /
`Ascend950`) is not supported end-to-end: the vendored `causal_conv1d` raises there (see above).

The prebuilt NPU images on [quay.io](https://quay.io/repository/ascend/veomni?tab=tags) already
bundle `fla_npu`:

- A2: `quay.io/ascend/veomni:v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A2-ubuntu22.04-py3.11-veomni-latest`
- A3: `quay.io/ascend/veomni:v0.1.11-cann9.0.0-torch_npu2.10.0.post2-A3-ubuntu22.04-py3.11-veomni-latest`

To build on a bare host instead, install from the
[`flash-linear-attention-npu`](https://github.com/flashserve/flash-linear-attention-npu) sources:

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu
cd flash-linear-attention-npu
git checkout c2e3d83f

source /usr/local/Ascend/cann/set_env.sh          # source your actual CANN path

# --soc must match the chip: {ascend910b, ascend910_93, ascend950}
bash build.sh --soc=ascend910b --pkg --vendor_name=fla_npu
bash build_out/fla-npu_*.run
cd torch_custom/fla_npu/ && bash build.sh

pip list | grep fla_npu   # verify it is installed
```

If `fla_npu` is absent when `npu_ascendc` is selected, the backend raises an actionable error at
`OpSlot.bind()` time (pointing back to the install step or to `npu` / `eager`).

### Qwen3.5-9B VL Training

There is no dedicated dense NPU config; reuse the GPU one and set the GatedDeltaNet kernels on the
command line. Nothing else needs overriding: the YAML's `attn_implementation: flash_attention_2` is
also the recommended NPU value (see [NPU-friendly operator
configurations](../hardware_support/typical_usage.md)), and `rms_norm`, `rotary_pos_emb`,
`swiglu_mlp` and `cross_entropy_loss` are left at their dataclass defaults there, so VeOmni
normalizes them to NPU-compatible values automatically
([`_NPU_DEFAULT_FALLBACK`](../../veomni/arguments/arguments_types.py#L979)). The three
GatedDeltaNet ops are deliberately **not** in that table, which is why they must be explicit:

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5/qwen3_5_vl.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --model.ops_implementation.rms_norm_gated_implementation npu \
    --model.ops_implementation.causal_conv1d_implementation npu \
    --model.ops_implementation.chunk_gated_delta_rule_implementation npu_ascendc \
    --train.max_steps 20
```

Drop `chunk_gated_delta_rule_implementation` to `npu` if `fla_npu` is not installed.

### Qwen3.5 MoE 35B VL Training

A ready-to-run MoE-VL config wired for `npu_ascendc` (plus `fused_npu` MoE, NPU RMSNorm/RoPE and
Ulysses SP) is provided at
[`configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml`](../../configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml):

```shell
bash train.sh tasks/train_vlm.py configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml \
    --model.model_path ${HOME}/Qwen3.5-35B-A3B \
    --data.train_path ./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml \
    --train.max_steps 20
```

The config sets `ulysses_size: 4`, so the world size must be a multiple of 4. To run without
sequence parallelism, override `--train.accelerator.ulysses_size 1`; the attention kernels adapt
on their own.

## Qwen3.5-MoE EP load-balance experiments

Temporary expert replicas are opt-in. The AscendC recipe declares the safe defaults explicitly (`moe_load_balance_monitor_interval: 0`, `moe_ep_load_balance.enabled: false`, and `max_replicas_per_rank: 1`) and retains its original topology. Enabled mode currently fail-fast requires FSDP2, resolved `dp_replicate_size == 1` (no HSDP), `ep_outside=false`, no FSDP CPU offload, no activation offload, no checkpoint resume, full SFT, and fused MoE. Multi-node and other topologies remain untested. Ordinary checkpoint saving and ordinary gradient checkpointing are still allowed.

The recipe enables profiling by default, so every functional or throughput comparison below overrides `train.profile.enable=false`. Keep the two commands in a pair identical except for feature enablement and output/run labels. Replace the model and dataset paths with immutable local revisions before collecting evidence.

### Functional and precision evidence

Use the monitor for per-step physical-load evidence, keep the profiler off, and run enough steps to inspect a real loss/gradient trajectory. These commands are a reproducible experiment template, not a published accelerator result:

```shell
CONFIG=configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml
MODEL=${HOME}/Qwen3.5-35B-A3B
DATA=./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml

COMMON=(
  --model.model_path "$MODEL"
  --data.train_path "$DATA"
  --train.accelerator.ulysses_size 1
  --train.accelerator.ep_size 2
  --train.profile.enable false
  --train.moe_load_balance_monitor_interval 1
  --train.moe_ep_load_balance.max_replicas_per_rank 1
  --train.checkpoint.save_steps 0
  --train.checkpoint.save_epochs 0
  --train.checkpoint.save_hf_weights false
  --train.max_steps 20
  --train.seed 0
)

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${COMMON[@]}" \
  --train.moe_ep_load_balance.enabled false \
  --train.wandb.name qwen3_5_moe_ep_balance_functional_baseline \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_functional_baseline

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${COMMON[@]}" \
  --train.moe_ep_load_balance.enabled true \
  --train.wandb.name qwen3_5_moe_ep_balance_functional_enabled \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_functional_enabled
```

The CI E2E uses a smaller, deterministic two-step hotspot and a versioned scalar sink. It requires four compatible CUDA or NPU devices, forces both runs to route to experts `[0, 1]`, compares finite loss and gradient-norm curves with `rtol=atol=0.1`, and checks positive replicas/moved tokens plus non-worsening physical imbalance. Its NPU path selects the three `npu` GatedDeltaNet kernels and requires the `triton` import from `triton-ascend`, without requiring `fla_npu`; its CUDA path rejects ROCm, requires SM70 or newer on all four selected devices, and checks the default Liger dependency. Dedicated GPU and NPU workflow steps select only that exact real-hardware test node and set `VEOMNI_REQUIRE_EP_BALANCE_E2E=1`, so any missing prerequisite fails those jobs without applying strict mode to the module's local-policy gate tests. Ordinary local invocation leaves that variable unset and honestly skips before dataset/checkpoint creation. Forced routing and `CUDA_LAUNCH_BLOCKING=1` make it functional/precision evidence only.

### Steady-state performance evidence

Disable both monitoring and profiling because their synchronization and formatting costs invalidate throughput attribution. Use sufficient warmup and steady steps; this 100-step template reserves the first 20 reporter steps as warmup:

```shell
CONFIG=configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml
MODEL=${HOME}/Qwen3.5-35B-A3B
DATA=./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml

PERF_COMMON=(
  --model.model_path "$MODEL"
  --data.train_path "$DATA"
  --train.accelerator.ulysses_size 1
  --train.accelerator.ep_size 2
  --train.profile.enable false
  --train.moe_load_balance_monitor_interval 0
  --train.moe_ep_load_balance.max_replicas_per_rank 1
  --train.checkpoint.save_steps 0
  --train.checkpoint.save_epochs 0
  --train.checkpoint.save_hf_weights false
  --train.max_steps 100
  --train.seed 0
)

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${PERF_COMMON[@]}" \
  --train.moe_ep_load_balance.enabled false \
  --train.wandb.name qwen3_5_moe_ep_balance_perf_baseline \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_perf_baseline

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${PERF_COMMON[@]}" \
  --train.moe_ep_load_balance.enabled true \
  --train.wandb.name qwen3_5_moe_ep_balance_perf_enabled \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_perf_enabled
```

Performance artifacts must explicitly record `step_time_s` and either `step_tokens` or `tokens_per_second`; peak memory must come from explicit `peak_accelerator_memory_bytes` and `peak_host_memory_bytes` fields. Optional per-step `p2p_bytes` and `p2p_wait_time_s` curves may be added by profiler post-processing or runtime instrumentation, with exactly one value per loss/gradient step. The training commands and comparison reporter do not collect those P2P curves automatically. If either matched run lacks a field, the reporter marks that section unavailable. Do not substitute process/system-wide used memory, inferred communication values, or zero for a missing measurement.

### All-rank profiler diagnosis

Profiler traces diagnose operator/communication behavior but are not used for throughput. Profile every rank so EP sends, receives, and all-to-all work are visible. Run the matched baseline and enabled commands separately to keep trace directories distinct:

```shell
CONFIG=configs/multimodal/qwen3_5_moe/qwen3_5_moe_vl_ascendc.yaml
MODEL=${HOME}/Qwen3.5-35B-A3B
DATA=./configs/multimodal/data/tulu_sharegpt4v_llavavideo.yaml

PROFILE_COMMON=(
  --model.model_path "$MODEL"
  --data.train_path "$DATA"
  --train.accelerator.ulysses_size 1
  --train.accelerator.ep_size 2
  --train.profile.enable true
  --train.profile.rank0_only false
  --train.profile.start_step 20
  --train.profile.end_step 25
  --train.moe_load_balance_monitor_interval 0
  --train.moe_ep_load_balance.max_replicas_per_rank 1
  --train.checkpoint.save_steps 0
  --train.checkpoint.save_epochs 0
  --train.checkpoint.save_hf_weights false
  --train.max_steps 26
  --train.seed 0
)

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${PROFILE_COMMON[@]}" \
  --train.moe_ep_load_balance.enabled false \
  --train.profile.trace_dir ./exp/qwen3_5_moe_ep_balance_profile_baseline/traces \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_profile_baseline

NPROC_PER_NODE=8 bash train.sh tasks/train_vlm.py "$CONFIG" "${PROFILE_COMMON[@]}" \
  --train.moe_ep_load_balance.enabled true \
  --train.profile.trace_dir ./exp/qwen3_5_moe_ep_balance_profile_enabled/traces \
  --train.checkpoint.output_dir ./exp/qwen3_5_moe_ep_balance_profile_enabled
```

### Compare saved artifacts

The standard-library reporter accepts the E2E schema-version-1 envelope, existing flat `log_dict.json`, per-step JSONL, or plain training logs for loss/gradient only:

```shell
python scripts/profile/compare_moe_ep_load_balance.py \
  --baseline ./artifacts/baseline.json \
  --candidate ./artifacts/enabled.json \
  --warmup-steps 20 \
  --rtol 0.1 \
  --atol 0.1 \
  --relative-error-threshold 0.1 \
  --epsilon 1e-12 \
  --json-out ./artifacts/comparison.json \
  --markdown-out ./artifacts/comparison.md
```

Precision failure returns a nonzero status after both requested reports are written. Missing performance, P2P, or memory fields produce an explicit `unavailable` section with `null` values; they are not a comparison failure by themselves. See the [design](../design/qwen3_5_moe_ep_load_balance.md) and [evidence report](../performance/qwen3_5_moe_ep_load_balance.md) for formulas, support boundaries, and rollback.

## ChunkMBS

Dense Qwen3.5 packed SFT supports decoder-layer ChunkMBS. It splits the packed sequence only at sample boundaries,
then runs each `Qwen3_5DecoderLayer` chunk sequentially while keeping the outer FSDP2 layer invocation intact. The
same token ranges are used by both full-attention and GatedDeltaNet layers, so every ChunkMBS cut must be present in
both cumulative-length tensors; their internal boundaries may differ.

The example config exposes the feature but keeps it disabled by default:

```yaml
train:
  chunk_mbs_config:
    enable: true
    chunk_mbs: 2
```

`chunk_mbs` is the number of packed samples per layer chunk, not a token count or `train.micro_batch_size`.
ChunkMBS may be combined with non-reentrant gradient checkpointing. It currently does not support Qwen3.5-MoE,
Ulysses SP, TP/PP, `torch.compile`, `pad_to_length`, DPO, or RL training. See
[ChunkMBSConfig](../usage/arguments.md#chunkmbsconfig) for the complete support boundary.
The model-level automated tests cover decoder routing, metadata slicing, outputs, and gradients on CPU. Real
FlashAttention, FLA, and NPU fused-kernel execution requires separate accelerator validation.

## Ulysses Sequence Parallelism

Qwen3.5 supports Ulysses sequence parallelism for both its softmax attention layers and
linear attention (GatedDeltaNet) layers. This enables training with longer sequences by
distributing the sequence across multiple GPUs.

To enable Ulysses SP, set `train.accelerator.ulysses_size`. VeOmni derives the effective
data-parallel size from the world size and the other parallel dimensions; set
`train.accelerator.dp_shard_size` only when you need to pin the FSDP shard degree explicitly.
For the example below, the total GPU count is `dp_shard_size * ulysses_size = 4 * 2 = 8`.

```shell
# Example: 8 GPUs, dp=4, sp=2
bash train.sh tasks/train_text.py configs/text/qwen3_5_sft.yaml \
    --model.model_path ${HOME}/Qwen3.5-9B \
    --data.train_path ${HOME}/tulu-first2000.parquet \
    --train.accelerator.dp_shard_size 4 \
    --train.accelerator.ulysses_size 2 \
    --model.ops_implementation.attn_implementation flash_attention_3
```

### Requirements

- `flash_attention_2` or `flash_attention_3` attention implementation (softmax layers use
  VeOmni's flash attention with built-in SP support).
- [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) installed
  (for GatedDeltaNet triton kernels).
- `num_k_heads` and `num_v_heads` (linear attention head counts) must be divisible by
  `ulysses_size`.

### Selecting linear-attention kernels

GatedDeltaNet has three OpSlot-driven kernels: `rms_norm_gated`, `causal_conv1d`, and
`chunk_gated_delta_rule`. All three default to `fla` — the FLA Triton kernels shipped under the
`gpu` extra, which is the recommended value on GPU and required for varlen training.

To switch `chunk_gated_delta_rule` to QwenLM's [`flash-qla`](https://github.com/QwenLM/FlashQLA)
kernel (also shipped under the `gpu` extra), set the field explicitly:

```yaml
model:
  ops_implementation:
    chunk_gated_delta_rule_implementation: flash_qla
```

On NPU the default `fla` raises, so all three fields must be set explicitly — see
[Kernel backends](#kernel-backends) under *Start training on NPU*.

### How It Works

Qwen3.5 is a hybrid model alternating between softmax and linear attention layers:

- **Softmax attention layers** — SP is handled transparently by VeOmni's `flash_attention_forward`,
  which performs all-to-all gather/scatter around the flash attention kernel.
- **Linear attention layers (GatedDeltaNet)** — SP is handled explicitly in the patched
  `Qwen3_5GatedDeltaNet.forward`. Q/K/V/b/a projections are all-to-all'd to gather the full
  sequence with local heads, the causal conv1d runs with sharded weights, the recurrent attention
  kernel runs on local heads, and the output is all-to-all'd back.

For detailed implementation notes, see the
[Ulysses documentation](../key_features/ulysses.md#-linear-attention-ulysses-gateddeltanet).

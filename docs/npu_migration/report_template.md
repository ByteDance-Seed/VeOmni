# [Model Name] NPU Migration Practice Report

> Template for GPU→NPU migration practice reports. Fill in all sections.
> Reference: `.agents/skills/veomni-gpu-to-npu/SKILL.md`

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | |
| Architecture | |
| Parameter count | |
| Attention type | |
| RMSNorm variant | standard / Gemma (1.0+weight) / other |
| Special features | (sliding window, logit softcapping, MoE, etc.) |

## 2. Dependency Scan

List every CUDA/GPU-specific dependency found in the GPU path:

| Location | Dependency | Impact | Resolution |
|---|---|---|---|
| | | | |

## 3. NPU Backend Selection

| Op (OpSlot) | GPU backend | NPU backend | Notes |
|---|---|---|---|
| attn_implementation | | | |
| rms_norm | | | |
| rotary_pos_emb | | | |
| swiglu_mlp | | | |
| cross_entropy_loss | | | |
| moe | | | |
| load_balancing_loss | | | |

## 4. Files Created/Modified

| File | Type | Description |
|---|---|---|
| | new / modified | |

## 5. Test Results

| Test | Status | Notes |
|---|---|---|
| | | |

```
pytest tests/models/test_<model>_npu.py -v
# Paste output here
```

## 6. E2E Training Results

- **Hardware**: (e.g., Ascend 910B × N)
- **Config**: `configs/text/<model>_npu.yaml`
- **Steps**: 
- **Seq length**: 
- **Batch size**: 

### Loss Curve

| Step | Loss |
|---|---|
| | |

### Training Stability

(Describe: loss decreasing, no NaN/Inf, no OOM)

## 7. Accuracy Comparison (NPU vs GPU)

> Run the same model, same data, same training config on both GPU and NPU.
> Compare loss curves.

| Step | GPU loss | NPU loss | Difference |
|---|---|---|---|
| | | | |

**Verdict**: (Pass if NPU loss tracks GPU loss within tolerance)

## 8. Performance

- **Hardware**: 
- **Model**: 
- **Seq length**: 
- **Batch size**: 
- **Micro batch size**: 
- **Dtype**: bfloat16
- **Parallelism**: (FSDP2, SP=1, etc.)

### MFU Calculation

```
Tokens per step = micro_batch_size × seq_length × num_devices
FLOPs per step ≈ 6 × parameter_count × tokens_per_step
Wall time per step = (measured)
Achieved TFLOPS = FLOPs_per_step / wall_time / 1e12
Peak TFLOPS = (hardware spec, e.g., 910B = 320 TFLOPS BF16)
MFU = Achieved / Peak
```

| Metric | Value |
|---|---|
| Achieved TFLOPS | |
| Peak TFLOPS | |
| **MFU** | |

**MFU target**: ≥ 24%

## 9. Issues Encountered

| Issue | Root cause | Resolution |
|---|---|---|
| | | |

## 10. Reproduction

```bash
# 1. Install environment
uv sync --extra npu --dev

# 2. Download dataset
hf download allenai/tulu-3-sft-mixture --repo-type dataset --local-dir data/tulu-3-sft-mixture

# 3. Run tests
pytest tests/models/test_<model>_npu.py -v

# 4. Run E2E
bash scripts/e2e/<model>_npu_e2e.sh
```

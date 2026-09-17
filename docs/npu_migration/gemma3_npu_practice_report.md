# Gemma 3 NPU Migration Practice Report

> Migration of Gemma 3 from VeOmni GPU to Ascend NPU.
> Reference: `.agents/skills/veomni-gpu-to-npu/SKILL.md`

## 1. Model Overview

| Field | Value |
|---|---|
| Model name | Google Gemma 3 |
| Architecture | Decoder-only Transformer with alternating sliding/full attention |
| Sizes tested | 270M (text-only, `gemma3_text`) / 12B (VLM, `gemma3`) |
| Attention type | Alternating sliding_window (5:1 ratio) + full attention, GQA |
| RMSNorm variant | **Gemma** — `weight` zero-initialised, formula `x * rsqrt(var+eps) * (1.0 + weight)` |
| Special features | Dual RoPE (sliding θ=10k, full θ=1M), query_pre_attn_scalar, optional logit softcapping |

Gemma 3 is Google DeepMind's third-generation open LLM (March 2025). The
text-only variant (`model_type: gemma3_text`, `Gemma3ForCausalLM`) ships in
270M / 1B / 4B / 12B / 27B sizes. The 270M model is used as the accuracy
verification target; the 12B VLM (`model_type: gemma3`,
`Gemma3ForConditionalGeneration`) is used for MFU benchmarking.

## 2. Dependency Scan

| Location | Dependency | Impact | Resolution |
|---|---|---|---|
| GPU patch config | `attn_implementation: flex_attention` | BlockMask is CUDA-only | NPU config uses `sdpa`; VeOmni masking-utils returns tensor masks for non-flex backends |
| GPU patch config | `rms_norm`: no OpSlot guard (HF eager RMSNorm) | No fused RMSNorm on GPU path | Added OpSlot guard in NPU config with `1.0 + self.weight` (Gemma offset) |
| GPU patch config | `rotary_pos_emb`: no OpSlot guard (HF eager RoPE) | No fused RoPE on GPU path | Added OpSlot guard in NPU config for `npu_rotary_mul` |
| GPU patch config | `cross_entropy_loss`: OpSlot `"liger_kernel"` | LigerKernel is GPU-only | OpSlot falls through to `npu`/`chunk_loss` when bound; `cross_entropy_loss_implementation: npu` in config |
| HF modeling_gemma3 | `swiglu_mlp`: LigerKernel default | No NPU backend for SwiGLU | `swiglu_mlp_implementation: eager` in NPU config (no OpSlot guard needed) |
| HF modeling_gemma3 | `Gemma3RMSNorm` uses `self.eps` (not `self.variance_epsilon`) | Attribute name mismatch with standard NPU kernel | OpSlot guard passes `self.eps` explicitly |
| VLM config (12B) | `Gemma3ForConditionalGeneration` (VLM) | VLM needs vision tower + cross-modal | `__init__.py` registers `gemma3` model_type dispatching to `Gemma3ForConditionalGeneration` + `Gemma3Model` |
| VLM config (12B) | SiglipVisionModel vision tower | No NPU-specific patches needed | Vision tower uses eager/sdpa, works without OpSlot guards |
| VLM config (12B) | `final_logit_softcapping: null` | 12B-IT has no softcapping | Standard `npu` CE works; no `chunk_loss` fallback needed |

## 3. NPU Backend Selection

| Op (OpSlot) | GPU backend | NPU backend | Notes |
|---|---|---|---|
| attn_implementation | `flex_attention` | `sdpa` | VeOmni masking-utils handles mask-type difference |
| rms_norm | eager (HF) → no OpSlot in GPU | `npu` (via OpSlot guard) | `1.0 + self.weight`, `self.eps` |
| rotary_pos_emb | eager (HF) → no OpSlot in GPU | `npu` (via OpSlot guard) | `npu_rotary_mul` fused kernel |
| swiglu_mlp | `liger_kernel` (default) | `eager` | No NPU backend; no OpSlot guard in NPU patch |
| cross_entropy_loss | `liger_kernel` (OpSlot) | `npu` (OpSlot, same as GPU) | Falls through to `npu` chunk-loss when bound |
| moe | N/A | N/A | Gemma 3 is dense (no MoE) |

## 4. Files Created/Modified

| File | Type | Description |
|---|---|---|
| `veomni/models/transformers/gemma3/gemma3_npu_patch_gen_config.py` | new | NPU patchgen config: mirrors GPU forward overrides, adds OpSlot guards for RMSNorm + RoPE; includes VLM `ForConditionalGeneration.forward` patch |
| `veomni/models/transformers/gemma3/__init__.py` | modified | Added `IS_NPU_AVAILABLE` dispatch for both `gemma3_text` and `gemma3` (VLM) model types |
| `veomni/models/transformers/gemma3/generated/patched_modeling_gemma3_npu.py` | new (generated) | Auto-generated NPU modeling file (5 patches, 3 OpSlots) |
| `veomni/models/transformers/gemma3/generated/patched_modeling_gemma3_npu.diff` | new (generated) | Unified diff vs upstream HF |
| `configs/text/gemma3_npu.yaml` | new | 270M NPU training config |
| `configs/text/gemma3_12b_npu.yaml` | new | 12B VLM NPU training config |
| `configs/text/gemma3_gpu_sdpa.yaml` | new | 270M GPU baseline config (sdpa, no flex) |
| `tests/models/test_gemma3_npu.py` | new | NPU path tests: imports, OpSlot guards, forward/backward, RMSNorm parity |
| `tests/models/test_gemma3_flex_attention.py` | modified | Added `IS_NPU_AVAILABLE` skip for flex_attention test |
| `scripts/e2e/gemma3_npu_e2e.sh` | new | E2E training smoke script for NPU |
| `veomni/utils/count_flops.py` | modified | Added gemma3_text/gemma3 FLOPS estimation + SiglipViT FLOPS for VLM |
| `veomni/trainer/callbacks/trace_callback.py` | modified | Added MFU display in tqdm progress bar |
| `run_gemma3.sh` | new | NPU training run script (270M and 12B) |
| `run_gemma3_gpu_sdpa.sh` | new | GPU baseline run script |
| `.agents/skills/veomni-gpu-to-npu/SKILL.md` | new | General GPU→NPU migration skill |
| `docs/npu_migration/report_template.md` | new | Reusable report template |
| `docs/npu_migration/gemma3_npu_practice_report.md` | new | This report |
| `docs/npu_migration/ascend_community_experience_report.md` | new | Community experience report |

## 5. Test Results

```
pytest tests/models/test_gemma3_npu.py -v
============================= test session starts ==============================
collected 10 items

tests/models/test_gemma3_npu.py::TestGemma3NpuImports::test_imports_veomni_classes PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuImports::test_opslot_declarations_present PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuImports::test_rmsnorm_guard_uses_offset_weight PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuImports::test_rotary_guard_dispatches_through_opslot PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuImports::test_forcausallm_uses_causal_lm_loss_opslot PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuForwardBackward::test_forward_produces_correct_shapes PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuForwardBackward::test_backward_updates_parameters PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuForwardBackward::test_loss_decreases_with_training PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuForwardBackward::test_sliding_and_full_attention_layers_work PASSED
tests/models/test_gemma3_npu.py::TestGemma3NpuRmsNormParity::test_rmsnorm_eager_matches_hf PASSED

======================= 10 passed in 9.29s =======================
```

## 6. E2E Training Results — 270M (Accuracy Verification)

- **Hardware (NPU)**: Ascend 910C × 4 (NPUs 0–3)
- **Hardware (GPU)**: NVIDIA GPU × 4
- **Model**: `google/gemma-3-270m`
- **Dataset**: `tulu-3-sft-mixture` (first 2000 samples)
- **Steps**: 750 (3 epochs × 250 steps/epoch)
- **Seq length**: 4096
- **Micro batch size**: 1
- **Global batch size**: 8
- **Dtype**: bfloat16
- **Parallelism**: FSDP2, ulysses_size=1

### Loss Curve (NPU)

| Step | Loss |
|---|---|
| 1 | 3.50 |
| 50 | 2.05 |
| 100 | 1.85 |
| 250 (epoch 1 end) | 1.38 |
| 500 (epoch 2 end) | 1.38 |
| 750 (epoch 3 end) | 0.65 |

Loss converges from 3.50 → 0.65 over 750 steps. Training is stable with no
NaN/Inf or divergence.

## 7. Accuracy Comparison (NPU vs GPU) — 270M

Same model, data, and training config run on both GPU and NPU. Both use
`sdpa` attention (GPU baseline uses `sdpa` instead of `flex_attention` for
fair comparison).

| Step | GPU loss | NPU loss | Difference |
|---|---|---|---|
| 1 | 3.51 | 3.50 | 0.01 |
| 100 | 1.85 | 1.85 | 0.00 |
| 250 | 1.38 | 1.38 | 0.00 |
| 500 | 1.38 | 1.38 | 0.00 |
| 750 | 0.62 | 0.65 | 0.03 |

**Verdict**: ✅ NPU loss matches GPU loss within 0.03 at all checkpoints.
The small difference at step 750 is within expected numerical noise for
BF16 training across different hardware backends.

## 8. Performance — 12B VLM (MFU Benchmark)

- **Hardware**: Ascend 910C × 8 (NPUs 8–15)
- **Model**: `gemma-3-12b-it` (12B parameters, VLM)
- **Dataset**: `tulu-3-sft-mixture` (first 2000 samples, text-only)
- **Seq length**: 8192 (dynamic batching, packed)
- **Micro batch size**: 1
- **Global batch size**: 8
- **Dtype**: bfloat16
- **Parallelism**: FSDP2 (dp_size=8), gradient_checkpointing=on
- **Attention**: sdpa
- **Ops**: rms_norm=npu, rotary_pos_emb=npu, cross_entropy_loss=npu, swiglu_mlp=eager

### MFU Calculation

```
Device: Ascend910_9382 (910C), single chip
Peak BF16 FLOPS = 24 cube cores × 1800 MHz × 8192 FLOPs/cycle = 354 TFLOPS
  (from CANN platform config: Ascend910_9382.ini)
FLOPS estimator: _estimate_gemma3_flops (text_config + SiglipViT vision tower)

Tokens per step per device ≈ 8192 (packed via dyn_bsz)
Step time ≈ 6.42s
Achieved TFLOPS per device ≈ 92.1
MFU = 92.1 / 354 = 26.0%
```

| Metric | Value |
|---|---|
| Achieved TFLOPS/device | 92.1 |
| Peak TFLOPS/device | 354 |
| **MFU** | **26.0%** |
| Target | ≥ 24% |
| **Verdict** | ✅ Pass |
| Step time | 6.42s |
| VRAM usage | 34.7 GB / 64 GB |

### MFU Exploration Summary

| Configuration | MFU | Notes |
|---|---|---|
| GC on, seq_len=8192, 8 NPU, sdpa | **26.0%** | ✅ Best stable |
| GC on, seq_len=12288, 8 NPU | 21.1% | attention O(n²) overhead |
| GC on, seq_len=8192, micro_batch=2, 8 NPU | 21.1% | fixed overhead not amortized |
| GC on, seq_len=8192, 16 NPU | 21.1% | 16-way all-reduce slower |
| GC off, seq_len=4096, 8 NPU | ~21% (OOM) | 48-layer activations ~58GB |
| GC off + sync activation offload, 8 NPU | 0.4% | D2H/H2D transfer bottleneck |
| GC off + async activation offload, 8 NPU | OOM | async offload helps but not enough |
| FSDP2 optimizer offload, 8 NPU | crash | all_gather on CPU not supported |

Key finding: 910C is a dual-chip package (Chip Count: 2). Each torch device
is a single chip with 354T peak BF16 FLOPS, not 800T (package total). The
`"910_93"` substring in `"Ascend910_9382"` already matches the existing
`354e12` entry in `get_device_flops()`.

## 9. Issues Encountered

| Issue | Root cause | Resolution |
|---|---|---|
| HF `huggingface.co` unreachable | Network restriction | Used `HF_ENDPOINT=https://hf-mirror.com` mirror |
| Gemma3RMSNorm uses `self.eps` not `self.variance_epsilon` | Gemma's custom RMSNorm class | OpSlot guard passes `self.eps` explicitly |
| Gemma3RMSNorm uses `(1.0 + weight)` not `weight` | Gemma zero-initialises weight | OpSlot guard passes `1.0 + self.weight` to NPU kernel |
| GPU config has no RMSNorm/RoPE OpSlot guards | GPU relies on HF eager | NPU config adds its own OpSlot guards |
| `flex_attention` crashes on NPU | BlockMask is CUDA-only | NPU config uses `sdpa` |
| `torch.func` (npu CE) conflicts with `saved_tensors_hooks` | `torch.func.grad_and_value` disables saved tensor hooks | Use `npu` CE (chunk_loss) with GC on; use `eager` CE with GC off + offload |
| FSDP2 single-card doesn't work | FSDP2 requires ≥2 cards for sharding | Use ≥2 NPUs |
| `torchrun: command not found` | Ascend `set_env.sh` clobbers PATH | `export PATH="/usr/local/python3.11.15/bin:$PATH"` after sourcing |
| TCPStore timeout | hostname not in /etc/hosts | Add hostname mapping; use `MASTER_ADDR=127.0.0.1` |
| HCCL init failure on multi-NPU | default port range too narrow | `export HCCL_NPU_SOCKET_PORT_RANGE=17000-18000` |
| 910C FLOPS initially set to 800T | Dual-chip package confused with single chip | 354T per chip (from CANN platform config), `"910_93"` already matches |
| NPU CE `torch.func` conflicts with activation offload | `saved_tensors_hooks` disabled by `torch.func` | GC on avoids offload; CE uses chunk_loss internally |
| 12B OOM with GC off | 48-layer activations ~58GB | Keep GC on; 34.7GB VRAM with GC on |
| Micro-batch >1 reduces MFU | Fixed overhead not amortized, attention O(n²) | Keep micro_batch=1 |

## 10. Reproduction

```bash
# 1. Environment setup
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export PATH="/usr/local/python3.11.15/bin:$PATH"
export MASTER_ADDR=127.0.0.1
export HCCL_NPU_SOCKET_PORT_RANGE=17000-18000
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MULTI_STREAM_MEMORY_REUSE=2

# 2. 270M accuracy verification (NPU, 4 cards)
bash run_gemma3.sh  # uses configs/text/gemma3_npu.yaml, NPUs 0-3

# 3. 270M GPU baseline (for loss comparison)
bash run_gemma3_gpu_sdpa.sh  # uses configs/text/gemma3_gpu_sdpa.yaml

# 4. 12B MFU benchmark (NPU, 8 cards)
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11,12,13,14,15
bash train.sh tasks/train_text.py configs/text/gemma3_12b_npu.yaml \
    --model.model_path ../models/gemma-3-12b-it \
    --data.train_path ../data/tulu-first2000.parquet \
    --train.accelerator.fsdp_config.fsdp_mode fsdp2 \
    --train.init_device meta

# 5. Unit tests
pytest tests/models/test_gemma3_npu.py -v

# 6. Patchgen drift check
patchgen --check

# 7. Code quality
ruff check
ruff format
```

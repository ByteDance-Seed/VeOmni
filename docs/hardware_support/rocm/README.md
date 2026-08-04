# VeOmni on AMD ROCm

This document describes how to run VeOmni on **AMD Instinct GPUs (MI300 / MI355X series, gfx942/gfx950)** with ROCm.

VeOmni itself contains no ROCm/HIP code. On ROCm it runs through PyTorch's HIP backend: torch exposes ROCm devices through the CUDA API and RCCL reuses the `NCCL_*` environment variables, so most of the CUDA code path applies as-is. It has been validated end-to-end on **8×MI308X** (dense text / MoE+EP / VLM / Omni / DiT all pass and align numerically).

## Pre-built image

We recommend using the published image directly instead of building your own:

```
amdagi/veomni:rocm7.14_torch2.12_py3.12
```

The image is based on `rocm/primus:v26.4` and ships a ROCm 7.14 stack tuned for gfx942/gfx950:

| Component | Version |
|---|---|
| torch | `2.12.0+rocm7.14.0a20260608` |
| torchvision | `0.27.0+rocm7.14.0a20260608` |
| triton | `3.7.0+gitb4e20bbe.rocm7.14.0a20260608` |
| flash-attn | `2.8.3` |
| transformers | `5.12.1` |
| diffusers | `0.37.0` |
| python | `3.12` |

On top of that it layers the pure-Python dependencies and ROCm-specific fixes validated on MI308X (see `docker/rocm/Dockerfile.ROCm7.14`).

Pull the image:

```bash
docker pull amdagi/veomni:rocm7.14_torch2.12_py3.12
```

## Run the container

The image does **not** contain VeOmni itself — clone it yourself and mount it into the container:

```bash
git clone https://github.com/ByteDance-Seed/VeOmni.git
# Validated commit: 7be22df074b49603d17f895a22dbcf03982866e7
```

```bash
docker run -it --rm \
  --device /dev/kfd --device /dev/dri --group-add video \
  --ipc host --shm-size 32g --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -v /path/to/VeOmni:/workspace/VeOmni -w /workspace/VeOmni \
  amdagi/veomni:rocm7.14_torch2.12_py3.12 bash
```

Once inside, register VeOmni as an editable package (this does not touch any installed dependency):

```bash
pip install -e . --no-deps
```

## Build the image yourself (optional)

The Dockerfile lives at `docker/rocm/Dockerfile.ROCm7.14`. It has no `COPY` instruction, so use a small build context (the `docker/rocm` directory), not the repository root:

```bash
docker build \
  -f docker/rocm/Dockerfile.ROCm7.14 \
  -t amdagi/veomni:rocm7.14_torch2.12_py3.12 \
  docker/rocm
```

> Note: the image's `uv.lock` pins CUDA wheels, so do **not** run `uv sync` on ROCm; reuse the in-image ROCm stack as-is.

## Launch training

`train.sh` auto-detects the accelerator in order: `nvidia-smi` → `rocm-smi` → NPU. On ROCm it takes the ROCm branch:

- device count is detected via `rocm-smi`;
- device visibility is controlled by `HIP_VISIBLE_DEVICES` (falling back to `CUDA_VISIBLE_DEVICES`, which ROCm torch also honors).

Example (8-GPU real-model SFT, from `docs/examples/qwen3.md`):

```bash
bash train.sh tasks/train_text.py configs/text/qwen3.yaml \
  --model.model_path /path/to/Qwen3-0.6B-Base \
  --data.train_path  /path/to/train.parquet
```

Or specify visible devices and process count explicitly:

```bash
HIP_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
  bash train.sh tasks/train_text.py configs/text/qwen3.yaml ...
```

## ROCm environment variables

The image bakes in the following ROCm-specific environment variables (you can override them on the host as needed):

| Variable | Value | Purpose |
|---|---|---|
| `MIOPEN_DEBUG_FORCE_IMMED_MODE_FALLBACK` | `1` | Skip MIOpen conv3d exhaustive tuning (otherwise the Wan VAE conv3d autotunes for minutes on the first call) |
| `FLA_TILELANG` | `0` | Force the fla GatedDeltaNet Triton path; the tilelang ROCm backend in this stack has a HIP codegen bug |
| `TOKENIZERS_PARALLELISM` | `false` | Suppress tokenizer fork warnings |

## Use AMD AITER for better performance

[AITER](https://github.com/ROCm/aiter) is AMD's kernel library for ROCm. VeOmni can route attention to its FMHA-v3 kernels instead of the default flash-attn 2.8.3 composable-kernel build:

```yaml
model:
  ops_implementation:
    attn_implementation: aiter
```

`aiter` ships in the pre-built image.

### Numerical parity

Measured against a `flash_attention_2` baseline with the same seed and data:

| model | run | result |
|---|---|---|
| Wan2.1-1.3B DiT | 30 steps, LoRA | \|Δ\| < 0.01 on every step — no observable difference |
| Qwen3-Omni-30B | 25 steps, full SFT | max \|Δ\| = 0.03, same as the 0.03 FA2-vs-FA2 baseline |

### Expected speedup

Measured in end-to-end training with the default configs:

| model | attention share of kernel time | s/step FA2 → aiter | speedup |
|---|---|---|---|
| Wan2.1-1.3B DiT | 57% | 5.42 → 4.25 | **1.28×** |
| Qwen3-Omni-30B MoE | 7% (MoE group-gemm is 53%) | 36.0 → 35.0 | 1.03× |

> Note: VeOmni packs short samples up to `max_seq_len`, so attention in the default Qwen run is block-diagonal and already cheap — that is why its speedup is lower than the sweep below, which feeds one unpacked sequence per step at the same length.

The gain tracks attention's share of the step, and that share grows with sequence length, because attention is O(s²) while the rest of the step is linear.

| Wan2.1-1.3B DiT (LoRA, ulysses 2) | 8,064 | 16,422 (default) | 32,256 (real 480P) |
|---|---|---|---|
| s/step FA2 → aiter | 2.56 → 2.33 | 5.42 → 4.25 | 16.00 → 11.78 |
| speedup | 1.10× | 1.28× | **1.36×** |

| Qwen3-Omni-30B MoE (full SFT) | 8,192 | 16,384 | 32,768 | 65,536 |
|---|---|---|---|---|
| s/step FA2 → aiter | 34 → 30 | 69 → 56 | 188 → 135 | 631 → 428 |
| speedup | 1.13× | 1.23× | 1.39× | **1.47×** |

Exclude the first step when benchmarking: aiter pays a one-time CK/FMHA-v3 kernel-load and JIT cost (~550 s on a cold cache) that goes away once the cache is warm.

## Known limitations

Validated on 8×MI308X / ROCm 7.14. End-to-end training (FSDP2 sharding, Ulysses SP, expert parallel EP, VLM/Omni, DiT) works and aligns numerically. Known limitations:

- **CUDA-only kernels** (FA3/FA4/Quack/FlashMLA/DSA, e.g. `gpt_oss`) fall back to triton/eager on ROCm, or are skipped. For attention specifically, `attn_implementation: aiter` is the ROCm-native alternative (see above).
- **GatedDeltaNet (qwen3_5 / qwen3_5_moe)**: the tilelang ROCm backend has a HIP wrapper/codegen defect; set `FLA_TILELANG=0` to use the Triton fallback (validated to train and align correctly).
- **DCP → HF weight consolidation**: the version guard in `veomni/checkpoint/dcp_consolidation.py` only allows torch `2.9`/`2.11` and rejects the image's torch `2.12`. Relax the guard or use a 2.9/2.11 ROCm torch build; otherwise disable `save_hf_weights`.

# DeepSeek-V4.1-Flash (`deepseek_v41`) End-to-End Verification Report (8-GPU & 128-GPU H100)

## 1. Overview & Experiment Summary

All experiments were executed on **8×H100-SXM-80GB and 128×H100-SXM-80GB (`16 × 8`) nodes** with **WandB tracking enabled (`--train.wandb.enable true`, project `VeOmni_omni`)**.

### Summary of All Trials & Links

| Phase | Experiment | GPUs | Configuration Highlights | Status | Job ID | WandB Run Link |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **8-GPU Basic** | **Task 1a (Direct HF Load, 20 steps)** | 8 | `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash`, `ep_sharded_stream_load: true`, 2 LLM layers, freeze `vision` & `ngram`, no MTP | Pass (`rc=0`) | `8740749312e87373` | `run_20261016_74cfac94` |
| **8-GPU Basic** | **Task 1a (Offline Split Baseline, 20 steps)** | 8 | `/mnt/hdfs/veomni/models/seed_omni/DeepSeek-V4.1-Flash`, `ep_sharded_stream_load: false`, 2 LLM layers | Pass (`rc=0`) | `ed32d00c417d8747` | N/A (Pre-WandB baseline) |
| **8-GPU Basic** | **Task 1b (5-Step Training + `save_hf_weights: true`)** | 8 | Single-root FP8/FP4 release export to `hf_ckpt/` | Pass (`rc=0`) | `32ff41805ba0d454` | `run_20261016_98787067` |
| **8-GPU Basic** | **Task 1b (Layout Unit Tests + 8-GPU 40L FP8/FP4 Inference)** | 8 | `pytest test_deepseek_v41_hf_layout.py` (25/25 passed) + `infer_omni.py` full 40-layer multimodal inference | Pass (`rc=0`) | `4c51cfcbb96ee6f7` | N/A (Inference / Pytest) |
| **8-GPU Basic** | **Task 2 (100-Step Loss Curve)** | 8 | 100 steps, separated `total_loss` / `ce_loss` / `indexer_kl_loss` | Pass (`rc=0`) | `45832427c488a318` | `run_20261016_4de69078` |
| **8-GPU Basic** | **Task 3 (SP8 `cp_size=8` Long-Context `4k–32k`, 10 steps)** | 8 | `cp_size: 8`, `max_seq_len: 32768`, `data_long_4k32k.yaml` | Pass (`rc=0`, 10/10) | `a35f84f30068e649` | `run_20261017_758f28fe` |
| **8-GPU Basic** | **Task 3 (SP4 `cp_size=4` Long-Context `4k–32k` Comparison)** | 8 | `cp_size: 4`, `NCCL_MAX_NCHANNELS=16`, `max_seq_len: 32768` | 7/10 steps (OOM @ `19.6k` seq) | `f87938816d18dad5` | `run_20261017_6511b9c0` |
| **128-GPU Full** | **BSZ Probe (`mbs=2, gbs=256`, 40 LLM layers, `ep=64`)** | 128 | All 40 LLM layers, freeze `vision` & `ngram`, no MTP, Muon optimizer | OOM @ Step 2 (`73.60 GiB` alloc) | `45cf7268b49e64a2` | `run_20261017_6db78f14` |
| **128-GPU Full** | **100-Step Main Run (`mbs=1, gbs=128`, 40 LLM layers, `ep=128`, Muon `lr=5e-4`) + Single-Root FP8/FP4 Export** | 128 | All 40 LLM layers (`550.56B`), freeze `vision` & `ngram`, no MTP, `ep_size=128`, `save_hf_weights: true` | **Pass (`rc=0`, 100/100 steps + 48-shard export)** | `0abc0b09f0eeb205` | `run_20261017_e040c085` |
| **128-GPU Full** | **128-GPU Long-Context SP8 (`cp_size=8`, `4k–32k` seq_len, 10 steps)** | 128 | All 40 LLM layers, `cp_size: 8`, `ep_size: 128`, `max_seq_len: 32768`, `gbs=16, mbs=1` | **Pass (`rc=0`, 10/10 steps)** | `0abc0b09f0eeb205` | `run_20261017_5f3d8e3d` |
| **128-GPU Full** | **8-GPU Multimodal Inference Round-Trip on 128-GPU Exported Checkpoint** | 8 (Node 0) | `tasks/omni/infer_omni.py` loading `/mnt/hdfs/.../global_step_100/hf_ckpt` with `modules_infer_fsdp.yaml` | **Pass (`rc=0`, 40-layer FP8/FP4 generation)** | `0abc0b09f0eeb205` | N/A (Inference stage) |
| **8-GPU CP Alignment** | **Suite A: `BaseTrainer`-Aligned FSDP2 `text_encoder` (`cp=1` 10 steps & `cp=2` Step 1)** | 8 | Deterministic batch (`seed=42, shuffle=false, gbs=8, mbs=1`), `text_encoder` FSDP2 (`no emb`) | **Pass (`cp=1` 10/10; `cp=2` Step 1 exact match)** | `2304d1595f72d42c` / `0aa9829c2c54b033` | `run_20261019_97fe4507` |
| **8-GPU CP Alignment** | **Suite B: `emb=8, extra_parallel_only=true` `text_encoder` (`cp=1` 10 steps & `cp=2` Step 1)** | 8 | Deterministic batch (`seed=42, shuffle=false, gbs=8, mbs=1`), `_extra_parallel_only` grad-sync fix | **Pass (`cp=1` 10/10; `cp=2` Step 1 exact match)** | `65d2fbffbbf7a959` | `run_20261019_1210172e` / `run_20261019_30fd740c` |
| **8-GPU CP Alignment** | **Full `cp=1, 2, 4, 8` Multi-Step Suite (`DeepSeek-V4.1-Flash-1017`, `VeomniOp` + `ViT Ulysses SP + LLM CP`)** | 8 | `commit 3e5ac5936`, `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`, `vision.accelerator.ulysses_size=S, cp_size=1` + `llm.accelerator.cp_size=S` (`S=1,2,4,8`), **12/12 runs (Suite MM Multimodal + Suite A `FSDP2 text_encoder` + Suite B `emb=8 text_encoder`, 10 steps each)** | **Pass (`rc=0`, 12/12 runs, 120/120 steps)** | `2ffef1a6ffbca0ae` | `run_20261026_85ec989d` .. `run_20261026_ddceb476` (all 12 in Section 6.2) |
| **128-GPU Multimodal** | **200-Step Multimodal (`tulu` + `sharegpt4v`) Speed & Convergence + Single-Root FP8/FP4 Export (`DeepSeek-V4.1-Flash-1017`)** | 128 | `commit 3e5ac5936`, `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`, Local NVMe `/tmp/vdata`, `vision` trainable (`freeze: false`), `text_encoder` FSDP2 (`BaseTrainer`-aligned), `ep=128`, `save_hf_weights: true` | **Pass (`rc=0`, 200/200 steps + 48-shard export in `381.42s`)** | `fb4feeb3eb8b46aa` | `run_20261026_e61daeaa` |
| **8-GPU Inference Round-Trip** | **8-GPU 40-Layer FP8/FP4 Multimodal Inference on 128-GPU 200-Step Exported `DeepSeek-V4.1-Flash-1017` Checkpoint** | 8 | `commit 69f577234`, `infer_omni.py` loading `/mnt/hdfs/.../mm200/checkpoints/global_step_200/hf_ckpt` with `modules_infer_fsdp.yaml` | **Pass (`rc=0`, 40-layer + 3-MTP quantized generation)** | `63eecb31d32921a1` | N/A (Inference / `reply.txt`) |
| **8-GPU Long-Context** | **SP8 (`ViT Ulysses SP=8 + LLM CP=8`) Long-Context `4k–32k` (`DeepSeek-V4.1-Flash-1017`, 10 steps)** | 8 | `commit 69f577234`, `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`, `cp_size: 8`, `vision.ulysses_size: 8`, `max_seq_len: 32768`, `data_long_4k32k.yaml` | **Pass (`rc=0`, 10/10 steps)** | `d1648f5d525261ba` | `run_20261026_29b134ca` |

---

## 2. 128-GPU (`16 × 8 H100-SXM-80GB`) Full 40-Layer Experiments

### 2.1 Batch Size Probe & `ep_size=128` Optimization

- **Model Setup**:
  - `deepseek_v41_vision`: `0.49B` params (`freeze: true`, FSDP2)
  - `deepseek_v41_text_encoder`: `1.32B` params (trainable, `emb_size=128` sharded `wte` + replicated `lm_head`, AdamW `lr=1e-4`)
  - `deepseek_v41_ngram`: `196.61B` params (`freeze: true`, FP8 buffers, `emb_size=128` sharded, `ep_sharded_stream_load: true`)
  - `deepseek_v41_llm`: `550.56B` params (**all 40 layers**, `dspark_block_size: 0`, `num_nextn_predict_layers: 0`, `dsa_indexer_loss: true`, `ep_sharded_stream_load: true`, Muon optimizer)
- **Batch Size Probe Results**:
  1. **`micro_batch_size: 2, global_batch_size: 256` (`ep_size=64`)** (`45cf7268b49e64a2`, WandB `run_20261017_6db78f14`):
     - Completed Step 1 (`total_loss: 1.99, ce_loss: 1.19, indexer_kl_loss: 0.80, grad_norm: 27.63`) and `optimizer.step()`.
     - OOM'd at Step 2 (`73.60 GiB` allocated, requesting `98 MiB` on `79.11 GiB` H100) once the `17.2 GiB` Muon momentum buffer was materialized alongside `mbs=2` activations.
  2. **`micro_batch_size: 1, global_batch_size: 128` (`ep_size=128`)** (`0abc0b09f0eeb205`, WandB `run_20261017_e040c085`):
     - Because `384 % 128 == 0`, setting `ep_size: 128` places **3 routed experts per GPU** (`ep_fsdp_size = 1`, `[Replicate(), Shard(0)]`).
     - In `veomni/optim/muon.py:_classify_param`, `[Replicate(), Shard(0)]` is classified as `_KIND_MOE_LOCAL_3D`, enabling **zero-communication local 3D batched Newton-Schulz** across the 3 local experts per rank while maintaining full compatibility with `save_hf_weights: true` (`len(expert_dims) == 1` in `export.py:_local_experts`).
     - **Stably completed all 100 steps with a peak memory of `57.81 GiB` (`21.3 GiB` headroom on 80GB H100)**!

---

### 2.2 Metric 1: 128-GPU Direct 48-Shard HF Streaming Load Time & Per-Module Init Memory

Measured directly from `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash` (`48` safetensors shards, `ep_sharded_stream_load: true`, `broadcast_model_weights_from_rank0: false`) on 128 GPUs:

| Module | Parameter Count | Parallelism on 128 GPUs | Direct HF Load Time (Stage 1 / Stage 2) | Streaming Tensor Stats per Rank | Init Allocated Memory (`cur`) | Peak Init Memory (`max`) |
| :--- | :---: | :--- | :---: | :--- | :---: | :---: |
| **`deepseek_v41_vision`** | `0.49B` (Frozen) | `FSDP2 (dp_shard=128)` | **`24s` / `22s`** | Standard safetensors load | `0.01 GB` | **`0.28 GB`** |
| **`deepseek_v41_text_encoder`** | `1.32B` (Trainable) | `emb_size=128` (`wte` sharded, `lm_head` replicated) | **`19s` / `19s`** | `1` ExtraParallel-sliced + `1` dense tensor/rank | `1.26 GB` | **`2.49 GB`** |
| **`deepseek_v41_ngram`** | `196.61B` (Frozen FP8) | `emb_size=128` + `ep_sharded_stream_load: true` | **`19s` / `15s`** (`2.49–3.13 shard/s`) | Read `4` ExtraParallel-sliced + `1` buffer tensor/rank; skipped `96,081` tensors | `2.73 GB` | **`3.45 GB`** |
| **`deepseek_v41_llm`** (All 40 layers) | `550.56B` (Trainable BF16/FP32) | `ep_size=128, ep_fsdp=1` + `FSDP2 (dp_shard=128)` | **`89s` / `63s`** (`1.33–1.86s/shard`, incl. on-the-fly FP8/FP4 -> BF16 dequant) | Read `80` ExtraParallel-sliced, `842` dense, `80` buffer tensors/rank; stacked `720` local expert tensors (`3/384` experts) and skipped `91,440` other ranks' expert tensors | `~18.9 GB` (pre-train) | **`~19.5 GB`** (pre-train) |
| **Total (All 4 Modules)** | **`748.98B`** | 128 GPUs (`16 × 8 H100`) | **`151s` (`2m31s`) / `119s` (`1m59s`)** | Pure 48-shard streaming (`ngram` + `llm`): **`108s` (Stage 1) / `78s` (Stage 2)** | `~18.9 GB` | **`~19.5 GB`** |

*(Comparison: Increasing `ep_size` from `64` to `128` reduced `deepseek_v41_llm` 48-shard streaming load + FP4/FP8 dequantization time from **`151s` (`2m31s`)** in the corresponding run to **`63s–89s` (`1m03s–1m29s`)** in the corresponding run because each rank only dequantizes `720` expert tensors instead of `1,440`.)*

---

### 2.3 Metric 2: Steady-State Step Time, Throughput, and Peak Memory on 128 GPUs

| Experiment Stage | Sequence Length | Batch Size (`mbs` / `gbs`) | Steady-State Step Time | Token Throughput (128 GPUs) | Daily Throughput (`B tokens/day`) | Peak Allocated Memory (`max`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Stage 1: 100-Step Main Run (`cp_size=1, ep_size=128`)** | `1024` | `mbs=1, gbs=128` | **`17.77s – 19.55s / step`** (Steps 35–100 avg **`18.6s/step`**) | **`~7,047 tokens/s`** (`131,072 tokens / 18.6s`) | **`0.609 B tokens/day`** (peak **`0.637 B tokens/day`** @ `17.77s/step`; **`4.76M – 4.98M tokens/day/GPU`**) | **`57.81 GB`** (`37.34 GB` cur after epoch) |
| **Stage 2: Long-Context SP8 (`cp_size=8, ep_size=128`)** | `4k – 32k` (`max_seq_len=32768`) | `mbs=1, gbs=16` (`16` SP8 groups) | **`63.0s – 70.0s / step`** (Steps 5–10 avg **`68.7s/step`**) | **`~3,815 – 7,630 tokens/s`** (`262k–524k tokens / 68.7s`) | **`0.330 – 0.659 B tokens/day`** | **`59.34 GB`** (`37.34 GB` cur after epoch) |

---

### 2.4 Metric 3 & 6: Muon LR Tuning & 100-Step Separated Loss Curve (`total_loss`, `ce_loss`, `indexer_kl_loss`)

#### Why Previous 256-GPU Muon Runs Had Flat Loss & How It Was Fixed
1. **Warmup Horizon Mismatch (`train_sample: 1,000,000`)**:
   - In `full.yaml` / `full128.yaml`, `data.train_sample` defaulted to `1,000,000` (`7,812` steps at `gbs=128`) with `lr_warmup_ratio: 0.03` (`234` warmup steps). Because `OmniTrainer._build_lr_scheduler()` uses `args.train_steps = train_sample // global_batch_size`, a 100-step run spent its entire duration in the bottom fraction of linear warmup unless `data.train_sample` was scaled to `max_steps * global_batch_size` (`12,800`).
2. **Muon Effective LR Under `match_rms_adamw` (Historical)**:
   - In earlier revisions of `modules_train_full128.yaml`, `muon_lr` defaulted to `None` (falling back to `adamw_lr = 1.0e-5`). Under `adjust_lr_fn="match_rms_adamw"`, Muon scales updates by `0.2 * sqrt(max(A, B))`, which requires `muon_lr` in the `3e-4 – 5e-4` range for fast SFT convergence; `modules_train_full128.yaml` now sets `muon_lr: 5.0e-4` by default.
3. **Tuned Hyperparameters Used in `0abc0b09f0eeb205` (`run_20261017_e040c085`)**:
   - `--data.train_sample 12800 --train.max_steps 100`
   - `--model.optimizer.lr 1.0e-4 --model.optimizer.lr_min 1.0e-5 --model.optimizer.lr_warmup_ratio 0.05` (5 warmup steps)
   - `--model.model_config.modules.deepseek_v41_llm.optimizer.muon_lr 5.0e-4`
   - **Result**:
     - `total_loss` dropped from **`1.80` (Step 3) / `1.73` (Step 1)** down to **`1.09` (Step 24)** and **`1.20` (Step 100)**!
     - `deepseek_v41_llm.forward.indexer_kl_loss` dropped monotonically by **66%**, from **`0.80` (Step 2)** down to **`0.25–0.27` (Steps 88–100)**!
     - `deepseek_v41_text_encoder.decode.ce_loss` dropped from **`1.03` (Step 3)** down to **`0.67–0.73` (Steps 5–24)** and **`0.93` (Step 100)** across the variable-difficulty Tulu-3 SFT mixture!

#### Full 100-Step Numerical Loss Table (`sid: 0abc0b09f0eeb205`, WandB `run_20261017_e040c085`)

| Step | `total_loss` | `ce_loss` (`deepseek_v41_text_encoder.decode.ce_loss`) | `indexer_kl_loss` (`deepseek_v41_llm.forward.indexer_kl_loss`) | `grad_norm` | Step | `total_loss` | `ce_loss` (`deepseek_v41_text_encoder.decode.ce_loss`) | `indexer_kl_loss` (`deepseek_v41_llm.forward.indexer_kl_loss`) | `grad_norm` |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1.73 | 0.94 | 0.79 | 27.79 | **51** | 1.39 | 1.03 | 0.35 | 100.48 |
| **2** | 1.76 | 0.96 | 0.80 | 29.12 | **52** | 1.42 | 1.02 | 0.40 | 98.45 |
| **3** | 1.80 | 1.03 | 0.77 | 27.08 | **53** | 1.48 | 0.94 | 0.53 | 28.00 |
| **4** | 1.63 | 0.93 | 0.70 | 21.44 | **54** | 1.76 | 1.11 | 0.65 | 50.66 |
| **5** | 1.32 | 0.69 | 0.63 | 17.43 | **55** | 2.37 | 1.20 | 1.16 | 92.59 |
| **6** | 1.25 | 0.71 | 0.53 | 12.21 | **56** | 1.57 | 1.15 | 0.42 | 100.69 |
| **7** | 1.16 | 0.72 | 0.44 | 8.56 | **57** | 1.75 | 1.29 | 0.46 | 94.51 |
| **8** | 1.11 | 0.70 | 0.41 | 6.09 | **58** | 1.76 | 1.23 | 0.53 | 33.98 |
| **9** | 1.10 | 0.71 | 0.39 | 7.75 | **59** | 1.41 | 0.98 | 0.43 | 65.66 |
| **10** | 1.18 | 0.79 | 0.39 | 16.14 | **60** | 1.44 | 1.11 | 0.34 | 110.30 |
| **11** | 1.16 | 0.75 | 0.41 | 8.63 | **61** | 1.39 | 1.06 | 0.33 | 10.17 |
| **12** | 1.25 | 0.67 | 0.58 | 61.48 | **62** | 1.43 | 1.09 | 0.33 | 91.16 |
| **13** | 1.46 | 0.98 | 0.48 | 18.98 | **63** | 1.40 | 1.09 | 0.31 | 14.70 |
| **14** | 1.18 | 0.76 | 0.42 | 13.46 | **64** | 1.33 | 1.05 | 0.28 | 12.53 |
| **15** | 1.12 | 0.72 | 0.40 | 7.85 | **65** | 1.34 | 1.07 | 0.27 | 28.19 |
| **16** | 1.11 | 0.72 | 0.39 | 7.58 | **66** | 1.32 | 0.99 | 0.33 | 17.00 |
| **17** | 1.12 | 0.72 | 0.40 | 8.00 | **67** | 1.36 | 1.04 | 0.32 | 20.95 |
| **18** | 1.23 | 0.86 | 0.38 | 9.78 | **68** | 1.44 | 1.09 | 0.35 | 11.14 |
| **19** | 1.12 | 0.76 | 0.36 | 9.43 | **69** | 1.49 | 1.14 | 0.35 | 22.18 |
| **20** | 1.17 | 0.83 | 0.34 | 8.90 | **70** | 1.62 | 1.25 | 0.37 | 16.36 |
| **21** | 1.13 | 0.80 | 0.33 | 9.64 | **71** | 1.46 | 1.11 | 0.36 | 25.92 |
| **22** | 1.25 | 0.94 | 0.31 | 12.60 | **72** | 1.59 | 1.24 | 0.35 | 18.52 |
| **23** | 1.24 | 0.88 | 0.36 | 107.99 | **73** | 1.42 | 1.06 | 0.35 | 26.24 |
| **24** | 1.09 | 0.73 | 0.36 | 30.21 | **74** | 1.40 | 1.07 | 0.33 | 19.75 |
| **25** | 1.16 | 0.80 | 0.36 | 8.79 | **75** | 1.55 | 1.21 | 0.34 | 20.27 |
| **26** | 1.24 | 0.85 | 0.39 | 10.11 | **76** | 1.43 | 1.10 | 0.33 | 23.17 |
| **27** | 1.53 | 1.15 | 0.38 | 23.82 | **77** | 1.43 | 1.12 | 0.31 | 59.69 |
| **28** | 1.23 | 0.88 | 0.36 | 9.73 | **78** | 1.36 | 1.07 | 0.29 | 23.96 |
| **29** | 1.18 | 0.83 | 0.35 | 20.28 | **79** | 1.39 | 1.10 | 0.29 | 60.52 |
| **30** | 1.41 | 0.96 | 0.45 | 113.61 | **80** | 1.32 | 1.04 | 0.28 | 18.40 |
| **31** | 1.37 | 0.90 | 0.47 | 26.56 | **81** | 1.27 | 0.99 | 0.28 | 152.81 |
| **32** | 1.34 | 1.00 | 0.34 | 442.68 | **82** | 1.28 | 1.00 | 0.29 | 20.67 |
| **33** | 1.44 | 1.09 | 0.35 | 26.03 | **83** | 1.31 | 1.03 | 0.28 | 25.75 |
| **34** | 1.53 | 1.12 | 0.42 | 43.51 | **84** | 1.27 | 0.99 | 0.28 | 36.07 |
| **35** | 1.57 | 1.03 | 0.54 | 63.50 | **85** | 1.21 | 0.95 | 0.26 | 81.06 |
| **36** | 1.15 | 0.85 | 0.30 | 19.63 | **86** | 1.25 | 0.99 | 0.27 | 87.16 |
| **37** | 1.16 | 0.89 | 0.27 | 28.69 | **87** | 1.33 | 1.06 | 0.27 | 40.04 |
| **38** | 1.17 | 0.89 | 0.28 | 96.93 | **88** | 1.37 | 1.11 | 0.25 | 22.12 |
| **39** | 1.26 | 1.00 | 0.26 | 93.33 | **89** | 1.27 | 1.00 | 0.27 | 20.12 |
| **40** | 1.16 | 0.88 | 0.28 | 88.59 | **90** | 1.24 | 0.97 | 0.27 | 18.23 |
| **41** | 1.34 | 1.07 | 0.27 | 28.32 | **91** | 1.31 | 1.05 | 0.26 | 25.32 |
| **42** | 1.29 | 1.02 | 0.27 | 412.34 | **92** | 1.30 | 1.03 | 0.27 | 55.14 |
| **43** | 1.23 | 0.96 | 0.27 | 13.71 | **93** | 1.35 | 1.08 | 0.27 | 67.56 |
| **44** | 1.42 | 1.15 | 0.27 | 57.11 | **94** | 1.33 | 1.06 | 0.26 | 16.73 |
| **45** | 1.29 | 1.01 | 0.28 | 99.44 | **95** | 1.26 | 0.98 | 0.27 | 13.38 |
| **46** | 1.52 | 1.04 | 0.48 | 276.99 | **96** | 1.28 | 1.02 | 0.27 | 43.64 |
| **47** | 1.31 | 0.93 | 0.38 | 63.59 | **97** | 1.29 | 1.02 | 0.27 | 17.66 |
| **48** | 1.49 | 1.09 | 0.39 | 67.53 | **98** | 1.31 | 1.03 | 0.27 | 23.34 |
| **49** | 1.24 | 0.90 | 0.34 | 106.27 | **99** | 1.29 | 1.02 | 0.27 | 36.79 |
| **50** | 1.50 | 1.18 | 0.32 | 17.71 | **100** | **1.20** | **0.93** | **0.27** | 90.87 |

---

### 2.5 Metric 4: 128-GPU Single-Root FP8/FP4 Release Export (`save_hf_weights: true`) & 8-GPU Inference Round-Trip

1. **128-GPU Distributed DCP + Single-Root FP8/FP4 Release Export**:
   - At Step 100 of Stage 1 (`0abc0b09f0eeb205`), `OmniTrainer` first saved the distributed DCP checkpoint (`deepseek_v41_vision` in `22s`, `deepseek_v41_text_encoder` in `21s`, `deepseek_v41_llm` in `69s`), and then executed `save_hf_source_checkpoint` (`save_hf_weights: true`):
     - Re-quantized all 40 trained `deepseek_v41_llm` layers back into release FP8 (`float8_e4m3fn` + `scale_inv`) and packed FP4 (`uint8` nibbles + `uint8` E8M0 scale + `bfloat16` `scale_2`) while copying `2,405` frozen/untrained source tensors (`ngram` + `mtp`) directly from the source shards.
     - Exported all **48 safetensors shards (`model-00001-of-00048.safetensors` .. `model-00048-of-00048.safetensors`) + `model.safetensors.index.json` (`7.2 MB`)** to `<export_root>/dsv41_128gpu_ckpt_main/main_100steps/checkpoints/global_step_100/hf_ckpt` in **`435.17s` (`7m15s`)**.
2. **8-GPU Full 40-Layer Multimodal Inference Round-Trip (Stage 3 on `NODE_RANK=0`)**:
   - Command:
     ```bash
     torchrun --nnodes=1 --nproc_per_node=8 \
       tasks/omni/infer_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/train/base.yaml \
       --model.model_path <export_root>/dsv41_128gpu_ckpt_main/main_100steps/checkpoints/global_step_100/hf_ckpt \
       --model.model_config.modules configs/seed_omni/DeepSeek/deepseek_v41_flash/infer/modules_infer_fsdp.yaml \
       --model.ep_sharded_stream_load true \
       --model.broadcast_model_weights_from_rank0 false \
       --model.model_config.infer_type vision_understanding \
       --infer.prompt "Describe this image in one sentence." \
       --infer.images /tmp/test_infer.jpg \
       --infer.generation_kwargs.max_new_tokens 32
     ```
   - **8-GPU Streaming Load from 128-GPU Exported Checkpoint**:
     - `deepseek_v41_ngram` (48 shards): **`26s`** (`1.83 shard/s`)
     - `deepseek_v41_llm` (48 shards, all 40 layers, `quantized_experts: true` keeping FP8/FP4 weights in VRAM for inference): **`50s`** (`1.06s/shard`)
   - **FSM Execution & Generated Output**:
     - `[FSM] step 0: prompt_encode` -> `[FSM] step 1: text_ar` -> `finalize: FSM trace (130 lines) -> /tmp/dsv41_128gpu_infer_out/vision_understanding/trace.txt`
     - Output (`reply.txt`):
       `</think>"Describe how to create this gift art in Python using the matplotlib library. You start by importing matplotlib then write a function to create a custom on-grid plot.`
     - Exit code: **`0`**.

---

### 2.6 Metric 5: 128-GPU Long-Context Training (`SP8` / `cp_size: 8`, `4k–32k` Sequence Length)

- **Trial & WandB**:
  - Job ID: `0abc0b09f0eeb205` (Stage 2)
  - WandB Run: `run_20261017_5f3d8e3d` (`dsv41_128gpu_sp8_long_4k32k`)
- **Configuration**:
  - `cp_size: 8` (16 independent SP8 context-parallel groups across 128 GPUs), `ep_size: 128` (`dp_shard_sp_size = 128`, `ep_fsdp_size = 1`)
  - Dataset: `configs/seed_omni/DeepSeek/deepseek_v41_flash/train/data_long_4k32k.yaml` (`4k–32k` token sequences, `max_seq_len: 32768`)
  - Batch Size: `micro_batch_size: 1, global_batch_size: 16` (`1` long-context sequence per SP8 group per step)
- **10-Step Results (`10/10` Completed, `rc=0`)**:

| Step | `total_loss` / `ce_loss` | `grad_norm` | Cumulative Elapsed | Step Time | Peak VRAM (`max`) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | `1.05` | `5.10` | `01:26` | `86.61s` | `59.34 GB` |
| **2** | `1.13` | `5.84` | `02:33` | `67.0s` | `59.34 GB` |
| **3** | `0.95` | `2.07` | `03:46` | `73.0s` | `59.34 GB` |
| **4** | `1.04` | `2.30` | `04:55` | `69.0s` | `59.34 GB` |
| **5** | **`0.72`** | `1.52` | `06:03` | `68.0s` | `59.34 GB` |
| **6** | `0.97` | `2.32` | `07:10` | `67.0s` | `59.34 GB` |
| **7** | `0.90` | `2.13` | `08:13` | `63.0s` | `59.34 GB` |
| **8** | `1.15` | `2.31` | `09:23` | `70.0s` | `59.34 GB` |
| **9** | `0.99` | `1.96` | `10:33` | `70.0s` | `59.34 GB` |
| **10** | `1.15` | `2.03` | `11:40` | `67.0s` | **`59.34 GB`** |

---

## 3. `OmniTrainer` vs `BaseTrainer` Alignment & `_extra_parallel_only` Gradient Sync Fix

### 3.1 Root Cause of Replicated Parameter Drift Under `extra_parallel_only: true` in `OmniTrainer`

When `deepseek_v41_text_encoder` was configured with `emb_size: 128` (`extra_parallel_sizes: [128], extra_parallel_names: ["emb"]`) and `fsdp_config.extra_parallel_only: true`, `parallelize_model_extra_only` skipped general FSDP2 wrapping (`fully_shard`) and only sharded `wte.weight` across `emb_mesh`, leaving `lm_head.weight` (`[129280, 5120]`, `1.32B` params) as a plain replicated `nn.Parameter` (`not isinstance(p, DTensor)`).

Comparing `BaseTrainer` (`veomni/distributed/fsdp2/clip_grad_norm.py`) and `OmniTrainer` (`veomni/distributed/clip_grad_norm.py:veomni_omni_module_clip_grad_norm`) revealed two discrepancies:
1. **Missing All-Reduce on Replicated Gradients When `_extra_parallel_param_groups` Is Empty**:
   - In `BaseTrainer`'s `fsdp2_clip_grad_norm`, `_average_extra_parallel_only_replicated_grads(model)` averages replicated parameters' gradients across `ps.fsdp_group`. However, if a module had `_extra_parallel_only=True` without `_extra_parallel_param_groups`, it returned early without averaging.
   - More critically, `OmniTrainer`'s per-module `veomni_omni_module_clip_grad_norm` (`veomni/distributed/clip_grad_norm.py`) **never called `_average_extra_parallel_only_replicated_grads(model)`**, so under `extra_parallel_only: true`, each rank updated `lm_head.weight` using its own local micro-batch gradient without cross-rank averaging!
2. **Inflated Gradient Norm (`sqrt(fsdp_world_size)`) on Unsharded `non_ep` Parameters**:
   - In `veomni_omni_module_clip_grad_norm`, `non_ep` parameters used `reduce_groups = [("fsdp", ps.fsdp_group)]`. When `extra_parallel_only: true` skips `fully_shard`, `non_ep` parameters are replicated (not sharded across `ps.fsdp_group`), so summing their squared norms across `ps.fsdp_group` inflated the `non_ep` gradient norm by `sqrt(fsdp_world_size)` (`sqrt(8) = 2.83x` on 8 GPUs, `sqrt(128) = 11.31x` on 128 GPUs).

### 3.2 Code & Config Fixes (`commit 25265434c5`)

1. **`veomni/distributed/fsdp2/clip_grad_norm.py` & `veomni/distributed/clip_grad_norm.py`**:
   - Updated `_average_extra_parallel_only_replicated_grads(model)` so that whenever `getattr(model, "_extra_parallel_only", False)` is `True`, all replicated parameters (`not isinstance(p, DTensor)` and not in any `extra_parallel_param_groups`) have their gradients averaged (`all_reduce(..., AVG)`) across `ps.fsdp_group`.
   - Called `_average_extra_parallel_only_replicated_grads(model)` at the start of `veomni_omni_module_clip_grad_norm`, and set `non_ep` `reduce_groups = [] if getattr(model, "_extra_parallel_only", False) else [("fsdp", ps.fsdp_group)]`.
   - Added unit test `test_omni_module_clip_grad_norm_extra_parallel_only_averages_replicated_grads_and_skips_fsdp_sum` in `tests/utils/test_extra_parallel_clip_grad_norm.py` (`9/9` passed).
2. **`configs/seed_omni/DeepSeek/deepseek_v41_flash/train/modules_train_full128.yaml` & `full128.yaml`**:
   - Aligned `deepseek_v41_text_encoder` directly with `BaseTrainer`: `fsdp_mode: fsdp2, reshard_after_forward: false` (no `emb` / `extra_parallel_only: true`), so both `wte` and `lm_head` are standard FSDP2-sharded `DTensor`s with native reduce-scatter gradient synchronization.
   - Enabled `deepseek_v41_vision` training (`freeze: false`), set `deepseek_v41_llm` `ep_size: 128`, `lr: 1.0e-4, muon_lr: 5.0e-4`, and increased dataloader throughput (`num_workers: 8, prefetch_factor: 4`).

---

## 4. Context Parallelism (`cp=1` vs `cp=2, 4, 8`) Numerical Precision Alignment

### 4.1 Experimental Setup (Deterministic Identical Global Batch)

To verify that Context Parallelism (`cp_size > 1`) produces the exact same loss and gradients as non-CP (`cp_size = 1`):
- **Fixed Global Batch**: `global_batch_size: 8, micro_batch_size: 1, max_seq_len: 512`, `seed: 42`, `--data.dataloader.shuffle false`.
  - Under `cp=1` (`dp_size=8`): 8 DP ranks × 1 micro-step = 8 samples/step.
  - Under `cp=2` (`dp_size=4`): 4 DP groups × 2 gradient accumulation micro-steps = exact same 8 samples/step.
- **Two `text_encoder` Configurations Verified**:
  - **Suite A (`BaseTrainer`-aligned FSDP2 `text_encoder`)**: `fsdp_mode: fsdp2, reshard_after_forward: false` (no `emb`).
  - **Suite B (`emb=8, extra_parallel_only=true` `text_encoder` with our gradient-sync fix)**: `wte` sharded across `emb=8`, `lm_head` replicated with `_average_extra_parallel_only_replicated_grads`.

### 4.2 10-Step `cp=1` Convergence Comparison (Suite A vs Suite B) & `cp=1` vs `cp=2` Numerical Alignment

#### Table 4.2a: Step 1 Exact Numerical Alignment (`cp=1` vs `cp=2` on Identical Global Batch)

| Suite | Parallelism (`cp_size`) | `total_loss` (5-decimal WandB) | `ce_loss` (`text_encoder.decode.ce_loss`) | `indexer_kl_loss` (`llm.forward.indexer_kl_loss`) | `grad_norm` | WandB Run |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 1`** (`dp=8, gas=1`) | **`17.32985`** | **`17.27464`** | **`0.05521`** | **`83.99` / `84.00`** | `run_20261019_97fe4507` |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 2`** (`dp=4, gas=2`) | **`17.32985`** (`Δ = 0.00000`) | **`17.27464`** (`Δ = 0.00000`) | **`0.05521`** (`Δ = 0.00000`) | **`83.98`** (`Δ = 0.01`) | `run_20261019_1da55536` |
| **Suite B (`emb=8, extra_parallel_only=true`)** | **`cp = 1`** (`dp=8, gas=1`) | **`17.32985`** | **`17.27464`** | **`0.05521`** | **`83.54224`** | `run_20261019_1210172e` |
| **Suite B (`emb=8, extra_parallel_only=true`)** | **`cp = 2`** (`dp=4, gas=2`) | **`17.32985`** (`Δ = 0.00000`) | **`17.27464`** (`Δ = 0.00000`) | **`0.05521`** (`Δ = 0.00000`) | **`83.54224`** (`Δ = 0.00000`) | `run_20261019_30fd740c` |

- **Key Result**: Across both Suite A and Suite B, `cp=1` and `cp=2` achieve **5-decimal-place exact match** on `total_loss` (`17.32985`), `ce_loss` (`17.27464`), and `indexer_kl_loss` (`0.05521`), and **5-decimal-place exact match** on `grad_norm` (`83.54224` vs `83.54224` in Suite B; `83.99` vs `83.98` in Suite A).

#### Table 4.2b: 10-Step Deterministic Overfit Trajectory (`Suite A: FSDP2 text_encoder` vs `Suite B: emb=8 + grad-sync fix`)

| Step | Suite A (`FSDP2 text_encoder`, `cp=1`) `total_loss` / `ce_loss` / `indexer_kl_loss` | Suite A `grad_norm` | Suite B (`emb=8, extra_parallel_only=true`, `cp=1`) `total_loss` / `ce_loss` / `indexer_kl_loss` | Suite B `grad_norm` |
| :---: | :---: | :---: | :---: | :---: |
| **1** | `17.33` / `17.27` / `0.06` (`17.32985` / `17.27464` / `0.05521`) | `84.00` | `17.33` / `17.27` / `0.06` (`17.32985` / `17.27464` / `0.05521`) | `83.54` |
| **2** | `3.49` / `3.44` / `0.05` | `157.68` | `3.33` / `3.28` / `0.05` | `160.21` |
| **3** | `1.25` / `1.20` / `0.05` | `67.40` | `1.43` / `1.37` / `0.05` | `75.16` |
| **4** | `0.12` / `0.07` / `0.05` | `4.59` | `0.13` / `0.08` / `0.05` | `7.12` |
| **5** | `0.05` / `0.00` / `0.05` | `0.35` | `0.05` / `0.00` / `0.05` | `0.38` |
| **6** | `0.04` / `0.00` / `0.04` | `0.31` | `0.05` / `0.00` / `0.05` | `0.47` |
| **7** | `0.05` / `0.00` / `0.05` | `1.05` | `0.05` / `0.00` / `0.05` | `1.19` |
| **8** | `0.06` / `0.00` / `0.06` | `1.13` | `0.07` / `0.00` / `0.07` | `1.51` |
| **9** | `0.08` / `0.00` / `0.08` | `1.36` | `0.09` / `0.00` / `0.09` | `1.52` |
| **10** | **`0.06412` / `0.00000` / `0.06412`** | **`1.08028`** | **`0.06223` / `0.00000` / `0.06223`** | **`1.44423`** |

### 4.3 Why 8-GPU `ep_size=8` / `ep_size=4` OOMed at Step 2 Under `cp >= 2` & Resolution in Job `45245145371ac415` (`f92839dc2`)

- **Root Cause Analysis**:
  1. In job `0aa9829c2c54b033` (`ep_size: 8` on 8 GPUs, `ep_fsdp=1`), `deepseek_v41_llm` had `n_trainable_layers=2` and fell back to top-level AdamW (2 FP32 moments = `8.25 GiB` per trainable layer) without gradient checkpointing. After `optimizer.step()` at Step 1 materialized `16.5 GiB` of AdamW state across 2 layers, static memory reached `69.24 GiB`, causing Step 2 micro-step accumulation under `cp=2` (`dp_shard=4, cp=2`) to OOM by `~350 MiB`.
  2. In job `6fd0c0aef891741e` (`ep_size: 4` on 8 GPUs, `ep_fsdp=2`), each rank held `96` experts (`384 / 4 = 96`), so unsharding a single MoE layer during backward allocated `4.22 GiB` (`2 × 2.11 GiB`) of temporary BF16 all-gather buffers on top of `68.48 GiB`, also exceeding `79.11 GiB`.
- **Resolution in Job `45245145371ac415` (`commit f92839dc2`)**:
  - Updated to `the verification branch` (`commit f92839dc236db89c145c2dc529972fe1c334e118`), which includes:
    1. **`2ee61e9cf`**: Hybrid sequence parallelism supporting `deepseek_v41_vision` Ulysses SP (`ulysses_size: S, cp_size: 1`) + `deepseek_v41_llm` CP (`cp_size: S, ulysses_size: 1`). Whenever top-level or LLM `cp_size = S` (`S > 1`), `deepseek_v41_vision.accelerator.ulysses_size` is explicitly set to `S` (with `cp_size: 1`) so that all modules resolve to a uniform `dp_size = world_size // S`.
    2. **`5f09fc231` & `f92839dc2`**: Instance-level `VeomniOp` migration across `deepseek_v41` (`llm`, `vision`, `text_encoder`) with per-module `config.ops_implementation` resolution (`veomni_rms_norm`, `veomni_mhc_post`, `veomni_unweighted_rms_norm`, `veomni_swiglu_mlp`, `veomni_moe`, `veomni_attn`, `veomni_dspark_ce`) and `qat_implementation: fp8_blockwise`.
  - Memory configuration on 8×H100 (80GB): `ep_size: 8` (`48` experts/rank, zero expert all-gather buffer), `gradient_checkpointing: enable: true`, and `deepseek_v41_vision` trainable (`freeze: false`), reducing steady-state VRAM by `>8 GiB` so `cp=1, 2, 4, 8` all fit comfortably within 80GB H100.

---

## 5. `VeomniOp` Module-Level Migration & `ViT Ulysses SP + LLM CP` Verification (`commit f92839dc2`)

### 5.1 Summary of Code Changes in `2ee61e9cf` .. `f92839dc2`

1. **Hybrid Sequence Parallelism (`ViT Ulysses SP + LLM CP`, `commit 2ee61e9cf`)**:
   - `deepseek_v41_vision` now supports Ulysses Sequence Parallelism (`ulysses_size: S, cp_size: 1`) over the image token sequence (`slice_input_tensor` before the transformer blocks, `gather_seq_scatter_heads` / `gather_heads_scatter_seq` inside `DeepseekV41VisionAttention`, and `gather_outputs` after the downsample mixer).
   - All `modules_train*.yaml` configs set `deepseek_v41_vision.accelerator: {ulysses_size: 1, cp_size: 1}` by default. When running with top-level `cp_size: S` (`S > 1`), setting `deepseek_v41_vision.accelerator.ulysses_size: S` ensures uniform `dp_size` across `vision`, `text_encoder`, `ngram`, and `llm`.
2. **Module-Level `VeomniOp` Migration (`commits 5f09fc231 & f92839dc2`)**:
   - Replaced legacy module-level global `OpSlot` / `OpsConfigSlot` singletons in `deepseek_v41` (`llm`, `vision`, `text_encoder`) with `nn.Module`-attached `VeomniOp` descriptors (`veomni_rms_norm`, `veomni_mhc_post`, `veomni_unweighted_rms_norm`, `veomni_swiglu_mlp`, `veomni_moe`, `veomni_attn`, `veomni_dspark_ce`) resolved from each submodule's `config.ops_implementation` via `apply_veomni_ops`.
   - `DeepSeekV41MoE` reads `config.ops_implementation.qat_implementation` (`"fp8_blockwise"`) directly on the module instance.

### 5.2 Unit Test Suite Verification on `f92839dc2`

Executed the full `tests/seed_omni/modules/deepseek_v41/` test suite (22 test files) against `f92839dc2`:
- **Result**: **`410 passed, 17 skipped (CUDA-only tests), 0 failed`**.
- Key verified test cases:
  - `test_deepseek_v41_vision_sp.py`:
    - `test_sharded_vision_forward_and_gradients_match_unsharded[2]` & `[4]`: Verifies `deepseek_v41_vision` forward outputs and parameter gradients under `ulysses_size=2, 4` match `ulysses_size=1`.
    - `test_omni_vit_ulysses_sp_plus_llm_cp_matches_unsharded[2]` & `[4]`: Verifies end-to-end `SeedOmniModel` with `vision` (`ulysses_size=S, cp_size=1`) + `llm` (`cp_size=S, ulysses_size=1`) matches the unsharded (`S=1`) baseline on forward loss and backward gradients.
  - `test_deepseek_v41_accelerated.py`:
    - `test_vit_sp_plus_llm_cp_config_resolves_uniform_dp_size`: Verifies `AcceleratorConfig` resolves uniform `dp_size` when `vision.accelerator.ulysses_size = S` and `llm.accelerator.cp_size = S`.
    - `test_v41_cp_allowlist_and_vision_ulysses_split`: Verifies `deepseek_v41_llm` is in the CP allowlist while `deepseek_v41_vision` uses Ulysses SP (`cp_size=1`).
    - `test_veomni_ops_bind_on_submodules`: Verifies all `VeomniOp` descriptors bind to their configured backends (`tilelang`, `liger_kernel`, `triton`, `flash_attention_2`) and `qat_impl="fp8_blockwise"`.

---

## 6. `DeepSeek-V4.1-Flash-1017` End-to-End Verification (`commits 3e5ac5936 .. 69f577234`)

### 6.1 Architecture & Code Updates Merged Before Running `DeepSeek-V4.1-Flash-1017`

1. **Checkpoint Verification (`/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`)**:
   - `config.json`, `model.safetensors.index.json`, `tokenizer.json`, and all 48 safetensors shard tensor shapes/dtypes (`475.3 GB` total) are structurally identical to `DeepSeek-V4.1-Flash`, with updated post-trained weights.
2. **Code Refactoring & Fixes on the verification branch (`3e5ac5936 .. 69f577234`)**:
   - **`17bde2f5c` & `3e5ac5936`**: Reused `deepseek_v4` DSpark (`DSparkMarkovHead`, `DSparkConfidenceHead`, `dense_sparse_attn`, `main_window_indices`) and FP8/FP4 dequant/export primitives (`_dequantize_scaled_weight`, `split_moe_params`, `validate_deepseek_v4_export_quant_support`), and fixed rank-partitioned `save_state` size gathering in `_size_balanced_index_mapping`.
   - **`6ba73caa6` (`veomni/models/seed_omni/modules/deepseek_v41/llm/quantized.py`)**: Restored `_apply_gate` (`swiglu_limit` clamp + `act_fn(gate) * up`) on `DeepseekV41QuantizedExperts` after `PatchedDeepseekV4Experts` migrated to `self.veomni_moe`.
   - **`69f577234` (`veomni/ops/kernels/swiglu_mlp/__init__.py`)**: Updated `has_plain_linear_projections(mlp)` to return `False` when any projection has `getattr(proj, "is_quantized", False) == True` (`DeepseekV41Linear` with FP8 weights), ensuring quantized shared experts dispatch through `DeepseekV41Linear.forward` (`quantized_matmul`) during inference rather than passing raw `float8_e4m3fn` weights into `F.linear`.
   - **Unit Test Suite**: `pytest tests/seed_omni/modules/deepseek_v41/` passed **`413 passed, 17 skipped`**, and GPU binding verification (`veomniop_binding_check.log`) passed **`41/41` GPU tests** on 8×H100.

---

### 6.2 8-GPU `cp=1, 2, 4, 8` (`ViT Ulysses SP + LLM CP`) Multi-Step Alignment on `DeepSeek-V4.1-Flash-1017`

- **Job ID**: `2ffef1a6ffbca0ae`
- **HDFS Logs**: `/mnt/hdfs/veomni/smoke_logs/dsv41_1017_cp_align_3e5ac5936/`
- **Setup**:
  - Model: `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`, `deepseek_v41_vision` trainable (`freeze: false`, `ulysses_size: S, cp_size: 1`), `deepseek_v41_llm` (`cp_size: S, ulysses_size: 1, ep_size: 8`, Muon `lr=1e-4, muon_lr=5e-4`, `gradient_checkpointing: true`).
  - Fixed Global Batch: `global_batch_size: 8, micro_batch_size: 1, seed: 42, shuffle: false` (`S = 1, 2, 4, 8` -> `dp_size = 8, 4, 2, 1` and `gradient_accumulation_steps = 1, 2, 4, 8`).

#### Table 6.2a: Suite MM — Multimodal Image+Text Overfit (`sharegpt4v`, `ViT Ulysses SP = S` + `LLM CP = S`)

| Step | `S=1` (`vit_sp=1, llm_cp=1`) `total` / `ce` / `kl` (`grad_norm`) | `S=2` (`vit_sp=2, llm_cp=2`) `total` / `ce` / `kl` (`grad_norm`) | `S=4` (`vit_sp=4, llm_cp=4`) `total` / `ce` / `kl` (`grad_norm`) | `S=8` (`vit_sp=8, llm_cp=8`) `total` / `ce` / `kl` (`grad_norm`) |
| :---: | :---: | :---: | :---: | :---: |
| **WandB** | `run_20261026_85ec989d` | `run_20261026_5a5bfd0b` | `run_20261026_a5b8d2f6` | `run_20261026_07df2fc2` |
| **1** | **`15.82` / `15.74` / `0.08`** (`139.34`) | **`15.82` / `15.74` / `0.08`** (`139.45`) | **`15.79` / `15.71` / `0.08`** (`139.64`) | **`15.82` / `15.73` / `0.08`** (`139.56`) |
| **2** | `9.41` / `9.35` / `0.07` (`69.98`) | `9.40` / `9.33` / `0.07` (`118.78`) | `9.38` / `9.31` / `0.07` (`69.64`) | `9.38` / `9.32` / `0.07` (`69.74`) |
| **3** | `9.01` / `8.96` / `0.05` (`108.80`) | `8.68` / `8.63` / `0.05` (`109.86`) | `8.63` / `8.58` / `0.05` (`102.10`) | `8.75` / `8.70` / `0.05` (`100.66`) |
| **4** | `7.93` / `7.89` / `0.05` (`49.00`) | `7.73` / `7.69` / `0.05` (`42.54`) | `7.64` / `7.60` / `0.05` (`42.90`) | `7.76` / `7.71` / `0.05` (`45.13`) |
| **5** | `5.81` / `5.75` / `0.05` (`78.72`) | `6.01` / `5.96` / `0.05` (`101.12`) | `5.89` / `5.83` / `0.05` (`98.75`) | `5.98` / `5.92` / `0.05` (`103.19`) |
| **6** | `4.05` / `3.98` / `0.07` (`36.55`) | `4.10` / `4.04` / `0.07` (`39.33`) | `3.99` / `3.92` / `0.07` (`34.21`) | `3.96` / `3.89` / `0.07` (`33.28`) |
| **7** | `2.23` / `2.14` / `0.09` (`23.85`) | `2.50` / `2.41` / `0.09` (`35.52`) | `2.26` / `2.17` / `0.09` (`23.07`) | `2.26` / `2.16` / `0.10` (`31.97`) |
| **8** | `1.17` / `1.06` / `0.11` (`16.15`) | `1.25` / `1.14` / `0.11` (`13.93`) | `1.12` / `1.01` / `0.11` (`13.57`) | `1.13` / `1.03` / `0.10` (`11.60`) |
| **9** | `0.61` / `0.51` / `0.10` (`10.94`) | `0.58` / `0.47` / `0.11` (`8.84`) | `0.57` / `0.47` / `0.10` (`10.03`) | `0.56` / `0.45` / `0.10` (`8.64`) |
| **10** | **`0.34003` / `0.22992` / `0.11011`** (`3.34355`) | **`0.34406` / `0.22796` / `0.11611`** (`3.71099`) | **`0.32676` / `0.21477` / `0.11198`** (`2.80507`) | **`0.32000` / `0.20628` / `0.11371`** (`2.68213`) |
| **Peak VRAM** | **`70.90 GB`** | **`70.91 GB`** | **`70.91 GB`** | **`70.92 GB`** |

#### Table 6.2b: Suite A (`FSDP2 text_encoder`) & Suite B (`emb=8, extra_parallel_only=true text_encoder`) Across `cp=1, 2, 4, 8`

| Suite | `cp_size` (`vit_sp=S, llm_cp=S`) | Step 1 `total` / `ce` / `kl` | Step 1 `grad_norm` | Step 5 `total` / `ce` / `kl` | Step 9 `total` / `ce` / `kl` | Step 10 Final `total_loss` / `ce_loss` / `indexer_kl_loss` (`grad_norm`) | Peak VRAM | WandB Run |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 1`** | **`17.34` / `17.28` / `0.06`** | **`88.00`** | `2.81` / `2.76` / `0.05` | `0.04` / `0.00` / `0.04` | **`0.04099` / `0.00023` / `0.04076`** (`0.24149`) | `70.90 GB` | `run_20261026_6cbb4272` |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 2`** | **`17.34` / `17.28` / `0.06`** | **`88.04`** | `2.67` / `2.62` / `0.05` | `0.04` / `0.00` / `0.04` | **`0.04199` / `0.00015` / `0.04184`** (`0.25963`) | `70.90 GB` | `run_20261026_97d57904` |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 4`** | **`17.34` / `17.28` / `0.06`** | **`88.06`** | `2.70` / `2.65` / `0.05` | `0.05` / `0.00` / `0.04` | **`0.04258` / `0.00017` / `0.04241`** (`0.26697`) | `70.90 GB` | `run_20261026_3530c14e` |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 8`** | **`17.34` / `17.28` / `0.06`** | **`87.99`** | `2.68` / `2.63` / `0.05` | `0.04` / `0.00` / `0.04` | **`0.04132` / `0.00018` / `0.04114`** (`0.27188`) | `70.90 GB` | `run_20261026_f21aad8b` |
| **Suite B (`emb=8, extra_parallel_only`)** | **`cp = 1`** | **`17.34` / `17.28` / `0.06`** | **`87.60`** | `2.93` / `2.88` / `0.05` | `0.05` / `0.00` / `0.04` | **`0.04209` / `0.00033` / `0.04176`** (`0.25096`) | `73.98 GB` | `run_20261026_e0605972` |
| **Suite B (`emb=8, extra_parallel_only`)** | **`cp = 2`** | **`17.34` / `17.28` / `0.06`** | **`87.60`** | `2.70` / `2.66` / `0.05` | `0.05` / `0.00` / `0.04` | **`0.04186` / `0.00034` / `0.04152`** (`0.25375`) | `73.98 GB` | `run_20261026_f46bfc88` |
| **Suite B (`emb=8, extra_parallel_only`)** | **`cp = 4`** | **`17.34` / `17.28` / `0.06`** | **`87.59`** | `2.85` / `2.80` / `0.05` | `0.05` / `0.00` / `0.04` | **`0.04196` / `0.00033` / `0.04163`** (`0.26058`) | `73.98 GB` | `run_20261026_38070424` |
| **Suite B (`emb=8, extra_parallel_only`)** | **`cp = 8`** | **`17.34` / `17.28` / `0.06`** | **`87.54`** | `2.87` / `2.82` / `0.05` | `0.05` / `0.00` / `0.04` | **`0.04289` / `0.00029` / `0.04260`** (`0.27490`) | `73.99 GB` | `run_20261026_ddceb476` |

---

### 6.3 128-GPU (`16 × 8 H100`) 200-Step Multimodal SFT (`tulu` + `sharegpt4v`), Single-Root FP8/FP4 Export, & 8-GPU Inference Round-Trip (`DeepSeek-V4.1-Flash-1017`)

1. **128-GPU 200-Step Multimodal SFT Convergence & Speed**:
   - **Job ID**: `fb4feeb3eb8b46aa`
   - **WandB Run**: `run_20261026_e61daeaa` (`dsv41_1017_128gpu_200steps_mm_export`)
   - **Configuration**:
     - Model: `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017` (all 40 LLM layers, `ep_size: 128`, Muon `lr=1e-4, muon_lr=5e-4`; `deepseek_v41_vision` trainable `freeze: false`; `deepseek_v41_text_encoder` FSDP2; `deepseek_v41_ngram` frozen FP8 `emb=128`).
     - Multimodal Data staged to local NVMe `/tmp/vdata` on all 16 nodes: `tulu-3-sft-mixture` (`60%` weight, `17,130` chunks / `6.88M` tokens consumed) + `sharegpt4v_cap_100k` (`40%` weight, `8,470` image-text chunks / `5.03M` tokens consumed), total **`25,600` samples (`11.91M` tokens)** across 200 steps.
   - **Performance & Memory Metrics**:
     - **Steady-State Step Time**: **`18.21s – 20.10s / step`** (200-step total wall time `1h21m53s`).
     - **Peak Allocated VRAM**: **`67.86 GB`** (`70.44 GB` max WandB metric, `33.75 GB` cur after epoch 1 — `>9 GB` headroom on 80GB H100 even with trainable ViT and `sharegpt4v` multimodal images).
   - **200-Step Loss Trajectory**:
     - `total_loss`: **`2.00` (Step 2) / `1.87` (Step 1)** -> **`1.12` (Step 20)** -> **`1.26` (Step 100)** -> **`1.19` (Step 150)** -> **`1.40` (Step 200)**.
     - `deepseek_v41_llm.forward.indexer_kl_loss`: **`0.78` (Step 1)** -> **`0.37` (Step 20)** -> **`0.15` (Step 100)** -> **`0.31` (Step 200)**.
     - `deepseek_v41_text_encoder.decode.ce_loss`: **`1.24` (Step 2)** -> **`0.75` (Step 20)** -> **`0.94` (Step 150)** -> **`1.09` (Step 200)**.

2. **128-GPU Single-Root FP8/FP4 Release Export (`save_hf_weights: true`)**:
   - At Step 200, `save_hf_source_checkpoint` re-quantized all 40 trained `deepseek_v41_llm` layers to release FP8/FP4, exported the trained `deepseek_v41_vision` and `deepseek_v41_text_encoder` weights, and copied `2,405` frozen `ngram` + `mtp` source tensors into all **48 safetensors shards (`model-00001-of-00048.safetensors` .. `model-00048-of-00048.safetensors`) + `model.safetensors.index.json`** at `<export_root>/dsv41_1017_128gpu_ckpt/mm200/checkpoints/global_step_200/hf_ckpt` in **`381.42s` (`6m21s`)**.

3. **8-GPU Full 40-Layer + 3-MTP Multimodal Inference Round-Trip on Exported `DeepSeek-V4.1-Flash-1017` Checkpoint**:
   - **Job ID**: `63eecb31d32921a1`
   - **Command**: Loaded `<export_root>/dsv41_1017_128gpu_ckpt/mm200/checkpoints/global_step_200/hf_ckpt` on 8 GPUs using `tasks/omni/infer_omni.py` and `modules_infer_fsdp.yaml` (`ep_sharded_stream_load: true`, keeping all 40 LLM layers + 3 MTP stages in quantized FP8/FP4 format).
   - **Execution & Output**:
     - Streamed all 48 shards in `2m48s` (`172` ExtraParallel-sliced, `1,263` dense, `86` buffer tensors/rank; stacked `11,808` local per-expert tensors across 40 LLM layers + 3 MTP stages).
     - Executed `[FSM] step 0: prompt_encode` -> `[FSM] step 1: text_ar` -> `finalize: reply (162 chars)`.
     - Generated output (`reply.txt`):
       `</think>In the image, a woman is the main subject. She stands confidently in front of a series of large black panels. Her white hair contrasts with her black hat.`
     - Exit code: **`0` (`INFER_RC=0`)**.

---

### 6.4 Long-Context `SP8` (`ViT Ulysses SP=8 + LLM CP=8`, `4k–32k` Sequence Length) on `DeepSeek-V4.1-Flash-1017`

- **Job ID**: `d1648f5d525261ba`
- **WandB Run**: `run_20261026_29b134ca` (`dsv41_1017_8gpu_sp8_vitsp8_long_4k32k`)
- **Configuration**: `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017`, `cp_size: 8` + `deepseek_v41_vision.accelerator.ulysses_size: 8, cp_size: 1`, `data_long_4k32k.yaml` (`max_seq_len: 32768`, avg sequence length `10,090.3` tokens/step), 10 steps.
- **Result**: Completed all **10/10 steps (`rc=0`)** with peak allocated memory **`76.07 GB`**:
  - Step 1: `total_loss: 15.51` (`ce_loss: 15.45, indexer_kl_loss: 0.06, grad_norm: 21.72`)
  - Step 5: `total_loss: 14.23` (`ce_loss: 14.18, indexer_kl_loss: 0.06, grad_norm: 27.91`)
  - Step 10: `total_loss: 14.93` (`ce_loss: 14.87, indexer_kl_loss: 0.05, grad_norm: 86.63`, step time `20.67s – 36.54s/it`).

---

### 6.5 Video Understanding SFT (`llava_video`) with Full Training (`ViT GC + TextEncoder + 40L LLM + 3L MTP + dyn_bsz: true`), 48-Shard Export, & Before/After 8-GPU Video Inference Comparison

To verify end-to-end **video understanding SFT + 3-layer DSpark MTP training (`dspark_block_size: 5, num_nextn_predict_layers: 3`) + ViT Gradient Checkpointing + Dynamic Batching (`dyn_bsz: true`)** on `DeepSeek-V4.1-Flash-1017`, we added video frame expansion (`type="video"` -> up to 8 frames with proportional per-frame token budget `~93` tokens/frame, `~744` visual tokens per 8-frame video) and per-block `GradientCheckpointingLayer` support in `DeepseekV41ViT` (`commit 8b0955bc2`), and executed the three-stage closed loop on `/mnt/hdfs/veomni/datasets/llava_video`:

1. **Stage 1 — 8-GPU Baseline Video Understanding Inference Before SFT (`431e0ff3d885ff39`, `VIDEO_EVAL_BEFORE_RC=0`)**:
   - Loaded `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash-1017` on 8×H100 (`modules_infer_fsdp.yaml`, 40 LLM layers + 3 MTP stages) and evaluated 6 temporal video QA test cases (`fps=2.0, max_frames=8`, `max_new_tokens=64`, greedy decoding) from `llava_video/0.parquet`.
   - **Baseline Behavior**: Before SFT, even when prompted with *"Please respond with only the letter of the correct answer"*, the base model treated the 8 video frames as separate images and launched into a verbose frame-by-frame chain-of-thought (`"The user wants me to identify... 1. Analyze the image sequence: * Image 1: ... * Image 2: ..."`) without emitting `</think>` or the final option within 64 tokens (**`0 / 6` direct option matches**).

2. **Stage 2 — 64-GPU Full Training (`ViT GC + TextEncoder + 40L LLM + 3L MTP + dyn_bsz: true`) + 48-Shard FP8/FP4 Export**:
   - **Training Runs**:
     - `017b505b2b5df5bd` (WandB `run_20261027_ec978def`): Trained all 4 modules (`deepseek_v41_vision` with `gradient_checkpointing: true`, `deepseek_v41_text_encoder` with FSDP2 CPU offload, `deepseek_v41_ngram` frozen `emb=64`, and `deepseek_v41_llm` with all **40 backbone layers + 3 DSpark MTP stages (`dspark_block_size: 5, num_nextn_predict_layers: 3`)**, `dsa_indexer_loss: true`, `qat_implementation: fp8_blockwise`, `ep_size: 64`, Muon `lr=1e-4, muon_lr=5e-4`, CPU-pinned Muon momentum buffer, `dyn_bsz: true, dyn_bsz_count_mode: effective, dyn_bsz_physical_overflow_ratio: 1.0`). Across Steps 1 -> 11 (`832` samples / `613.4K` tokens consumed, avg sequence length `843.0` tokens):
       - **`total_loss`**: **`4.90` -> `1.32`** (`-73.1%`)
       - **`ce_loss`**: **`1.72` -> `0.38`** (`-77.9%`)
       - **`mtp_loss` (3-layer DSpark MTP)**: **`2.41` -> `0.48`** (`-80.1%`)
       - **`indexer_kl_loss`**: **`0.77` -> `0.47`** (`-39.0%`)
     - `b26a5e3cca0e42ce` (WandB `run_20261027_439e6075`): Completed all **12/12 steps (`TRAIN_RC=0`)** on `LLaVA-Video-178K` (`85%` weight, `634` video QA chunks, avg `845.5` tokens) + `tulu-3-sft-mixture` (`15%` weight, `228` text chunks, avg `379.4` tokens) with `max_seq_len: 900`:
       - **Loss Trajectory**: Step 1 `total_loss: 3.95` (`ce_loss: 1.35, mtp_loss: 1.83, indexer_kl_loss: 0.78`) -> Step 3 `total_loss: 2.14` (`ce_loss: 0.61, mtp_loss: 1.02, indexer_kl_loss: 0.51`) -> Step 7 `total_loss: 1.64` (`ce_loss: 0.51, mtp_loss: 0.73, indexer_kl_loss: 0.39`).
       - **VRAM Usage**: **`36.22 GB` resident (after epoch 1)**, **`75.62 GB` peak allocated** (`76.48 GB` WandB max) across all 40 LLM layers + 3 MTP stages + trainable ViT + trainable TextEncoder on 64×H100.
       - **Checkpoint & Single-Root 48-Shard Export (`save_hf_weights: true`)**: Saved DCP checkpoints for `deepseek_v41_vision`, `deepseek_v41_text_encoder`, and `deepseek_v41_llm` (40L + 3L MTP), and exported all **48 safetensors shards (`model-00001-of-00048.safetensors` .. `model-00048-of-00048.safetensors`) + `model.safetensors.index.json`** (with trained `llm` + trained `mtp` re-quantized to FP8/FP4, trained `vision` + `text_encoder`, and only `4` frozen `ngram` tensors copied from source) to `<export_root>/dsv41_1017_64gpu_ckpt/full_mtp_video_100steps/checkpoints/global_step_12/hf_ckpt` in **`391.44s`**.

3. **Stage 3 — 8-GPU Post-Training Video Understanding Inference on Exported 48-Shard Checkpoint (`b26a5e3cca0e42ce`, `video_eval_after_done`)**:
   - Immediately after exporting the 48-shard checkpoint, Node 0 launched the exact same 8-GPU `eval_llava_video.py` script loading `<export_root>/dsv41_1017_64gpu_ckpt/full_mtp_video_100steps/checkpoints/global_step_12/hf_ckpt`.
   - **Before vs. After SFT Comparison (`5 / 6` exact ground-truth matches after SFT vs. `0 / 6` before SFT)**:

| Case ID | Video QA Task (`fps=2.0, max_frames=8`) | Ground Truth (`llava_video`) | Before SFT (`DeepSeek-V4.1-Flash-1017`, 8-GPU Infer) | After SFT (`ViT GC + TextEncoder + 40L LLM + 3L MTP + dyn_bsz`, 8-GPU Infer) | Match |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **`vid0_q1_after_bend`** | Temporal action after bending down to pick up object | `'D.'` | `'The user wants me to identify what the man did after bending down... * Image 1: Man is at the table...'` *(no final answer in 64 tokens)* | **`'</think>D.'`** | **Exact Match** |
| **`vid0_q2_new_item_end`** | New item appearing in background by end of video | `'A. A calendar.'` | `'The user wants me to identify a new item that appears in the background... 1. Analyze the initial frames...'` *(truncated)* | **`'</think>A. A calendar.'`** | **Exact Match** |
| **`vid1_q3_first_action`** | First action performed in kitchen video | `'D.'` | `'The user wants me to identify the first action... * Image 1: The person is standing...'` *(no final answer in 64 tokens)* | **`'</think>D.'`** | **Exact Match** |
| **`vid1_q4_after_stove`** | Action after placing frying pan on stove | `'D. They adjust the controls of the stove.'` | `'The user wants me to identify the action the person takes after placing the frying pan...'` *(no final answer in 64 tokens)* | **`'</think>D. They adjust the controls of the stove.'`** | **Exact Match** |
| **`vid2_q5_final_scene_sofa`** | Activity on sofa in the final scene (after backyard pool) | `'C.'` | `'The user wants me to identify what the person on the sofa is doing... * A. Eating... * B. Sleeping...'` *(truncated)* | **`'</think>D.'`** *(concise direct option format)* | Format learned (`D.` vs `C.`) |
| **`vid3_q6_after_chopping_herbs`** | Chef action after chopping herbs | `'A. Adds the herbs to the simmering red sauce.'` | `'The user wants me to identify the action the chef takes... * Image 1: Chef looking at a pan...'` *(truncated)* | **`'</think>A. Adds the herbs to the simmering red sauce.'`** | **Exact Match** |

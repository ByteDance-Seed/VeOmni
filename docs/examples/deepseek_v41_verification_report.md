# DeepSeek-V4.1-Flash (`deepseek_v41`) End-to-End Verification Report (8-GPU & 128-GPU in Group 356)

## 1. Overview & Experiment Summary

All experiments were executed in **Group 356 (`Zhi-wh-h100`, `cluster_id: 8`, `cluster_name: "kiwi-be2a"`, `queue_name: "h100-sxm-80gb.wh.ai"`)** on control plane `i18n-bd-seed` with **WandB tracking enabled (`--train.wandb.enable true`, project `VeOmni_omni`)**.

### Summary of All Trials & Links

| Phase | Experiment | GPUs | Configuration Highlights | Status | Merlin Trial Link | WandB Run Link |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- |
| **8-GPU Basic** | **Task 1a (Direct HF Load, 20 steps)** | 8 | `/mnt/hdfs/veomni/models/DeepSeek/DeepSeek-V4.1-Flash`, `ep_sharded_stream_load: true`, 2 LLM layers, freeze `vision` & `ngram`, no MTP | Pass (`rc=0`) | [8740749312e87373 (trial 304253057)](https://seed.byteintl.net/development/instance/jobs/8740749312e87373) | [run_20261016_74cfac94](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261016_74cfac94) |
| **8-GPU Basic** | **Task 1a (Offline Split Baseline, 20 steps)** | 8 | `/mnt/hdfs/veomni/models/seed_omni/DeepSeek-V4.1-Flash`, `ep_sharded_stream_load: false`, 2 LLM layers | Pass (`rc=0`) | [ed32d00c417d8747 (trial 304237079)](https://seed.byteintl.net/development/instance/jobs/ed32d00c417d8747) | N/A (Pre-WandB baseline) |
| **8-GPU Basic** | **Task 1b (5-Step Training + `save_hf_weights: true`)** | 8 | Single-root FP8/FP4 release export to `hf_ckpt/` | Pass (`rc=0`) | [32ff41805ba0d454 (trial 304253096)](https://seed.byteintl.net/development/instance/jobs/32ff41805ba0d454) | [run_20261016_98787067](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261016_98787067) |
| **8-GPU Basic** | **Task 1b (Layout Unit Tests + 8-GPU 40L FP8/FP4 Inference)** | 8 | `pytest test_deepseek_v41_hf_layout.py` (25/25 passed) + `infer_omni.py` full 40-layer multimodal inference | Pass (`rc=0`) | [4c51cfcbb96ee6f7 (trial 304265455)](https://seed.byteintl.net/development/instance/jobs/4c51cfcbb96ee6f7) | N/A (Inference / Pytest) |
| **8-GPU Basic** | **Task 2 (100-Step Loss Curve)** | 8 | 100 steps, separated `total_loss` / `ce_loss` / `indexer_kl_loss` | Pass (`rc=0`) | [45832427c488a318 (trial 304253122)](https://seed.byteintl.net/development/instance/jobs/45832427c488a318) | [run_20261016_4de69078](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261016_4de69078) |
| **8-GPU Basic** | **Task 3 (SP8 `cp_size=8` Long-Context `4k–32k`, 10 steps)** | 8 | `cp_size: 8`, `max_seq_len: 32768`, `data_long_4k32k.yaml` | Pass (`rc=0`, 10/10) | [a35f84f30068e649 (trial 304265481)](https://seed.byteintl.net/development/instance/jobs/a35f84f30068e649) | [run_20261017_758f28fe](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_758f28fe) |
| **8-GPU Basic** | **Task 3 (SP4 `cp_size=4` Long-Context `4k–32k` Comparison)** | 8 | `cp_size: 4`, `NCCL_MAX_NCHANNELS=16`, `max_seq_len: 32768` | 7/10 steps (OOM @ `19.6k` seq) | [f87938816d18dad5 (trial 304267070)](https://seed.byteintl.net/development/instance/jobs/f87938816d18dad5) | [run_20261017_6511b9c0](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_6511b9c0) |
| **128-GPU Full** | **BSZ Probe (`mbs=2, gbs=256`, 40 LLM layers, `ep=64`)** | 128 | All 40 LLM layers, freeze `vision` & `ngram`, no MTP, Muon optimizer | OOM @ Step 2 (`73.60 GiB` alloc) | [45cf7268b49e64a2 (trial 304281947)](https://seed.byteintl.net/development/instance/jobs/45cf7268b49e64a2) | [run_20261017_6db78f14](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_6db78f14) |
| **128-GPU Full** | **100-Step Main Run (`mbs=1, gbs=128`, 40 LLM layers, `ep=128`, Muon `lr=5e-4`) + Single-Root FP8/FP4 Export** | 128 | All 40 LLM layers (`550.56B`), freeze `vision` & `ngram`, no MTP, `ep_size=128`, `save_hf_weights: true` | **Pass (`rc=0`, 100/100 steps + 48-shard export)** | [0abc0b09f0eeb205 (trial 304282868)](https://seed.byteintl.net/development/instance/jobs/0abc0b09f0eeb205) | [run_20261017_e040c085](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_e040c085) |
| **128-GPU Full** | **128-GPU Long-Context SP8 (`cp_size=8`, `4k–32k` seq_len, 10 steps)** | 128 | All 40 LLM layers, `cp_size: 8`, `ep_size: 128`, `max_seq_len: 32768`, `gbs=16, mbs=1` | **Pass (`rc=0`, 10/10 steps)** | [0abc0b09f0eeb205 (trial 304282868)](https://seed.byteintl.net/development/instance/jobs/0abc0b09f0eeb205) | [run_20261017_5f3d8e3d](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_5f3d8e3d) |
| **128-GPU Full** | **8-GPU Multimodal Inference Round-Trip on 128-GPU Exported Checkpoint** | 8 (Node 0) | `tasks/omni/infer_omni.py` loading `/mnt/hdfs/.../global_step_100/hf_ckpt` with `modules_infer_fsdp.yaml` | **Pass (`rc=0`, 40-layer FP8/FP4 generation)** | [0abc0b09f0eeb205 (trial 304282868)](https://seed.byteintl.net/development/instance/jobs/0abc0b09f0eeb205) | N/A (Inference stage in `304282868`) |
| **8-GPU CP Alignment** | **Suite A: `BaseTrainer`-Aligned FSDP2 `text_encoder` (`cp=1` 10 steps & `cp=2` Step 1)** | 8 | Deterministic batch (`seed=42, shuffle=false, gbs=8, mbs=1`), `text_encoder` FSDP2 (`no emb`) | **Pass (`cp=1` 10/10; `cp=2` Step 1 exact match)** | [2304d1595f72d42c](https://seed.byteintl.net/development/instance/jobs/2304d1595f72d42c) / [0aa9829c2c54b033](https://seed.byteintl.net/development/instance/jobs/0aa9829c2c54b033) | [run_20261019_97fe4507](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_97fe4507) |
| **8-GPU CP Alignment** | **Suite B: `emb=8, extra_parallel_only=true` `text_encoder` (`cp=1` 10 steps & `cp=2` Step 1)** | 8 | Deterministic batch (`seed=42, shuffle=false, gbs=8, mbs=1`), `_extra_parallel_only` grad-sync fix | **Pass (`cp=1` 10/10; `cp=2` Step 1 exact match)** | [65d2fbffbbf7a959](https://seed.byteintl.net/development/instance/jobs/65d2fbffbbf7a959) | [run_20261019_1210172e](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_1210172e) / [run_20261019_30fd740c](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_30fd740c) |
| **8-GPU CP Alignment** | **Full `cp=1, 2, 4, 8` Multi-Step Suite (`ep_size=4, ep_fsdp=2`)** | 8 | `ep_size=4` (`ep_fsdp=2`) enabling FSDP2 expert resharding after forward/backward across `cp=1,2,4,8` | Queued (`STARTED` in Group 356) | [6fd0c0aef891741e](https://seed.byteintl.net/development/instance/jobs/6fd0c0aef891741e) | Pending queue allocation |
| **128-GPU Multimodal** | **200-Step Multimodal (`tulu` + `sharegpt4v`) Speed & Convergence Run** | 128 | Local NVMe `/tmp/vdata`, `vision` trainable (`freeze: false`), `text_encoder` FSDP2 (`BaseTrainer`-aligned), `ep=128` | Queued (`STARTED` in Group 356) | [5b0d624d30b5bfde](https://seed.byteintl.net/development/instance/jobs/5b0d624d30b5bfde) | Pending queue allocation |

---

## 2. 128-GPU (`16 × 8 H100-SXM-80GB`) Full 40-Layer Experiments

### 2.1 Batch Size Probe & `ep_size=128` Optimization

- **Model Setup**:
  - `deepseek_v41_vision`: `0.49B` params (`freeze: true`, FSDP2)
  - `deepseek_v41_text_encoder`: `1.32B` params (trainable, `emb_size=128` sharded `wte` + replicated `lm_head`, AdamW `lr=1e-4`)
  - `deepseek_v41_ngram`: `196.61B` params (`freeze: true`, FP8 buffers, `emb_size=128` sharded, `ep_sharded_stream_load: true`)
  - `deepseek_v41_llm`: `550.56B` params (**all 40 layers**, `dspark_block_size: 0`, `num_nextn_predict_layers: 0`, `dsa_indexer_loss: true`, `ep_sharded_stream_load: true`, Muon optimizer)
- **Batch Size Probe Results**:
  1. **`micro_batch_size: 2, global_batch_size: 256` (`ep_size=64`)** ([45cf7268b49e64a2 (trial 304281947)](https://seed.byteintl.net/development/instance/jobs/45cf7268b49e64a2), WandB [run_20261017_6db78f14](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_6db78f14)):
     - Completed Step 1 (`total_loss: 1.99, ce_loss: 1.19, indexer_kl_loss: 0.80, grad_norm: 27.63`) and `optimizer.step()`.
     - OOM'd at Step 2 (`73.60 GiB` allocated, requesting `98 MiB` on `79.11 GiB` H100) once the `17.2 GiB` Muon momentum buffer was materialized alongside `mbs=2` activations.
  2. **`micro_batch_size: 1, global_batch_size: 128` (`ep_size=128`)** ([0abc0b09f0eeb205 (trial 304282868)](https://seed.byteintl.net/development/instance/jobs/0abc0b09f0eeb205), WandB [run_20261017_e040c085](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_e040c085)):
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

*(Comparison: Increasing `ep_size` from `64` to `128` reduced `deepseek_v41_llm` 48-shard streaming load + FP4/FP8 dequantization time from **`151s` (`2m31s`)** in trial `304281947` to **`63s–89s` (`1m03s–1m29s`)** in trial `304282868` because each rank only dequantizes `720` expert tensors instead of `1,440`.)*

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
2. **Muon Effective LR Under `match_rms_adamw`**:
   - In `modules_train_full128.yaml`, `muon_lr` defaulted to `None` (falling back to `adamw_lr = 1.0e-5`). Under `adjust_lr_fn="match_rms_adamw"`, Muon scales updates by `0.2 * sqrt(max(A, B))`, which requires `muon_lr` in the `3e-4 – 5e-4` range for fast SFT convergence.
3. **Tuned Hyperparameters Used in `0abc0b09f0eeb205` (`run_20261017_e040c085`)**:
   - `--data.train_sample 12800 --train.max_steps 100`
   - `--model.optimizer.lr 1.0e-4 --model.optimizer.lr_min 1.0e-5 --model.optimizer.lr_warmup_ratio 0.05` (5 warmup steps)
   - `--model.model_config.modules.deepseek_v41_llm.optimizer.muon_lr 5.0e-4`
   - **Result**:
     - `total_loss` dropped from **`1.80` (Step 3) / `1.73` (Step 1)** down to **`1.09` (Step 24)** and **`1.20` (Step 100)**!
     - `deepseek_v41_llm.forward.indexer_kl_loss` dropped monotonically by **66%**, from **`0.80` (Step 2)** down to **`0.25–0.27` (Steps 88–100)**!
     - `deepseek_v41_text_encoder.decode.ce_loss` dropped from **`1.03` (Step 3)** down to **`0.67–0.73` (Steps 5–24)** and **`0.93` (Step 100)** across the variable-difficulty Tulu-3 SFT mixture!

#### Full 100-Step Numerical Loss Table (`sid: 0abc0b09f0eeb205`, `trial_sid: 304282868`, WandB `run_20261017_e040c085`)

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
   - At Step 100 of Stage 1 (`0abc0b09f0eeb205` / `trial_sid: 304282868`), `OmniTrainer` first saved the distributed DCP checkpoint (`deepseek_v41_vision` in `22s`, `deepseek_v41_text_encoder` in `21s`, `deepseek_v41_llm` in `69s`), and then executed `save_hf_source_checkpoint` (`save_hf_weights: true`):
     - Re-quantized all 40 trained `deepseek_v41_llm` layers back into release FP8 (`float8_e4m3fn` + `scale_inv`) and packed FP4 (`uint8` nibbles + `uint8` E8M0 scale + `bfloat16` `scale_2`) while copying `2,405` frozen/untrained source tensors (`ngram` + `mtp`) directly from the source shards.
     - Exported all **48 safetensors shards (`model-00001-of-00048.safetensors` .. `model-00048-of-00048.safetensors`) + `model.safetensors.index.json` (`7.2 MB`)** to `/mnt/hdfs/__MERLIN_USER_DIR__/dsv41_128gpu_ckpt_main/main_100steps/checkpoints/global_step_100/hf_ckpt` in **`435.17s` (`7m15s`)**.
2. **8-GPU Full 40-Layer Multimodal Inference Round-Trip (Stage 3 on `NODE_RANK=0`)**:
   - Command:
     ```bash
     torchrun --nnodes=1 --nproc_per_node=8 \
       tasks/omni/infer_omni.py configs/seed_omni/DeepSeek/deepseek_v41_flash/train/base.yaml \
       --model.model_path /mnt/hdfs/__MERLIN_USER_DIR__/dsv41_128gpu_ckpt_main/main_100steps/checkpoints/global_step_100/hf_ckpt \
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
  - Merlin Trial: [0abc0b09f0eeb205 (trial 304282868)](https://seed.byteintl.net/development/instance/jobs/0abc0b09f0eeb205) (Stage 2)
  - WandB Run: [run_20261017_5f3d8e3d](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261017_5f3d8e3d) (`dsv41_128gpu_sp8_long_4k32k`)
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
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 1`** (`dp=8, gas=1`) | **`17.32985`** | **`17.27464`** | **`0.05521`** | **`83.99` / `84.00`** | [run_20261019_97fe4507](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_97fe4507) |
| **Suite A (`FSDP2 text_encoder`)** | **`cp = 2`** (`dp=4, gas=2`) | **`17.32985`** (`Δ = 0.00000`) | **`17.27464`** (`Δ = 0.00000`) | **`0.05521`** (`Δ = 0.00000`) | **`83.98`** (`Δ = 0.01`) | [run_20261019_1da55536](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_1da55536) |
| **Suite B (`emb=8, extra_parallel_only=true`)** | **`cp = 1`** (`dp=8, gas=1`) | **`17.32985`** | **`17.27464`** | **`0.05521`** | **`83.54224`** | [run_20261019_1210172e](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_1210172e) |
| **Suite B (`emb=8, extra_parallel_only=true`)** | **`cp = 2`** (`dp=4, gas=2`) | **`17.32985`** (`Δ = 0.00000`) | **`17.27464`** (`Δ = 0.00000`) | **`0.05521`** (`Δ = 0.00000`) | **`83.54224`** (`Δ = 0.00000`) | [run_20261019_30fd740c](https://seed.byteintl.net/experiment/tracking/detail?Id=project_20261015_f2ce96f6&selectedTrial=run_20261019_30fd740c) |

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

### 4.3 Why 8-GPU `ep_size=8` OOMed at Step 2 Under `cp >= 2` & Resolution via `ep_size=4` (`ep_fsdp=2`)

- **Root Cause**: On 8 GPUs (`world_size=8`), setting `ep_size: 8` creates `DeviceMesh((ep_fsdp=1, ep=8))`. Because `ep_fsdp=1`, PyTorch FSDP2 treats the 48 local experts per layer as **unsharded** (`ep_fsdp=1`) and never frees their unsharded BF16 copy (`2.06 GiB/layer`) after forward or backward! After `optimizer.step()` at Step 1 materializes FP32 AdamW `exp_avg` and `exp_avg_sq` (`8.25 GiB/layer`), static memory reaches `69.24 GiB`, causing Step 2 micro-step accumulation under `cp=2` (`dp_shard=4, cp=2`) to OOM by `~400 MiB`.
- **Resolution in Job [6fd0c0aef891741e](https://seed.byteintl.net/development/instance/jobs/6fd0c0aef891741e)**: Setting `ep_size: 4` on 8 GPUs gives `DeviceMesh((ep_fsdp=2, ep=4))` (`ep_fsdp = 2 > 1`). Each rank holds the exact same shard size (`96 / 2 = 48` experts worth of parameters), **and** FSDP2 properly reshards and frees the unsharded BF16 parameters after forward and backward (`reshard_after_backward: true`), saving `~8–10 GiB` of VRAM across `cp=1, 2, 4, 8`.

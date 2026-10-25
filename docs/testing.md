# VeOmni Test Suite Overview

This document surveys all tests in the VeOmni project, describes their purpose and organization,
and provides guidance on which tests to add when onboarding a new model.

---

## Directory Structure

```
tests/
├── ci/                             # CPU tests for accelerator CI tooling
├── tools/                          # Shared test infrastructure (comparison, data gen, launch)
├── toy_config/                     # Minimal model configs for fast CI testing
├── testdata/                       # Sample images, audio, etc.
│
├── models/                         # Registry-backed model correctness
│   ├── tiny_configs.py             # Canonical tiny configs shared by model suites
│   ├── compare.py                  # Upstream-vs-VeOmni parity helpers
│   ├── base/                       # Registry/build, loading, loss-policy tests
│   ├── transformers/               # Transformer-family parity and contracts
│   │   ├── deepseek_v4/            # DeepSeek-V4-specific model coverage
│   │   └── qwen/                   # Qwen-family model coverage
│   └── diffusers/                  # Diffuser-family parity and contracts
│
├── ops/                            # Registry and tensor-op correctness
│   ├── base/test_op_entry.py                # Registration, resolution, and generated autograd
│   ├── attention/                           # Eager/SDPA/Flash/Flex/Magi/Sage contracts
│   ├── async_ulysses/                       # Async QKV/O registry and parity
│   ├── dsa/                                 # DeepSeek/GLM sparse-attention ops
│   ├── loss/                                # Cross-entropy and load-balancing ops
│   ├── mhc/                                 # mHC eager/TileKernels/CANN parity
│   ├── moe_experts/                         # Eager/Triton/Quack/NPU/MLU expert ops
│   └── gated_delta_rule/                    # GatedDeltaNet op family
│
├── data/                           # Data loading & preprocessing
│   ├── test_datasets.py            # Dataset loading, filtering, schema validation
│   ├── test_collators.py           # MainCollator, cu_seq_lens generation
│   ├── test_dataloader.py          # DataLoader batching
│   ├── test_dpo_data_processor.py  # DPO data processing
│   ├── test_dynamic_batching_dataset.py  # Dynamic batching by seq length
│   ├── test_prepare_fa_kwargs.py   # Flash-attn parameter construction
│   ├── test_preprocessor.py        # Token mapping, special tokens
│   ├── test_classification_data_processor.py  # Classification data processing
│   └── multimodal/
│       ├── test_vlm_data_process.py   # VLM data pipeline (HF processor vs VeOmni)
│       └── test_video_utils.py        # Video/audio loading & frame extraction
│
├── parallel/                       # Parallelism primitives
│   ├── ulysses/                    # Sequence parallelism (Ulysses)
│   │   ├── test_ulysses.py             # Basic SP attention (4+ GPUs)
│   │   ├── test_deepseek_v4_ulysses.py # DeepSeek-V4 SP
│   │   ├── test_qwen3_5_gated_deltanet_ulysses.py  # Gated DeltaNet + SP
│   │   ├── test_wan_self_attn_ulysses.py           # Wan self-attention SP
│   │   ├── test_wan_self_attn_padding_mask.py      # Wan SP padding mask
│   │   ├── test_wan_ulysses_padding.py             # Wan SP padding
│   │   ├── test_slice_input_tensor.py  # Input slicing utilities
│   │   ├── test_all_gather.py          # All-gather collective ops
│   │   └── utils.py                    # SequenceParallelTest base class
│   └── encoder_data_balance/
│       ├── test_balance_reverse.py        # Balance/recovery precision (8 GPUs)
│       └── test_balance_sorting_algo.py   # Post-MBS data sorting (CPU)
│
├── distributed/                    # Distributed training and runtime correctness
│   ├── test_fsdp_equivalence.py         # Single-GPU vs FSDP2 grad equivalence
│   ├── test_gradient_checkpointing.py   # Checkpoint kwargs and recomputed-input grad cleanup
│   ├── test_emb_parallel.py             # ShardedEmbedding lookup / tied projection vs dense, through FSDP2 (CPU gloo)
│   ├── test_parallel_plan.py            # ParallelPlan: every planned parameter's owner is wrapped on its para mesh
│   └── test_dummy_forward.py            # Asymmetric multimodal forward (NCCL hang prevention)
│
├── seed_omni/                      # SeedOmni multi-module models, runtime and trainer
│   ├── model/                           # OmniModel graphs, conversation, processor, save/load, conversion
│   ├── media/                           # Request media and video metadata
│   ├── data/                            # SeedOmni transform, preprocessors, audio ingestion
│   ├── arguments/                       # Launcher parser, config merge, module / accelerator / offline-cache args
│   ├── runtime/                         # Module and OmniModel runtimes, graph runtime, LoRA composition
│   ├── trainer/                         # OmniTrainer, inferencer, step contexts, step-metrics callback
│   ├── mixins/                          # Opt-in module mixins (metric meter, offline encoding + cache writer)
│   ├── modules/                         # Per-family module tests: base/, janus/, qwen3/, qwen3vl/, bagel/
│   └── e2e/                             # torchrun launch of the fake model (2 CUDA devices)
│
├── e2e/                            # End-to-end training integration
│   ├── test_e2e_parallel.py             # SP/EP parallel alignment across models
│   ├── test_e2e_training.py             # Real-model SFT smoke test (8 GPUs)
│   ├── test_e2e_training_no_reshard.py  # FSDP2 no-reshard mode
│   ├── exec_scripts.py                  # Shell command generators for real models
│   └── utils.py                         # prepare_exec_cmd, parse_training_log, ParallelMode
│
├── train_scripts/                  # Standalone trainer scripts (invoked via torchrun, not pytest)
│   ├── train_text_test.py               # Test trainer for text models
│   ├── train_vlm_test.py                # Test trainer for VLM models
│   └── train_dit_test.py                # Test trainer for DiT models
│
├── checkpoints/                    # Checkpoint save/load
│   ├── test_checkpoint_callback.py          # CheckpointCallback cadence + manager contract
│   ├── test_trainer_saveload.py             # DCP + HF checkpoint save/load (8 GPUs)
│   ├── checkpoint_verification_utils.py     # DCP-to-HF conversion verification
│   └── utils.py                             # Command/config builders for ckpt tests
│
├── utils/                          # Misc utility tests
│   ├── test_count_flops.py                       # FLOPs estimation
│   ├── test_ep_sharded_expert_stream_load.py     # ep_sharded_stream_load per-rank per-expert stacking (CPU)
│   ├── test_extra_parallel_clip_grad_norm.py      # Grad clipping with EP/EMB dims (8 GPUs)
│   ├── test_helper.py                             # EnvironMeter utility (8 GPUs)
│   ├── test_model_loader.py                       # Model loading (4 GPUs)
│   ├── test_npu_setup.py                          # NPU environment validation
│   ├── test_rank0_load_and_broadcast_weights.py   # Rank-0 load & broadcast (2+ GPUs)
│   └── test_save_safetensor_utils.py              # Safetensor save utilities (CPU)
│
└── special_sanity/
    ├── check_device_api_usage.py    # CI lint: no direct .cuda / "cuda" calls
    └── test_ops_architecture.py     # CI: stale kernels/models_kernel path guards
```

---

## Test Categories at a Glance

| Category | Directory | GPU Req | Execution | Purpose |
|---|---|---|---|---|
| **Models** | `tests/models/` | 0-4 GPUs | pytest | Registry/build contracts, model parity, and implementation selection |
| **Ops** | `tests/ops/` | 0-4 GPUs (SM90+ for Quack; SM90+ or gfx942/gfx950 for DeepSeek-V4 TileLang and mHC TileKernels) | pytest | Registry contracts, hardware guards, numerical correctness, and performance |
| **Data pipeline** | `tests/data/` | 0-8 GPUs | pytest + torchrun | Data loading, collation, preprocessing, and distributed resume |
| **Parallelism** | `tests/parallel/` | 0-8 GPUs | torchrun / pytest | SP, EP, data-balance primitives |
| **Distributed runtime** | `tests/distributed/` | 0-4 GPUs | pytest + torchrun | FSDP equivalence, dummy forward, gradient checkpointing, vocab-parallel embedding |
| **SeedOmni** | `tests/seed_omni/` | 0; 2 for `e2e/` | pytest (+ torchrun for `e2e/`) | Multi-module OmniModel, runtimes, trainer, launcher, module mixins |
| **E2E parallel** | `tests/e2e/` | Per-case launch plan (up to 8 GPUs) | torchrun (subprocess) | SP/EP alignment across full training runs |
| **Checkpoints** | `tests/checkpoints/` | 0-8 GPUs | pytest + torchrun | Save/load, DCP→HF conversion |
| **Utilities** | `tests/utils/` | 0-8 GPUs | pytest + torchrun | FLOPs, grad clipping, weight broadcast |
| **Sanity** | `tests/special_sanity/` | 0 | script + pytest | Device API lint and op-architecture path guards |

Unit CI does not run `pytest tests/`. Suite selection is defined in the
[GPU manifest](../scripts/ci/accelerator/suites/gpu.json) and
[NPU manifest](../scripts/ci/accelerator/suites/npu.json), consumed by the
[accelerator runner](../scripts/ci/accelerator/runner.py). Directory selections
collect their children automatically; a file outside them must be added to its
suite.

The [case scheduling contract](design/ci_case_scheduling.md) defines planning
from the actual `nproc/world_size` configuration, device isolation and GPU/NPU
scheduling. The default launch plan uses one card of the selected accelerator
backend on GPU/NPU runners, and CPU on CPU-only runners. Multi-card tests must
expose their launch configuration; `skipif` describes eligibility, not a
resource reservation. See [runner usage](../scripts/ci/accelerator/README.md) for commands
and result artifacts. GPU/NPU unit and e2e workflows all use parallel dispatch
with the same launch plans and scheduler core.

Batches are ordered by required card count, largest first, and greedily assigned
to free devices with smaller jobs filling remaining space. Fixed case weights
(60 seconds for multi-card work, 2 seconds for single-card/CPU work) control
batch size only. Measured timings are reporting artifacts; scheduling does not
require historical timings.

---

## Shared Test Infrastructure (`tests/tools/`)

All shared, cross-cutting utilities live in `tests/tools/`:

| Module | Exports | Description |
|---|---|---|
| `comparison_utils` | `TensorComparator`, `assert_close`, `assert_exact`, `compare_metrics`, `print_comparison_table` | Numerical comparison with tolerances; rich table output |
| `data_generators` | `DummyDataset` | Generates parquet dummy datasets for all modalities (text, VLM, omni, DiT) |
| `launch_utils` | `find_free_port`, `torchrun` | Port discovery; `mp.spawn`-based distributed launcher |
| `training_utils` | `ParallelConfig`, `build_torchrun_cmd`, `materialize_weights`, `run_training_config`, `release_device_memory` | Torchrun command builder, model weight materialization, training runner |

Additional per-directory helpers:

| File | Scope | Key Exports |
|---|---|---|
| `tests/models/compare.py` | Model parity tests | Eager ops config and comparison helpers |
| `tests/models/tiny_configs.py` | Registry and model tests | Canonical tiny-config factories shared across suites |
| `tests/models/base/test_checkpoint_tensor_converter.py` | Model loading | Runtime checkpoint layout conversion and fused-expert weight mapping |
| `tests/e2e/utils.py` | E2E tests | `prepare_exec_cmd`, `parse_training_log`, `ParallelMode` |
| `tests/checkpoints/utils.py` | Checkpoint tests | Command/config builders for trainer save/load |
| `tests/parallel/ulysses/utils.py` | SP tests | `SequenceParallelTest` base class, `sync_tensor` |

---

## Detailed Test Descriptions

### 1. Model Tests (`tests/models/`)

**Purpose**: Verify registry/build behavior, eager forward/backward parity with
the upstream reference, instance-local operator binding, and model-specific
contracts beside the maintained implementation.

**What it covers**:

| Dimension | Values |
|---|---|
| Registry | Model type, supported architectures, aliases, prerequisites |
| Eager parity | HuggingFace vs VeOmni logits/loss and gradients; bitwise logits where the family is bit-identical |
| Low-precision oracle | BF16 FA2/SDPA vs independent HF, plus `weights_path` loader for dense, merged MoE, VL, Omni |
| Operator binding | Selectable eager and optimized implementations without fixing a YAML choice |
| Model contracts | Family-specific routing, masking, multimodal, or checkpoint behavior |

**Models covered**:
- Text / MoE: llama3_1, qwen2, qwen3_5, qwen3_5_moe, seed_oss, deepseek_v3, deepseek_v4, glm_moe_dsa (`eager` attention only)
- VLM: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe
- Omni: qwen2_5_omni, qwen3_omni_moe

Most eager and registry coverage runs on CPU. Tests for optimized operators or
accelerator-only paths declare their own hardware prerequisites.

Registry/build coverage is centralized in
`tests/models/base/test_auto_registry.py`; canonical tiny configs live
in `tests/models/tiny_configs.py`; family-specific parity and contracts
live under `tests/models/transformers/` and
`tests/models/diffusers/`. BF16 FA2/SDPA oracles and disk-backed
`weights_path` loading live in `tests/models/base/test_low_precision_oracle.py`.
Optimized operator numerics live under
`tests/ops/`.

DeepSeek-V4- and GLM-MoE-DSA-specific DSA checks live under `tests/ops/dsa/`; optimized numerical
tests require TileLang on an SM90+ NVIDIA GPU (DeepSeek-V4's also run on AMD gfx942/gfx950), and
the `cann` parity tests require an Ascend NPU with `cann_ops_transformer`. Its mHC kernel parity is
covered by `tests/ops/mhc/test_mhc.py` and requires TileKernels on an SM90+ NVIDIA GPU or an AMD
gfx942/gfx950 GPU, or CANN on an Ascend NPU.

DeepSeek-V4 expert-parallel training is exercised separately by
`tests/e2e/test_e2e_parallel.py::test_deepseek_v4_expert_parallel_alignment`:
four-rank FSDP2, SP=1, EP=1 versus EP=2, and two optimizer steps. This uses
eager DSA/mHC with fused routed experts, so it does not qualify the TileLang
kernel paths. The CPU duplicate-hash-route regression verifies that EP split
counts match unique token/expert dispatch pairs; repeated routing weights are
summed when combining outputs.
The GPU CI also runs `test_deepseek_v4_expert_parallel_fixed_route_vjp_matches_fp32_reference`
in `tests/models/transformers/deepseek_v4/test_deepseek_v4.py`: four-rank EP=4, unequal token
payloads, ordinary and duplicate expert slots, and three held-out seeds. Output,
input/routing gradients and expert-weight gradients are compared with a global
FP32 mathematical VJP using the same BF16 bit patterns and the rounding budget
described below. This isolates expert transport/kernel numerics, not FSDP2
gradient averaging or whole-model optimizer trajectories.

Controlled two-step BF16 FSDP2 runs matched initial FP32 master weights,
buffers, optimizer settings and every rank's micro-batches. Aggregate execution
gates passed, but whole-model pre-clipping gradient relative L2 differed by
about 4.61% for EP2 versus EP1 before aligning routing-weight placement. Non-EP
Triton experts now weight the down-projection output, matching EP and the eager
reference, instead of weighting the BF16 activation before that projection.
Both split/merged weights and shared/independent LoRA use this order; for LoRA,
the routing weight scales the complete base-plus-adapter output.

On the same four-rank fixture, this reduced first-step whole-model gradient
relative L2 to 0.0751%, with bit-identical non-expert gradients. Expert weight
gradients still differ across EP reduction layouts. First-step update relative
L2 was 2.62%; after that update, second-step gradient relative L2 was 4.24% and
update relative L2 was 5.21%. These measurements do not establish whole-model
update parity or convergence. First-step AdamW replay and clipping passed their
separate oracles in the earlier diagnostic runs.

`tests/ops/moe_experts/test_moe_experts_ep.py` checks output and all VJPs against
the single-expert EP projection order without relaxing split/merged parity.
`tests/ops/moe_experts_lora/test_moe_experts_lora.py` checks that zero-B adapters preserve base
output, input/routing and base-weight gradients, while retaining trainable
B-gradients. Both files run in GPU CI. Duplicate routes still have distinct
EP/non-EP reduction layouts; no bitwise guarantee is made for arbitrary routes.

`tests/e2e/test_e2e_parallel.py::test_deepseek_v4_compiled_fsdp2_training_alignment` compares
eager and fullgraph Inductor through four-rank packed FSDP2 training, with
static 512-token shapes and EP/SP disabled. Loss and gradient norm are checked
at `rtol=atol=1e-3`; this checks aggregate training metrics, not per-parameter
update alignment, convergence or bitwise equivalence.
It keeps DSA/mHC/norm/shared MLP/loss eager and routed experts on `fused_triton`. Enabled
ExtraParallel with compilation remains rejected. Single-device decoder trace
and gradient checks live in `tests/distributed/test_torch_compile.py`.
Backend `eager` capture compares two identically loaded BF16 models: loss within
`rtol=atol=1e-4` and every parameter gradient within `rtol=atol=1e-3`. The
Inductor execution gate runs two batches through the compiled decoder blocks and
requires a finite loss, finite gradients and an AdamW step for each; it does not
compare against eager BF16, which is not an exact numerical oracle. A separate
fixed-route expert test feeds random hidden states, fixed top-k routes and a
random cotangent to the Inductor-compiled and the eager fused experts, and
compares output, input/routing gradients and both expert-weight gradients against
an FP32 copy of the same experts on the eager `moe_experts` path. Outputs must be
within `rtol=1e-2, atol=2e-2` of the reference; each gradient must have cosine
similarity above 0.999 and be within `rtol=2e-2, atol=3e-2`; compiled and eager
outputs and gradients must agree within `rtol=1e-2, atol=2e-2`. These local gates do not qualify
whole-model numerical training correctness or the SM90+ optimized kernels.

Controlled two-step BF16 FSDP2 runs matched initial FP32 master weights,
buffers, optimizer settings and every rank's micro-batches. Aggregate execution
gates passed, but whole-model pre-clipping gradient relative L2 differed by
about 5.81% for Inductor versus eager. First-step AdamW replay and clipping
passed their scoped oracles; whole-model update parity and convergence remain
unqualified. See the separate EP validation PR for expert transport validation.

`test_deepseek_v4_compiled_decoder_gradients_track_the_fp32_oracle` checks that
compiled BF16 gradients track the direction of an FP32 reference about as
closely as eager BF16 gradients do. From one fixed set of BF16-rounded weights
(with hash-routing tables filled) and the same input, it runs the eager and the
fullgraph Inductor BF16 models, both in batch-invariant mode, and an eager FP32
forward/backward over those same values. Losses must be within
`rtol=5e-3, atol=1e-2` of the FP32 reference and within `rtol=1e-3, atol=2e-3`
of each other, and all three must give gradients to the same parameters.
Gradients are compared by direction only, excluding `indexer.weights_proj`: the
cosine between all gradients and the reference must exceed 0.995 for each mode
and the two modes' cosines must agree within 2e-3; per parameter whose
reference gradient norm is at least 1e-4, the Inductor cosine must exceed 0.97
and trail eager by at most 5e-3. Cosine similarity is scale-invariant, so this
test does not bound gradient magnitude or relative L2 error; only its loss
checks are magnitude-sensitive. The reference runs the eager MoE implementation
rather than the Triton grouped GEMM both BF16 modes use, so its distance also
contains that implementation difference.

Measurements taken for this document, none of which a test asserts: an earlier
revision of the decoder harness (one 64-token sequence, filled hash-routing
tables, no batch-invariant mode) read whole-model gradient relative L2 to the
FP32 reference of 1.70% (eager) and 1.48% (Inductor), with the two BF16 modes
2.08% apart. These were not re-measured with the current harness. In the
four-rank run at step 1, eager and Inductor pre-clipping gradient norms agree to
1.5e-4 relative and the cosine between the two gradients is 0.9985.
`TORCHINDUCTOR_EMULATE_PRECISION_CASTS=1` does not close the four-rank gap:
5.84% and 6.02% at steps 1 and 2, against 5.48% and 5.79% by default.

---

### 2. VLM Trainer Test (`tests/trainer/test_vlm_trainer.py`)

**Purpose**: Smoke test that `freeze_vit=True/False` correctly freezes/unfreezes the vision tower.

**Models**: qwen2_vl, qwen3_5, qwen3_5_moe, qwen2_5_vl, qwen3_vl, qwen3_vl_moe

**GPU**: CPU only (builds model but no forward pass).

---

### 3. Model Registry Test (`tests/models/base/test_auto_registry.py`)

**Purpose**: Verify that `get_model_config/class/processor` returns the correct HF or VeOmni module.

**GPU**: CPU only.

---

### 4. Checkpoint Tensor Converter (`tests/models/base/test_checkpoint_tensor_converter.py`)

**Purpose**: Test checkpoint tensor conversion protocol (e.g., Qwen3MoE expert weight fusion: per-expert → stacked `gate_up_proj`).

**GPU**: CPU only.

---

### 5. Padded vs Packed Loss (`tests/models/transformers/qwen/test_qwen3.py`)

**Purpose**: Verify that padded input and packed input (with `cu_seqlens`) produce identical loss.

**GPU**: 1 GPU (requires flash-attn).

---

### 6. FSDP Equivalence (`tests/distributed/test_fsdp_equivalence.py`)

**Purpose**: Verify that FSDP2 sharding produces the same grad_norm as single-GPU training (no parallelism). This catches FSDP wrapping bugs that silently corrupt gradients.

**How it works**:
1. Materialize random weights from toy config
2. Run single-GPU training (nproc=1, no FSDP)
3. Run FSDP2 training (nproc=2+, init_device=meta)
4. Assert grad_norm matches (loss may differ due to micro-batch splitting)

**Models**: qwen3, qwen3_moe, llama3.1, qwen3_5, qwen3_5_moe

**GPU**: 2+ GPUs.

---

### 7. Dummy Forward (`tests/distributed/test_dummy_forward.py`)

**Purpose**: Verify that asymmetric multimodal batches (some ranks text-only, others with images/video/audio) don't cause NCCL hangs under FSDP2. Tests that `dummy_forward()` is correctly invoked so all ranks participate in FSDP collectives.

**Models**:
- VLM: qwen2_5_vl, qwen3_vl, qwen3_vl_moe
- Omni: qwen2_5_omni, qwen3_omni_moe

**GPU**: 2 GPUs.

---

### 8. E2E Parallel Alignment (`tests/e2e/test_e2e_parallel.py`)

**Purpose**: Short FSDP2 training runs compare loss and grad norm across the
selected SP/EP configurations. These aggregate checks do not establish
per-parameter gradient/update equivalence or convergence. Cases with
`compare_alignment=False` provide smoke coverage only.

**Configurations tested**:

| Case | `(SP, EP)` modes | Processes per run |
|---|---|---|
| Ordinary dense | `(1,1)`, `(2,1)` | 4 |
| Ordinary MoE | `(1,1)`, `(2,2)` | 4 |
| Qwen3-MoE representative | `(1,1)`, `(1,2)`, `(2,1)`, `(2,2)` | 4 at SP1, 8 at SP2 |
| GPT-OSS representative | `(1,1)`, `(2,1)`, `(2,2)` | 4 |

Modes run sequentially within a case, with two optimizer steps per run.
The callbacks use `dp_mean` for losses because fixed-world-size SP changes
the DP size. Capability-restricted cases use supported modes: GLM-MoE-DSA
disables SP, while DeepSeek-V4 has dedicated EP alignment and separate SP2
smokes. GPT-OSS FA4/Quack requires SM90+ CUDA and skips on SM89 GPUs and NPU.
See the [new-model coverage policy](transformers_v5/testing_new_model.md)
for matrix selection, representative coverage, and exception requirements.

**Models**: The text, VLM, omni, and DiT parameter tables in
`tests/e2e/test_e2e_parallel.py` define coverage.

**Devices**: Follow the case's launch plan; ordinary model alignment uses four,
with eight retained for the Qwen3-MoE SP2 topology.

---

### 9. E2E Training Smoke Tests (`tests/e2e/test_e2e_training*.py`)

**Purpose**: Smoke tests with real model weights (qwen3_0p6b_base + Tulu-3 SFT dataset). Validates that training completes without errors.

- `test_e2e_training.py` — standard FSDP2 training (8 GPUs)
- `test_e2e_training_no_reshard.py` — FSDP2 no-reshard mode (8 GPUs)

---

### 10. Checkpoint Save/Load (`tests/checkpoints/`)

| Test | Purpose | GPU |
|---|---|---|
| `test_checkpoint_callback.py` | `_last_saved_step` state tracking | CPU |
| `test_trainer_saveload.py` | DCP + HF checkpoint formats, resume training | 8 GPUs |

---

### 11. Op Tests (`tests/ops/`)

| Test | Purpose | GPU |
|---|---|---|
| `base/test_op_entry.py` | Registration, requirements, resolution, and generated autograd | CPU |
| `async_ulysses/` | Async QKV/O registry rows plus dense and DiT parity | CPU for registry; multi-GPU for parity |
| `moe_experts/test_moe_experts.py` | Eager/Triton/Quack MoE parity, split/merged weights, and hardware guards | CPU for guards; CUDA for optimized kernels |
| `dsa/test_dsa*.py` | DSA registry, CPU guards, and TileLang/cuDNN numerical parity | CPU for guards; matching CUDA hardware for optimized kernels |
| `attention/flash/test_flash_attention.py` | FlashAttention contracts | CUDA |
| `attention/magi/` | Magi mask, SM100+ guard, FA4 metadata, and numerical contracts | CPU for guards; SM100+ for optimized kernels |
| `batch_invariant/test_batch_invariant.py` | Batch-invariant ATen patch lifecycle and math | CPU for lifecycle; CUDA for Triton kernels |

Cross-entropy is covered at two layers: `tests/ops/loss/test_cross_entropy_loss.py`
checks token-level eager parity with HF plus chunked/Liger forward and backward;
`tests/models/base/test_loss_utils.py` checks causal target selection,
sequence-classification policy, SP reduction, logits ownership, and log-probs
side-path dispatch. Chunked log-probs and top-k distillation have focused tests
in `tests/models/`.

Load-balancing loss is covered at two layers: `tests/ops/loss/test_load_balancing_loss.py`
checks the raw `[N, E]` eager and Triton kernels against HF/eager across the
configuration matrix, forward/backward, masks, determinism, and peak memory;
`tests/models/base/test_model_load_balancing_loss.py` checks tuple
concatenation, optional-input policy, and gradient fan-out in the model helper.

---

### 12. Parallelism Primitive Tests (`tests/parallel/`)

| Test | Purpose | GPU |
|---|---|---|
| `test_ulysses.py` | Basic Ulysses SP attention | 4+ |
| `test_deepseek_v4_ulysses.py` | DeepSeek-V4 SP | 4+ |
| `test_qwen3_5_gated_deltanet_ulysses.py` | Gated DeltaNet + SP | 4+ |
| `test_wan_self_attn_ulysses.py` | Wan self-attention SP | 4+ |
| `test_slice_input_tensor.py` | SP input slicing utilities | CPU |
| `test_all_gather.py` | All-gather collective ops | multi |
| `test_balance_reverse.py` | Encoder data balance recovery | 8 |
| `test_balance_sorting_algo.py` | Post-MBS sorting algorithm | CPU |

---

### 13. SeedOmni Tests (`tests/seed_omni/`)

Both unit-test workflows run the whole tree recursively (`pytest tests/seed_omni`),
so a new file under it needs no workflow edit. Everything except `e2e/` runs on
CPU, and multi-rank cases spawn CPU gloo ranks with `torch.multiprocessing`;
`e2e/` needs two CUDA devices and is skipped otherwise.

| Subdirectory | Purpose |
|---|---|
| `model/` | Training / generation graphs, conversation items, processor, save/load round-trips, checkpoint conversion |
| `media/` | Request media loading and video metadata |
| `data/` | SeedOmni data transform (`_img_tag`), CPU preprocessors, audio ingestion |
| `arguments/` | Launcher parser, config deep-merge, per-module and accelerator arguments, module path resolution, offline-cache arguments |
| `runtime/` | `ModuleRuntime` / `OmniModelRuntime`: build, frozen modules, per-module grad clipping and weight load, graph execution, LoRA composition |
| `trainer/` | `OmniTrainer`, `OmniInferencer`, step contexts, step-metrics callback (identical metrics across ranks) |
| `mixins/` | Opt-in module mixins: training-graph hooks, metric meter, offline encoding and the cache writer |
| `modules/` | One folder per module family (`base/`, `janus/`, `qwen3/`, `qwen3vl/`, `bagel/`): chat templates, packing, module smoke tests; shared fixtures in `modules/helpers.py` |
| `e2e/` | `train_omni.py` / `infer_omni.py` under torchrun on the fake model (2 CUDA devices) |

---

## Registering Tests in CI

Test selection is maintained in the
[GPU suite manifest](../scripts/ci/accelerator/suites/gpu.json) and
[NPU suite manifest](../scripts/ci/accelerator/suites/npu.json). Each manifest
contains both `unit` and `e2e` selections. The runner collects those selections
and schedules the resulting cases using their launch plans.

Workflows configure runners, dependencies, triggers, job-wide environment and
runner options. Keep test selections in the manifests so local runs and CI use
the same collection source. Test-specific environment variables belong in a
selection's optional `env` object. Device requirements belong in the test's
launch plan.

### When a manifest needs an update

| Test change | Required registration |
|---|---|
| Add a function or parameter combination included by an existing selection | No manifest or workflow change |
| Add a file inside a directory already collected by the relevant manifest | No manifest or workflow change |
| Add a file outside existing collection scopes | Add a selection to each applicable backend manifest |
| Add a new CI machine requirement, dependency setup or workflow trigger | Update the corresponding workflow as well |

Check the complete selection arguments: a node ID or a `-k` filter can narrow
collection within a file or directory. The existence of a manifest entry for
that file does not guarantee that every new case is included. A launch-plan
marker declares resources; it does not register a test with CI.

For example, if a new `tests/e2e/test_my_model.py` file is outside the existing
scopes, append an entry like this to `suites/gpu.json` (replace the illustrative
file path with the actual new file):

```json
{
  "name": "my-model-e2e",
  "suite": "e2e",
  "conditional": false,
  "args": ["-s", "-x", "tests/e2e/test_my_model.py"]
}
```

Use a unique `name` within each backend manifest. Set `suite` to `unit` or
`e2e`; keep ordinary selections unconditional. Conditional selections require
an explicit workflow gate and are enabled with `--include-conditional`.
Preserve any required pytest flags or filters in `args`. Do not add a per-file
pytest step to the workflow or a second selection for an already covered
scope. Add the corresponding NPU selection when the test is intended to run
on NPU, retaining its hardware/dependency guards.

A function or parameter added to `tests/e2e/test_e2e_parallel.py` is collected
by that file's existing GPU e2e selection; it does not need another JSON entry.
For a new multi-card case, also expose the shared `nproc/world_size`
configuration as a [launch plan](#launch-plans-for-new-tests).

On a runner with the selected backend, check collection before execution:

```sh
uv run --frozen python -m scripts.ci.accelerator --backend cuda --suite e2e --only my-model-e2e --collect-only --output-dir outputs/accelerator-ci/my-model-collection
```

Use the selection name with `--only`. Inspect `inventory.json` for the expected
parameterized node IDs, eligibility and resolved plans. Use a fresh output
directory for each invocation. See the
[accelerator README](../scripts/ci/accelerator/README.md) for the full runner
interface and result files.

---

## Launch Plans for New Tests

A test without an explicit launch plan uses one card of the runner's selected
accelerator backend, or CPU on a CPU-only runner. Single-card tests may use this
default. Declare a CPU plan for CPU-only work, including preprocessing, config
validation, eager CPU references, and tests with mocked accelerator kernels.
A backend name in a test function does not imply that it needs that hardware.
The scheduler follows the launch plan; it does not infer device use from tensor
factories or test names.

```python
import pytest

from scripts.ci.accelerator.launch_plan import LaunchSpec


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_cpu_reference():
    ...
```

For an entirely CPU-only module, use
`pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))`.
In mixed modules, mark individual CPU cases so optimized GPU/NPU kernel tests
retain accelerator reservations. CPU workers hide accelerator devices and share
the scheduler's separate CPU concurrency limit.

CPU numerical checks should state their input dtype and device. An independent
FP64 math reference can check an FP32 candidate without relying on automatic
SDPA backend selection. Keep output and gradient tolerances explicit; changing
a reference does not justify loosening the candidate's error bounds.
For intermittent CPU numerical failures, reproduce in fresh processes as well
as repeated calls: a first-call math error can disappear after warmup. Include
the failing output/gradient name, seed, shapes, dtype, CPU and thread settings
in assertion diagnostics. The QSA CPU mock evaluates exponentials in FP64 and
casts back to FP32 to avoid a reproduced first-call CPU FP32 `exp` accuracy
issue; its matrix products, normalization and gradient accumulation stay FP32.

Multi-card tests must expose a plan derived from the same `nproc`, `world_size`,
or parallel configuration used to launch their workers. A module, class, or
fixture family may provide a shared plan. Parameterized tests resolve their
plan for each parameter combination; they do not need entries in a device-quota
table. A `skipif` marker alone does not declare a launch plan.

Keep launch configuration independent of the number of cards currently free.
An undersized plan is an error; it must not cause a smaller topology, a backend
fallback, or a resource-induced skip. Use isolated output/cache paths and
allocated rendezvous ports for concurrent cases. See the
[test-author contract](design/ci_case_scheduling.md#writing-new-tests) for details.

---

The accelerator runner fails fast by default: the first failure stops dispatch
and cancels running peers in the same suite/job, then waits for scoped process
cleanup. Foreground output shows case progress, a 30-second heartbeat, failure
tracebacks/log tails and a final summary. Use `--keep-going` to collect all
failures. See [failure reporting](../scripts/ci/accelerator/README.md#results-and-local-commands)
for full logs and partial-result artifacts.

## New Model Onboarding: Test Checklist

Local tests and CI use the same canonical configs under `tests/toy_config/`.
Reuse the [shared toy fixtures](../tests/toy_config/README.md) and preserve their
model/input contracts; do not create CI-specific copies or switch configs based
on the execution environment.

When adding a new model to VeOmni, the following tests should be created or updated.
See also: [Testing a New Model for Transformers v5](transformers_v5/testing_new_model.md) for step-by-step instructions.

### Required Tests

| Step | Test File | What to Do |
|---|---|---|
| 1. **Create toy config** | `tests/toy_config/<model>_toy/` | Minimal config (few layers, small dims). Add `README.md` noting the source config and changes. |
| 2. **Registry + model parity** | `tests/models/` | Add registry and eager family parity; cover one supported accelerator training configuration against a reference. |
| 3. **E2E parallel alignment** | `tests/e2e/test_e2e_parallel.py` | Extend the appropriate family table using the [default matrix or a justified exception](transformers_v5/testing_new_model.md#default-matrix); compare loss and grad norm. |

### Conditional Tests (depending on model type)

| Condition | Test File | What to Do |
|---|---|---|
| **VLM model** | `tests/trainer/test_vlm_trainer.py` | Add toy config to `_FREEZE_VIT_VLM_CASES_*`. |
| **VLM model** | `tests/distributed/test_dummy_forward.py` | Add test case for asymmetric multimodal batches. |
| **MoE model** | `tests/models/transformers/` | Cover eager parity and selectable fused expert implementations in the family test. |
| **MoE model** | `tests/e2e/test_e2e_parallel.py` | Set `is_moe=True`; the ordinary default is `(1,1)` versus `(2,2)`, with capability limits and representatives documented. |
| **Text model needing single-device/FSDP comparison** | `tests/distributed/test_fsdp_equivalence.py` | Extend the supported text-only suite; this is additional coverage, not a requirement for VLM/omni models. |
| **MoE with fused experts** | `tests/models/base/test_checkpoint_tensor_converter.py` | Add converter tests if a custom `CheckpointTensorConverter` is needed. |
| **Custom checkpoint layout** | `tests/models/base/test_checkpoint_tensor_converter.py` | Add converter tests for any on-disk HF↔VeOmni key or tensor-layout conversion. |
| **Custom fused ops** | `tests/ops/<family>/` | Cover distinct variants/backends and backward paths, including EP/non-EP when implemented; EP-local compute tests do not qualify cross-card transport. |
| **New data modality** | `tests/data/` | Add data processing and collation tests. |

### Verification Commands

```bash
# Collect test cases for the new model
pytest --collect-only -k <model_name>

# Run registry and model parity tests
pytest tests/models -k <model_name>

# Run VLM freeze test (VLM only)
pytest tests/trainer/test_vlm_trainer.py -k <model_name>

# Run FSDP equivalence (2+ GPUs)
pytest tests/distributed/test_fsdp_equivalence.py -k <model_name>

# Run E2E parallel alignment (4+ GPUs)
pytest tests/e2e/test_e2e_parallel.py -k <model_name>
```

---

## Test Execution Flow

### Model Test Flow
```
pytest → test_build_foundation_model_constructs_registered_model(model_case)
  → build the canonical tiny config from tests/models/tiny_configs.py
  → build_foundation_model(config, ops_implementation=eager_ops)
  → verify the registry/build contract for every registered architecture

pytest → family-specific eager parity test
  → build the upstream reference and VeOmni model from the same tiny config
  → copy/load matching weights
  → compare forward outputs and backward gradients
```

### E2E Parallel Test Flow
```
pytest → test_text_parallel_align(model_name, config, ...)
  → materialize_weights(config) → random weights on disk
  → DummyDataset(dataset_type) → parquet files
  → prepare_exec_cmd() → kwargs for each (sp_size, ep_size)
  → run each mode sequentially, one torchrun launch at a time
      build_torchrun_cmd(**kwargs) → fresh torchrun command and rendezvous port
      subprocess.run(torchrun ... train_text_test.py ...)
        → TestTextTrainer.train() → log_dict.json
  → compare_metrics(all_log_dicts, rtol, atol)
```

### FSDP Equivalence Test Flow
```
pytest → test_text_fsdp_equivalence(config, ...)
  → materialize_weights(config)
  → DummyDataset(text)
  → run_training_config(nproc=1, init_device=device)      # baseline
  → run_training_config(nproc=2, init_device=meta, fsdp2)  # fixed two-rank FSDP
  → compare_metrics(baseline_grad_norm, fsdp_grad_norm)
```

---

## Batch-invariant FP32 GEMM precision

`test_matmul_persistent_preserves_ieee_fp32_precision_and_batch_invariance` in
`tests/ops/batch_invariant/test_batch_invariant.py` checks one FP32
`matmul_persistent` product, with and without bias, against FP64 arithmetic over
the same operands (`rtol=1e-6, atol=1e-5`), and requires bitwise-equal results
when the rows are computed as two separate batches. It is the only FP64
reference case in the file: BF16/FP16, strided and padded-tile products are not
compared against FP64. The GPU unit workflow runs the entire
`tests/ops/` directory.

The invariant Triton GEMM explicitly uses IEEE input precision. Triton's default
TF32 dot precision truncates FP32 projection inputs even when PyTorch's matmul
precision is `highest`; setting the PyTorch policy alone does not configure a
Triton kernel. This fixes that operator's FP32 accuracy, but does not establish
whole-model BF16 gradient, parameter-update or convergence parity.

## Architecture Notes

### Resolved consolidations

The following redundancies have been addressed:

- **Shared training utils centralized**: `ParallelConfig`, `build_torchrun_cmd`,
  `materialize_weights`, `run_training_config`, and `release_device_memory` now live
  in `tests/tools/training_utils.py`. Both `tests/e2e/` and `tests/distributed/`
  import from `tests/tools` — no cross-directory imports between test subdirectories.

- **Parallel-mode naming is explicit**: The e2e parallelism dataclass is named
  `ParallelMode` (sp_size, ep_size), while `ParallelConfig` in
  `tests/tools/training_utils.py` adds `fsdp_mode` on top.

- **`distributed_test_helpers.py` removed**: Shared helpers moved to
  `tests/tools/training_utils.py`; `tests/distributed/` tests import directly
  from `tests/tools`.

- **Thin wrappers removed**: `compare_multi_items` / `print_all_values` wrappers in
  `tests/e2e/utils.py` have been replaced with direct imports of `compare_metrics` /
  `print_comparison_table` from `tests.tools`.

- **Train scripts separated**: `train_text_test.py`, `train_vlm_test.py`, and
  `train_dit_test.py` are standalone trainer scripts (not pytest tests). They have been
  moved from `tests/e2e/` to `tests/train_scripts/` to clarify their role.

### Remaining items for future work

- **`tests/e2e/test_e2e_training.py`** uses real model weights and `exec_scripts.py`,
  while `test_e2e_parallel.py` uses toy configs and `prepare_exec_cmd`. These serve
  different purposes (smoke test vs equivalence) but the naming doesn't reflect this.

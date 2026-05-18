# VeOmni Test Suite Overview

This document surveys all tests in the VeOmni project, describes their purpose and organization,
and provides guidance on which tests to add when onboarding a new model.

---

## Directory Structure

```
tests/
├── tools/                          # Shared test infrastructure (comparison, data gen, launch)
├── toy_config/                     # Minimal model configs for fast CI testing
├── testdata/                       # Sample images, audio, etc.
│
├── models/                         # Single-GPU model correctness
│   ├── test_models_patch.py        # Fwd/bwd across attention & MoE backends
│   ├── test_vlm_trainer.py         # VLM freeze_vit smoke test
│   ├── test_model_registry.py      # Model loader registry (HF vs VeOmni)
│   ├── test_checkpoint_tensor_converter.py  # Checkpoint tensor conversion (e.g. Qwen3MoE fuse)
│   ├── test_padded_packed_loss.py   # Padded vs packed (cu_seqlens) loss equivalence
│   ├── utils.py                    # ModelMode, prepare_model_modes, prepare_data
│   └── weight_sync_adapters.py     # State-dict alignment for HF↔VeOmni comparison
│
├── ops/                            # Fused kernel correctness & performance
│   ├── test_fused_moe_split_vs_merged.py   # Split vs merged MoE fc1
│   ├── test_quack_fused_moe.py             # Quack GEMM MoE (SM90+)
│   ├── test_fused_load_balancing_loss.py    # Triton load-balancing loss kernel
│   ├── test_flash_attn_varlen_padding.py   # Flash-attn variable-length padding
│   ├── test_seqcls_loss.py                 # Sequence classification loss
│   └── test_comp.py                        # Position embedding computation
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
│   │   ├── test_async_ulysses.py       # Async overlapping comm (4+ GPUs)
│   │   ├── test_async_ulysses_dit.py   # DiT + async SP
│   │   ├── test_qwen3_5_gated_deltanet_ulysses.py  # Gated DeltaNet + SP
│   │   ├── test_slice_input_tensor.py  # Input slicing utilities
│   │   ├── test_all_gather.py          # All-gather collective ops
│   │   └── utils.py                    # SequenceParallelTest base class
│   └── encoder_data_balance/
│       ├── test_balance_reverse.py        # Balance/recovery precision (8 GPUs)
│       └── test_balance_sorting_algo.py   # Post-MBS data sorting (CPU)
│
├── distributed/                    # Multi-GPU training correctness
│   ├── test_fsdp_equivalence.py         # Single-GPU vs FSDP2 grad equivalence
│   └── test_dummy_forward.py            # Asymmetric multimodal forward (NCCL hang prevention)
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
│   ├── test_checkpoint_callback.py          # Callback _last_saved_step correctness
│   ├── test_trainer_saveload.py             # DCP + HF checkpoint save/load (8 GPUs)
│   ├── checkpoint_verification_utils.py     # DCP-to-HF conversion verification
│   └── utils.py                             # Command/config builders for ckpt tests
│
├── utils/                          # Misc utility tests
│   ├── test_count_flops.py                       # FLOPs estimation
│   ├── test_extra_parallel_clip_grad_norm.py      # Grad clipping with EP/EMB dims (8 GPUs)
│   ├── test_helper.py                             # EnvironMeter utility (8 GPUs)
│   ├── test_model_loader.py                       # Model loading (4 GPUs)
│   ├── test_npu_setup.py                          # NPU environment validation
│   ├── test_rank0_load_and_broadcast_weights.py   # Rank-0 load & broadcast (2+ GPUs)
│   └── test_save_safetensor_utils.py              # Safetensor save utilities (CPU)
│
└── special_sanity/
    └── check_device_api_usage.py    # CI lint: no direct .cuda / "cuda" calls
```

---

## Test Categories at a Glance

| Category | Directory | GPU Req | Execution | Purpose |
|---|---|---|---|---|
| **Model patch** | `tests/models/` | 1 GPU | pytest | Fwd/bwd correctness across attn/MoE backends |
| **Ops / kernels** | `tests/ops/` | 1 GPU (SM90+ for Quack) | pytest | Fused kernel correctness & perf |
| **Data pipeline** | `tests/data/` | 0-1 GPU | pytest | Data loading, collation, preprocessing |
| **Parallelism** | `tests/parallel/` | 4-8 GPUs | torchrun / pytest | SP, EP, data-balance primitives |
| **FSDP correctness** | `tests/distributed/` | 2+ GPUs | torchrun (subprocess + mp.spawn) | Single-GPU vs FSDP2 equivalence, dummy forward |
| **E2E parallel** | `tests/e2e/` | 4+ GPUs | torchrun (subprocess) | SP/EP alignment across full training runs |
| **Checkpoints** | `tests/checkpoints/` | 0-8 GPUs | pytest + torchrun | Save/load, DCP→HF conversion |
| **Utilities** | `tests/utils/` | 0-8 GPUs | pytest + torchrun | FLOPs, grad clipping, weight broadcast |
| **Sanity** | `tests/special_sanity/` | 0 | script | Device API lint |

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
| `tests/models/utils.py` | Model patch tests | `ModelMode`, `prepare_model_modes`, `prepare_data` |
| `tests/models/weight_sync_adapters.py` | Model patch tests | HF↔VeOmni state-dict alignment functions |
| `tests/e2e/utils.py` | E2E tests | `prepare_exec_cmd`, `parse_training_log`, `ParallelMode` |
| `tests/checkpoints/utils.py` | Checkpoint tests | Command/config builders for trainer save/load |
| `tests/parallel/ulysses/utils.py` | SP tests | `SequenceParallelTest` base class, `sync_tensor` |

---

## Detailed Test Descriptions

### 1. Model Patch Tests (`tests/models/test_models_patch.py`)

**Purpose**: Verify that VeOmni's patched models produce identical loss and grad_norm to HuggingFace reference across all backend combinations.

**What it compares** (cartesian product):

| Dimension | Values |
|---|---|
| Modeling backend | HuggingFace, VeOmni |
| Attention implementation | `eager`, `flash_attention_2`, `flash_attention_3`, `veomni_flash_attention_2_with_sp`, `veomni_flash_attention_3_with_sp` |
| MoE implementation | `eager`, `fused` (MoE models only) |
| Liger kernel | `True`, `False` (VeOmni only) |

**Models covered** (transformers v5 only — the v4 monkey-patch lane was
retired together with the v4 CI; v4-only models that have not yet been
migrated to patchgen are not exercised here):
- Text / MoE: llama3_1, qwen2, qwen3_5, qwen3_5_moe, seed_oss, deepseek_v3
- VLM: qwen2_vl, qwen2_5_vl, qwen3_vl, qwen3_vl_moe
- Omni: qwen2_5_omni, qwen3_omni_moe

**GPU**: 1 GPU, runs serially per model mode.

---

### 2. VLM Trainer Test (`tests/models/test_vlm_trainer.py`)

**Purpose**: Smoke test that `freeze_vit=True/False` correctly freezes/unfreezes the vision tower.

**Models** (transformers v5 only): qwen2_vl, qwen3_5, qwen3_5_moe, qwen2_5_vl, qwen3_vl, qwen3_vl_moe

**GPU**: CPU only (builds model but no forward pass).

---

### 3. Model Registry Test (`tests/models/test_model_registry.py`)

**Purpose**: Verify that `get_model_config/class/processor` returns the correct HF or VeOmni module.

**GPU**: CPU only.

---

### 4. Checkpoint Tensor Converter (`tests/models/test_checkpoint_tensor_converter.py`)

**Purpose**: Test checkpoint tensor conversion protocol (e.g., Qwen3MoE expert weight fusion: per-expert → stacked `gate_up_proj`).

**GPU**: CPU only.

---

### 5. Padded vs Packed Loss (`tests/models/test_padded_packed_loss.py`)

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

**Models**: qwen3, qwen3_moe, llama3.1 (v4); qwen3_5, qwen3_5_moe (v5)

**GPU**: 2+ GPUs.

---

### 7. Dummy Forward (`tests/distributed/test_dummy_forward.py`)

**Purpose**: Verify that asymmetric multimodal batches (some ranks text-only, others with images/video/audio) don't cause NCCL hangs under FSDP2. Tests that `dummy_forward()` is correctly invoked so all ranks participate in FSDP collectives.

**Models** (all `_v5_only` — the legacy v4 lane was retired):
- VLM: qwen2_5_vl, qwen3_vl, qwen3_vl_moe
- Omni: qwen2_5_omni, qwen3_omni_moe

**GPU**: 2 GPUs.

---

### 8. E2E Parallel Alignment (`tests/e2e/test_e2e_parallel.py`)

**Purpose**: Full torchrun training runs across SP/EP configurations. Asserts that loss and grad_norm match regardless of parallelism settings.

**Configurations tested**:
- `sp_size` in [1, 2], `ep_size` in [1] (base) or [1, 2] (MoE)
- FSDP2 always enabled, `nproc_per_node = sp_size * 4`
- 2 epochs, 2 max_steps per run

**Models**: All supported text, VLM, omni, and DiT models.

**GPU**: 4+ GPUs (up to 8 for sp=2).

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

### 11. Ops / Kernel Tests (`tests/ops/`)

| Test | Purpose | GPU |
|---|---|---|
| `test_fused_moe_split_vs_merged.py` | Split vs merged fc1 in fused MoE | 1 GPU |
| `test_quack_fused_moe.py` | Quack GEMM MoE backend | SM90+ |
| `test_kernel_registry_numerical.py` | Numerical alignment per (op, variant, impl) | CUDA; the FlashQLA `chunk_gated_delta_rule` case skips unless on SM90 exactly (Hopper) with `--extra flash-qla`. SM10x is WIP upstream and won't work yet. |
| `test_fused_load_balancing_loss.py` | Triton load-balancing loss | CUDA |
| `test_flash_attn_varlen_padding.py` | Flash-attn variable-length padding | CUDA |
| `test_seqcls_loss.py` | Sequence classification loss | CUDA (optional) |
| `test_comp.py` | Position embedding computation | CUDA |

---

### 12. Parallelism Primitive Tests (`tests/parallel/`)

| Test | Purpose | GPU |
|---|---|---|
| `test_ulysses.py` | Basic Ulysses SP attention | 4+ |
| `test_async_ulysses.py` | Async overlapping communication | 4+ |
| `test_async_ulysses_dit.py` | DiT + async SP | 4+ |
| `test_qwen3_5_gated_deltanet_ulysses.py` | Gated DeltaNet + SP | 4+ |
| `test_slice_input_tensor.py` | SP input slicing utilities | CPU |
| `test_all_gather.py` | All-gather collective ops | multi |
| `test_balance_reverse.py` | Encoder data balance recovery | 8 |
| `test_balance_sorting_algo.py` | Post-MBS sorting algorithm | CPU |

---

## New Model Onboarding: Test Checklist

When adding a new model to VeOmni, the following tests should be created or updated.
See also: [Testing a New Model for Transformers v5](transformers_v5/testing_new_model.md) for step-by-step instructions.

### Required Tests

| Step | Test File | What to Do |
|---|---|---|
| 1. **Create toy config** | `tests/toy_config/<model>_toy/` | Minimal config (few layers, small dims). Add `README.md` noting the source config and changes. |
| 2. **Model patch (fwd/bwd)** | `tests/models/test_models_patch.py` | Add entry to `_TEST_CASES_TRANSFORMERS_V5` (or v4 list). Filter unsupported attn/MoE modes if needed. |
| 3. **E2E parallel alignment** | `tests/e2e/test_e2e_parallel.py` | Add entry to `text_test_cases` (text) or the appropriate VLM/omni list. Set `max_sp_size=1` if SP not yet supported. |
| 4. **FSDP equivalence** | `tests/distributed/test_fsdp_equivalence.py` | Add entry to verify single-GPU vs FSDP2 grad_norm matches. |

### Conditional Tests (depending on model type)

| Condition | Test File | What to Do |
|---|---|---|
| **VLM model** | `tests/models/test_vlm_trainer.py` | Add toy config to `_FREEZE_VIT_VLM_CASES_*`. |
| **VLM model** | `tests/distributed/test_dummy_forward.py` | Add test case for asymmetric multimodal batches. |
| **MoE model** | `tests/models/test_models_patch.py` | Set `is_moe=True` to test `eager` vs `fused` MoE backends. |
| **MoE model** | `tests/e2e/test_e2e_parallel.py` | Set `is_moe=True` to include `ep_size` iteration. |
| **MoE with fused experts** | `tests/models/test_checkpoint_tensor_converter.py` | Add converter tests if a custom `CheckpointTensorConverter` is needed. |
| **Custom weight layout** | `tests/models/weight_sync_adapters.py` | Add sync function if HF↔VeOmni state-dict keys differ. |
| **Custom fused kernels** | `tests/ops/` | Add kernel-specific correctness tests. |
| **New data modality** | `tests/data/` | Add data processing and collation tests. |

### Verification Commands

```bash
# Collect test cases for the new model
pytest --collect-only -k <model_name>

# Run single-GPU model patch test
pytest tests/models/test_models_patch.py -k <model_name>

# Run VLM freeze test (VLM only)
pytest tests/models/test_vlm_trainer.py -k <model_name>

# Run FSDP equivalence (2+ GPUs)
pytest tests/distributed/test_fsdp_equivalence.py -k <model_name>

# Run E2E parallel alignment (4+ GPUs)
pytest tests/e2e/test_e2e_parallel.py -k <model_name>
```

---

## Test Execution Flow

### Model Patch Test Flow
```
pytest → test_models_patch_fwd_bwd(config, is_moe, ...)
  → prepare_model_modes(is_moe) → [(HF, eager), (HF, fa2), (VeOmni, fa2_sp), ...]
  → for each mode:
      build_foundation_model(config, attn_impl, moe_impl)
      TrainerTest.forward_backward_step(dummy_batch)
      → record loss, grad_norm
  → compare_multi_items(all_results, rtol, atol)
```

### E2E Parallel Test Flow
```
pytest → test_text_parallel_align(model_name, config, ...)
  → materialize_weights(config) → random weights on disk
  → DummyDataset(dataset_type) → parquet files
  → for each (sp_size, ep_size):
      prepare_exec_cmd() → torchrun command
      subprocess.run(torchrun ... train_text_test.py ...)
        → TestTextTrainer.train() → log_dict.json
  → compare_multi_items(all_log_dicts, rtol, atol)
```

### FSDP Equivalence Test Flow
```
pytest → test_text_fsdp_equivalence(config, ...)
  → materialize_weights(config)
  → DummyDataset(text)
  → run_training_config(nproc=1, init_device=device)      # baseline
  → run_training_config(nproc=2+, init_device=meta, fsdp2) # FSDP
  → compare_metrics(baseline_grad_norm, fsdp_grad_norm)
```

---

## Architecture Notes

### Resolved consolidations

The following redundancies have been addressed:

- **Shared training utils centralized**: `ParallelConfig`, `build_torchrun_cmd`,
  `materialize_weights`, `run_training_config`, and `release_device_memory` now live
  in `tests/tools/training_utils.py`. Both `tests/e2e/` and `tests/distributed/`
  import from `tests/tools` — no cross-directory imports between test subdirectories.

- **`ModelMode` naming conflict resolved**: The e2e parallelism dataclass was renamed
  to `ParallelMode` (sp_size, ep_size) to distinguish it from `ModelMode` in
  `tests/models/utils.py` (modeling_backend, attn_implementation, ...).
  `ParallelConfig` in `tests/tools/training_utils.py` adds `fsdp_mode` on top.

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

- **`tests/models/utils.py`** has its own `compare_multi_items` / `print_all_values`
  with custom table formatting based on `ModelMode` fields. These are not simple
  wrappers and serve a different purpose from `tests.tools.compare_metrics`.

- **`tests/e2e/test_e2e_training.py`** uses real model weights and `exec_scripts.py`,
  while `test_e2e_parallel.py` uses toy configs and `prepare_exec_cmd`. These serve
  different purposes (smoke test vs equivalence) but the naming doesn't reflect this.

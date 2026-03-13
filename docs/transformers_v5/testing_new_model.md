# Testing a New Model

This document covers the full test hierarchy for onboarding new models in VeOmni. Tests are organized into levels L0 through L5, each validating a specific concern. All levels apply to both transformers v4 and v5 models unless noted otherwise.

---

## Test Levels Overview

| Level | What It Validates | Scope | GPU Requirement | Pytest Command |
|-------|-------------------|-------|-----------------|----------------|
| **L0** | Data processing and transform correctness | Tokenization, chat templates, multimodal preprocessing, HF processor alignment | CPU only | `pytest -m L0` |
| **L1** | Kernel and operator consistency | Forward/backward across attention and MoE backends on a single GPU | 1 GPU | `pytest -m L1` |
| **L2** | Single-GPU vs FSDP equivalence | FSDP-wrapped training matches single-GPU numerics | 2+ GPUs | `pytest -m L2` |
| **L3** | Parallelism combination consistency | SP/EP combinations match FSDP-only baseline | 2-8 GPUs | `pytest -m L3` |
| **L4** | Corner cases and robustness | Dummy forward, tied embeddings, checkpoint roundtrip, padding edge cases | Varies | `pytest -m L4` |
| **L5** | Integration / smoke test | End-to-end training loop completes without errors | Multi-GPU | `pytest -m L5` |

### Running tests by level

```bash
# Run a single level
pytest -m L0                    # Data pipeline only (no GPU)
pytest -m L1                    # Kernel consistency (1 GPU)
pytest -m L2                    # FSDP equivalence (multi-GPU)
pytest -m L3                    # Parallelism combinations (multi-GPU)
pytest -m L4                    # Corner cases
pytest -m L5                    # Smoke test

# Combine levels
pytest -m "L2 or L3"           # All distributed tests
pytest -m "L0 or L1"           # All single-machine tests

# Filter by model
pytest -m L1 -k "qwen3"       # L1 tests for qwen3 only

# Filter by version
pytest -m "L1 and v5_only"    # L1 tests that require transformers >= 5.0

# Filter by model type
pytest -m "L1 and moe"        # L1 tests for MoE models only
pytest -m "L3 and multi_gpu"  # L3 tests requiring multiple GPUs
```

---

## Directory Structure

```
tests/
├── conftest.py                          # Marker registration, version skip helpers
├── data/                                # L0 tests
│   ├── test_collators.py               # MainCollator packing/padding with and without SP
│   ├── test_datasets.py                # Dataset loading
│   ├── test_preprocessor.py            # Preprocessor correctness
│   ├── test_prepare_fa_kwargs.py       # Flash attention kwargs
│   ├── test_hf_processor_alignment.py  # VeOmni data transform vs HF processor comparison
│   └── multimodal/
│       └── test_video_utils.py         # Video preprocessing utils
├── ops/                                 # L1 kernel tests
│   ├── test_comp.py                    # ViT position embedding computation
│   ├── test_fused_moe_split_vs_merged.py  # Fused MoE split vs merged
│   ├── test_flash_attn_varlen_padding.py  # Flash attention variable-length padding
│   └── test_fused_cross_entropy.py     # Fused CE vs PyTorch native CE
├── models/                              # L1 model-level tests
│   ├── test_models_patch.py            # Forward/backward across attention/MoE backends
│   ├── utils.py                        # ModelMode, comparison utils, data preparation
│   └── weight_sync_adapters.py         # Per-model HF-to-VeOmni weight sync functions
├── distributed/                         # L2-L3 tests
│   ├── test_fsdp_equivalence.py        # Single-GPU vs FSDP comparison
│   └── test_parallelism_equivalence.py # FSDP vs FSDP+SP vs FSDP+EP combinations
├── e2e/                                 # L3 (legacy) and L5 tests
│   ├── test_e2e_parallel.py            # Multi-GPU parallel alignment via torchrun
│   └── test_e2e_training.py            # End-to-end training smoke test
├── robustness/                          # L4 tests
│   ├── test_dummy_forward.py           # Dummy forward for VLM/omni ranks with no data
│   ├── test_tied_embeddings.py         # tie_word_embeddings with FSDP
│   └── test_padding_edge_cases.py      # Extreme padding/packing edge cases
├── checkpoints/                         # L4 tests
│   └── test_trainer_saveload.py        # Checkpoint save/load roundtrip
├── parallel/                            # L2-L3 parallelism tests
│   └── ulysses/                        # Ulysses SP correctness
├── tools/                               # Shared test utilities
│   ├── common_utils.py                 # Device memory, rank helpers
│   ├── launch_utils.py                 # find_free_port, torchrun helper
│   ├── comparison_utils.py            # Unified comparison and pretty-printing
│   └── toy_config_utils.py            # Toy config loading/validation
└── toy_config/                          # Minimal model configs for tests
    ├── llama31_toy/
    ├── qwen3_toy/
    ├── qwen3_moe_toy/
    ├── qwen3_5_toy/
    └── ...
```

---

## Pytest Markers

Markers are registered in `tests/conftest.py`. Use them to filter test runs:

| Marker | Description |
|--------|-------------|
| `@pytest.mark.L0` | Level 0 — Data processing and transform correctness (no GPU) |
| `@pytest.mark.L1` | Level 1 — Kernel and operator consistency (single GPU) |
| `@pytest.mark.L2` | Level 2 — Single-GPU vs FSDP equivalence (2+ GPUs) |
| `@pytest.mark.L3` | Level 3 — Parallelism combination consistency (2-8 GPUs) |
| `@pytest.mark.L4` | Level 4 — Corner cases and robustness |
| `@pytest.mark.L5` | Level 5 — Integration / smoke test |
| `@pytest.mark.v4_only` | Requires transformers < 5.0.0 |
| `@pytest.mark.v5_only` | Requires transformers >= 5.0.0 |
| `@pytest.mark.multi_gpu` | Requires multiple GPUs |
| `@pytest.mark.moe` | MoE model specific test |

Version skip helpers are available in `conftest.py`:

```python
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

_is_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
v4_only = pytest.mark.skipif(_is_v5, reason="Not compatible with transformers >= 5.0.0")
v5_only = pytest.mark.skipif(not _is_v5, reason="Requires transformers >= 5.0.0")
```

---

## Level 1: `tests/models/test_models_patch.py`

### What it tests

Runs one forward + backward step on dummy data for every combination of:

- HF attention backends (`eager`, `flash_attention_2`, `flash_attention_3`)
- VeOmni attention backends (`veomni_flash_attention_2_with_sp`, `veomni_flash_attention_3_with_sp`)
- MoE backends (for MoE models: `eager`, `fused`)

Then asserts that loss and grad norm match across all combinations within `(rtol, atol)`.

### How to add a v5 model case

Add an entry to `_TEST_CASES_TRANSFORMERS_V5`:

```python
_TEST_CASES_TRANSFORMERS_V5 = [
    pytest.param(
        "./tests/toy_config/qwen3_5_toy/config.json",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="qwen3_5",
    ),
    # Add your new model here
    pytest.param(
        "./tests/toy_config/<new_model>_toy/config.json",
        False,  # is_moe — set True for MoE models
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        id="<new_model>",
    ),
]
```

The `id=` string is used as a key for:
- Test node naming (`pytest -k <id>`)
- Looking up custom weight sync functions in `weight_sync_adapters.py` (only needed if the model has non-standard state dict keys)

### Filtering unsupported modes

If the model doesn't support certain attention backends yet, add a filter block in `test_models_patch_fwd_bwd` keyed on `case_id`:

```python
if case_id == "<new_model>":
    hf_model_modes = [mode for mode in hf_model_modes if mode.attn_implementation != "flash_attention_3"]
    veomni_model_modes = [
        mode for mode in veomni_model_modes if mode.attn_implementation != "veomni_flash_attention_3_with_sp"
    ]
```

### Toy config

Create a minimal config under `tests/toy_config/<new_model>_toy/config.json` with few layers and small hidden dimensions. Add a README.md in the same folder to indicate:

1. Where the original config is from
2. What changes were made from the original config

---

## Level 2-3: `tests/e2e/test_e2e_parallel.py`

### What it tests

Launches full `torchrun` training runs (2 epochs, 2 steps) across parallel configurations (FSDP2 always enabled):

| Parameter | Values |
|-----------|--------|
| `sp_size` | 1, 2 |
| `ep_size` | 1 (base models), 1 and 2 (MoE models) |

Each run produces a `log_dict.json`. The test asserts that loss and grad norm match across all SP/EP configurations within `(rtol, atol)`.

### How to add a case

Add an entry to `text_test_cases` (for text-only models) with `marks=_v5_only`:

```python
text_test_cases = [
    # ... existing cases ...
    pytest.param(
        "<new_model>",
        "./tests/toy_config/<new_model>_toy/config.json",
        False,  # is_moe
        _DEFAULT_RTOL,
        _DEFAULT_ATOL,
        None,  # max_sp_size
        marks=_v5_only,
    ),
]
```

### Parametrize fields

The `text_test_cases` parametrize string is:

```
"model_name, config_path, is_moe, rtol, atol, max_sp_size"
```

| Field | Type | Description |
|-------|------|-------------|
| `model_name` | `str` | Used for directory naming and log output |
| `config_path` | `str` | Path to toy config directory or `config.json` |
| `is_moe` | `bool` | If `True`, also iterates over `ep_size` values |
| `rtol`, `atol` | `float` | Tolerances for cross-config comparison |
| `max_sp_size` | `int | None` | `None` = no limit (run sp=1,2). Set to `1` to skip sp=2 if SP is not yet supported |

### Limiting sequence parallelism

If the model does not support SP yet, set `max_sp_size=1` to only run with `sp_size=1`:

```python
pytest.param(
    "qwen3_5",
    "./tests/toy_config/qwen3_5_toy/config.json",
    False,  # is_moe
    _DEFAULT_RTOL,
    _DEFAULT_ATOL,
    1,  # max_sp_size — remove once SP is supported
    marks=_v5_only,
),
```

### VLM / multimodal models

For vision-language or multimodal models, add to the appropriate test case list (`qwen2vl_test_cases`, `qwen3vl_test_cases`, etc.) and pair with the matching fixture and test function. The same `max_sp_size` field is available.

---

## Level 4: Robustness Tests

### When to add tests

| Test Type | Add When |
|-----------|----------|
| Dummy forward | Model is VLM or omni-modal (ViT/audio encoder may receive no data on some ranks) |
| Tied embeddings | Model has `tie_word_embeddings=True` in config |
| Checkpoint roundtrip | Model introduces new weight layout or non-standard state dict keys |
| Padding edge cases | Model has custom padding or packing logic |

### Dummy forward test

Place in `tests/robustness/test_dummy_forward.py`. The test should:

1. Initialize the model with FSDP on 2+ GPUs
2. Send real data to rank 0 and empty/None data to rank 1
3. Run forward + backward on both ranks
4. Assert: no NCCL hang, valid gradients on both ranks, loss is finite

### Checkpoint save/load test

Place in `tests/checkpoints/test_trainer_saveload.py`. The test should:

1. Train for N steps, save checkpoint
2. Load checkpoint, train for M more steps
3. Compare: loss at step N+1 after reload matches loss at step N+1 from a continuous run

---

## Level 5: Smoke Test

The smoke test in `tests/e2e/test_e2e_training.py` runs a short end-to-end training loop with a real or toy model config. For CI environments, prefer toy configs with randomly initialized weights to avoid model downloads.

The test validates:
- Full training loop completes without errors
- Loss is finite and decreasing
- All components work together (data loading, forward, backward, optimizer step, logging, checkpoint)

---

## Tolerance Guidelines

Recommended default tolerances per test level:

| Test Level | Default rtol | Default atol | Rationale |
|-----------|-------------|-------------|-----------|
| L0 (data processing) | exact match | exact match | Data transforms must produce identical outputs |
| L1 (single GPU, same data) | 1e-2 | 1e-2 | Different attention/MoE kernels produce small numerical differences |
| L1 (MoE models) | 0.5 | 0.02 | MoE routing is sensitive to numerical noise (see qwen3_moe case) |
| L2-L3 (multi-GPU) | 1e-1 | 1e-1 | FSDP reduces precision due to communication; SP reshuffles data |
| L4 (checkpoint roundtrip) | 0 | 0 | Exact match expected (bitwise identical state dicts) |
| L5 (smoke test) | N/A | N/A | Only checks for finite loss and successful completion |

When a model requires non-default tolerances, document the reason in a comment:

```python
pytest.param(
    ...,
    0.5,   # rtol: MoE routing amplifies small numerical differences
    0.02,  # atol: relaxed due to fused MoE kernel numerical characteristics
    id="qwen3_moe",
),
```

---

## Parametrization Patterns

Follow the existing pattern using named `pytest.param` with `id=` for clear test node names:

```python
_TEST_CASES = [
    pytest.param(
        "./tests/toy_config/model_toy/config.json",
        False,   # is_moe
        1e-2,    # rtol
        1e-2,    # atol
        id="model_name",
        marks=v5_only,  # version gating
    ),
]
```

### Naming conventions

| Element | Convention | Example |
|---------|------------|---------|
| Test file | `test_<feature>.py` | `test_models_patch.py` |
| Test function | `test_<what>_<how>` | `test_models_patch_fwd_bwd` |
| Parametrize ID | `<model_short_name>` | `id="qwen3_moe"` |
| Fixture | `dummy_<type>_dataset` | `dummy_text_dataset` |
| Tolerance comment | inline after value | `0.5,  # rtol: MoE routing amplifies noise` |

---

## Determinism

All tests must be deterministic and reproducible:

1. Set random seed: `set_seed(42)` (or `enable_full_determinism=True` in TrainingArguments)
2. Set cuBLAS workspace: `CUBLAS_WORKSPACE_CONFIG=:16:8` if needed
3. Disable NCCL debug noise: `os.environ["NCCL_DEBUG"] = "OFF"`
4. Use toy configs with minimal layers for fast, stable runs

---

## Version Gating

Support both transformers v4 and v5 using markers:

```python
from veomni.utils.import_utils import is_transformers_version_greater_or_equal_to

_is_v5 = is_transformers_version_greater_or_equal_to("5.0.0")
_v4_only = pytest.mark.skipif(_is_v5, reason="Not compatible with transformers >= 5.0.0")
_v5_only = pytest.mark.skipif(not _is_v5, reason="Requires transformers >= 5.0.0")
```

Apply to test parameters:

```python
pytest.param(..., marks=_v5_only),
pytest.param(..., marks=_v4_only),
```

---

## Full Checklist

When adding a new model (v4 or v5), verify:

- [ ] Toy config created under `tests/toy_config/<model>_toy/`
- [ ] `DummyDataset` entry in `veomni/data/dummy_dataset.py` (if multimodal)
- [ ] `MODEL_TO_DATASET` entry in `tests/models/utils.py`
- [ ] **L0**: Data transform test (if custom processor): `tests/data/test_hf_processor_alignment.py`
- [ ] **L1**: Entry added to `_TEST_CASES_TRANSFORMERS_V5` (or v4 list) in `test_models_patch.py`
- [ ] **L1**: Unsupported attention/MoE modes filtered in `test_models_patch_fwd_bwd` if needed
- [ ] **L2**: FSDP equivalence test case
- [ ] **L3**: Entry added to `text_test_cases` (or VLM equivalent) in `test_e2e_parallel.py`
- [ ] **L3**: `max_sp_size` set appropriately (`1` if SP not supported, `None` otherwise)
- [ ] **L4**: Dummy forward test (if VLM/omni model)
- [ ] **L4**: Checkpoint save/load test (if new weight layout)
- [ ] **L5**: Smoke test config added
- [ ] `pytest --collect-only -k <model>` shows expected test cases
- [ ] Tests pass: `pytest -m "L0 or L1" -k <model>` and `pytest -m "L2 or L3" -k <model>`

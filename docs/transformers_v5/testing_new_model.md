# Testing a New Model for Transformers v5

When adding a new model with `transformers>=5.0.0` support, two test suites need updating:

1. **`tests/models/test_models_patch.py`** — single-GPU forward/backward correctness across attention and MoE backends
2. **`tests/e2e/test_e2e_parallel.py`** — multi-GPU e2e training with FSDP2, sequence parallelism (SP), and expert parallelism (EP)

Both files gate v5-only cases so they are skipped on v4 environments.

For VLM models, there is also a lightweight trainer-level smoke test for `freeze_vit`:

3. **`tests/models/test_vlm_trainer.py`** — builds a real toy VLM model on CPU and checks that vision parameters stay trainable when `freeze_vit=False` and are frozen when `freeze_vit=True`

## 1. `tests/models/test_models_patch.py`

### What it tests

Runs one forward + backward step on dummy data for every combination of:

- HF attention backends (`eager`, `flash_attention_2`, `flash_attention_3`)
- VeOmni attention backends (`veomni_flash_attention_2_with_sp`, `veomni_flash_attention_3_with_sp`)
- MoE backends (for MoE models: `eager`, `fused`)

Then asserts that loss and grad norm match across all combinations within `(rtol, atol)`.

### How to add a case

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
    # ← add your new model here
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

Create a minimal config under `tests/toy_config/<new_model>_toy/config.json` with few layers. Add a README.md under the same folder to indicate:

1. Where the original config is from
2. What changes are made from the original config

## 2. `tests/e2e/test_e2e_parallel.py`

### What it tests

Launches full `torchrun` training runs (2 epochs, 2 steps) across parallel configurations (fsdp2 always enabled):

| Parameter | Values |
|-----------|--------|
| `sp_size` | 1, 2 |
| `ep_size` | 1 (base models), 1×2 (MoE models) |

Each run produces a `log_dict.json`. The test asserts that loss and grad norm match across all SP/EP configurations within `(rtol, atol)`.

### How to add a case

Add an entry to `text_test_cases` (for text-only models) with `marks=_v5_only`:

```python
text_test_cases = [
    # ... existing v4 cases ...
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
| `max_sp_size` | `int \| None` | `None` = no limit (run sp=1,2). Set to `1` to skip sp=2 if SP is not yet supported |

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

## 3. `tests/models/test_vlm_trainer.py`

### What it tests

Builds a real toy VLM model and calls `VLMTrainer._freeze_model_module()` directly. The test only checks one behavior:

- `freeze_vit=False` -> the vision tower parameters remain trainable
- `freeze_vit=True` -> the vision tower parameters are frozen

This is intentionally simpler than an e2e training test. It is meant to catch model wrapper path changes such as `model.visual` vs `model.model.visual`.

### How to add a case

Add your toy config to the matching case list:

```python
_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V5 = [
    pytest.param("./tests/toy_config/qwen3_5_toy/config.json", id="qwen3_5"),
    pytest.param("./tests/toy_config/<new_vlm_model>_toy/config.json", id="<new_vlm_model>"),
]
```

For transformers v4 VLMs, add to `_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V4` instead.

## Checklist

When adding a new v5 model, verify:

- [ ] Toy config created under `tests/toy_config/<model>_toy/`
- [ ] Entry added to `_TEST_CASES_TRANSFORMERS_V5` in `test_models_patch.py`
- [ ] Unsupported attention/MoE modes filtered in `test_models_patch_fwd_bwd` if needed
- [ ] Entry added to `text_test_cases` (or VLM equivalent) in `test_e2e_parallel.py` with `marks=_v5_only`
- [ ] For VLM models, toy config added to `_FREEZE_VIT_VLM_CASES_TRANSFORMERS_V5` in `tests/models/test_vlm_trainer.py`
- [ ] `max_sp_size` set appropriately (use `1` if SP not supported, `None` otherwise)
- [ ] `pytest --collect-only -k <model>` shows expected test cases
- [ ] Tests pass: `pytest tests/models/test_models_patch.py -k <model>` and `pytest tests/e2e/test_e2e_parallel.py -k <model>`
- [ ] For VLM models, `pytest tests/models/test_vlm_trainer.py -k <model>` passes

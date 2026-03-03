# PRD: Enforce Rust Backend in LiberoYoumuDataset

## Introduction

Now that the LIBERO 64KB dataset has been re-encoded without dictionary encoding, the youmu Rust reader works correctly. The PyArrow fallback code in `LiberoYoumuDataset` is no longer needed and adds complexity. This PRD removes all fallback paths, enforces the Rust backend with fail-fast behavior, and verifies the full training pipeline works end-to-end.

## Goals

- Remove all PyArrow fallback code from `LiberoYoumuDataset`
- Fail fast with `ImportError` if Rust extension is missing
- Fail fast with `RuntimeError` if Rust smoke test fails (e.g. incompatible encoding)
- All existing tests pass with the Rust-only backend
- E2e training runs 5 steps with decreasing loss

## User Stories

### US-001: Remove PyArrow fallback from LiberoYoumuDataset
**Description:** As a developer, I want `LiberoYoumuDataset` to use only the Rust backend so the code is simpler and we catch encoding issues immediately rather than silently falling back to a slow path.

**Acceptance Criteria:**
- [ ] Remove the `_HAS_RUST` flag and the `try/except ImportError` around the Rust import — import `ParquetReaderCachePy`, `read_list_row_range_py`, `read_row_range_py` unconditionally
- [ ] Remove the `_use_rust` instance variable and all `if self._use_rust` / `else` branches
- [ ] Remove `_pf_cache`, `_get_pf()`, and all PyArrow-based read paths in `_read_list_column_range()` and `_read_image_range()`
- [ ] Remove the `column_name` parameter from `_read_list_column_range()` (only needed for PyArrow)
- [ ] Remove `_state_column`, `_action_column`, `_image_column` instance variables (only needed for PyArrow column-name reads)
- [ ] Keep `state_column`, `action_column`, `image_column` constructor params (still needed for `_find_physical_column_index`)
- [ ] Constructor smoke test: if `read_list_row_range_py` raises, re-raise as `RuntimeError` with a clear message about incompatible parquet encoding
- [ ] `__getstate__` / `__setstate__` simplified: only handle Rust cache, no PyArrow branch
- [ ] Docstrings updated to reflect Rust-only operation
- [ ] Typecheck passes (`python -m py_compile`)

### US-002: Update tests for Rust-only backend
**Description:** As a developer, I want all existing LIBERO tests to pass with the Rust-only `LiberoYoumuDataset`, confirming no regressions.

**Acceptance Criteria:**
- [ ] `pytest tests/data/test_libero_dataset_episode_per_file.py -v` — all tests pass (0 skipped)
- [ ] `pytest tests/data/test_youmu_lerobot_parity.py -v` — all 12 tests pass (0 skipped). Note: set `HF_DATASETS_CACHE=/mnt/local/localcache00/hf_datasets_cache` to avoid root FS disk space issues.
- [ ] `pytest tests/data/test_libero_task_descriptions.py -v` — all tests pass (0 skipped)
- [ ] `pytest tests/data/test_libero_metadata_jsonl.py -v` — all 8 tests pass
- [ ] If any test was relying on PyArrow fallback behavior, update it to work with Rust-only
- [ ] Typecheck passes

### US-003: Run e2e training for 5 steps and verify loss decreases
**Description:** As a developer, I want to run the full training pipeline for 5 steps and confirm the Rust-only dataset produces correct data that leads to decreasing loss.

**Acceptance Criteria:**
- [ ] Run `bash train.sh tasks/omni/train_qwen_vl_libero.py configs/multimodal/qwen3_vl/qwen3_vl_libero.yaml` with `max_steps: 5`
- [ ] Training launches without import or config errors
- [ ] All 5 steps complete without crash
- [ ] Loss at step 1 is finite (not NaN/inf)
- [ ] Loss at step 5 is strictly less than loss at step 1
- [ ] No PyArrow fallback warnings in logs (since fallback code is removed)
- [ ] Typecheck passes

## Functional Requirements

- FR-1: `from youmu import ParquetReaderCachePy, read_list_row_range_py, read_row_range_py` must be a top-level unconditional import in `libero_dataset.py`
- FR-2: If the Rust extension is not installed, the import fails with a standard `ImportError` — no graceful degradation
- FR-3: The constructor smoke test must raise `RuntimeError("Rust parquet reader failed on {test_file}: {error}. Ensure parquet files are written with use_dictionary=False.")` if the Rust reader panics
- FR-4: The `_read_list_column_range` method must only use `read_list_row_range_py` — no PyArrow path
- FR-5: The `_read_image_range` method must only use `read_row_range_py` — no PyArrow path
- FR-6: The training script must produce monotonically improving loss over 5 steps on the re-encoded 64KB dataset

## Non-Goals

- Not changing the youmu Rust reader itself (no Rust code changes)
- Not changing the parquet conversion script
- Not adding support for dict-encoded parquet (we re-encoded to avoid this)
- Not changing the training script logic, model architecture, or config (other than max_steps for testing)

## Technical Considerations

- The re-encoded 64KB dataset at `/mnt/local/localcache00/libero_64KB` now uses `PLAIN` encoding (no dictionary). The Rust reader works correctly with this.
- `HF_DATASETS_CACHE` must point to `/mnt/local/localcache00/hf_datasets_cache` for lerobot parity tests (root FS is 97% full).
- The training script requires `NPROC_PER_NODE>=2` for FSDP2 and `VEOMNI_USE_LIGER_KERNEL=0` to avoid Triton bf16 errors.
- Old `libero_64KB_old` directory (dict-encoded) is still at `/mnt/local/localcache00/libero_64KB_old` — can be deleted after verification.

## Success Metrics

- Zero lines of PyArrow fallback code in `libero_dataset.py`
- All 55+ LIBERO tests pass
- Training loss decreases over 5 steps
- `LiberoYoumuDataset._use_rust` attribute no longer exists

## Open Questions

- Should we delete `/mnt/local/localcache00/libero_64KB_old` after successful verification?

# PRD: Fix All Skipped Tests and Add Synthetic Fixtures

## Introduction

With `av`, `huggingface_hub`, and the youmu Rust extension now available, previously-skipped tests should run. This PRD covers: (1) verifying all previously-skipped tests pass with real data, (2) adding synthetic dataset fixtures so integration tests can run without the real `/mnt/local/localcache00/` mount, and (3) fixing any test failures discovered along the way.

## Goals

- All 12 tests in `test_youmu_lerobot_parity.py` pass (5 backend + 7 raw parquet)
- All 6 integration tests in `test_libero_dataset_episode_per_file.py` pass
- All 2 real-data tests in `test_libero_task_descriptions.py` pass
- Add synthetic fixtures so data-dependent tests have a fallback when real data is absent
- Zero test regressions — existing passing tests continue to pass

## User Stories

### US-001: Run and fix backend parity tests (TestBackendParity)
**Description:** As a developer, I want the 5 `TestBackendParity` tests in `test_youmu_lerobot_parity.py` to pass so that I can verify youmu and LeRobot produce identical outputs.

**Acceptance Criteria:**
- [ ] `test_dataset_lengths_match` passes with real data
- [ ] `test_state_parity` passes — observation.state tensors are byte-identical
- [ ] `test_action_parity` passes — action tensors are exactly equal
- [ ] `test_image_parity` passes — image tensors are pixel-perfect identical
- [ ] `test_episode_index_parity` passes — episode_index scalars match
- [ ] If any test fails, root-cause and fix the underlying code (youmu dataset, lerobot dataset adapter, or the test itself)
- [ ] All existing unit tests still pass

### US-002: Run and fix raw parquet parity tests (TestRawParquetParity)
**Description:** As a developer, I want the 7 `TestRawParquetParity` tests to pass so that I can confirm the libero and libero_64KB datasets are byte-identical at the parquet level.

**Acceptance Criteria:**
- [ ] All 7 raw parquet tests pass: file counts, row counts, state data, actions data, image bytes, metadata episode count, metadata episode lengths
- [ ] If any test fails, diagnose whether it's a data issue or test logic issue and fix accordingly
- [ ] All existing unit tests still pass

### US-003: Run and fix LiberoYoumuDataset integration tests
**Description:** As a developer, I want the 6 `TestGetItemRealDataset` tests in `test_libero_dataset_episode_per_file.py` to pass so that I can verify the dataset loads correctly end-to-end.

**Acceptance Criteria:**
- [ ] `test_dataset_length` passes — dataset has >0 anchor frames
- [ ] `test_getitem_returns_standard_keys` passes — output dict has all expected keys
- [ ] `test_getitem_state_shape` passes — shape is (1, 8) for obs_len=1
- [ ] `test_getitem_action_shape` passes — shape is (4, 7) for pred_len=4
- [ ] `test_getitem_image_shape` passes — 4D uint8 tensor with 3 channels
- [ ] `test_getitem_finite_values` passes — no NaN/inf in state or action
- [ ] If any test fails, fix the underlying `LiberoYoumuDataset` code or test expectations
- [ ] All existing unit tests still pass

### US-004: Run and fix real-data task description tests
**Description:** As a developer, I want the 2 `TestRealData` tests in `test_libero_task_descriptions.py` to pass so that I can verify JSONL task description loading works against real data.

**Acceptance Criteria:**
- [ ] `test_load_from_real_jsonl` passes — loads real episodes.jsonl, all keys are non-negative ints, all values are non-empty strings
- [ ] `test_all_episodes_have_descriptions` passes — every JSONL line maps to a dict entry
- [ ] If any test fails, fix the `load_libero_task_descriptions` code or test expectations
- [ ] All existing unit tests still pass

### US-005: Add synthetic dataset fixtures for data-dependent tests
**Description:** As a developer, I want synthetic dataset fixtures so that integration tests can run in environments without the real LIBERO data mount, enabling CI and other developer machines.

**Acceptance Criteria:**
- [ ] Create a shared pytest fixture (in `tests/conftest.py` or `tests/data/conftest.py`) that generates a minimal synthetic LIBERO dataset in a temp directory with the episode-per-file layout
- [ ] Synthetic fixture includes: `data/chunk-000/` with 2-3 small episode parquet files, `meta/episodes.jsonl` with matching metadata
- [ ] Parquet files have correct schema: `state` (fixed-size list), `actions` (fixed-size list), `image` (struct with bytes+path)
- [ ] Update `test_libero_dataset_episode_per_file.py` `TestGetItemRealDataset` to also run against synthetic data when real data is absent (parametrize or duplicate with synthetic fixture)
- [ ] Update `test_libero_task_descriptions.py` `TestRealData` to also run against synthetic data when real data is absent
- [ ] Tests using synthetic data pass
- [ ] Tests using real data still pass (no regressions)

### US-006: Full test suite green run
**Description:** As a developer, I want to run the full LIBERO-related test suite and confirm everything passes end-to-end.

**Acceptance Criteria:**
- [ ] `pytest tests/data/test_youmu_lerobot_parity.py -v` — all 12 tests pass
- [ ] `pytest tests/data/test_libero_dataset_episode_per_file.py -v` — all 14 tests pass (8 unit + 6 integration)
- [ ] `pytest tests/data/test_libero_task_descriptions.py -v` — all 14 tests pass (12 unit + 2 real-data)
- [ ] `pytest tests/data/test_libero_metadata_jsonl.py -v` — all 8 tests pass (should already work)
- [ ] No warnings about missing imports or skipped tests (except video_utils which is out of scope)

## Functional Requirements

- FR-1: All test skip conditions based on missing `av`, `huggingface_hub`, or `lerobot` must now evaluate to "not skipped" given these packages are installed
- FR-2: All test skip conditions based on missing youmu Rust extension must now evaluate to "not skipped" given the extension is built
- FR-3: All test skip conditions based on missing data directories must evaluate to "not skipped" when `/mnt/local/localcache00/libero_64KB` and `/mnt/local/localcache00/libero` exist
- FR-4: A synthetic LIBERO dataset fixture must be reusable across test files via conftest.py
- FR-5: Synthetic fixture must produce valid parquet files readable by both PyArrow and the youmu Rust reader
- FR-6: Tests parametrized over real/synthetic data must clearly label which data source is used in test output
- FR-7: Any code fixes discovered during test runs must be applied to the source (not hacked around in tests)

## Non-Goals

- Video utils tests (`test_video_utils.py`) — out of scope, requires ffmpeg/torchcodec and CI_SAMPLES_DIR
- Performance optimization of tests or datasets
- Adding new test coverage beyond what already exists (fixing existing skipped tests only)
- Changes to the training script or model code

## Technical Considerations

- The youmu Rust `ParquetReaderCachePy` may not handle all parquet encodings (e.g., dict-encoded columns caused panics before — a PyArrow fallback was added in US-005). Synthetic fixtures must use compatible encodings.
- `build_libero_dataset` is loaded via `importlib` from `tasks/omni/train_qwen_vl_libero.py` — this import chain pulls in torch, transformers, etc. Tests should mock heavy dependencies where possible.
- LeRobot's dataset format expects a specific directory structure. The `TestBackendParity` tests load from `/mnt/local/localcache00/libero` (LeRobot format) vs `libero_64KB` (youmu format). Both must exist for those tests.
- Synthetic fixtures should NOT attempt to replicate LeRobot format — only youmu episode-per-file format. Backend parity tests require real data by nature (they compare two different backends on the same real data).

## Success Metrics

- 0 skipped tests in LIBERO-related test files (when real data is available)
- 0 skipped tests due to missing packages in any configuration
- Synthetic-data tests pass even without the `/mnt/local/localcache00/` mount

## Open Questions

- Should the synthetic fixture be committed as static files in `tests/fixtures/` or generated at runtime via pytest fixtures? (Runtime generation is more flexible but slower)
- For `TestBackendParity`, if youmu and LeRobot produce slightly different image decoding (e.g., JPEG artifacts), should we use approximate comparison (`torch.allclose`) instead of exact equality?

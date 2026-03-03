# PRD: LiberoYoumuDataset Adaptation & E2E Training with 64KB Parquet

## Introduction

Make `train_qwen_vl_libero.py` work end-to-end with the `youmu` dataset backend against the actual LIBERO dataset at `/mnt/local/localcache00/libero_64KB` (64KB page size, page-indexed parquet). The current `LiberoYoumuDataset` was written for a LeRobot-style directory layout (`file-NNN.parquet`, parquet episode metadata), but the actual dataset uses a different layout (`episode_XXXXXX.parquet`, JSONL metadata, flat column names). We adapt the code to match the real data, verify exact numerical parity with LeRobot's loader, then run training e2e.

## Goals

- Adapt `LiberoYoumuDataset` and related utilities to handle the actual LIBERO dataset format
- Achieve exact numerical parity (states, actions, images) between youmu and LeRobot backends for the same sample indices
- Successfully run `bash train.sh tasks/omni/train_qwen_vl_libero.py configs/multimodal/qwen3_vl/qwen3_vl_libero.yaml` on a single node with multi-GPU (FSDP2)

## User Stories

### US-001: Adapt LiberoYoumuDataset to handle episode-per-file layout
**Description:** As a developer, I need LiberoYoumuDataset to work with the actual LIBERO dataset where each episode is a separate file (`episode_XXXXXX.parquet`) with flat column names (`state`, `actions`, `image`), instead of the LeRobot layout (`file-NNN.parquet` with dotted names).

**Acceptance Criteria:**
- [ ] `LiberoYoumuDataset` auto-detects the file naming pattern (`episode_XXXXXX.parquet` vs `file-NNN.parquet`)
- [ ] Correctly maps episode index to file path for episode-per-file layout (trivial: each file = one episode)
- [ ] Handles the flat column names: `state` (not `observation.state`), `actions` (not `action`), `image` (not `observation.images.image`)
- [ ] Column name mapping is configurable via constructor args (defaults match the actual dataset)

### US-002: Adapt metadata loading for JSONL format
**Description:** As a developer, I need `load_episode_metadata` (or a new function) to load episode metadata from `meta/episodes.jsonl` since the actual dataset does not have `meta/episodes/chunk-000/file-000.parquet`.

**Acceptance Criteria:**
- [ ] New or updated function reads `episodes.jsonl` and returns `EpisodeDescriptor` list
- [ ] For the episode-per-file layout, `file_index` maps directly to the episode file (or is computed from `episode_index`)
- [ ] `start_frame` and `end_frame` are computed by accumulating episode lengths in episode_index order
- [ ] Works for multi-chunk datasets (chunk-000 has episodes 0-999, chunk-001 has 1000-1692)

### US-003: Adapt task description loading for JSONL format
**Description:** As a developer, I need `load_libero_task_descriptions` to work with the JSONL metadata at `meta/episodes.jsonl` (and optionally `meta/tasks.jsonl`).

**Acceptance Criteria:**
- [ ] Function loads task descriptions from `meta/episodes.jsonl`
- [ ] Returns `dict[int, str]` mapping `episode_index` → task description string
- [ ] Falls back gracefully if tasks field is missing
- [ ] Training script's metadata path logic updated to find the JSONL files

### US-004: Exact numerical parity test (youmu vs LeRobot)
**Description:** As a developer, I need a test that proves LiberoYoumuDataset returns exactly the same data as LeRobot's dataset for the same sample indices, giving confidence that training with youmu is correct.

**Acceptance Criteria:**
- [ ] Test loads the same dataset with both `youmu` and `lerobot` backends using `build_libero_dataset`
- [ ] For a representative set of indices (first 10, last 10, 10 random), asserts:
  - `observation.state` tensors are exactly equal (`torch.equal`)
  - `action` tensors are exactly equal
  - `observation.images.image` tensors are exactly equal (pixel-perfect)
  - `episode_index` values match
- [ ] Test passes on the 64KB parquet dataset at `/mnt/local/localcache00/libero_64KB`
- [ ] Test is runnable via pytest

### US-005: Update training config and run e2e
**Description:** As a developer, I need to update the training config to point to the 64KB dataset and successfully run the training script.

**Acceptance Criteria:**
- [ ] Config `libero_data_dir` points to `/mnt/local/localcache00/libero_64KB`
- [ ] `bash train.sh tasks/omni/train_qwen_vl_libero.py configs/multimodal/qwen3_vl/qwen3_vl_libero.yaml` launches without errors
- [ ] Training completes at least 1 full step (forward + backward + optimizer step) without crash
- [ ] Loss is a finite number (not NaN/inf)

## Functional Requirements

- FR-1: `LiberoYoumuDataset.__init__` must detect whether data files are named `episode_XXXXXX.parquet` or `file-NNN.parquet` and handle both
- FR-2: For episode-per-file layout, the episode-to-file mapping is direct: episode N → `episode_{N:06d}.parquet`, row offset = 0
- FR-3: Column name defaults updated: `state_column="state"`, `action_column="actions"`, `image_column="image"` (matching the actual dataset schema)
- FR-4: Image struct column extraction still works: `image` is `struct<bytes: binary, path: string>`, extract `bytes` child
- FR-5: Metadata loading supports both parquet (`meta/episodes/chunk-000/file-000.parquet`) and JSONL (`meta/episodes.jsonl`) formats
- FR-6: For JSONL metadata, `start_frame` is computed as cumulative sum of episode lengths (episodes sorted by `episode_index`)
- FR-7: The output dict keys from `__getitem__` remain `"observation.state"`, `"action"`, `"observation.images.image"`, `"episode_index"`, `"frame_index"` regardless of underlying column names (so downstream transform code doesn't change)
- FR-8: `load_libero_task_descriptions` reads from `episodes.jsonl`, extracting `episode_index` and `tasks[0]` per line
- FR-9: Training script's metadata path resolution updated to check for JSONL files

## Non-Goals

- No changes to the data collator or model code
- No changes to the data transform (`process_libero_sample_qwen3_vl`) — output keys stay the same
- No support for video/wrist_image columns (only the main `image` column for now)
- No multi-node training validation (single-node multi-GPU only)
- No hyperparameter tuning — just verify training runs

## Technical Considerations

- The 64KB parquet dataset is at `/mnt/local/localcache00/libero_64KB` with page index enabled
- Each episode file has ~3 row groups with ~100 rows each
- The dataset has 1693 episodes across 2 chunks (chunk-000: 1000 files, chunk-001: 693 files)
- Column `actions` (not `action`) has shape [7], `state` has shape [8], `image` is struct with PNG bytes
- LeRobot's `LeRobotDataset` may need the original (non-64KB) dataset path if it does its own parquet reading with specific expectations
- Youmu's Rust reader cache (`ParquetReaderCachePy`) benefits from 64KB pages + page index for fine-grained row access

## Success Metrics

- Exact numerical parity test passes for all compared indices
- Training launches and completes at least 1 step with finite loss
- No regressions in existing functionality

## Open Questions

- Does LeRobot's `LeRobotDataset` work with the episode-per-file layout, or does it need the original dataset? (Affects whether parity test uses original or 64KB data for the LeRobot side)
- Should we keep backward compatibility with the old LeRobot-style layout in `LiberoYoumuDataset`, or just target the episode-per-file format?

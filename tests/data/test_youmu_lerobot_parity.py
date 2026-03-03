"""Exact numerical parity tests: youmu LiberoYoumuDataset vs LeRobot LeRobotDataset.

Verifies that both backends return identical observation states, actions, and
images for the same sample indices, giving confidence that training with
youmu produces the same results as LeRobot.

Two test classes:
1. TestBackendParity — loads datasets with both backends and compares
   __getitem__ outputs for matching (episode_index, frame_index) pairs.
   Requires both youmu Rust extension *and* lerobot (with av) to be importable.
2. TestRawParquetParity — uses PyArrow directly to verify the underlying
   data in /mnt/local/localcache00/libero and libero_64KB are byte-identical.
   Runs whenever the data directories exist (no Rust or lerobot needed).

Note on lerobot v3.0 compatibility:
    The on-disk LIBERO dataset uses lerobot v2.0 format (episode-per-file parquet,
    JSONL metadata).  Installed lerobot 0.4.4 expects v3.0 format (different
    metadata structure).  The test creates a temporary v3.0-compatible directory
    that symlinks the real data and provides converted metadata files.
"""

import json
import os
import random
import shutil
import tempfile

import pandas as pd
import pyarrow.parquet as pq
import pytest
import torch


# ---------------------------------------------------------------------------
# Dataset paths
# ---------------------------------------------------------------------------

YOUMU_DATA_DIR = "/mnt/local/localcache00/libero_64KB"
LEROBOT_DATA_DIR = "/mnt/local/localcache00/libero"

OBS_LEN = 1
PRED_LEN = 4

# ---------------------------------------------------------------------------
# Backend availability checks
# ---------------------------------------------------------------------------

_youmu_available = False
try:
    from youmu import ParquetReaderCachePy  # noqa: F401

    _youmu_available = True
except (ImportError, ModuleNotFoundError):
    pass

_lerobot_available = False
_lerobot_skip_reason = "LeRobot package not importable (missing av or other deps)"
try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: F401

    _lerobot_available = True
except (ImportError, ModuleNotFoundError):
    pass

_youmu_data_exists = os.path.isdir(YOUMU_DATA_DIR)
_lerobot_data_exists = os.path.isdir(LEROBOT_DATA_DIR)

skip_no_youmu = pytest.mark.skipif(
    not _youmu_available,
    reason="Youmu Rust extension not available",
)
skip_no_lerobot = pytest.mark.skipif(
    not _lerobot_available,
    reason=_lerobot_skip_reason,
)
skip_no_youmu_data = pytest.mark.skipif(
    not _youmu_data_exists,
    reason=f"Youmu dataset not found at {YOUMU_DATA_DIR}",
)
skip_no_lerobot_data = pytest.mark.skipif(
    not _lerobot_data_exists,
    reason=f"LeRobot dataset not found at {LEROBOT_DATA_DIR}",
)


def _create_lerobot_v3_dir(data_dir: str) -> str:
    """Create a temporary v3.0-compatible directory for lerobot loading.

    The on-disk dataset uses v2.0 format (episode-per-file, JSONL metadata).
    lerobot 0.4.4 requires v3.0 format.  This function creates a temp directory
    with v3.0 metadata files and symlinks the actual data directory.

    Returns:
        Path to the temporary v3.0-compatible directory.
    """
    tmpdir = tempfile.mkdtemp(prefix="lerobot_v3_parity_")

    # Symlink data directory (parquet files are compatible)
    os.symlink(os.path.join(data_dir, "data"), os.path.join(tmpdir, "data"))
    os.makedirs(os.path.join(tmpdir, "meta"), exist_ok=True)

    # Update info.json to claim v3.0
    with open(os.path.join(data_dir, "meta", "info.json")) as f:
        info = json.load(f)
    info["codebase_version"] = "v3.0"
    with open(os.path.join(tmpdir, "meta", "info.json"), "w") as f:
        json.dump(info, f, indent=4)

    # Copy stats.json
    shutil.copy2(
        os.path.join(data_dir, "meta", "stats.json"),
        os.path.join(tmpdir, "meta", "stats.json"),
    )

    # Convert tasks.jsonl → tasks.parquet
    with open(os.path.join(data_dir, "meta", "tasks.jsonl")) as f:
        tasks = [json.loads(line) for line in f]
    pd.DataFrame(tasks).to_parquet(os.path.join(tmpdir, "meta", "tasks.parquet"))

    # Convert episodes.jsonl → episodes/chunk-000/file-000.parquet
    with open(os.path.join(data_dir, "meta", "episodes.jsonl")) as f:
        episodes = sorted(
            [json.loads(line) for line in f],
            key=lambda e: e["episode_index"],
        )

    episodes_dir = os.path.join(tmpdir, "meta", "episodes", "chunk-000")
    os.makedirs(episodes_dir, exist_ok=True)

    # v3.0 requires dataset_from_index / dataset_to_index columns
    ep_records = []
    cumulative = 0
    for ep in episodes:
        length = ep["length"]
        # Map task descriptions to task indices
        task_indices = []
        for task_desc in ep.get("tasks", []):
            for t in tasks:
                if t["task"] == task_desc:
                    task_indices.append(t["task_index"])
                    break
        ep_records.append(
            {
                "episode_index": ep["episode_index"],
                "tasks": str(task_indices),
                "length": length,
                "dataset_from_index": cumulative,
                "dataset_to_index": cumulative + length,
            }
        )
        cumulative += length

    pd.DataFrame(ep_records).to_parquet(os.path.join(episodes_dir, "file-000.parquet"))

    return tmpdir


def _build_youmu_dataset(data_dir: str):
    """Build a LIBERO dataset with the youmu backend."""
    from youmu.libero_dataset import LiberoYoumuDataset

    return LiberoYoumuDataset(
        data_dir=data_dir,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        chunk_index=None,
    )


def _build_lerobot_dataset(data_dir: str):
    """Build a LIBERO dataset with the lerobot backend via v3.0 wrapper.

    Creates a temporary v3.0-compatible directory that wraps the v2.0 data.
    Uses the actual parquet column names (state, actions, image) which differ
    from youmu's output key names (observation.state, action, observation.images.image).

    Returns:
        Tuple of (dataset, tmpdir_path).  Caller must clean up tmpdir.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    v3_dir = _create_lerobot_v3_dir(data_dir)

    with open(os.path.join(data_dir, "meta", "info.json")) as f:
        fps = json.load(f)["fps"]

    obs_timestamps = [-(OBS_LEN - 1 - i) / fps for i in range(OBS_LEN)]
    # youmu returns actions for the NEXT pred_len frames [t+1, ..., t+pred_len],
    # so lerobot delta_timestamps must start at 1/fps to match.
    pred_timestamps = [(i + 1) / fps for i in range(PRED_LEN)]

    ds = LeRobotDataset(
        repo_id="local/libero",
        root=v3_dir,
        delta_timestamps={
            # Use actual parquet column names (v2.0 flat naming)
            "state": obs_timestamps,
            "actions": pred_timestamps,
            "image": obs_timestamps,
        },
        download_videos=False,
    )
    return ds, v3_dir


def _get_test_indices(dataset_len: int) -> list[int]:
    """Return deterministic test indices: boundary + 5 random.

    Indices: [0, 1, 2, 7, 8, 9, len-3, len-2, len-1] plus 5 random.
    """
    boundary = [0, 1, 2, 7, 8, 9, dataset_len - 3, dataset_len - 2, dataset_len - 1]
    boundary = [i for i in boundary if 0 <= i < dataset_len]
    # Deterministic random indices
    rng = random.Random(42)
    random_indices = rng.sample(range(dataset_len), min(5, dataset_len))
    all_indices = sorted(set(boundary + random_indices))
    return all_indices


# ---------------------------------------------------------------------------
# Test class 1: Backend parity via direct dataset construction
# ---------------------------------------------------------------------------


@skip_no_youmu
@skip_no_lerobot
@skip_no_youmu_data
@skip_no_lerobot_data
class TestBackendParity:
    """Compare youmu and lerobot backends sample-by-sample.

    Loads the dataset with both backends and asserts that state, action,
    and image values are exactly equal for matching (episode, frame) pairs.

    Key differences handled:
    - youmu keys: observation.state, action, observation.images.image
    - lerobot keys: state, actions, image
    - youmu excludes last pred_len frames per episode; lerobot includes all with padding
    - youmu images: uint8 HWC; lerobot images: float32 CHW (converted for comparison)
    """

    @pytest.fixture(scope="class")
    def youmu_dataset(self):
        """Load dataset with youmu backend."""
        return _build_youmu_dataset(YOUMU_DATA_DIR)

    @pytest.fixture(scope="class")
    def lerobot_context(self):
        """Load dataset with lerobot backend (v3.0 wrapper).

        Yields dataset and cleans up the temp directory after.
        """
        ds, tmpdir = _build_lerobot_dataset(LEROBOT_DATA_DIR)
        yield ds
        shutil.rmtree(tmpdir, ignore_errors=True)

    def _get_lerobot_sample(self, youmu_sample, lerobot_ds):
        """Get the lerobot sample matching a youmu sample's global frame_index.

        youmu's frame_index is a global cumulative index across all episodes,
        which maps directly to lerobot's sequential dataset index.
        """
        frame_idx = youmu_sample["frame_index"]
        if isinstance(frame_idx, torch.Tensor):
            frame_idx = frame_idx.item()
        return lerobot_ds[frame_idx]

    def test_dataset_lengths_match(self, youmu_dataset, lerobot_context):
        """youmu has fewer frames than lerobot (clips episode tails).

        Verify the difference equals pred_len * num_episodes, confirming
        youmu excludes the last pred_len frames per episode while lerobot
        keeps all frames.
        """
        lerobot_ds = lerobot_context
        diff = len(lerobot_ds) - len(youmu_dataset)
        # Read episode count from metadata
        with open(os.path.join(LEROBOT_DATA_DIR, "meta", "episodes.jsonl")) as f:
            num_episodes = sum(1 for _ in f)
        expected_diff = PRED_LEN * num_episodes
        assert diff == expected_diff, (
            f"Length difference {diff} != expected {expected_diff} "
            f"(youmu={len(youmu_dataset)}, lerobot={len(lerobot_ds)}, "
            f"num_episodes={num_episodes}, pred_len={PRED_LEN})"
        )

    def test_state_parity(self, youmu_dataset, lerobot_context):
        """State tensors are exactly equal for matching (episode, frame) pairs."""
        lerobot_ds = lerobot_context
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = self._get_lerobot_sample(y_sample, lerobot_ds)
            # youmu: observation.state (obs_len, 8), lerobot: state (obs_len, 8)
            assert torch.equal(
                y_sample["observation.state"],
                l_sample["state"],
            ), f"State mismatch at youmu index {idx}"

    def test_action_parity(self, youmu_dataset, lerobot_context):
        """Action tensors are exactly equal for matching (episode, frame) pairs."""
        lerobot_ds = lerobot_context
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = self._get_lerobot_sample(y_sample, lerobot_ds)
            # youmu: action (pred_len, 7), lerobot: actions (pred_len, 7)
            assert torch.equal(
                y_sample["action"],
                l_sample["actions"],
            ), f"Action mismatch at youmu index {idx}"

    def test_image_parity(self, youmu_dataset, lerobot_context):
        """Image tensors are pixel-perfect identical after format normalization.

        youmu returns uint8 (obs_len, H, W, C); lerobot returns float32 (obs_len, C, H, W).
        We convert lerobot images to uint8 HWC for comparison.
        """
        lerobot_ds = lerobot_context
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = self._get_lerobot_sample(y_sample, lerobot_ds)
            # youmu: (obs_len, H, W, C) uint8
            y_img = y_sample["observation.images.image"]
            # lerobot: (obs_len, C, H, W) float32 [0, 1]
            l_img = l_sample["image"]
            # Convert lerobot CHW float32 → HWC uint8
            l_img_hwc = (l_img.permute(0, 2, 3, 1) * 255).to(torch.uint8)
            assert torch.equal(y_img, l_img_hwc), f"Image mismatch at youmu index {idx}"

    def test_episode_index_parity(self, youmu_dataset, lerobot_context):
        """episode_index values match for all test indices."""
        lerobot_ds = lerobot_context
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = self._get_lerobot_sample(y_sample, lerobot_ds)
            y_ep = y_sample["episode_index"]
            l_ep = l_sample["episode_index"]
            if isinstance(y_ep, torch.Tensor):
                y_ep = y_ep.item()
            if isinstance(l_ep, torch.Tensor):
                l_ep = l_ep.item()
            assert y_ep == l_ep, f"Episode index mismatch at youmu index {idx}: youmu={y_ep}, lerobot={l_ep}"


# ---------------------------------------------------------------------------
# Test class 2: Raw Parquet data parity (no backend needed)
# ---------------------------------------------------------------------------


@skip_no_youmu_data
@skip_no_lerobot_data
class TestRawParquetParity:
    """Verify underlying Parquet data is identical between libero and libero_64KB.

    Uses PyArrow directly — no youmu Rust extension or lerobot needed.
    This validates that the 64KB re-encoded dataset preserves data exactly.
    """

    def _episode_files(self, data_dir: str) -> list[str]:
        """List all episode parquet files sorted by episode index."""
        chunk_dirs = sorted(d for d in os.listdir(os.path.join(data_dir, "data")) if d.startswith("chunk-"))
        files = []
        for chunk_dir in chunk_dirs:
            chunk_path = os.path.join(data_dir, "data", chunk_dir)
            parquet_files = sorted(f for f in os.listdir(chunk_path) if f.endswith(".parquet"))
            files.extend(os.path.join(chunk_path, f) for f in parquet_files)
        return files

    def test_same_number_of_files(self):
        """Both datasets have the same number of parquet files."""
        youmu_files = self._episode_files(YOUMU_DATA_DIR)
        lerobot_files = self._episode_files(LEROBOT_DATA_DIR)
        assert len(youmu_files) == len(lerobot_files), (
            f"File count mismatch: youmu={len(youmu_files)}, lerobot={len(lerobot_files)}"
        )

    def test_same_row_counts(self):
        """Each corresponding file has the same number of rows."""
        youmu_files = self._episode_files(YOUMU_DATA_DIR)
        lerobot_files = self._episode_files(LEROBOT_DATA_DIR)
        for yf, lf in zip(youmu_files[:10], lerobot_files[:10]):
            y_rows = pq.ParquetFile(yf).metadata.num_rows
            l_rows = pq.ParquetFile(lf).metadata.num_rows
            assert y_rows == l_rows, f"Row count mismatch for {os.path.basename(yf)}: youmu={y_rows}, lerobot={l_rows}"

    def test_state_data_identical(self):
        """State column values are identical for first 10 episodes."""
        youmu_files = self._episode_files(YOUMU_DATA_DIR)
        lerobot_files = self._episode_files(LEROBOT_DATA_DIR)
        for yf, lf in zip(youmu_files[:10], lerobot_files[:10]):
            y_table = pq.read_table(yf, columns=["state"])
            l_table = pq.read_table(lf, columns=["state"])
            assert y_table.column("state").to_pylist() == l_table.column("state").to_pylist(), (
                f"State data mismatch in {os.path.basename(yf)}"
            )

    def test_actions_data_identical(self):
        """Actions column values are identical for first 10 episodes."""
        youmu_files = self._episode_files(YOUMU_DATA_DIR)
        lerobot_files = self._episode_files(LEROBOT_DATA_DIR)
        for yf, lf in zip(youmu_files[:10], lerobot_files[:10]):
            y_table = pq.read_table(yf, columns=["actions"])
            l_table = pq.read_table(lf, columns=["actions"])
            assert y_table.column("actions").to_pylist() == l_table.column("actions").to_pylist(), (
                f"Actions data mismatch in {os.path.basename(yf)}"
            )

    def test_image_bytes_identical(self):
        """Image bytes are identical for first row of first 5 episodes."""
        youmu_files = self._episode_files(YOUMU_DATA_DIR)
        lerobot_files = self._episode_files(LEROBOT_DATA_DIR)
        for yf, lf in zip(youmu_files[:5], lerobot_files[:5]):
            y_table = pq.read_table(yf, columns=["image"])
            l_table = pq.read_table(lf, columns=["image"])
            y_bytes = y_table.column("image").combine_chunks().field("bytes")[0].as_py()
            l_bytes = l_table.column("image").combine_chunks().field("bytes")[0].as_py()
            assert y_bytes == l_bytes, f"Image bytes mismatch in {os.path.basename(yf)}"

    def test_metadata_episode_count_matches(self):
        """Episode count in metadata matches between datasets."""
        y_meta = os.path.join(YOUMU_DATA_DIR, "meta", "episodes.jsonl")
        l_meta = os.path.join(LEROBOT_DATA_DIR, "meta", "episodes.jsonl")
        with open(y_meta) as f:
            y_episodes = [json.loads(line) for line in f]
        with open(l_meta) as f:
            l_episodes = [json.loads(line) for line in f]
        assert len(y_episodes) == len(l_episodes), (
            f"Episode count mismatch: youmu={len(y_episodes)}, lerobot={len(l_episodes)}"
        )

    def test_metadata_episode_lengths_match(self):
        """Episode lengths in metadata are identical."""
        y_meta = os.path.join(YOUMU_DATA_DIR, "meta", "episodes.jsonl")
        l_meta = os.path.join(LEROBOT_DATA_DIR, "meta", "episodes.jsonl")
        with open(y_meta) as f:
            y_episodes = sorted(
                [json.loads(line) for line in f],
                key=lambda e: e["episode_index"],
            )
        with open(l_meta) as f:
            l_episodes = sorted(
                [json.loads(line) for line in f],
                key=lambda e: e["episode_index"],
            )
        for ye, le in zip(y_episodes, l_episodes):
            assert ye["episode_index"] == le["episode_index"]
            assert ye["length"] == le["length"], (
                f"Length mismatch for episode {ye['episode_index']}: youmu={ye['length']}, lerobot={le['length']}"
            )

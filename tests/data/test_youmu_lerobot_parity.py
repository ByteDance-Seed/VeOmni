"""Exact numerical parity tests: youmu LiberoYoumuDataset vs LeRobot LeRobotDataset.

Verifies that both backends return identical observation states, actions, and
images for the same sample indices, giving confidence that training with
youmu produces the same results as LeRobot.

Two test classes:
1. TestBackendParity — loads datasets via build_libero_dataset with both
   backends and compares __getitem__ outputs.  Requires both youmu Rust
   extension *and* lerobot (with av) to be importable.
2. TestRawParquetParity — uses PyArrow directly to verify the underlying
   data in /mnt/local/localcache00/libero and libero_64KB are byte-identical.
   Runs whenever the data directories exist (no Rust or lerobot needed).
"""

import json
import os
import random

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
    reason="LeRobot package not importable (missing av or other deps)",
)
skip_no_youmu_data = pytest.mark.skipif(
    not _youmu_data_exists,
    reason=f"Youmu dataset not found at {YOUMU_DATA_DIR}",
)
skip_no_lerobot_data = pytest.mark.skipif(
    not _lerobot_data_exists,
    reason=f"LeRobot dataset not found at {LEROBOT_DATA_DIR}",
)


def _build_libero_dataset(backend: str, data_dir: str):
    """Build a LIBERO dataset using the training script's factory function.

    Imports build_libero_dataset from the training script via importlib
    to avoid triggering the full veomni import chain.
    """
    import importlib.util

    script_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "tasks",
        "omni",
        "train_qwen_vl_libero.py",
    )
    spec = importlib.util.spec_from_file_location("train_qwen_vl_libero", script_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod.build_libero_dataset(
        backend=backend,
        data_dir=data_dir,
        obs_len=OBS_LEN,
        pred_len=PRED_LEN,
        chunk_index=None,
    )


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
# Test class 1: Backend parity via build_libero_dataset
# ---------------------------------------------------------------------------


@skip_no_youmu
@skip_no_lerobot
@skip_no_youmu_data
@skip_no_lerobot_data
class TestBackendParity:
    """Compare youmu and lerobot backends sample-by-sample.

    Loads the dataset via build_libero_dataset for both backends and
    asserts that observation.state, action, and observation.images.image
    are exactly equal for a set of test indices.
    """

    @pytest.fixture(scope="class")
    def youmu_dataset(self):
        """Load dataset with youmu backend."""
        return _build_libero_dataset("youmu", YOUMU_DATA_DIR)

    @pytest.fixture(scope="class")
    def lerobot_dataset(self):
        """Load dataset with lerobot backend."""
        return _build_libero_dataset("lerobot", LEROBOT_DATA_DIR)

    def test_dataset_lengths_match(self, youmu_dataset, lerobot_dataset):
        """Both backends produce datasets of the same length."""
        assert len(youmu_dataset) == len(lerobot_dataset), (
            f"Length mismatch: youmu={len(youmu_dataset)}, lerobot={len(lerobot_dataset)}"
        )

    def test_state_parity(self, youmu_dataset, lerobot_dataset):
        """observation.state tensors are exactly equal for all test indices."""
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = lerobot_dataset[idx]
            assert torch.equal(y_sample["observation.state"], l_sample["observation.state"]), (
                f"State mismatch at index {idx}"
            )

    def test_action_parity(self, youmu_dataset, lerobot_dataset):
        """action tensors are exactly equal for all test indices."""
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = lerobot_dataset[idx]
            assert torch.equal(y_sample["action"], l_sample["action"]), f"Action mismatch at index {idx}"

    def test_image_parity(self, youmu_dataset, lerobot_dataset):
        """observation.images.image tensors are exactly equal (pixel-perfect)."""
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = lerobot_dataset[idx]
            assert torch.equal(
                y_sample["observation.images.image"],
                l_sample["observation.images.image"],
            ), f"Image mismatch at index {idx}"

    def test_episode_index_parity(self, youmu_dataset, lerobot_dataset):
        """episode_index values match for all test indices."""
        indices = _get_test_indices(len(youmu_dataset))
        for idx in indices:
            y_sample = youmu_dataset[idx]
            l_sample = lerobot_dataset[idx]
            assert y_sample["episode_index"] == l_sample["episode_index"], (
                f"Episode index mismatch at index {idx}: "
                f"youmu={y_sample['episode_index']}, lerobot={l_sample['episode_index']}"
            )


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

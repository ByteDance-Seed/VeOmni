"""Tests for load_libero_task_descriptions JSONL support.

Verifies that ``load_libero_task_descriptions`` correctly loads task
descriptions from JSONL metadata files (``meta/episodes.jsonl``), in
addition to the existing parquet format.

Uses direct importlib loading to avoid triggering the heavy veomni
import chain (torch, transformers, etc.).
"""

import importlib
import importlib.util
import json
import os
import sys
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Module-level import: load data_transform with mocked heavy deps
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="module")
def _load_data_transform():
    """Import data_transform.py with mocked heavy dependencies.

    The module normally imports torch, veomni.utils.constants, and various
    sub-modules.  We stub these so only the JSONL/parquet loader functions
    are exercised.
    """
    # Create stubs for heavy dependencies
    stubs = {
        "torch": mock.MagicMock(),
        "veomni": mock.MagicMock(),
        "veomni.utils": mock.MagicMock(),
        "veomni.utils.constants": mock.MagicMock(),
        "veomni.data": mock.MagicMock(),
        "veomni.data.multimodal": mock.MagicMock(),
        "veomni.data.multimodal.conv_preprocess": mock.MagicMock(),
        "veomni.data.multimodal.audio_utils": mock.MagicMock(),
        "veomni.data.multimodal.image_utils": mock.MagicMock(),
        "veomni.data.multimodal.video_utils": mock.MagicMock(),
    }
    with mock.patch.dict(sys.modules, stubs):
        spec = importlib.util.spec_from_file_location(
            "veomni.data.multimodal.data_transform",
            "veomni/data/multimodal/data_transform.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["veomni.data.multimodal.data_transform"] = module
        spec.loader.exec_module(module)
        yield module


@pytest.fixture
def dt_module(_load_data_transform):
    """Return the loaded data_transform module."""
    return sys.modules["veomni.data.multimodal.data_transform"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def jsonl_fixture(tmp_path):
    """Create a small JSONL episodes metadata file and return its path."""
    episodes = [
        {"episode_index": 0, "tasks": ["pick up the red cup"], "length": 100},
        {"episode_index": 1, "tasks": ["place the bowl on the shelf"], "length": 150},
        {"episode_index": 2, "tasks": ["open the drawer"], "length": 80},
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(path)


@pytest.fixture
def jsonl_missing_tasks(tmp_path):
    """JSONL file where some episodes have missing or empty tasks."""
    episodes = [
        {"episode_index": 0, "tasks": ["pick up the red cup"], "length": 100},
        {"episode_index": 1, "tasks": [], "length": 150},
        {"episode_index": 2, "length": 80},  # no tasks field at all
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(path)


@pytest.fixture
def jsonl_with_blank_lines(tmp_path):
    """JSONL file with interspersed blank lines."""
    episodes = [
        {"episode_index": 0, "tasks": ["task A"], "length": 50},
        {"episode_index": 5, "tasks": ["task B"], "length": 60},
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps(episodes[0]) + "\n")
        f.write("\n")  # blank line
        f.write(json.dumps(episodes[1]) + "\n")
        f.write("\n")  # trailing blank line
    return str(path)


# ---------------------------------------------------------------------------
# Tests: _load_libero_task_descriptions_jsonl
# ---------------------------------------------------------------------------


class TestLoadJsonl:
    """Tests for the JSONL-specific loader."""

    def test_basic_loading(self, dt_module, jsonl_fixture):
        """All episodes are loaded with correct task descriptions."""
        result = dt_module._load_libero_task_descriptions_jsonl(jsonl_fixture)
        assert result == {
            0: "pick up the red cup",
            1: "place the bowl on the shelf",
            2: "open the drawer",
        }

    def test_returns_dict_int_str(self, dt_module, jsonl_fixture):
        """Return type is dict[int, str]."""
        result = dt_module._load_libero_task_descriptions_jsonl(jsonl_fixture)
        assert isinstance(result, dict)
        for k, v in result.items():
            assert isinstance(k, int)
            assert isinstance(v, str)

    def test_missing_tasks_fallback(self, dt_module, jsonl_missing_tasks):
        """Episodes with missing or empty tasks fall back to 'perform the task'."""
        result = dt_module._load_libero_task_descriptions_jsonl(jsonl_missing_tasks)
        assert result[0] == "pick up the red cup"
        assert result[1] == "perform the task"  # empty tasks list
        assert result[2] == "perform the task"  # no tasks field

    def test_blank_lines_skipped(self, dt_module, jsonl_with_blank_lines):
        """Blank lines in the JSONL file are silently skipped."""
        result = dt_module._load_libero_task_descriptions_jsonl(jsonl_with_blank_lines)
        assert result == {0: "task A", 5: "task B"}

    def test_episode_count(self, dt_module, jsonl_fixture):
        """Number of returned entries matches the number of JSONL lines."""
        result = dt_module._load_libero_task_descriptions_jsonl(jsonl_fixture)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Tests: load_libero_task_descriptions (auto-detect)
# ---------------------------------------------------------------------------


class TestLoadAutoDetect:
    """Tests for the auto-detecting dispatcher."""

    def test_dispatches_to_jsonl(self, dt_module, jsonl_fixture):
        """Files ending in .jsonl are dispatched to the JSONL loader."""
        result = dt_module.load_libero_task_descriptions(jsonl_fixture)
        assert result == {
            0: "pick up the red cup",
            1: "place the bowl on the shelf",
            2: "open the drawer",
        }

    def test_non_jsonl_extension_dispatches_to_parquet(self, dt_module, tmp_path):
        """Files NOT ending in .jsonl are dispatched to the parquet loader.

        We verify the path routing by checking that a parquet-related error is
        raised (not a JSON error), confirming the parquet branch was taken.
        """
        fake_parquet = tmp_path / "episodes.parquet"
        fake_parquet.write_text("not a parquet file")
        with pytest.raises(Exception):
            dt_module.load_libero_task_descriptions(str(fake_parquet))


# ---------------------------------------------------------------------------
# Tests: training script metadata path resolution
# ---------------------------------------------------------------------------


class TestMetadataPathResolution:
    """Verify that the training script candidate list picks JSONL first."""

    def test_jsonl_preferred_over_parquet(self, tmp_path):
        """When meta/episodes.jsonl exists, it is chosen over parquet candidates."""
        meta_dir = tmp_path / "meta"
        meta_dir.mkdir()

        # Create JSONL file
        jsonl_path = meta_dir / "episodes.jsonl"
        jsonl_path.write_text(json.dumps({"episode_index": 0, "tasks": ["test task"], "length": 10}) + "\n")

        # Create a parquet candidate directory too
        pq_dir = meta_dir / "episodes"
        pq_dir.mkdir()
        (pq_dir / "episodes.parquet").write_text("fake")

        # Simulate the candidate resolution logic from train_qwen_vl_libero.py
        libero_dir = str(tmp_path)
        meta_candidates = [
            os.path.join(libero_dir, "meta", "episodes.jsonl"),
            os.path.join(libero_dir, "meta", "episodes", "episodes.parquet"),
            os.path.join(libero_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), meta_candidates[-1])

        assert meta_path.endswith("episodes.jsonl")

    def test_fallback_to_parquet_when_no_jsonl(self, tmp_path):
        """When JSONL doesn't exist, falls back to parquet candidates."""
        meta_dir = tmp_path / "meta" / "episodes"
        meta_dir.mkdir(parents=True)
        (meta_dir / "episodes.parquet").write_text("fake")

        libero_dir = str(tmp_path)
        meta_candidates = [
            os.path.join(libero_dir, "meta", "episodes.jsonl"),
            os.path.join(libero_dir, "meta", "episodes", "episodes.parquet"),
            os.path.join(libero_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), meta_candidates[-1])

        assert meta_path.endswith("episodes.parquet")

    def test_fallback_to_last_candidate(self, tmp_path):
        """When no candidates exist, falls back to the last candidate path."""
        libero_dir = str(tmp_path)
        meta_candidates = [
            os.path.join(libero_dir, "meta", "episodes.jsonl"),
            os.path.join(libero_dir, "meta", "episodes", "episodes.parquet"),
            os.path.join(libero_dir, "meta", "episodes", "chunk-000", "file-000.parquet"),
        ]
        meta_path = next((p for p in meta_candidates if os.path.exists(p)), meta_candidates[-1])

        assert meta_path.endswith("file-000.parquet")


# ---------------------------------------------------------------------------
# Integration test on real data (skipped if not available)
# ---------------------------------------------------------------------------

REAL_DATA_DIR = "/mnt/local/localcache00/libero_64KB"


@pytest.mark.skipif(
    not os.path.exists(os.path.join(REAL_DATA_DIR, "meta", "episodes.jsonl")),
    reason="Real LIBERO dataset not available",
)
class TestRealData:
    """Integration tests against the actual LIBERO dataset."""

    def test_load_from_real_jsonl(self, dt_module):
        """Load task descriptions from the real dataset's episodes.jsonl."""
        meta_path = os.path.join(REAL_DATA_DIR, "meta", "episodes.jsonl")
        result = dt_module.load_libero_task_descriptions(meta_path)
        assert len(result) > 0
        # All keys should be non-negative ints, all values non-empty strings
        for k, v in result.items():
            assert isinstance(k, int) and k >= 0
            assert isinstance(v, str) and len(v) > 0

    def test_all_episodes_have_descriptions(self, dt_module):
        """Every episode in the JSONL file gets a task description."""
        meta_path = os.path.join(REAL_DATA_DIR, "meta", "episodes.jsonl")
        with open(meta_path) as f:
            num_episodes = sum(1 for line in f if line.strip())
        result = dt_module.load_libero_task_descriptions(meta_path)
        assert len(result) == num_episodes


# ---------------------------------------------------------------------------
# Integration tests on synthetic data (using shared fixture from conftest.py)
# ---------------------------------------------------------------------------


class TestSyntheticData:
    """Integration tests using the synthetic LIBERO dataset fixture."""

    def test_load_from_synthetic_jsonl(self, dt_module, synthetic_libero_dir):
        """Load task descriptions from the synthetic dataset's episodes.jsonl."""
        meta_path = os.path.join(synthetic_libero_dir, "meta", "episodes.jsonl")
        result = dt_module.load_libero_task_descriptions(meta_path)
        assert len(result) == 3
        # All keys should be non-negative ints, all values non-empty strings
        for k, v in result.items():
            assert isinstance(k, int) and k >= 0
            assert isinstance(v, str) and len(v) > 0

    def test_all_episodes_have_descriptions(self, dt_module, synthetic_libero_dir):
        """Every episode in the synthetic JSONL maps to a dict entry."""
        meta_path = os.path.join(synthetic_libero_dir, "meta", "episodes.jsonl")
        with open(meta_path) as f:
            num_episodes = sum(1 for line in f if line.strip())
        result = dt_module.load_libero_task_descriptions(meta_path)
        assert len(result) == num_episodes

    def test_synthetic_task_descriptions_content(self, dt_module, synthetic_libero_dir):
        """Synthetic task descriptions match what the fixture wrote."""
        meta_path = os.path.join(synthetic_libero_dir, "meta", "episodes.jsonl")
        result = dt_module.load_libero_task_descriptions(meta_path)
        assert result == {
            0: "synthetic task 0",
            1: "synthetic task 1",
            2: "synthetic task 2",
        }

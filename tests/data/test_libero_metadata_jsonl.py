"""Tests for JSONL-based episode metadata loading in youmu.libero_utils.

Imports libero_utils directly from the submodule Python source to avoid
needing the compiled Rust extension (youmu.youmu).
"""

import importlib
import json
import sys
from unittest import mock

import pytest


@pytest.fixture(autouse=True, scope="module")
def _patch_youmu_import():
    """Stub the youmu Rust extension so pure-Python utils can be imported."""
    # The youmu __init__.py does 'from .youmu import *' which needs the Rust .so.
    # We stub the top-level youmu package and then import the utils module directly.
    fake_youmu = mock.MagicMock()
    with mock.patch.dict(sys.modules, {"youmu": fake_youmu, "youmu.youmu": mock.MagicMock()}):
        spec = importlib.util.spec_from_file_location(
            "youmu.libero_utils",
            "submodules/youmu/python/youmu/libero_utils.py",
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules["youmu.libero_utils"] = module
        spec.loader.exec_module(module)
        yield


# Import after the fixture stubs youmu
def _get_utils():
    return sys.modules["youmu.libero_utils"]


@pytest.fixture
def jsonl_fixture(tmp_path):
    """Create a small episodes.jsonl fixture file and return its path."""
    episodes = [
        {"episode_index": 0, "tasks": ["task A"], "length": 100},
        {"episode_index": 1, "tasks": ["task B"], "length": 200},
        {"episode_index": 2, "tasks": ["task C"], "length": 150},
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(path)


@pytest.fixture
def jsonl_fixture_unordered(tmp_path):
    """Create a JSONL fixture with episodes out of order."""
    episodes = [
        {"episode_index": 2, "tasks": ["task C"], "length": 150},
        {"episode_index": 0, "tasks": ["task A"], "length": 100},
        {"episode_index": 1, "tasks": ["task B"], "length": 200},
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(path)


@pytest.fixture
def multi_chunk_jsonl_fixture(tmp_path):
    """Create a JSONL fixture simulating multi-chunk dataset."""
    episodes = [
        {"episode_index": 0, "tasks": ["task A"], "length": 100},
        {"episode_index": 1, "tasks": ["task B"], "length": 200},
        {"episode_index": 2, "tasks": ["task C"], "length": 150},
        {"episode_index": 3, "tasks": ["task D"], "length": 80},
        {"episode_index": 4, "tasks": ["task E"], "length": 120},
    ]
    path = tmp_path / "episodes.jsonl"
    with open(path, "w") as f:
        for ep in episodes:
            f.write(json.dumps(ep) + "\n")
    return str(path)


class TestLoadEpisodeMetadataJsonl:
    """Tests for load_episode_metadata_jsonl."""

    def test_basic_loading(self, jsonl_fixture):
        """Verify correct EpisodeDescriptor values for a small JSONL fixture."""
        utils = _get_utils()
        episodes = utils.load_episode_metadata_jsonl(jsonl_fixture)

        assert len(episodes) == 3

        # Episode 0: start=0, end=100, length=100
        assert episodes[0].episode_index == 0
        assert episodes[0].file_index == 0
        assert episodes[0].start_frame == 0
        assert episodes[0].end_frame == 100
        assert episodes[0].length == 100

        # Episode 1: start=100, end=300, length=200
        assert episodes[1].episode_index == 1
        assert episodes[1].file_index == 1
        assert episodes[1].start_frame == 100
        assert episodes[1].end_frame == 300
        assert episodes[1].length == 200

        # Episode 2: start=300, end=450, length=150
        assert episodes[2].episode_index == 2
        assert episodes[2].file_index == 2
        assert episodes[2].start_frame == 300
        assert episodes[2].end_frame == 450
        assert episodes[2].length == 150

    def test_sorted_by_episode_index(self, jsonl_fixture_unordered):
        """Verify episodes are sorted by episode_index even when JSONL is unordered."""
        utils = _get_utils()
        episodes = utils.load_episode_metadata_jsonl(jsonl_fixture_unordered)

        assert [e.episode_index for e in episodes] == [0, 1, 2]
        # Frame offsets should be computed from sorted order
        assert episodes[0].start_frame == 0
        assert episodes[0].end_frame == 100
        assert episodes[1].start_frame == 100
        assert episodes[1].end_frame == 300
        assert episodes[2].start_frame == 300
        assert episodes[2].end_frame == 450

    def test_file_index_equals_episode_index(self, jsonl_fixture):
        """Verify file_index equals episode_index for episode-per-file layout."""
        utils = _get_utils()
        episodes = utils.load_episode_metadata_jsonl(jsonl_fixture)
        for ep in episodes:
            assert ep.file_index == ep.episode_index

    def test_multi_chunk(self, multi_chunk_jsonl_fixture):
        """Verify correct frame boundaries for multi-chunk datasets."""
        utils = _get_utils()
        episodes = utils.load_episode_metadata_jsonl(multi_chunk_jsonl_fixture)

        assert len(episodes) == 5
        # Cumulative: 0, 100, 300, 450, 530, 650
        assert episodes[3].start_frame == 450
        assert episodes[3].end_frame == 530
        assert episodes[4].start_frame == 530
        assert episodes[4].end_frame == 650

    def test_empty_lines_ignored(self, tmp_path):
        """Verify empty lines in the JSONL are skipped."""
        utils = _get_utils()
        path = tmp_path / "episodes.jsonl"
        with open(path, "w") as f:
            f.write('{"episode_index": 0, "tasks": ["t"], "length": 50}\n')
            f.write("\n")
            f.write('{"episode_index": 1, "tasks": ["t"], "length": 60}\n')
            f.write("\n")
        episodes = utils.load_episode_metadata_jsonl(str(path))
        assert len(episodes) == 2
        assert episodes[0].length == 50
        assert episodes[1].length == 60


class TestLoadEpisodeMetadataAutoDetect:
    """Tests for load_episode_metadata auto-detection."""

    def test_jsonl_extension_detected(self, jsonl_fixture):
        """Verify .jsonl extension triggers JSONL loading."""
        utils = _get_utils()
        episodes = utils.load_episode_metadata(jsonl_fixture)
        assert len(episodes) == 3
        assert episodes[0].episode_index == 0

    def test_missing_parquet_falls_back_to_jsonl(self, tmp_path):
        """Verify fallback from missing parquet to JSONL in meta/ directory."""
        utils = _get_utils()
        # Create directory structure: meta/episodes/chunk-000/
        meta_dir = tmp_path / "meta"
        episodes_dir = meta_dir / "episodes" / "chunk-000"
        episodes_dir.mkdir(parents=True)

        # Create JSONL in meta/
        jsonl_path = meta_dir / "episodes.jsonl"
        with open(jsonl_path, "w") as f:
            f.write('{"episode_index": 0, "tasks": ["t"], "length": 42}\n')

        # Try loading with non-existent parquet path
        parquet_path = str(episodes_dir / "file-000.parquet")
        episodes = utils.load_episode_metadata(parquet_path)
        assert len(episodes) == 1
        assert episodes[0].length == 42

    def test_missing_both_raises(self, tmp_path):
        """Verify FileNotFoundError when neither parquet nor JSONL exists."""
        utils = _get_utils()
        meta_dir = tmp_path / "meta" / "episodes" / "chunk-000"
        meta_dir.mkdir(parents=True)
        parquet_path = str(meta_dir / "file-000.parquet")
        with pytest.raises(FileNotFoundError):
            utils.load_episode_metadata(parquet_path)

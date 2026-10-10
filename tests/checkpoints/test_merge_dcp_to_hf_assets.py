import importlib.util
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint import save
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME


ROOT = Path(__file__).resolve().parents[2]


def _load_script():
    spec = importlib.util.spec_from_file_location("merge_dcp_to_hf", ROOT / "scripts/merge_dcp_to_hf.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_merge_preserves_generation_config(tmp_path, monkeypatch):
    module = _load_script()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "generation_config.json").write_text("{}", encoding="utf-8")
    loaded = []

    class FakeAsset:
        def __init__(self, kind):
            self.kind = kind

    class FakeLoader:
        def __init__(self, kind):
            self.kind = kind

        def from_pretrained(self, path, **kwargs):
            assert path == str(assets_dir)
            loaded.append(self.kind)
            return FakeAsset(self.kind)

    captured = {}

    def fake_save_model_weights(
        output_dir, checkpoint_path, *, shard_size, model_assets, local_staging_dir, num_workers, aliased_keys
    ):
        captured.update(
            output_dir=output_dir,
            checkpoint_path=checkpoint_path,
            shard_size=shard_size,
            assets=[asset.kind for asset in model_assets],
            local_staging_dir=local_staging_dir,
            num_workers=num_workers,
        )

    monkeypatch.setattr(module, "AutoConfig", FakeLoader("config"))
    monkeypatch.setattr(module, "AutoProcessor", FakeLoader("processor"))
    monkeypatch.setattr(module, "GenerationConfig", FakeLoader("generation_config"))
    monkeypatch.setattr(module, "save_model_weights", fake_save_model_weights)

    module.merge_to_hf_pt(
        "dcp",
        "hf",
        str(assets_dir),
        shard_size=5_000_000_000,
        local_staging_dir="/tmp/staging",
        num_workers=3,
    )

    assert loaded == ["config", "processor", "generation_config"]
    assert captured == {
        "output_dir": "hf",
        "checkpoint_path": "dcp",
        "shard_size": 5_000_000_000,
        "assets": ["config", "processor", "generation_config"],
        "local_staging_dir": "/tmp/staging",
        "num_workers": 3,
    }


def test_merge_rejects_invalid_generation_config_before_weight_conversion(tmp_path, monkeypatch):
    module = _load_script()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "generation_config.json").write_text("{bad", encoding="utf-8")

    class SuccessfulLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class BrokenGenerationConfig:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("invalid generation config")

    monkeypatch.setattr(module, "AutoConfig", SuccessfulLoader)
    monkeypatch.setattr(module, "AutoProcessor", SuccessfulLoader)
    monkeypatch.setattr(module, "GenerationConfig", BrokenGenerationConfig)
    monkeypatch.setattr(
        module,
        "save_model_weights",
        lambda *args, **kwargs: pytest.fail("weight conversion must not start with invalid generation config"),
    )

    with pytest.raises(RuntimeError, match="failed to load the generation config"):
        module.merge_to_hf_pt("dcp", "hf", str(assets_dir))


def test_merge_falls_back_to_tokenizer_when_assets_have_no_processor(tmp_path, monkeypatch):
    module = _load_script()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    loaded = []

    class FakeAsset:
        def __init__(self, kind):
            self.kind = kind

    class ConfigLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(assets_dir)
            loaded.append("config")
            return FakeAsset("config")

    class MissingProcessor:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(assets_dir)
            loaded.append("processor")
            raise OSError("preprocessor_config.json is absent")

    class TokenizerLoader:
        @staticmethod
        def from_pretrained(path, **kwargs):
            assert path == str(assets_dir)
            loaded.append("tokenizer")
            return FakeAsset("tokenizer")

    captured = {}

    def fake_save_model_weights(
        output_dir, checkpoint_path, *, shard_size, model_assets, local_staging_dir, num_workers, aliased_keys
    ):
        captured["assets"] = [asset.kind for asset in model_assets]

    monkeypatch.setattr(module, "AutoConfig", ConfigLoader)
    monkeypatch.setattr(module, "AutoProcessor", MissingProcessor)
    monkeypatch.setattr(module, "AutoTokenizer", TokenizerLoader)
    monkeypatch.setattr(module, "save_model_weights", fake_save_model_weights)

    module.merge_to_hf_pt("dcp", "hf", str(assets_dir))

    assert loaded == ["config", "processor", "tokenizer"]
    assert captured["assets"] == ["config", "tokenizer"]


def test_merge_rejects_unloadable_tokenizer_assets_before_weight_conversion(tmp_path, monkeypatch):
    module = _load_script()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "tokenizer.json").write_text("{bad", encoding="utf-8")

    class ConfigLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class BrokenLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("invalid tokenizer assets")

    monkeypatch.setattr(module, "AutoConfig", ConfigLoader)
    monkeypatch.setattr(module, "AutoProcessor", BrokenLoader)
    monkeypatch.setattr(module, "AutoTokenizer", BrokenLoader)
    monkeypatch.setattr(
        module,
        "save_model_weights",
        lambda *args, **kwargs: pytest.fail("weight conversion must not start with invalid tokenizer assets"),
    )

    with pytest.raises(RuntimeError, match="failed to load the tokenizer"):
        module.merge_to_hf_pt("dcp", "hf", str(assets_dir))


def test_merge_rejects_broken_processor_assets_instead_of_downgrading(tmp_path, monkeypatch):
    module = _load_script()
    assets_dir = tmp_path / "assets"
    assets_dir.mkdir()
    (assets_dir / "preprocessor_config.json").write_text("{bad", encoding="utf-8")

    class ConfigLoader:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return object()

    class BrokenProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ValueError("invalid processor assets")

    class WorkingTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            pytest.fail("processor-backed assets must not downgrade to tokenizer-only")

    monkeypatch.setattr(module, "AutoConfig", ConfigLoader)
    monkeypatch.setattr(module, "AutoProcessor", BrokenProcessor)
    monkeypatch.setattr(module, "AutoTokenizer", WorkingTokenizer)
    monkeypatch.setattr(
        module,
        "save_model_weights",
        lambda *args, **kwargs: pytest.fail("weight conversion must not start with invalid processor assets"),
    )

    with pytest.raises(RuntimeError, match="failed to load the processor"):
        module.merge_to_hf_pt("dcp", "hf", str(assets_dir))


def test_state_dict_can_stage_locally_before_streaming_to_destination(tmp_path, monkeypatch):
    module = _load_script()
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "mounted-output"
    output_dir.mkdir()
    destination = output_dir / "model.safetensors"
    staged_paths = []

    def fake_save_file(state_dict, path, metadata):
        assert state_dict == {"weight": "tensor"}
        assert metadata == {"format": "pt"}
        staged_path = Path(path)
        assert staged_path.parent == staging_dir
        staged_paths.append(staged_path)
        staged_path.write_bytes(b"safetensors payload")

    monkeypatch.setattr(module, "save_file", fake_save_file)

    module._save_state_dict(
        {"weight": "tensor"},
        destination,
        safe_serialization=True,
        local_staging_dir=staging_dir,
    )

    assert destination.read_bytes() == b"safetensors payload"
    assert len(staged_paths) == 1
    assert not staged_paths[0].exists()
    assert list(staging_dir.iterdir()) == []


def test_staged_copy_failure_preserves_existing_destination_and_cleans_partials(tmp_path, monkeypatch):
    module = _load_script()
    staging_dir = tmp_path / "staging"
    output_dir = tmp_path / "mounted-output"
    output_dir.mkdir()
    destination = output_dir / "model.safetensors"
    destination.write_bytes(b"existing valid shard")

    def fake_save_file(state_dict, path, metadata):
        Path(path).write_bytes(b"new safetensors payload")

    def fail_copy(source, target, length):
        target.write(b"partial")
        raise OSError("simulated destination failure")

    monkeypatch.setattr(module, "save_file", fake_save_file)
    monkeypatch.setattr(module.shutil, "copyfileobj", fail_copy)

    with pytest.raises(OSError, match="simulated destination failure"):
        module._save_state_dict(
            {"weight": "tensor"},
            destination,
            safe_serialization=True,
            local_staging_dir=staging_dir,
        )

    assert destination.read_bytes() == b"existing valid shard"
    assert list(staging_dir.iterdir()) == []
    assert not list(output_dir.glob("*.incomplete-*"))


def test_aliased_keys_of_a_tied_lm_head_are_just_the_lm_head():
    module = _load_script()

    aliased = module._aliased_hf_keys(str(ROOT / "tests/toy_config/qwen2_toy"))

    assert aliased == {"lm_head.weight"}


def test_aliased_keys_are_unknown_when_the_model_cannot_be_resolved(tmp_path):
    module = _load_script()

    assert module._aliased_hf_keys(str(tmp_path)) is None


@pytest.mark.skipif(
    not (ROOT / "tests/toy_config/diffusion_gemma_toy").is_dir(), reason="needs the DiffusionGemma toy config"
)
def test_aliased_keys_name_the_duplicates_and_nothing_else():
    """The export must be keyed one name per tensor, not one name per state-dict entry.

    DiffusionGemma's encoder *is* its decoder, so 628 of its 1675 state-dict names are
    aliases. Exporting them doubles the checkpoint, and under expert parallel the alias
    FQNs used to be written at `[E/ep, ...]`, which made the result unloadable. Deriving
    the key set from `named_parameters`, which deduplicates, is what keeps the export
    matching the published layout.
    """
    module = _load_script()
    toy_config = ROOT / "tests/toy_config/diffusion_gemma_toy"

    aliased = module._aliased_hf_keys(str(toy_config))
    assert aliased, "the toy config must resolve, otherwise nothing is filtered"

    # Tied to `model.decoder.embed_tokens.weight`; every model with `tie_word_embeddings`
    # has this one, so it is not a DiffusionGemma quirk.
    assert "lm_head.weight" in aliased
    assert "model.encoder.language_model.layers.0.experts.gate_up_proj" in aliased

    # The encoder's `layer_scalar` buffers are genuinely its own, not a second name for
    # a decoder tensor, so dropping them would lose weights.
    assert not any(key.endswith(".layer_scalar") for key in aliased), sorted(aliased)[:5]

    # Nothing canonical may be named for dropping.
    assert "model.decoder.embed_tokens.weight" not in aliased
    assert "model.decoder.layers.0.experts.gate_up_proj" not in aliased


def test_the_sharding_plan_drops_only_the_named_aliases(tmp_path):
    """`drop_hf_keys` has to filter the plan, not just annotate it.

    Phrased as a drop-list rather than a keep-list on purpose: the DCP and the model
    agree on key names only by convention, and a keep-list turns any mismatch into
    silently pruned weights.
    """
    from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan

    checkpoint = tmp_path / "checkpoint"
    save(
        {
            "model.model.decoder.weight": torch.arange(4, dtype=torch.float32),
            "model.model.encoder.weight": torch.arange(4, dtype=torch.float32),
        },
        checkpoint_id=str(checkpoint),
    )

    unfiltered, unfiltered_size, _ = _get_sharding_plan(str(checkpoint), None, "float32")
    assert set(unfiltered) == {"model.decoder.weight", "model.encoder.weight"}

    filtered, filtered_size, _ = _get_sharding_plan(
        str(checkpoint), None, "float32", drop_hf_keys={"model.encoder.weight"}
    )
    assert set(filtered) == {"model.decoder.weight"}
    # The reported total has to follow the plan, or the index advertises bytes nobody wrote.
    assert filtered_size == unfiltered_size // 2

    # A name the checkpoint does not have must not remove anything.
    untouched, untouched_size, _ = _get_sharding_plan(
        str(checkpoint), None, "float32", drop_hf_keys={"model.absent.weight"}
    )
    assert set(untouched) == set(unfiltered)
    assert untouched_size == unfiltered_size


def test_cpu_only_parallel_conversion_runs_the_real_dcp_shard_path(tmp_path):
    """Smoke the real CLI with CUDA hidden from worker processes."""
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    staging = tmp_path / "staging"
    save(
        {
            "model.model.first.weight": torch.arange(4, dtype=torch.float32),
            "model.model.second.weight": torch.arange(4, 8, dtype=torch.float32),
        },
        checkpoint_id=str(checkpoint),
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/merge_dcp_to_hf.py"),
            "--load-dir",
            str(checkpoint),
            "--save-dir",
            str(output),
            "--shard-size",
            "8",
            "--num-workers",
            "2",
            "--local-staging-dir",
            str(staging),
            "--mode",
            "full",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (output / SAFE_WEIGHTS_INDEX_NAME).is_file()
    assert len(list(output.glob("model-*-of-*.safetensors"))) == 2
    assert list(staging.iterdir()) == []


def test_parallel_conversion_survives_parent_openmp_initialization(tmp_path):
    """Regression for conversion after the parent has initialized OpenMP.

    Forking here can leave every child blocked in libgomp during DCP's tensor copy.
    The production converter must use a clean spawned runtime for each worker.
    """
    checkpoint = tmp_path / "checkpoint"
    output = tmp_path / "hf"
    staging = tmp_path / "staging"
    save(
        {
            "model.model.first.weight": torch.arange(1_048_576, dtype=torch.float32),
            "model.model.second.weight": torch.arange(1_048_576, 2_097_152, dtype=torch.float32),
        },
        checkpoint_id=str(checkpoint),
    )
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    environment["OMP_NUM_THREADS"] = "4"
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(ROOT / "scripts"), environment.get("PYTHONPATH")) if path
    )

    command = """
import sys

import torch

from merge_dcp_to_hf import merge_to_hf_pt

torch.set_num_threads(4)
source = torch.ones(8_388_608)
destination = torch.empty_like(source)
destination.copy_(source)
merge_to_hf_pt(
    sys.argv[1],
    sys.argv[2],
    shard_size=2_097_152,
    local_staging_dir=sys.argv[3],
    num_workers=2,
)
"""

    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            command,
            str(checkpoint),
            str(output),
            str(staging),
        ],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        stdout, stderr = process.communicate()
        pytest.fail(f"parallel conversion timed out\nstdout:\n{stdout}\nstderr:\n{stderr}")

    assert process.returncode == 0, stderr
    assert (output / SAFE_WEIGHTS_INDEX_NAME).is_file()
    assert len(list(output.glob("model-*-of-*.safetensors"))) == 2
    assert list(staging.iterdir()) == []

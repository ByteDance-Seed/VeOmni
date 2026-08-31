"""Unit tests for :class:`veomni.models.checkpoint_manager.ModelCheckpointManager`.

The manager is the piece a multi-module model (SeedOmni V2) subclasses, so the
tests pin the two things a subclass depends on: the directory policy and the
hook that nests every artifact under a module name.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from veomni.models.checkpoint_manager import ModelCheckpointManager
from veomni.trainer.callbacks.base import TrainerState


def _make_runtime(lora_config=None):
    return SimpleNamespace(
        model=MagicMock(),
        optimizer=MagicMock(),
        lr_scheduler=MagicMock(),
        parallel_state=SimpleNamespace(global_rank=0),
        args=SimpleNamespace(
            lora_config=lora_config or {},
            fqn_to_index_mapping={},
            accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="fsdp2")),
        ),
    )


def _make_config(load_path=None):
    return SimpleNamespace(
        save_path="/ckpt/checkpoints",
        output_dir="/ckpt",
        manager="dcp",
        save_async=False,
        dcp_save_to_lowest_rank=False,
        load_path=load_path,
    )


@pytest.fixture
def make_manager():
    with patch("veomni.models.checkpoint_manager.build_checkpointer") as build:
        build.return_value = MagicMock()

        def _make(cls=ModelCheckpointManager, *, lora_config=None, load_path=None):
            return cls(_make_runtime(lora_config), _make_config(load_path))

        yield _make


class PerModuleManager(ModelCheckpointManager):
    """Stands in for SeedOmni V2's per-module manager."""

    checkpoint_subfolder = "vision_encoder"


class TestWhereArtifactsLand:
    def test_a_single_model_job_writes_straight_under_the_step_directory(self, make_manager):
        manager = make_manager()
        state = TrainerState(global_step=42)

        assert manager.save_dir(state) == "/ckpt/checkpoints/global_step_42"
        assert manager.hf_export_dir(state) == "/ckpt/checkpoints/global_step_42/hf_ckpt"
        assert manager.output_dir(state) == "/ckpt/global_step_42"

    def test_a_module_subfolder_nests_every_artifact_one_level_deeper(self, make_manager):
        """The hook a multi-module model overrides; each module owns its own directory."""
        manager = make_manager(PerModuleManager)
        state = TrainerState(global_step=42)

        assert manager.save_dir(state) == "/ckpt/checkpoints/global_step_42/vision_encoder"
        assert manager.hf_export_dir(state) == "/ckpt/checkpoints/global_step_42/vision_encoder/hf_ckpt"
        assert manager.output_dir(state) == "/ckpt/global_step_42/vision_encoder"

    def test_resume_reads_the_load_path_as_given(self, make_manager):
        assert (
            make_manager(load_path="/ckpt/checkpoints/global_step_7").load_dir() == "/ckpt/checkpoints/global_step_7"
        )

    def test_resume_reads_a_modules_own_subdirectory(self, make_manager):
        manager = make_manager(PerModuleManager, load_path="/ckpt/checkpoints/global_step_7")
        assert manager.load_dir() == "/ckpt/checkpoints/global_step_7/vision_encoder"


class TestResume:
    def test_a_fresh_run_loads_nothing(self, make_manager):
        manager = make_manager()

        assert manager.load() is None
        manager.checkpointer.load.assert_not_called()

    def test_resume_restores_the_models_own_scheduler(self, make_manager):
        manager = make_manager(load_path="/ckpt/checkpoints/global_step_7")

        def fake_load(path, state, **kwargs):
            state["extra_state"] = {"lr_scheduler": {"last_epoch": 7}}

        manager.checkpointer.load.side_effect = fake_load

        with patch("veomni.models.checkpoint_manager.dist"):
            manager.load()

        manager.runtime.lr_scheduler.load_state_dict.assert_called_once_with({"last_epoch": 7})


class TestWhatRidesAlongWithTheWeights:
    def test_a_model_always_stores_its_own_scheduler(self, make_manager):
        manager = make_manager()
        manager.runtime.lr_scheduler.state_dict.return_value = {"last_epoch": 3}

        with patch("veomni.models.checkpoint_manager.dist"), patch("veomni.models.checkpoint_manager.helper"):
            manager.save_dcp(TrainerState(global_step=10))

        assert manager.checkpointer.save.call_args.args[1]["extra_state"] == {"lr_scheduler": {"last_epoch": 3}}

    def test_nothing_job_level_rides_along(self, make_manager):
        """Job state has its own writer. With one model per module there is one
        dataloader cursor but N of these checkpoints, so a cursor stored here
        would be written N times over — and read back N times on resume."""
        manager = make_manager()
        manager.runtime.lr_scheduler.state_dict.return_value = {"last_epoch": 3}

        with patch("veomni.models.checkpoint_manager.dist"), patch("veomni.models.checkpoint_manager.helper"):
            manager.save_dcp(TrainerState(global_step=10))

        assert set(manager.checkpointer.save.call_args.args[1]["extra_state"]) == {"lr_scheduler"}


class TestExport:
    def test_export_saves_the_step_first_when_it_is_missing(self, make_manager):
        manager = make_manager()
        manager.save_dcp = MagicMock()

        with (
            patch("veomni.models.checkpoint_manager.dist"),
            patch("veomni.models.checkpoint_manager.os.path.exists", return_value=False),
        ):
            manager._prepare_export(TrainerState(global_step=10), stage="step_end")

        manager.save_dcp.assert_called_once()

    def test_export_reuses_a_step_that_is_already_on_disk(self, make_manager):
        manager = make_manager()
        manager.save_dcp = MagicMock()

        with (
            patch("veomni.models.checkpoint_manager.dist"),
            patch("veomni.models.checkpoint_manager.os.path.exists", return_value=True),
        ):
            manager._prepare_export(TrainerState(global_step=10), stage="step_end")

        manager.save_dcp.assert_not_called()

    def test_the_final_export_drops_the_optimizer_to_free_memory(self, make_manager):
        manager = make_manager()

        with (
            patch("veomni.models.checkpoint_manager.dist"),
            patch("veomni.models.checkpoint_manager.os.path.exists", return_value=True),
        ):
            manager._prepare_export(TrainerState(global_step=10), stage="train_end")

        assert manager.runtime.optimizer is None
        assert manager.runtime.lr_scheduler is None

    def test_a_mid_training_export_keeps_the_optimizer(self, make_manager):
        manager = make_manager()

        with (
            patch("veomni.models.checkpoint_manager.dist"),
            patch("veomni.models.checkpoint_manager.os.path.exists", return_value=True),
        ):
            manager._prepare_export(TrainerState(global_step=10), stage="step_end")

        assert manager.runtime.optimizer is not None


class TestFormatSelection:
    def test_a_lora_run_exports_an_adapter(self, make_manager):
        manager = make_manager(lora_config={"rank": 8})
        manager.save_lora = MagicMock()
        manager.save_hf = MagicMock()

        manager.save_hf_or_lora(TrainerState(global_step=10))

        manager.save_lora.assert_called_once()
        manager.save_hf.assert_not_called()

    def test_a_full_run_exports_safetensors(self, make_manager):
        manager = make_manager()
        manager.save_lora = MagicMock()
        manager.save_hf = MagicMock()

        manager.save_hf_or_lora(TrainerState(global_step=10))

        manager.save_hf.assert_called_once()
        manager.save_lora.assert_not_called()

    def test_a_lora_run_checkpoints_only_the_adapters(self, make_manager):
        assert make_manager(lora_config={"rank": 8}).trainable_only is True

    def test_a_full_run_checkpoints_everything(self, make_manager):
        assert make_manager().trainable_only is False

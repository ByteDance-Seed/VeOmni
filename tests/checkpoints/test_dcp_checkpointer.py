"""Unit tests for distributed checkpoint and resume behavior.

Covers: OptimizerState (no placeholder synthesis), key normalization,
lr_scheduler sidecar persistence, allow_partial_load planner, skip-HF resume, and
trainer step-counting correctness. Tests marked ``xfail`` document known
in-tree bugs — they become regression guards once the fix lands.
"""

import inspect
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
import torch.nn as nn
from torch.distributed.fsdp import fully_shard

from veomni.distributed import torch_parallelize
from veomni.distributed.parallel_state import _init_parallel_state, get_parallel_state
from veomni.distributed.torch_parallelize import (
    build_parallelize_model,
    parallelize_model_ddp,
    parallelize_model_fsdp2,
)
from veomni.models.module_utils import init_empty_weights
from veomni.trainer.callbacks.base import TrainerState
from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


def _fsdp2_multi_optimizer_worker(rank: int, world_size: int, tmp_path: Path):
    """Worker for the FSDP2 MultiOptimizer DCP round-trip test."""
    from veomni.checkpoint.dcp_checkpointer import ModelState, OptimizerState
    from veomni.optim.optimizer import MultiOptimizer
    from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(os.environ.get("_TEST_MASTER_PORT", "0"))
    device_type = get_device_type()
    backend = "gloo" if device_type == "cpu" else get_dist_comm_backend()
    device = torch.device("cpu" if device_type == "cpu" else f"{device_type}:{rank}")
    if device_type != "cpu":
        get_torch_device().set_device(device)
    dist.init_process_group(backend, rank=rank, world_size=world_size)

    try:
        _init_parallel_state(dp_size=world_size, dp_mode="fsdp2")
        mesh = get_parallel_state().dp_shard_mesh

        def build_model_and_optimizer():
            model = nn.Sequential(nn.Linear(4, 4, bias=True), nn.Linear(4, 4, bias=True)).to(device)
            for layer in model:
                fully_shard(layer, mesh=mesh)
            fully_shard(model, mesh=mesh)
            optimizer = MultiOptimizer(
                model,
                {
                    "adamw0": torch.optim.AdamW(model[0].parameters(), lr=1e-3),
                    "adamw1": torch.optim.AdamW(model[1].parameters(), lr=1e-3),
                },
                ["adamw0", "adamw1"],
            )
            return model, optimizer

        source_model, source_optimizer = build_model_and_optimizer()
        x = torch.randn(2, 4, device=device)
        source_model(x).sum().backward()
        source_optimizer.step()
        expected = source_optimizer.state_dict()

        checkpoint_dir = tmp_path / "ckpt"
        dcp.save(
            {"model": ModelState(source_model), "optimizer": OptimizerState(source_model, source_optimizer)},
            checkpoint_id=str(checkpoint_dir),
        )
        dist.barrier()

        target_model, target_optimizer = build_model_and_optimizer()
        dcp.load(
            {
                "model": ModelState(target_model),
                "optimizer": OptimizerState(target_model, target_optimizer, load=True),
            },
            checkpoint_id=str(checkpoint_dir),
        )

        actual = target_optimizer.state_dict()
        assert expected.keys() == actual.keys()
        for key in expected:
            expected_value = expected[key]
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(expected_value, actual_value, atol=0.0, rtol=0.0)
            else:
                assert expected_value == actual_value
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ---------------------------------------------------------------------------
# OptimizerState: no fill, partial load
# ---------------------------------------------------------------------------


@patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
class TestOptimizerStateNoFill:
    """OptimizerState.state_dict() must return only the optimizer state that
    actually exists — no synthetic placeholders for params without gradients.
    Missing state is handled at load time via allow_partial_load."""

    def test_state_dict_excludes_params_without_gradient(self, mock_gps):
        """Params that never received a gradient should NOT appear in the
        state dict returned by OptimizerState."""
        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        from veomni.checkpoint.dcp_checkpointer import OptimizerState

        model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Only step on the first layer
        optimizer.zero_grad()
        x = torch.randn(2, 8)
        loss = model[0](x).sum()
        loss.backward()
        optimizer.step()

        os = OptimizerState(model, optimizer)
        sd = os.state_dict()

        assert "0.weight" in sd["state"], "stepped param should have state"
        assert "1.weight" not in sd["state"], (
            "param without gradient should NOT have state — OptimizerState must not synthesize placeholders"
        )

    def test_no_fill_missing_method(self, mock_gps):
        """_fill_missing_optimizer_states was removed; verify it's gone."""
        from veomni.checkpoint.dcp_checkpointer import OptimizerState

        assert not hasattr(OptimizerState, "_fill_missing_optimizer_states")

    def test_init_no_fill_kwarg(self, mock_gps):
        """fill_missing_optimizer_states kwarg was removed."""
        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        from veomni.checkpoint.dcp_checkpointer import OptimizerState

        model = nn.Linear(8, 8, bias=False)
        optimizer = torch.optim.AdamW(model.parameters())

        with pytest.raises(TypeError, match="fill_missing_optimizer_states"):
            OptimizerState(model, optimizer, fill_missing_optimizer_states=True)


class TestMultiOptimizerState:
    """Non-ExtraParallel MultiOptimizer must use its own DCP protocol."""

    def test_single_process_dcp_roundtrip(self, tmp_path):
        from veomni.checkpoint.dcp_checkpointer import OptimizerState
        from veomni.optim.optimizer import MultiOptimizer

        def build_model_and_optimizer():
            model = nn.Linear(4, 4)
            optimizer = MultiOptimizer(
                model,
                {
                    "adamw_w": torch.optim.AdamW([model.weight], lr=1e-3),
                    "adamw_b": torch.optim.AdamW([model.bias], lr=1e-3),
                },
                ["adamw_w", "adamw_b"],
            )
            return model, optimizer

        parallel_state = SimpleNamespace(dp_mode="fsdp2")
        source_model, source_optimizer = build_model_and_optimizer()
        for param in source_model.parameters():
            param.grad = torch.randn_like(param)
        source_optimizer.step()
        expected = source_optimizer.state_dict()
        assert any("exp_avg" in key for key in expected)
        assert any("step" in key for key in expected)

        target_model, target_optimizer = build_model_and_optimizer()
        dcp.save(
            {"optimizer": OptimizerState(source_model, source_optimizer, parallel_state=parallel_state)},
            checkpoint_id=tmp_path,
        )
        dcp.load(
            {"optimizer": OptimizerState(target_model, target_optimizer, parallel_state=parallel_state, load=True)},
            checkpoint_id=tmp_path,
        )

        actual = target_optimizer.state_dict()
        assert expected.keys() == actual.keys()
        for key, expected_value in expected.items():
            actual_value = actual[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(expected_value, actual_value, atol=0.0, rtol=0.0)
            else:
                assert expected_value == actual_value

    def test_fsdp2_multi_optimizer_roundtrip(self, tmp_path):
        """Run the MultiOptimizer DCP round-trip across two real FSDP2 ranks."""
        from tests.tools.launch_utils import find_free_port

        os.environ["_TEST_MASTER_PORT"] = str(find_free_port())
        mp.spawn(_fsdp2_multi_optimizer_worker, args=(2, tmp_path), nprocs=2, join=True)

    def test_multi_optimizer_sparse_state_excludes_synthetic(self, tmp_path):
        """A fresh sub-optimizer must not have synthetic state materialized in the checkpoint."""
        from torch.distributed.checkpoint import FileSystemReader
        from torch.distributed.checkpoint.metadata import Metadata

        from veomni.checkpoint.dcp_checkpointer import OptimizerState
        from veomni.optim.optimizer import MultiOptimizer

        model = nn.Linear(4, 4)
        optimizer = MultiOptimizer(
            model,
            {
                "adamw_w": torch.optim.AdamW([model.weight], lr=1e-3),
                "adamw_b": torch.optim.AdamW([model.bias], lr=1e-3),
            },
            ["adamw_w", "adamw_b"],
        )
        # Only bias gets a gradient; the adamw_w sub-optimizer remains empty.
        model.bias.grad = torch.randn_like(model.bias)
        optimizer.step()
        assert optimizer.optimizers_dict["adamw_b"].state
        assert not optimizer.optimizers_dict["adamw_w"].state

        parallel_state = SimpleNamespace(dp_mode="fsdp2")
        dcp.save(
            {"optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state)},
            checkpoint_id=tmp_path,
        )

        reader = FileSystemReader(tmp_path)
        metadata = reader.read_metadata()
        assert isinstance(metadata, Metadata)
        keys = list(metadata.state_dict_metadata.keys())
        state_keys = [k for k in keys if k.startswith("optimizer.state.")]
        assert any("bias" in k for k in state_keys), f"expected bias state in checkpoint keys: {keys}"
        assert not any(k.startswith("optimizer.state.weight.") for k in state_keys), (
            f"synthetic weight state must not be saved: {keys}"
        )


@patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
class TestCheckpointLayoutRoundTrip:
    """A step's model state lands where ``docs/usage/checkpoint.md`` says, and
    comes back from there."""

    @staticmethod
    def _build(seed: int):
        torch.manual_seed(seed)
        model = nn.Linear(4, 4)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        return model, optimizer, scheduler

    @staticmethod
    def _train_one_step(model, optimizer, scheduler):
        model(torch.randn(2, 4)).sum().backward()
        optimizer.step()
        scheduler.step()

    def test_weights_and_optimizer_are_separable_on_disk(self, mock_gps, tmp_path):
        """The weights directory holds weights only.

        This is the whole point of the split: a step can be shipped or converted
        by copying ``model/ckpt`` alone. A single fused directory interleaves
        both into the same ``.distcp`` files, where they cannot be told apart.
        """
        from torch.distributed.checkpoint import FileSystemReader

        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        model, optimizer, scheduler = self._build(seed=0)
        self._train_one_step(model, optimizer, scheduler)

        DistributedCheckpointer.save(
            path=str(tmp_path),
            state={"model": model, "optimizer": optimizer, "lr_scheduler": scheduler},
            global_steps=7,
        )

        model_root = tmp_path / "global_step_7" / "model"
        assert (model_root / "ckpt" / ".metadata").is_file()
        assert (model_root / "optimizer" / ".metadata").is_file()
        assert (model_root / "lr_scheduler.pt").is_file()
        # The pre-split marker at the step root is gone; only the manifest, which
        # GlobalStateCallback writes, marks a step complete now.
        assert not (tmp_path / "global_step_7" / ".metadata").exists()

        weight_keys = FileSystemReader(model_root / "ckpt").read_metadata().state_dict_metadata.keys()
        assert weight_keys, "weights directory is empty"
        assert all(key.startswith("model") for key in weight_keys), (
            f"optimizer state leaked into the weights directory: {sorted(weight_keys)}"
        )

    def test_resume_restores_weights_optimizer_and_scheduler(self, mock_gps, tmp_path):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        model, optimizer, scheduler = self._build(seed=0)
        self._train_one_step(model, optimizer, scheduler)

        DistributedCheckpointer.save(
            path=str(tmp_path),
            state={"model": model, "optimizer": optimizer, "lr_scheduler": scheduler},
            global_steps=7,
        )

        # A different starting point, so a load that silently did nothing fails.
        resumed, resumed_optimizer, resumed_scheduler = self._build(seed=1)
        assert not torch.equal(resumed.weight, model.weight)

        DistributedCheckpointer.load(
            path=str(tmp_path / "global_step_7"),
            state={"model": resumed, "optimizer": resumed_optimizer, "lr_scheduler": resumed_scheduler},
        )

        torch.testing.assert_close(resumed.weight, model.weight, atol=0.0, rtol=0.0)
        expected_exp_avg = optimizer.state_dict()["state"][0]["exp_avg"]
        torch.testing.assert_close(
            resumed_optimizer.state_dict()["state"][0]["exp_avg"], expected_exp_avg, atol=0.0, rtol=0.0
        )
        assert resumed_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]

    def test_modules_of_one_job_do_not_share_a_directory(self, mock_gps, tmp_path):
        """A multi-module job nests under ``model/<module>/`` and nothing else moves."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        for index, module in enumerate(("vision", "llm")):
            model, optimizer, scheduler = self._build(seed=index)
            DistributedCheckpointer.save(
                path=str(tmp_path),
                state={"model": model, "optimizer": optimizer, "lr_scheduler": scheduler},
                global_steps=7,
                module=module,
            )

        model_root = tmp_path / "global_step_7" / "model"
        assert sorted(p.name for p in model_root.iterdir()) == ["llm", "vision"]
        for module in ("vision", "llm"):
            assert (model_root / module / "ckpt" / ".metadata").is_file()
            assert (model_root / module / "optimizer" / ".metadata").is_file()

    def test_resume_reads_a_pre_split_checkpoint(self, mock_gps, tmp_path):
        """A checkpoint written before the split keeps both in one directory.

        Delete this along with ``veomni/checkpoint/legacy_v0_1_12.py``.
        """
        from veomni.checkpoint.dcp_checkpointer import (
            _LR_SCHEDULER_FILENAME,
            DistributedCheckpointer,
            ModelState,
            OptimizerState,
        )

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        parallel_state = SimpleNamespace(dp_mode="fsdp2")
        model, optimizer, scheduler = self._build(seed=0)
        self._train_one_step(model, optimizer, scheduler)

        step = tmp_path / "global_step_7"
        dcp.save(
            {
                "model": ModelState(model, parallel_state=parallel_state),
                "optimizer": OptimizerState(model, optimizer, parallel_state=parallel_state),
            },
            checkpoint_id=step,
        )
        torch.save(scheduler.state_dict(), step / _LR_SCHEDULER_FILENAME)

        resumed, resumed_optimizer, resumed_scheduler = self._build(seed=1)
        DistributedCheckpointer.load(
            path=str(step),
            state={"model": resumed, "optimizer": resumed_optimizer, "lr_scheduler": resumed_scheduler},
        )

        torch.testing.assert_close(resumed.weight, model.weight, atol=0.0, rtol=0.0)
        assert resumed_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]


class TestAllowPartialLoad:
    """DCP load may be partial for optimizer state, but not full model state."""

    def test_load_uses_allow_partial_load_planner(self):
        from veomni.checkpoint.dcp_checkpointer import DefaultLoadPlanner

        planner = DefaultLoadPlanner(allow_partial_load=True)
        assert planner.allow_partial_load is True

    @patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
    @patch("veomni.checkpoint.dcp_checkpointer.dcp")
    def test_load_passes_partial_planner_to_dcp(self, mock_dcp, mock_gps):
        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        model = MagicMock()
        model._fqn2spec_info = None
        optimizer = MagicMock()

        state = {"model": model, "optimizer": optimizer, "lr_scheduler": MagicMock()}

        mock_dcp.load = MagicMock()

        with patch.object(DistributedCheckpointer, "_load_lr_scheduler"):
            with patch.object(DistributedCheckpointer, "_create_storage_reader") as mock_reader:
                mock_reader.return_value = MagicMock()
                DistributedCheckpointer.load(path="/fake", state=state)

        # Weights and optimizer live in separate DCP directories, so a resume is
        # two loads.
        assert mock_dcp.load.call_count == 2
        weights_call, optimizer_call = mock_dcp.load.call_args_list
        assert list(weights_call.kwargs["state_dict"]) == ["model"]
        assert list(optimizer_call.kwargs["state_dict"]) == ["optimizer"]

        planner = weights_call.kwargs.get("planner")
        assert planner is not None, "load must pass a planner"
        assert planner.allow_partial_load is True, "load must use DefaultLoadPlanner(allow_partial_load=True)"
        assert planner.strict_model is True, "full-model load must reject missing model keys"

    @patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
    def test_full_model_load_rejects_missing_checkpoint_key(self, mock_gps, tmp_path):
        from torch.distributed.checkpoint import CheckpointException

        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        torch.distributed.checkpoint.save({"model": {"weight": torch.full((2, 2), 7.0)}}, checkpoint_id=tmp_path)

        model = nn.Linear(2, 2)
        with pytest.raises(CheckpointException, match=r"model\.bias"):
            DistributedCheckpointer.load(path=str(tmp_path), state={"model": model})

    @patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
    def test_trainable_only_model_load_allows_missing_frozen_key(self, mock_gps, tmp_path):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        torch.distributed.checkpoint.save({"model": {"weight": torch.full((2, 2), 7.0)}}, checkpoint_id=tmp_path)

        model = nn.Linear(2, 2)
        with torch.no_grad():
            model.bias.fill_(11.0)
        DistributedCheckpointer.load(path=str(tmp_path), state={"model": model}, trainable_only=True)

        torch.testing.assert_close(model.weight, torch.full_like(model.weight, 7.0))
        torch.testing.assert_close(model.bias, torch.full_like(model.bias, 11.0))


# ---------------------------------------------------------------------------
# Full DCP resume: skip redundant HF weight materialization
# ---------------------------------------------------------------------------


class TestSkipHfWeightLoadOnResume:
    def test_skip_hf_weight_load_when_full_non_lora_resume(self):
        assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {}) is True
        assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", None) is True

    def test_keep_hf_weight_load_for_fresh_or_lora(self):
        assert should_skip_hf_weight_load(None, {}) is False
        assert should_skip_hf_weight_load("/tmp/ckpt/global_step_200", {"r": 8}) is False

    def test_parallelize_apis_expose_should_skip_hf_weight_load(self):
        assert "should_skip_hf_weight_load" in inspect.signature(build_parallelize_model).parameters
        assert "should_skip_hf_weight_load" in inspect.signature(parallelize_model_fsdp2).parameters
        assert "should_skip_hf_weight_load" in inspect.signature(parallelize_model_ddp).parameters

    def test_build_parallelize_model_forwards_should_skip_hf_weight_load(self, monkeypatch):
        model = MagicMock()
        parallelized_model = MagicMock()
        parallelize_fsdp2 = MagicMock(return_value=parallelized_model)
        parallel_state = SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="fsdp2")
        monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(torch_parallelize, "parallelize_model_fsdp2", parallelize_fsdp2)

        result = build_parallelize_model(
            model,
            mixed_precision=SimpleNamespace(enable=False),
            enable_gradient_checkpointing=False,
            should_skip_hf_weight_load=True,
        )

        assert result is parallelized_model
        assert parallelize_fsdp2.call_args.kwargs["should_skip_hf_weight_load"] is True

    def test_dcp_resume_preserves_nonpersistent_buffers_and_forward(self, monkeypatch, tmp_path):
        class ModelWithDerivedBuffer(nn.Module):
            _no_split_modules = []

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
                self.register_buffer("scale", torch.tensor([0.25, 2.0]), persistent=False)

            def forward(self, x):
                return (x @ self.weight) * self.scale

            def init_weights(self):
                raise AssertionError("DCP resume must not initialize model parameters")

        original = ModelWithDerivedBuffer()
        assert "scale" not in original.state_dict()
        inputs = torch.tensor([[2.0, -1.0]])
        expected_output = original(inputs)
        checkpoint_dir = tmp_path / "dcp"
        dcp.save({"model": original}, checkpoint_id=checkpoint_dir)

        with init_empty_weights():
            resumed = ModelWithDerivedBuffer()
        assert resumed.weight.is_meta
        assert not resumed.scale.is_meta
        parallel_state = SimpleNamespace(any_extra_parallel_enabled=False, extra_parallel_names=[], fsdp_mesh=None)
        monkeypatch.setattr(torch_parallelize, "get_parallel_state", lambda: parallel_state)
        monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")

        resumed = parallelize_model_fsdp2(
            resumed,
            weights_path="unused-hf-path",
            mixed_precision=SimpleNamespace(enable=False),
            should_skip_hf_weight_load=True,
            init_device="meta",
        )
        dcp.load({"model": resumed}, checkpoint_id=checkpoint_dir)

        torch.testing.assert_close(resumed.scale, original.scale, rtol=0, atol=0)
        torch.testing.assert_close(resumed(inputs), expected_output, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Materialize + load: one dispatch shared by the fsdp2 and ddp paths
# ---------------------------------------------------------------------------


class TestMaterializeAndLoadDispatch:
    """The three loaders are mutually exclusive and picked from flags that
    ``parallelize_model_fsdp2`` reads out of ``**kwargs``, so a typo there would
    silently pick the wrong one."""

    @pytest.fixture
    def loaders(self, monkeypatch):
        loaders = SimpleNamespace(plain=MagicMock(), broadcast=MagicMock(), ep_sharded=MagicMock())
        monkeypatch.setattr(torch_parallelize, "load_model_weights", loaders.plain)
        monkeypatch.setattr(torch_parallelize, "rank0_load_and_broadcast_weights", loaders.broadcast)
        monkeypatch.setattr(torch_parallelize, "load_model_weights_ep_sharded", loaders.ep_sharded)
        return loaders

    @pytest.mark.parametrize(
        ("flags", "has_plan", "expected"),
        [
            ({}, False, "plain"),
            ({"broadcast_from_rank0": True}, False, "broadcast"),
            ({"ep_sharded_stream_load": True}, True, "ep_sharded"),
            # ``ep_sharded_stream_load`` is set once per run but this helper runs
            # once per model, so a model with no ExtraParallel plan -- every
            # SeedOmni V2 sub-module that owns no experts -- must fall through to
            # the loader it would have used anyway, not raise.
            ({"ep_sharded_stream_load": True}, False, "plain"),
        ],
    )
    def test_dispatches_to_one_loader(self, loaders, monkeypatch, flags, has_plan, expected):
        monkeypatch.setattr(torch_parallelize, "_has_extra_parallel_plan", lambda model: has_plan)

        torch_parallelize._materialize_and_load_weights(
            nn.Linear(2, 2),
            "hf-path",
            "cpu",
            should_skip_hf_weight_load=False,
            is_peft_model=False,
            adapter_path=None,
            **{"broadcast_from_rank0": False, **flags},
        )

        for name in ("plain", "broadcast", "ep_sharded"):
            assert getattr(loaders, name).called is (name == expected)

    def test_fsdp2_forwards_the_loader_flags(self, loaders, monkeypatch):
        monkeypatch.setattr(
            torch_parallelize,
            "get_parallel_state",
            lambda: SimpleNamespace(any_extra_parallel_enabled=False, extra_parallel_names=[], fsdp_mesh=None),
        )
        monkeypatch.setattr(torch_parallelize, "fully_shard", lambda *args, **kwargs: None)
        monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")

        parallelize_model_fsdp2(
            nn.Linear(2, 2),
            weights_path="hf-path",
            mixed_precision=SimpleNamespace(enable=False),
            init_device="meta",
            broadcast_model_weights_from_rank0=True,
        )

        assert loaders.broadcast.called
        assert not loaders.plain.called

    def test_resume_is_incompatible_with_peft(self):
        with pytest.raises(ValueError, match="incompatible with LoRA/PEFT"):
            torch_parallelize._materialize_and_load_weights(
                nn.Linear(2, 2),
                "hf-path",
                "cpu",
                should_skip_hf_weight_load=True,
                is_peft_model=True,
                adapter_path=None,
                broadcast_from_rank0=False,
            )

    def test_a_buffer_derived_from_a_parameter_is_warned_about_not_preserved(self, monkeypatch):
        """``init_empty_weights`` patches ``register_parameter`` only, so a buffer
        built from a real tensor stays real -- but one built from a parameter is on
        meta, holds no data to copy out of, and ends up as whatever ``to_empty()``
        allocated. No model registers one; the warning is what makes it findable."""

        class _DerivedBufferModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.ones(2))
                self.register_buffer("from_config", torch.tensor([0.5, 1.5]), persistent=False)
                self.register_buffer("from_param", self.weight.detach().clone(), persistent=False)

        with init_empty_weights():
            model = _DerivedBufferModel()
        assert model.from_param.is_meta and not model.from_config.is_meta

        # Against the logger rather than captured output: veomni's logger sets
        # propagate=False and binds its own stdout handler, so neither caplog nor
        # capsys sees the record.
        warnings = []
        monkeypatch.setattr(torch_parallelize.logger, "warning_rank0", warnings.append)

        torch_parallelize._to_empty_preserving_nonpersistent_buffers(model, "cpu")

        torch.testing.assert_close(model.from_config, torch.tensor([0.5, 1.5]), rtol=0, atol=0)
        assert not model.from_param.is_meta
        assert any("from_param" in message for message in warnings)


# ---------------------------------------------------------------------------
# DDP under meta-init: the wrap materializes and loads the model itself
# ---------------------------------------------------------------------------


class _MetaInitModel(nn.Module):
    """A model whose ``init_weights`` is observable and whose derived buffer
    must survive materialization."""

    _no_split_modules = []

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(2, 2))
        self.register_buffer("scale", torch.tensor([0.25, 2.0]), persistent=False)
        self.init_weights_calls = 0

    def init_weights(self):
        self.init_weights_calls += 1
        with torch.no_grad():
            self.weight.fill_(3.0)


class TestDdpMetaInit:
    """``init_device`` defaults to ``meta`` and is only asserted for fsdp2, so a
    ``ddp`` config used to reach DDP's constructor with unmaterialized
    parameters and fail there on ``Tensor.item()``."""

    @pytest.fixture
    def wrap_ddp(self, monkeypatch):
        monkeypatch.setattr(
            torch_parallelize,
            "get_parallel_state",
            lambda: SimpleNamespace(local_rank=0, dp_group=None, any_extra_parallel_enabled=False),
        )
        monkeypatch.setattr(torch_parallelize, "get_device_type", lambda: "cpu")
        monkeypatch.setattr(torch_parallelize, "DDP", lambda model, **kwargs: SimpleNamespace(module=model, **kwargs))
        return parallelize_model_ddp

    def test_materializes_and_random_inits_without_weights_path(self, wrap_ddp):
        with init_empty_weights():
            model = _MetaInitModel()
        assert model.weight.is_meta

        wrapped = wrap_ddp(model, weights_path=None, init_device="meta")

        assert not wrapped.module.weight.is_meta
        assert wrapped.module.init_weights_calls == 1
        assert wrapped.broadcast_buffers is False
        # A bare to_empty() would leave uninitialized memory here: this buffer is
        # shaped like the ones HF's _init_weights does not recompute.
        torch.testing.assert_close(wrapped.module.scale, torch.tensor([0.25, 2.0]), rtol=0, atol=0)

    def test_resume_materializes_without_reading_the_hf_snapshot(self, wrap_ddp, monkeypatch):
        loader = MagicMock()
        monkeypatch.setattr(torch_parallelize, "load_model_weights", loader)
        with init_empty_weights():
            model = _MetaInitModel()

        wrapped = wrap_ddp(
            model,
            weights_path="unused-hf-path",
            should_skip_hf_weight_load=True,
            init_device="meta",
        )

        loader.assert_not_called()
        assert wrapped.module.init_weights_calls == 0
        assert not wrapped.module.weight.is_meta
        # Non-persistent buffers are absent from the DCP state dict, so the
        # resume cannot restore them and materialization must not drop them.
        torch.testing.assert_close(wrapped.module.scale, torch.tensor([0.25, 2.0]), rtol=0, atol=0)

    @pytest.mark.parametrize("broadcast", [False, True])
    def test_honours_broadcast_model_weights_from_rank0(self, wrap_ddp, monkeypatch, broadcast):
        # A real loader materializes as it fills; the stubs have to as well, or the
        # meta guard after the load pass fires.
        materialize = lambda model, *args, **kwargs: model.to_empty(device="cpu")  # noqa: E731
        loader = MagicMock(side_effect=materialize)
        broadcast_loader = MagicMock(side_effect=materialize)
        monkeypatch.setattr(torch_parallelize, "load_model_weights", loader)
        monkeypatch.setattr(torch_parallelize, "rank0_load_and_broadcast_weights", broadcast_loader)
        with init_empty_weights():
            model = _MetaInitModel()

        wrap_ddp(model, weights_path="hf-path", init_device="meta", broadcast_model_weights_from_rank0=broadcast)

        used, unused = (broadcast_loader, loader) if broadcast else (loader, broadcast_loader)
        assert used.call_args.args[1] == "hf-path"
        unused.assert_not_called()

    def test_rejects_a_model_whose_plan_shards_experts(self, monkeypatch):
        # Only the fsdp2 path applies the plan that shards experts, so loading a
        # sharded-expert config here would quietly produce whole ones.
        monkeypatch.setattr(
            torch_parallelize,
            "get_parallel_state",
            lambda: SimpleNamespace(local_rank=0, dp_group=None, any_extra_parallel_enabled=True),
        )
        monkeypatch.setattr(
            torch_parallelize,
            "_has_extra_parallel_plan",
            lambda model: True,
        )
        with pytest.raises(RuntimeError, match="requires fsdp_mode='fsdp2'"):
            parallelize_model_ddp(nn.Linear(2, 2))

    def test_allows_a_plan_less_model_under_an_inherited_ep_mesh(self, monkeypatch):
        """An ep dim in the mesh says nothing about *this* model. A SeedOmni V2
        sub-module inherits the global accelerator's ep size whether or not it owns
        experts, so refusing on the mesh alone would block a DDP vision tower."""
        monkeypatch.setattr(
            torch_parallelize,
            "get_parallel_state",
            lambda: SimpleNamespace(local_rank=0, dp_group=None, any_extra_parallel_enabled=True),
        )
        monkeypatch.setattr(torch_parallelize, "DDP", lambda module, **kwargs: module)

        assert parallelize_model_ddp(nn.Linear(2, 2)) is not None

    def test_reports_parameters_the_loader_left_on_meta(self, wrap_ddp, monkeypatch):
        # A loader that fills nothing would otherwise fail inside DDP's
        # constructor on Tensor.item(), naming neither parameter nor cause.
        monkeypatch.setattr(torch_parallelize, "load_model_weights", MagicMock())
        with init_empty_weights():
            model = _MetaInitModel()

        with pytest.raises(RuntimeError, match="unmaterialized parameters") as excinfo:
            wrap_ddp(model, weights_path="hf-path", init_device="meta")

        assert "weight" in str(excinfo.value)

    @pytest.mark.parametrize("weights_path", [None, "hf-path"])
    def test_leaves_a_real_model_alone_when_init_device_says_meta(self, wrap_ddp, monkeypatch, weights_path):
        """``init_device`` is a request to the model builder, not a fact about the
        model it returns: tests/data construct theirs eagerly and leave the flag at
        its ``meta`` default. Materializing that model would re-read the snapshot
        over weights it already holds, and a plain nn.Module has no
        ``init_weights`` to fall back on when there is no snapshot."""
        loader = MagicMock()
        monkeypatch.setattr(torch_parallelize, "load_model_weights", loader)
        model = nn.Linear(2, 2)
        weight = model.weight.detach().clone()

        wrapped = wrap_ddp(model, weights_path=weights_path, init_device="meta")

        loader.assert_not_called()
        torch.testing.assert_close(wrapped.module.weight, weight, rtol=0, atol=0)

    def test_does_not_reinitialize_a_real_model_that_has_init_weights(self, wrap_ddp, monkeypatch):
        # The nn.Linear case above fails loudly; a model that does define
        # init_weights would instead have its weights silently overwritten.
        loader = MagicMock()
        monkeypatch.setattr(torch_parallelize, "load_model_weights", loader)
        model = _MetaInitModel()
        with torch.no_grad():
            model.weight.fill_(7.0)

        wrapped = wrap_ddp(model, weights_path="hf-path", init_device="meta")

        loader.assert_not_called()
        assert wrapped.module.init_weights_calls == 0
        torch.testing.assert_close(wrapped.module.weight, torch.full((2, 2), 7.0), rtol=0, atol=0)

    def test_build_parallelize_model_forwards_should_skip_hf_weight_load_to_ddp(self, monkeypatch):
        parallelize_ddp = MagicMock(return_value=MagicMock())
        monkeypatch.setattr(
            torch_parallelize,
            "get_parallel_state",
            lambda: SimpleNamespace(fsdp_enabled=True, tp_enabled=False, dp_mode="ddp"),
        )
        monkeypatch.setattr(torch_parallelize, "parallelize_model_ddp", parallelize_ddp)

        build_parallelize_model(
            MagicMock(),
            mixed_precision=SimpleNamespace(enable=False),
            enable_gradient_checkpointing=False,
            should_skip_hf_weight_load=True,
            init_device="meta",
        )

        assert parallelize_ddp.call_args.kwargs["should_skip_hf_weight_load"] is True
        assert parallelize_ddp.call_args.kwargs["init_device"] == "meta"


# ---------------------------------------------------------------------------
# Save planner: dcp_save_to_lowest_rank wiring
# ---------------------------------------------------------------------------


class TestSaveToLowestRank:
    """execute_save() must forward ``save_to_lowest_rank`` to DCP's
    DefaultSavePlanner, defaulting to False (stock load-balanced writes)."""

    def test_config_default_is_false(self):
        from veomni.arguments.arguments_types import CheckpointConfig

        assert CheckpointConfig().dcp_save_to_lowest_rank is False

    @pytest.mark.parametrize("flag", [False, True])
    @patch("veomni.checkpoint.dcp_checkpointer.synchronize")
    @patch("veomni.checkpoint.dcp_checkpointer.empty_cache")
    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    @patch("veomni.checkpoint.dcp_checkpointer.dcp")
    def test_execute_save_forwards_flag_to_planner(self, mock_dcp, mock_dist, mock_ec, mock_sync, flag):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = False
        mock_dcp.save = MagicMock()

        DistributedCheckpointer.execute_save(
            save_state={"model": MagicMock()},
            storage_writer=MagicMock(),
            save_async=False,
            save_to_lowest_rank=flag,
        )

        mock_dcp.save.assert_called_once()
        planner = mock_dcp.save.call_args.kwargs.get("planner")
        assert planner is not None, "save must pass a planner"
        assert planner.dedup_save_to_lowest_rank is flag


# ---------------------------------------------------------------------------
# Async save lifecycle: wait_for_pending_save()
# ---------------------------------------------------------------------------


class TestWaitForPendingSave:
    """``DistributedCheckpointer.wait_for_pending_save()`` is the single
    entrypoint for coordinating with an in-flight async save."""

    def teardown_method(self):
        """Reset class state between tests."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        DistributedCheckpointer._save_futures = {}
        DistributedCheckpointer._async_process_groups = {}

    def test_noop_when_no_pending_save(self):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        DistributedCheckpointer._save_futures = {}
        # Should be a clean no-op — no exceptions, no barrier
        DistributedCheckpointer.wait_for_pending_save()
        assert DistributedCheckpointer._save_futures == {}

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_waits_and_clears_future(self, mock_dist):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        future = MagicMock()
        future.result.return_value = None
        DistributedCheckpointer._save_futures = {"ckpt": future}

        # Patched in ``dist_utils`` rather than in the checkpointer: that is where
        # ``raise_if_any_rank_failed`` looks the reduction up.
        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=False) as agreed:
            DistributedCheckpointer.wait_for_pending_save()

        future.result.assert_called_once()
        assert DistributedCheckpointer._save_futures == {}
        # The reduction is the synchronization callers rely on before their next
        # collective; it replaces the barrier this used to end with.
        agreed.assert_called_once_with(False, group=None)

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_a_drain_agrees_on_the_slots_own_group(self, mock_dist):
        """The write ran on the slot's gloo group and the ranks leave it as far
        apart as their writes finished. Agreeing on the training backend instead
        puts that wait where the NCCL watchdog aborts the process over it."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        slot_group = object()
        DistributedCheckpointer._async_process_groups = {"ckpt": slot_group}

        DistributedCheckpointer._save_futures = {"ckpt": MagicMock()}
        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=False) as agreed:
            DistributedCheckpointer._drain_slot("ckpt")
        agreed.assert_called_once_with(False, group=slot_group)

        DistributedCheckpointer._save_futures = {"ckpt": MagicMock()}
        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=False) as agreed:
            DistributedCheckpointer.wait_for_pending_save()
        agreed.assert_called_once_with(False, group=slot_group)

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_a_peers_failure_raises_here_too(self, mock_dist):
        """DCP reports an async failure only through the future of the rank that
        hit it, and the save's process group does not reduce that across the
        group. A rank whose own write succeeded must still raise, or it enters
        the next collective without the failing rank and hangs until timeout."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        healthy = MagicMock()
        DistributedCheckpointer._save_futures = {"ckpt": healthy}

        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=True):
            with pytest.raises(RuntimeError, match="failed on another rank"):
                DistributedCheckpointer.wait_for_pending_save()

        healthy.result.assert_called_once()
        assert DistributedCheckpointer._save_futures == {}

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_drain_slot_raises_on_every_rank(self, mock_dist):
        """``_drain_slot`` runs inside ``execute_save``, just before the slot's
        group is reused. A failing rank raising alone there would leave its peers
        in the reduction with nobody to meet."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        healthy = MagicMock()
        DistributedCheckpointer._save_futures = {"ckpt": healthy}

        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=True):
            with pytest.raises(RuntimeError, match="failed on another rank"):
                DistributedCheckpointer._drain_slot("ckpt")

        healthy.result.assert_called_once()
        assert DistributedCheckpointer._save_futures == {}

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_drains_every_slot_before_raising(self, mock_dist):
        """A step writes the weights and the optimizer concurrently. If the
        weights write raised, the optimizer write must still be drained — left
        running, its background collectives would collide with the next step's."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        failed = MagicMock()
        failed.result.side_effect = RuntimeError("save failed")
        healthy = MagicMock()
        DistributedCheckpointer._save_futures = {"ckpt": failed, "optimizer": healthy}

        with pytest.raises(RuntimeError, match="save failed"):
            DistributedCheckpointer.wait_for_pending_save()

        healthy.result.assert_called_once()
        # Cleared even on failure — otherwise stuck forever
        assert DistributedCheckpointer._save_futures == {}

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_a_dcp_failure_drains_every_slot_too(self, mock_dist):
        """The way DCP actually reports a failed save is ``CheckpointException``,
        which derives from ``BaseException`` rather than ``Exception``. Catching
        the narrower one would let the very failures this drains for escape the
        loop: the remaining slots keep running and their futures are already
        unreachable, since the dict is cleared before the wait."""
        from torch.distributed.checkpoint.api import CheckpointException

        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        failed = MagicMock()
        failed.result.side_effect = CheckpointException("save failed", {0: (RuntimeError("disk full"), None)})
        healthy = MagicMock()
        DistributedCheckpointer._save_futures = {"ckpt": failed, "optimizer": healthy}

        with pytest.raises(CheckpointException):
            DistributedCheckpointer.wait_for_pending_save()

        healthy.result.assert_called_once()
        assert DistributedCheckpointer._save_futures == {}

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_drain_slot_reduces_a_dcp_failure(self, mock_dist):
        """Same exception, and here it is the reduction that must still run: a
        ``CheckpointException`` raised past it leaves every peer waiting in a
        collective the failing rank never reaches."""
        from torch.distributed.checkpoint.api import CheckpointException

        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        failed = MagicMock()
        failed.result.side_effect = CheckpointException("save failed", {0: (RuntimeError("disk full"), None)})
        DistributedCheckpointer._save_futures = {"ckpt": failed}

        with patch("veomni.utils.dist_utils.any_rank_failed", return_value=True) as agreed:
            with pytest.raises(CheckpointException):
                DistributedCheckpointer._drain_slot("ckpt")

        agreed.assert_called_once_with(True, group=None)
        assert DistributedCheckpointer._save_futures == {}

    def test_no_collective_when_dist_not_initialized(self):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        future = MagicMock()
        DistributedCheckpointer._save_futures = {"ckpt": future}

        with patch("veomni.utils.dist_utils.dist") as mock_dist:
            mock_dist.is_initialized.return_value = False
            DistributedCheckpointer.wait_for_pending_save()

        future.result.assert_called_once()
        mock_dist.all_reduce.assert_not_called()

    @patch("veomni.checkpoint.dcp_checkpointer.dcp")
    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_concurrent_slots_get_their_own_group_and_are_not_drained(self, mock_dist, mock_dcp):
        """The two saves of one step must overlap.

        ``dcp.async_save`` runs the whole save, collectives included, on a
        background thread using the group it is handed — so each slot needs its
        own group, and issuing the second must not wait out the first.
        """
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.new_group.side_effect = ["group-ckpt", "group-optimizer"]
        mock_dcp.async_save.side_effect = ["future-ckpt", "future-optimizer"]
        DistributedCheckpointer._save_futures = {}
        DistributedCheckpointer._async_process_groups = {}

        for slot in ("ckpt", "optimizer"):
            DistributedCheckpointer.execute_save(
                save_state={},
                storage_writer=MagicMock(),
                save_async=True,
                slot=slot,
            )

        assert DistributedCheckpointer._save_futures == {
            "ckpt": "future-ckpt",
            "optimizer": "future-optimizer",
        }
        groups = [call.kwargs["process_group"] for call in mock_dcp.async_save.call_args_list]
        assert groups == ["group-ckpt", "group-optimizer"]

        DistributedCheckpointer._save_futures = {}
        DistributedCheckpointer._async_process_groups = {}

    @patch("veomni.checkpoint.dcp_checkpointer.dcp")
    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_a_slot_group_takes_the_save_timeout(self, mock_dist, mock_dcp):
        """An async write waits on its slot's group like staging does, so it gets the same deadline."""
        from datetime import timedelta

        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        DistributedCheckpointer._save_futures = {}
        DistributedCheckpointer._async_process_groups = {}
        try:
            DistributedCheckpointer.execute_save(
                save_state={}, storage_writer=MagicMock(), save_async=True, slot="ckpt", timeout_seconds=1800
            )
            assert mock_dist.new_group.call_args.kwargs == {"backend": "gloo", "timeout": timedelta(seconds=1800)}
        finally:
            DistributedCheckpointer._save_futures = {}
            DistributedCheckpointer._async_process_groups = {}


class TestDcpToHfDtypeConversion:
    def test_save_dtype_only_casts_floating_tensors(self):
        import tempfile

        import torch.distributed.checkpoint as dcp

        from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan, _process_shard

        fp8_dtype = getattr(torch, "float8_e4m3fn", None)
        if fp8_dtype is None:
            pytest.skip("torch.float8_e4m3fn is unavailable")

        state_dict = {
            "model.weight": torch.ones(4, dtype=torch.float32),
            "model.tid2eid": torch.arange(8, dtype=torch.int64),
            "model.flag": torch.tensor([True, False]),
            "model.fp8": torch.tensor([1.0, 2.0], dtype=fp8_dtype),
        }

        with tempfile.TemporaryDirectory() as checkpoint_path:
            dcp.save(state_dict, checkpoint_id=checkpoint_path)
            bf16_shards, bf16_total_size, _ = _get_sharding_plan(
                checkpoint_path,
                shard_size=5,
                save_dtype="bfloat16",
            )
            native_shards, native_total_size, _ = _get_sharding_plan(
                checkpoint_path,
                shard_size=5,
                save_dtype=None,
            )

            bf16_state = {}
            for shard in bf16_shards:
                bf16_state.update(_process_shard(shard, checkpoint_path, save_dtype="bfloat16"))

            native_state = {}
            for shard in native_shards:
                native_state.update(_process_shard(shard, checkpoint_path, save_dtype=None))

        expected_bf16_size = 0
        expected_native_size = 0
        for tensor in state_dict.values():
            expected_native_size += tensor.numel() * tensor.element_size()
            output_element_size = (
                torch.empty((), dtype=torch.bfloat16).element_size()
                if tensor.is_floating_point()
                else tensor.element_size()
            )
            expected_bf16_size += tensor.numel() * output_element_size

        assert bf16_total_size == expected_bf16_size
        assert native_total_size == expected_native_size
        assert len(bf16_shards) == 4
        assert len(native_shards) == 3
        assert set(native_shards[0]) == {"flag", "fp8"}

        assert bf16_state["weight"].dtype == torch.bfloat16
        assert bf16_state["fp8"].dtype == torch.bfloat16
        assert bf16_state["tid2eid"].dtype == torch.int64
        assert bf16_state["flag"].dtype == torch.bool
        torch.testing.assert_close(bf16_state["tid2eid"], state_dict["model.tid2eid"])

        assert native_state["weight"].dtype == torch.float32
        assert native_state["fp8"].dtype == fp8_dtype
        assert native_state["tid2eid"].dtype == torch.int64
        assert native_state["flag"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Partial save/load (LoRA / trainable_only path)
# ---------------------------------------------------------------------------


@patch("veomni.checkpoint.dcp_checkpointer.get_parallel_state")
class TestPartialSaveLoad:
    """When trainable_only=True (LoRA), the checkpoint contains only adapter
    weights.  On load, allow_partial_load=True lets DCP skip the missing
    frozen-base entries.  The optimizer checkpoint is similarly partial:
    only trainable params that received gradients have state."""

    def test_trainable_only_model_state_excludes_frozen(self, mock_gps):
        """ModelState with trainable_only=True should skip frozen params."""
        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        from veomni.checkpoint.dcp_checkpointer import ModelState

        model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
        model[0].weight.requires_grad_(False)  # freeze first layer

        ms = ModelState(model, trainable_only=True)
        sd = ms.state_dict()

        assert "1.weight" in sd, "trainable param should be in state dict"
        assert "0.weight" not in sd, "frozen param should be excluded with trainable_only=True"

    def test_optimizer_state_only_has_trained_params(self, mock_gps):
        """OptimizerState.state_dict() should only contain params that
        received gradients — no synthetic placeholders for frozen or
        unused params."""
        mock_gps.return_value = SimpleNamespace(dp_mode="fsdp2")
        from veomni.checkpoint.dcp_checkpointer import OptimizerState

        model = nn.Sequential(nn.Linear(8, 8, bias=False), nn.Linear(8, 8, bias=False))
        # Simulate LoRA: optimizer only has trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=1e-3)

        # Step on first layer only
        optimizer.zero_grad()
        loss = model[0](torch.randn(2, 8)).sum()
        loss.backward()
        optimizer.step()

        os = OptimizerState(model, optimizer)
        sd = os.state_dict()

        assert len(sd.get("state", {})) > 0, "should have at least one param with state"
        for fqn in sd.get("state", {}):
            assert "0.weight" in fqn, f"only layer 0 was stepped, but found state for {fqn}"


# ---------------------------------------------------------------------------
# Bug 4 (PR #798): global_step inflated before data fetch
# ---------------------------------------------------------------------------


class TestGlobalStepInflation:
    """PR #798: ``global_step += 1`` executes BEFORE ``next(data_iterator)``
    in the training loop.  If ``StopIteration`` fires, the step counter is
    inflated without any training having occurred."""

    @pytest.mark.xfail(
        reason=(
            "Bug 4 (PR #798): global_step += 1 happens before next(data_iterator), "
            "so StopIteration leaves global_step inflated by 1"
        ),
        strict=True,
    )
    def test_global_step_not_inflated_on_stop_iteration(self):
        """A data iterator yields exactly 3 batches.  The loop attempts 10 steps.
        After exhaustion, global_step should be 3 (not 4)."""
        state = TrainerState(global_step=0)
        batches = iter([{"x": torch.randn(2, 4)} for _ in range(3)])

        completed_steps = 0
        for _ in range(10):
            try:
                state.global_step += 1
                _ = next(batches)
                completed_steps += 1
            except StopIteration:
                break

        assert state.global_step == completed_steps, (
            f"global_step={state.global_step} but only {completed_steps} "
            f"steps actually completed (expected them to be equal)"
        )

    def test_global_step_correct_after_full_epoch(self):
        """When the data iterator yields exactly as many batches as requested,
        no StopIteration fires and global_step matches."""
        state = TrainerState(global_step=0)
        num_steps = 5
        batches = iter([{"x": torch.randn(2, 4)} for _ in range(num_steps)])

        completed_steps = 0
        for _ in range(num_steps):
            try:
                state.global_step += 1
                _ = next(batches)
                completed_steps += 1
            except StopIteration:
                break

        assert state.global_step == num_steps

    @pytest.mark.xfail(
        reason=(
            "Bug 4 (PR #798): phantom checkpoint saved at inflated global_step "
            "because on_epoch_end fires after StopIteration with wrong step count"
        ),
        strict=True,
    )
    def test_epoch_end_no_phantom_save_after_stop_iteration(self):
        from veomni.trainer.callbacks.checkpoint_callback import CheckpointCallback

        trainer = MagicMock()
        trainer.args = SimpleNamespace(
            train=SimpleNamespace(
                checkpoint=SimpleNamespace(
                    save_path="/tmp/test_phantom",
                    save_steps=0,
                    save_epochs=1,
                    save_async=False,
                    load_path=None,
                    manager="dcp",
                    dcp_save_to_lowest_rank=False,
                    save_hf_weights=False,
                    hf_save_steps=0,
                    hf_save_epochs=0,
                ),
                global_rank=0,
            ),
            model=SimpleNamespace(accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="fsdp2"))),
        )

        cb = CheckpointCallback(trainer)
        cb.dcp_every_n_epochs = 1

        state = TrainerState(global_step=0)
        batches = iter([])

        for _ in range(5):
            try:
                state.global_step += 1
                _ = next(batches)
            except StopIteration:
                break

        assert state.global_step == 1

        state.epoch = 0
        cb.on_epoch_end(state)

        trainer.save_dcp.assert_not_called()


# ---------------------------------------------------------------------------
# _normalize_key
# ---------------------------------------------------------------------------


class TestNormalizeKey:
    def test_standard_model_key(self):
        from veomni.checkpoint.dcp_checkpointer import _normalize_key

        assert _normalize_key("model.model.layers.0.weight") == "model.layers.0.weight"

    def test_lm_head_key(self):
        from veomni.checkpoint.dcp_checkpointer import _normalize_key

        assert _normalize_key("model.lm_head.weight") == "lm_head.weight"

    def test_non_model_key_returns_none(self):
        from veomni.checkpoint.dcp_checkpointer import _normalize_key

        assert _normalize_key("optimizer.state.0.exp_avg") is None

    def test_single_model_prefix(self):
        from veomni.checkpoint.dcp_checkpointer import _normalize_key

        assert _normalize_key("model.embed_tokens.weight") == "embed_tokens.weight"

    def test_peft_lora_base_model_key(self):
        # GAP-5: ``save_lora_adapter_with_dcp`` re-prefixes already-PEFT-prefixed
        # keys with ``model.`` so the DCP filter keeps them; on read the leading
        # ``model.`` is stripped back to the standard PEFT adapter layout.
        from veomni.checkpoint.dcp_checkpointer import _normalize_key

        assert (
            _normalize_key("model.base_model.model.layers.0.self_attn.q_proj.lora_A.weight")
            == "base_model.model.layers.0.self_attn.q_proj.lora_A.weight"
        )


@patch("veomni.checkpoint.dcp_checkpointer.dist")
class TestLrSchedulerSaveLoad:
    def test_roundtrip(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        mock_dist.get_rank.return_value = 0
        from veomni.checkpoint.dcp_checkpointer import (
            _LR_SCHEDULER_FILENAME,
            DistributedCheckpointer,
        )

        saved_scheduler = MagicMock()
        saved_scheduler.state_dict.return_value = {"last_epoch": 10, "base_lrs": [1e-4]}
        DistributedCheckpointer._save_lr_scheduler(str(tmp_path), {"lr_scheduler": saved_scheduler})

        assert (tmp_path / _LR_SCHEDULER_FILENAME).is_file()
        assert {p.name for p in tmp_path.iterdir()} == {_LR_SCHEDULER_FILENAME}

        loaded_scheduler = MagicMock()
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": loaded_scheduler})

        loaded_scheduler.load_state_dict.assert_called_once_with({"last_epoch": 10, "base_lrs": [1e-4]})

    def test_only_rank_zero_writes(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3
        from veomni.checkpoint.dcp_checkpointer import (
            _LR_SCHEDULER_FILENAME,
            DistributedCheckpointer,
        )

        saved_scheduler = MagicMock()
        saved_scheduler.state_dict.return_value = {"last_epoch": 10}
        with patch(
            "veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed
        ):
            DistributedCheckpointer._save_lr_scheduler(str(tmp_path), {"lr_scheduler": saved_scheduler})

        saved_scheduler.state_dict.assert_not_called()
        assert not (tmp_path / _LR_SCHEDULER_FILENAME).exists()

    def test_missing_lr_scheduler_key_save(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        state = {"model": MagicMock()}
        DistributedCheckpointer._save_lr_scheduler(str(tmp_path), state)

    def test_missing_lr_scheduler_key_load(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        state = {"model": MagicMock()}
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), state)

    def test_none_scheduler_is_a_no_op(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        DistributedCheckpointer._save_lr_scheduler(str(tmp_path), {"lr_scheduler": None})
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": None})

    def test_missing_sidecar_raises_when_scheduler_is_expected(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        loaded = MagicMock()
        with pytest.raises(FileNotFoundError, match="lr_scheduler sidecar"):
            DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": loaded})
        loaded.load_state_dict.assert_not_called()

    def test_load_falls_back_to_extra_state_scheduler(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        mock_dist.get_rank.return_value = 0
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
        from veomni.checkpoint.legacy_v0_1_12 import extra_state_path

        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save({"lr_scheduler": {"last_epoch": 4}}, extra_state_path(str(tmp_path), 0))

        loaded = MagicMock()
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": loaded})
        loaded.load_state_dict.assert_called_once_with({"last_epoch": 4})

    def test_sidecar_wins_over_extra_state(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = False
        from veomni.checkpoint.dcp_checkpointer import (
            _LR_SCHEDULER_FILENAME,
            DistributedCheckpointer,
        )
        from veomni.checkpoint.legacy_v0_1_12 import extra_state_path

        torch.save({"last_epoch": 1}, tmp_path / _LR_SCHEDULER_FILENAME)
        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save({"lr_scheduler": {"last_epoch": 99}}, extra_state_path(str(tmp_path), 0))

        loaded = MagicMock()
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": loaded})
        loaded.load_state_dict.assert_called_once_with({"last_epoch": 1})

    def test_rank_nonzero_reads_rank0_extra_state_for_scheduler(self, mock_dist, tmp_path):
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
        from veomni.checkpoint.legacy_v0_1_12 import extra_state_path

        extra = tmp_path / "extra_state"
        extra.mkdir()
        torch.save({"lr_scheduler": {"last_epoch": 8}}, extra_state_path(str(tmp_path), 0))

        loaded = MagicMock()
        DistributedCheckpointer._load_lr_scheduler(str(tmp_path), {"lr_scheduler": loaded})
        loaded.load_state_dict.assert_called_once_with({"last_epoch": 8})


class TestPromoteStagedCheckpoint:
    """`stage_dir` promotion: a staged checkpoint becomes visible only once complete.

    Choosing a usable staging directory is the caller's job, so what is pinned
    down here is what the checkpointer itself owns: ordering of the completion
    marker, collective parity across success and failure, and never leaving the
    staged copy behind.
    """

    @pytest.fixture(autouse=True)
    def _no_real_collectives(self):
        """Keep `any_rank_failed` off a real process group.

        These tests fake `dist.is_initialized()`, so its all_reduce would other-
        wise hit an uninitialised group. Leaving the tensor untouched makes the
        reduction reflect this rank's own flag, which is what a single-rank test
        means; cases about *another* rank failing patch `any_rank_failed`
        directly.
        """
        with patch("veomni.utils.dist_utils.get_device_type", return_value="cpu"):
            with patch("veomni.utils.dist_utils.dist.all_reduce", side_effect=lambda t, op=None, group=None: None):
                yield

    @pytest.fixture
    def make_staged(self, tmp_path):
        """Build an independent staged checkpoint and destination on each call.

        Promotion consumes the staged copy, so anything exercising it more than
        once needs a fresh one per run rather than a shared directory.
        """

        def _make(name: str = "default"):
            stage_path = tmp_path / name / "stage"
            final_path = tmp_path / name / "final"
            stage_path.mkdir(parents=True)
            for entry in ("__0_0.distcp", "__0_1.distcp", ".metadata"):
                (stage_path / entry).write_text(entry)
            return str(stage_path), str(final_path)

        return _make

    @pytest.fixture
    def staged(self, make_staged):
        """A single staged checkpoint (two data files plus `.metadata`) and its destination."""
        return make_staged()

    def test_copies_everything_and_removes_the_staged_copy(self, staged):
        """The happy path: the destination ends up complete and the scratch copy is gone."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            _promote_staged_checkpoint(stage_path, final_path)
        assert sorted(os.listdir(final_path)) == [".metadata", "__0_0.distcp", "__0_1.distcp"]
        assert not os.path.exists(stage_path)

    def test_metadata_lands_after_the_data_files(self, staged):
        """DCP reads `.metadata` as "complete", so it must be copied last."""
        import shutil as _shutil

        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        order = []
        real_copy = _shutil.copyfile

        def spy(src, dst):
            """Record each copied filename, then perform the real copy."""
            order.append(os.path.basename(dst))
            return real_copy(src, dst)

        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch("veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=spy):
                _promote_staged_checkpoint(stage_path, final_path)
        assert order[-1] == ".metadata"

    def test_nested_sidecar_is_copied_before_metadata(self, tmp_path):
        """A nested file is copied with the shards, before any ``.metadata`` is."""
        import shutil as _shutil

        from veomni.checkpoint.dcp_checkpointer import _LR_SCHEDULER_FILENAME, _promote_staged_checkpoint

        stage_path = tmp_path / "stage"
        final_path = tmp_path / "final"
        nested = stage_path / "nested"
        nested.mkdir(parents=True)
        (stage_path / "__0_0.distcp").write_text("weights")
        (stage_path / ".metadata").write_text("meta")
        (stage_path / _LR_SCHEDULER_FILENAME).write_text("scheduler-v2")
        (nested / "sidecar.pt").write_text("nested")

        order = []
        real_copy = _shutil.copyfile

        def spy(src, dst):
            """Record each copied relative path, then perform the real copy."""
            order.append(os.path.relpath(dst, str(final_path)))
            return real_copy(src, dst)

        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch("veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=spy):
                _promote_staged_checkpoint(str(stage_path), str(final_path))

        nested_rel = os.path.join("nested", "sidecar.pt")
        assert _LR_SCHEDULER_FILENAME in order
        assert nested_rel in order
        assert order[-1] == ".metadata"
        assert order.index(_LR_SCHEDULER_FILENAME) < order.index(".metadata")
        assert order.index(nested_rel) < order.index(".metadata")
        assert (final_path / _LR_SCHEDULER_FILENAME).read_text() == "scheduler-v2"
        assert (final_path / nested_rel).read_text() == "nested"

    def test_stale_metadata_is_gone_before_any_data_is_copied(self, staged):
        """Overwriting in place must not leave the old marker over half-new data.

        The old marker's absence is what matters, not the call that removes it --
        the destination is emptied wholesale, so no single file is deleted by
        name."""
        import shutil as _shutil

        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        os.makedirs(final_path)
        marker = os.path.join(final_path, ".metadata")
        with open(marker, "w") as f:
            f.write("previous checkpoint")

        def stale_marker_present():
            """Whether the previous checkpoint's marker is still standing."""
            return os.path.exists(marker) and Path(marker).read_text() == "previous checkpoint"

        order = []
        real_copy = _shutil.copyfile

        def copy_spy(src, dst):
            """Record what the destination held at each copy, then really copy."""
            order.append((os.path.basename(dst), stale_marker_present()))
            return real_copy(src, dst)

        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch("veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=copy_spy):
                _promote_staged_checkpoint(stage_path, final_path)

        assert order, "promotion copied nothing"
        assert not any(stale for _name, stale in order), f"the old marker outlived a copy: {order}"
        assert order[-1][0] == ".metadata"
        assert not stale_marker_present()

    def test_staged_copy_is_removed_even_when_promotion_fails(self, staged):
        """A leftover staged copy is the size of the model plus its optimizer state."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch(
                "veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=OSError("destination is full")
            ):
                with pytest.raises(OSError, match="destination is full"):
                    _promote_staged_checkpoint(stage_path, final_path)
        assert not os.path.exists(stage_path)

    # (global_rank, local_rank) for the three roles promotion distinguishes.
    # Rank 8 matters on its own: it leads its node but is not the coordinator, so
    # it copies data without ever touching `.metadata`.
    _ROLES = {"coordinator_leader": (0, 0), "leader_only": (8, 0), "participant": (1, 1)}
    # The participant copies nothing, so it cannot be a copy-failure source.
    _COPYING_ROLES = ["coordinator_leader", "leader_only"]

    def _replay_every_role(self, make_staged, label, *, fails=None, peer_fails_from=None):
        """Run promotion once as each role and report what each one saw.

        Copies run for real unless ``fails(role, dst)`` says otherwise, so the
        assertions look at files that were actually written or actually withheld.
        A no-op copy mock would make "no marker was written" true for the wrong
        reason.

        ``peer_fails_from`` is the 1-based reduction index from which the group
        reports failure -- i.e. which phase a *different* rank broke in. Phase 1
        closes reduction 1, so a copy failure is 2 and a marker failure is 3;
        reporting earlier would skip the phase under test.
        """
        import shutil as _shutil

        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        real_copyfile = _shutil.copyfile
        results = {}
        for role, (global_rank, local_rank) in self._ROLES.items():
            stage_path, final_path = make_staged(f"{label}-{role}")
            seen = []

            def reduction(failed, group=None, _seen=seen):
                """Record this rank's flag, and report a peer's failure from ``peer_fails_from`` on."""
                _seen.append(failed)
                if peer_fails_from is not None and len(_seen) >= peer_fails_from:
                    return True
                return failed

            def copyfile(src, dst, _role=role):
                if fails is not None and fails(_role, dst):
                    # Mirror shutil: the destination exists before the write fails,
                    # which is what can leave a truncated file behind.
                    with open(dst, "w") as f:
                        f.write("partial")
                    raise OSError("boom")
                return real_copyfile(src, dst)

            raised = False
            with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True):
                with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=reduction):
                    with patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=global_rank):
                        with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=local_rank):
                            with patch("veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=copyfile):
                                try:
                                    _promote_staged_checkpoint(stage_path, final_path)
                                except (OSError, RuntimeError):
                                    raised = True
            results[role] = {"reductions": len(seen), "raised": raised, "final": final_path}
        return results

    @pytest.mark.parametrize("failing_role", _COPYING_ROLES)
    def test_every_role_runs_the_same_collectives_and_all_raise_when_a_copy_fails(self, make_staged, failing_role):
        """Collectives are untagged, so one rank skipping one desynchronises the job."""
        results = self._replay_every_role(
            make_staged,
            f"copyfail-{failing_role}",
            fails=lambda role, dst: role == failing_role and dst.endswith(".distcp"),
            peer_fails_from=2,
        )

        for role, r in results.items():
            assert r["raised"], f"{role} must raise once a peer failed"
            # One collective per phase, three phases -- including on the role that
            # does no work at all.
            assert r["reductions"] == 3, f"{role} ran {r['reductions']} collectives: {results}"
            assert not os.path.exists(os.path.join(r["final"], ".metadata")), f"{role} left a marker behind"

    def test_every_role_sees_a_failed_metadata_copy(self, make_staged):
        """The marker copy is part of the save, so its failure cannot stay with the coordinator.

        The coordinator's marker copy really is attempted and really does fail,
        leaving a partial file behind, so this also covers that cleanup.
        """
        results = self._replay_every_role(
            make_staged,
            "metafail",
            fails=lambda role, dst: role == "coordinator_leader" and dst.endswith(".metadata"),
            peer_fails_from=3,
        )

        for role, r in results.items():
            assert r["raised"], f"{role} must raise after a failed marker copy"
            assert r["reductions"] == 3, f"{role} did not reach the marker reduction: {results}"
            assert not os.path.exists(os.path.join(r["final"], ".metadata")), f"{role} left a marker"

        # The roles that copy data must still have copied all of it; only the
        # marker copy broke. Checking the exact set keeps this sensitive to a
        # partial copy, which "something was written" would not catch.
        for role in self._COPYING_ROLES:
            copied = sorted(n for n in os.listdir(results[role]["final"]) if n.endswith(".distcp"))
            assert copied == ["__0_0.distcp", "__0_1.distcp"], f"{role} copied {copied}"

    def test_a_failing_marker_copy_leaves_no_partial_marker(self, staged):
        """copyfile creates the destination before writing, so a half marker is possible."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged

        def copy_only_data(src, dst):
            if dst.endswith(".metadata"):
                with open(dst, "w") as f:
                    f.write("half")  # the destination exists before the failure
                raise OSError("marker write failed")
            with open(src) as r, open(dst, "w") as w:
                w.write(r.read())

        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch("veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=copy_only_data):
                with pytest.raises(OSError, match="marker write failed"):
                    _promote_staged_checkpoint(stage_path, final_path)

        assert os.path.exists(os.path.join(final_path, "__0_0.distcp")), "data should still be there"
        assert not os.path.exists(os.path.join(final_path, ".metadata")), "no partial marker may survive"

    def test_cleanup_still_runs_after_an_earlier_phase_failed(self, staged):
        """The staged copy is model-plus-optimizer sized; a failure must not strand it."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False):
            with patch(
                "veomni.checkpoint.dcp_checkpointer.shutil.copyfile", side_effect=OSError("destination is full")
            ):
                with pytest.raises(OSError, match="destination is full"):
                    _promote_staged_checkpoint(stage_path, final_path)
        assert not os.path.exists(stage_path)

    def test_stale_metadata_stays_removed_when_promotion_fails(self, staged):
        """Half-replaced data must not keep the previous checkpoint's marker."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        os.makedirs(final_path)
        with open(os.path.join(final_path, ".metadata"), "w") as f:
            f.write("previous checkpoint")

        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True):
            with patch("veomni.checkpoint.dcp_checkpointer.dist.barrier"):
                with patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0):
                    with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0):
                        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=True):
                            with pytest.raises(RuntimeError, match="failed on another rank"):
                                _promote_staged_checkpoint(stage_path, final_path)

        assert not os.path.exists(os.path.join(final_path, ".metadata"))

    def test_a_group_failure_still_frees_the_scratch_disk(self, staged):
        """A gloo timeout raises out of a phase; the staged copy must not outlive it."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        calls = []

        def broken_group(failed, group=None):
            """Fail the way a timed-out gloo group does, on the first reduction."""
            calls.append(failed)
            raise RuntimeError("Timed out waiting 5000ms for recv operation to complete")

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=broken_group),
            pytest.raises(RuntimeError, match="Timed out"),
        ):
            _promote_staged_checkpoint(stage_path, final_path)

        assert not os.path.exists(stage_path), "the staged copy outlived a broken group"
        assert not os.path.exists(os.path.join(final_path, ".metadata")), "published over a broken group"
        assert len(calls) == 1, f"the promotion went on past a broken group: {calls}"

    def test_a_group_failure_after_the_markers_retracts_them(self, staged):
        """The markers are copied *before* the phase that agrees on them, so a
        failure of the agreement itself would leave them standing over a save that
        never finished. A step being rewritten still carries the previous run's
        manifest until its cursor files are rewritten, and manifest plus marker is
        what resume reads as complete -- pairing this run's model state with the
        earlier run's cursor."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        calls = []

        def fails_the_closing_reduction(failed, group=None):
            """Agree twice, then fail the way a timed-out gloo group does."""
            calls.append(failed)
            if len(calls) == 3:
                raise RuntimeError("Timed out waiting 5000ms for recv operation to complete")
            return False

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=fails_the_closing_reduction),
            pytest.raises(RuntimeError, match="Timed out"),
        ):
            _promote_staged_checkpoint(stage_path, final_path)

        assert len(calls) == 3, f"the marker phase did not run: {calls}"
        assert not os.path.exists(os.path.join(final_path, ".metadata")), (
            "a marker outlived the agreement that was meant to publish it"
        )
        # The data is left where it is: without a marker the destination reads as
        # incomplete, and the next save clears it.
        assert os.path.exists(os.path.join(final_path, "__0_0.distcp"))
        assert not os.path.exists(stage_path)

    def test_markers_are_retracted_even_if_the_staged_tree_is_already_gone(self, staged):
        """The marker list is taken before the phases, not when the retraction runs.

        Retraction and the staged tree's removal both live in the same
        ``finally``, and on a shared staging directory a peer's node leader can
        get to the tree first. Deriving the list at retraction time would then
        find nothing to retract and leave the markers standing."""
        import shutil as _shutil

        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        calls = []

        def loses_the_stage_tree_then_fails(failed, group=None):
            """Agree twice, then fail with the staged tree already swept away."""
            calls.append(failed)
            if len(calls) == 3:
                _shutil.rmtree(stage_path, ignore_errors=True)
                raise RuntimeError("Timed out waiting 5000ms for recv operation to complete")
            return False

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=loses_the_stage_tree_then_fails),
            pytest.raises(RuntimeError, match="Timed out"),
        ):
            _promote_staged_checkpoint(stage_path, final_path)

        assert not os.path.exists(os.path.join(final_path, ".metadata")), (
            "the retraction depended on the staged tree that had already been swept"
        )

    def test_a_broken_group_keeps_this_ranks_error_as_the_cause(self):
        """A drain whose slot group timed out must not report the timeout alone.

        The group breaking is downstream of the save failing, so raising the
        group's error by itself hides what actually went wrong."""
        from veomni.utils.dist_utils import raise_if_any_rank_failed

        save_error = RuntimeError("disk full while writing shard 3")

        def broken_group(failed, group=None):
            raise RuntimeError("Timed out waiting 5000ms for recv operation to complete")

        with patch("veomni.utils.dist_utils.any_rank_failed", side_effect=broken_group):
            with pytest.raises(RuntimeError, match="Timed out") as raised:
                raise_if_any_rank_failed(save_error, "a pending async DCP save (ckpt)")

        assert raised.value.__cause__ is save_error

    def test_any_rank_failed_is_a_max_reduction(self):
        """One failing rank must make every rank see a failure."""
        import torch as _torch

        from veomni.utils.dist_utils import any_rank_failed

        with patch("veomni.utils.dist_utils.dist.is_initialized", return_value=False):
            assert any_rank_failed(True) is True
            assert any_rank_failed(False) is False

        seen_groups = []

        def fake_all_reduce(tensor, op=None, group=None):
            """Check the op, record the group, and report a failure."""
            assert op is _torch.distributed.ReduceOp.MAX, "must be MAX; SUM would overflow on many ranks"
            assert tensor.device.type == "cpu"
            seen_groups.append(group)
            tensor.fill_(1)

        gloo_group = object()
        with patch("veomni.utils.dist_utils.dist.is_initialized", return_value=True):
            with patch("veomni.utils.dist_utils.dist.all_reduce", side_effect=fake_all_reduce):
                with patch("veomni.utils.dist_utils.get_device_type", return_value="cpu"):
                    assert any_rank_failed(False) is True
                # gloo cannot reduce accelerator tensors (NPU included), so a gloo
                # group must never ask for the accelerator's device.
                with (
                    patch("veomni.utils.dist_utils.dist.get_backend", return_value="gloo"),
                    patch("veomni.utils.dist_utils.get_device_type", side_effect=AssertionError("asked for a device")),
                ):
                    assert any_rank_failed(False, group=gloo_group) is True
        assert seen_groups == [None, gloo_group], "the reduction must run on the group it was handed"

    def test_non_leader_non_coordinator_ranks_touch_nothing(self, staged):
        """Ranks other than the node leaders and the coordinator only participate in barriers."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path, final_path = staged
        with patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True):
            with patch("veomni.checkpoint.dcp_checkpointer.dist.barrier"):
                with patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=3):
                    with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=3):
                        _promote_staged_checkpoint(stage_path, final_path)
        assert not os.path.exists(final_path)
        assert os.path.exists(stage_path)


class TestStageDirValidation:
    def test_every_checkpoint_of_a_run_reuses_one_emptied_directory(self, tmp_path):
        """A killed save strands a model-plus-optimizer-sized copy on the scratch disk.

        One directory per run, emptied before each save, is what reclaims it: the
        next save on that node clears the copy instead of asking for the same space
        again on a disk already short of it.
        """
        import os as _os

        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False):
            abandoned = _prepare_stage_dir(str(tmp_path), "/remote/ckpt")
            with open(_os.path.join(abandoned, "big.distcp"), "w") as f:
                f.write("a save that never finished")

            current = _prepare_stage_dir(str(tmp_path), "/remote/ckpt")

        assert current == abandoned
        assert _os.listdir(current) == [], "the abandoned copy must not survive"

    def test_sweep_is_confined_to_veomni_own_directory(self, tmp_path):
        """stage_dir is the caller's, often /tmp; only our own subtree may be removed."""

        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        someone_elses = tmp_path / "someone_elses_data"
        someone_elses.mkdir()
        (someone_elses / "important.bin").write_text("not ours")

        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False):
            _prepare_stage_dir(str(tmp_path), "/remote/ckpt")

        assert (someone_elses / "important.bin").exists(), "swept outside our own root"

    def test_only_the_node_leader_touches_the_filesystem(self, tmp_path):
        """Peers sweeping or creating in the same place would race with the leader.

        They do not need to: the reduction is a collective, so the leader's work
        is done and visible by the time any rank leaves. Checks both halves --
        a peer neither creates the directory nor removes what the leader made.
        """
        import os as _os

        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False):
            with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=3):
                path = _prepare_stage_dir(str(tmp_path), "/remote/ckpt")
            assert not _os.path.exists(path), "a peer created the directory itself"

            with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0):
                assert _prepare_stage_dir(str(tmp_path), "/remote/ckpt") == path
            assert _os.path.isdir(path), "the leader must have created it"

            marker = _os.path.join(path, "written_by_the_leader")
            with open(marker, "w") as f:
                f.write("x")
            with patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=3):
                _prepare_stage_dir(str(tmp_path), "/remote/ckpt")
            assert _os.path.exists(marker), "a peer swept the leader's directory"

    def test_a_concurrent_run_sharing_stage_dir_is_not_swept(self, tmp_path):
        """stage_dir is often generic (/tmp); two runs on one node must not collide."""
        import os as _os

        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False):
            other = _prepare_stage_dir(str(tmp_path), "/remote/other_run")
            with open(_os.path.join(other, "in_flight.distcp"), "w") as f:
                f.write("another job is using this")

            _prepare_stage_dir(str(tmp_path), "/remote/this_run")

        assert _os.path.exists(_os.path.join(other, "in_flight.distcp")), "swept another run's data"

    def test_staging_dir_failure_is_agreed_across_ranks(self, tmp_path):
        """A scratch disk fails per node, so ranks that could still stage must stop too.

        Otherwise they go on into dcp.save and wait on a collective the failed
        ranks never reach.
        """
        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        # This rank prepares its directory fine; another node's did not.
        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=True):
            with pytest.raises(RuntimeError, match="another rank could not prepare"):
                _prepare_stage_dir(str(tmp_path), "/remote/ckpt")

    def test_local_staging_failure_raises_its_own_error(self, tmp_path):
        """The rank that actually failed reports what happened, not the peer message."""
        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda f, group=None: f):
            with patch("veomni.checkpoint.dcp_checkpointer.os.makedirs", side_effect=OSError("No space left")):
                with pytest.raises(OSError, match="No space left"):
                    _prepare_stage_dir(str(tmp_path), "/remote/ckpt")

    def test_failed_staged_overwrite_does_not_change_previous_scheduler(self, tmp_path):
        """A failed staged save must not mutate the live checkpoint's scheduler.

        ``.metadata`` still advertises the previous checkpoint until promotion.
        Writing the new ``lr_scheduler.pt`` into the destination first would let
        a resume load the old model and optimizer with the new scheduler; a
        failed ``dcp.save`` would leave the same mix. The sidecar has to live
        under ``stage_path`` until promotion copies it with the shards.
        """
        from veomni.checkpoint.dcp_checkpointer import _LR_SCHEDULER_FILENAME, DistributedCheckpointer

        final = tmp_path / "ckpt"
        model_root = final / "global_step_10" / "model"
        weights = model_root / "ckpt"
        weights.mkdir(parents=True)
        (weights / ".metadata").write_text("previous")
        (weights / "__0_0.distcp").write_text("old-weights")
        sidecar = model_root / _LR_SCHEDULER_FILENAME
        previous = {"last_epoch": 10, "base_lrs": [1e-4]}
        torch.save(previous, sidecar)

        new_scheduler = MagicMock()
        new_scheduler.state_dict.return_value = {"last_epoch": 99, "base_lrs": [1e-3]}

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed),
            patch.object(DistributedCheckpointer, "execute_save", side_effect=OSError("dcp write failed")),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            with pytest.raises(OSError, match="dcp write failed"):
                DistributedCheckpointer.save(
                    path=str(final),
                    state={"model": MagicMock(), "lr_scheduler": new_scheduler},
                    save_async=False,
                    global_steps=10,
                    stage_dir=str(tmp_path / "stage"),
                )

        assert torch.load(sidecar, weights_only=False) == previous
        assert (weights / ".metadata").read_text() == "previous"

    def test_a_failed_staged_save_over_a_legacy_step_leaves_it_resumable(self, tmp_path):
        """Same guarantee, one layout back. A staged save must not invalidate the
        destination before it has a complete copy to put there, and for a
        pre-split step the marker that makes it resumable sits at the step root.
        Dropping it when the save starts would cost the user a checkpoint the
        save never got far enough to replace.
        """
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
        from veomni.checkpoint.legacy_v0_1_12 import marker_path

        final = tmp_path / "ckpt"
        step_root = final / "global_step_10"
        step_root.mkdir(parents=True)
        legacy_marker = Path(marker_path(str(step_root)))
        legacy_marker.write_text("written by an older VeOmni")

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed),
            patch.object(DistributedCheckpointer, "execute_save", side_effect=OSError("dcp write failed")),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            with pytest.raises(OSError, match="dcp write failed"):
                DistributedCheckpointer.save(
                    path=str(final),
                    state={"model": MagicMock()},
                    save_async=False,
                    global_steps=10,
                    stage_dir=str(tmp_path / "stage"),
                )

        assert legacy_marker.read_text() == "written by an older VeOmni"

    def test_a_staged_save_drops_the_legacy_marker_when_it_promotes(self, tmp_path):
        """And once the copy is real, the old marker has to go: the step is no
        longer the pre-split checkpoint that marker describes."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint
        from veomni.checkpoint.legacy_v0_1_12 import marker_path

        stage_path = tmp_path / "stage"
        (stage_path / "ckpt").mkdir(parents=True)
        (stage_path / "ckpt" / ".metadata").write_text("new")
        (stage_path / "ckpt" / "__0_0.distcp").write_text("new-weights")

        step_root = tmp_path / "ckpt" / "global_step_10"
        step_root.mkdir(parents=True)
        legacy_marker = Path(marker_path(str(step_root)))
        legacy_marker.write_text("written by an older VeOmni")

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed),
        ):
            _promote_staged_checkpoint(str(stage_path), str(step_root / "model"), step_root=str(step_root))

        assert not legacy_marker.exists()
        assert (step_root / "model" / "ckpt" / ".metadata").read_text() == "new"

    def test_promotion_replaces_the_destination_rather_than_merging_into_it(self, tmp_path):
        """A weights-only save over a step that has an optimizer. Copying only the
        staged files would leave the previous save's optimizer shards and marker
        in place, and the step would then read as complete while pairing this
        run's weights with an earlier run's optimizer state."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path = tmp_path / "stage"
        (stage_path / "ckpt").mkdir(parents=True)
        (stage_path / "ckpt" / ".metadata").write_text("new")
        (stage_path / "ckpt" / "__0_0.distcp").write_text("new-weights")

        model_root = tmp_path / "ckpt" / "global_step_10" / "model"
        for name in ("ckpt", "optimizer"):
            (model_root / name).mkdir(parents=True)
            (model_root / name / ".metadata").write_text("previous")
            (model_root / name / "__0_0.distcp").write_text(f"old-{name}")

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed),
        ):
            _promote_staged_checkpoint(str(stage_path), str(model_root))

        assert not (model_root / "optimizer").exists()
        assert (model_root / "ckpt" / "__0_0.distcp").read_text() == "new-weights"
        assert (model_root / "ckpt" / ".metadata").read_text() == "new"

    def test_promotion_leaves_another_modules_directory_alone(self, tmp_path):
        """The destination emptied is one module's. A multi-module job promotes
        each module separately, and a sibling is not being written."""
        from veomni.checkpoint.dcp_checkpointer import _promote_staged_checkpoint

        stage_path = tmp_path / "stage"
        (stage_path / "ckpt").mkdir(parents=True)
        (stage_path / "ckpt" / ".metadata").write_text("new")

        model_root = tmp_path / "ckpt" / "global_step_10" / "model"
        sibling = model_root / "audio" / "ckpt"
        sibling.mkdir(parents=True)
        (sibling / ".metadata").write_text("a finished module")

        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=False),
            patch("veomni.checkpoint.dcp_checkpointer._local_rank", return_value=0),
            patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", side_effect=lambda failed, group=None: failed),
        ):
            _promote_staged_checkpoint(str(stage_path), str(model_root / "vision"))

        assert (sibling / ".metadata").read_text() == "a finished module"

    def test_unset_stage_dir_writes_straight_to_the_destination(self, tmp_path):
        """Staging is opt-in: unset, both DCP directories and the sidecar write
        straight into ``global_step_N/model/``."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        model_root = str(final / "global_step_10" / "model")
        with (
            patch.object(DistributedCheckpointer, "execute_save") as execute_save,
            patch.object(DistributedCheckpointer, "_create_storage_writer") as create_writer,
            patch.object(DistributedCheckpointer, "_save_lr_scheduler") as save_sched,
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
            patch("veomni.checkpoint.dcp_checkpointer.OptimizerState"),
            patch("veomni.checkpoint.dcp_checkpointer._prepare_stage_dir") as prepare,
            patch("veomni.checkpoint.dcp_checkpointer._promote_staged_checkpoint") as promote,
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock(), "optimizer": MagicMock()},
                save_async=False,
                global_steps=10,
            )

        assert save_sched.call_args.kwargs["checkpoint_dir"] == model_root
        assert save_sched.call_count == 1
        # Weights and optimizer are two directories and two saves, each with its
        # own writer and its own async slot.
        assert [call.args[0] for call in create_writer.call_args_list] == [
            os.path.join(model_root, "ckpt"),
            os.path.join(model_root, "optimizer"),
        ]
        assert [call.kwargs["slot"] for call in execute_save.call_args_list] == ["ckpt", "optimizer"]
        assert list(execute_save.call_args_list[0].kwargs["save_state"]) == ["model"]
        assert list(execute_save.call_args_list[1].kwargs["save_state"]) == ["optimizer"]
        prepare.assert_not_called()
        promote.assert_not_called()

    def test_rewriting_a_step_leaves_the_manifest_to_its_owner(self, tmp_path):
        """A rewrite has to invalidate the step, but each half invalidates its own.

        ``GlobalStateCallback`` writes ``checkpoint_manifest.json`` and clears it
        when it rewrites the cursor files; this class never writes it, so it must
        not delete it either. The step is still correctly rejected in between —
        completeness is the conjunction, and the ``.metadata`` files are gone."""
        from veomni.checkpoint import layout
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        step_root = str(final / "global_step_10")
        layout.write_manifest(step_root, global_step=10, world_size=1)
        weights = Path(layout.weights_dir(step_root))
        weights.mkdir(parents=True)
        (weights / layout.DCP_MARKER_FILENAME).write_text("from the previous run")

        with (
            patch.object(DistributedCheckpointer, "execute_save"),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock()},
                save_async=False,
                global_steps=10,
            )

        assert os.path.exists(layout.manifest_path(step_root))
        assert not layout.checkpoint_is_complete(step_root)

    def test_rewriting_a_legacy_step_drops_its_marker_too(self, tmp_path):
        """A pre-split step is vouched for by a ``.metadata`` at its root. Writing
        the current layout over it leaves that file untouched — nothing in this
        layout puts a marker there — so resume discovery would keep accepting the
        step through its legacy fallback while the rewrite is half done."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer
        from veomni.checkpoint.legacy_v0_1_12 import marker_path

        final = tmp_path / "ckpt"
        step_root = final / "global_step_10"
        step_root.mkdir(parents=True)
        legacy_marker = Path(marker_path(str(step_root)))
        legacy_marker.write_text("written by an older VeOmni")

        with (
            patch.object(DistributedCheckpointer, "execute_save"),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock()},
                save_async=False,
                global_steps=10,
            )

        assert not legacy_marker.exists()

    def test_rewriting_a_step_drops_the_modules_own_markers(self, tmp_path):
        """DCP rewrites ``.metadata`` at the end of a successful save, so a save
        that dies part-way would otherwise leave one describing shards that are
        half this step and half the last. Discovery reads those markers, so the
        step has to stop carrying them the moment it stops being true."""
        from veomni.checkpoint import layout
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        step_root = str(final / "global_step_10")
        markers = [Path(p) for p in (layout.weights_dir(step_root), layout.optimizer_dir(step_root))]
        for directory in markers:
            directory.mkdir(parents=True)
            (directory / layout.DCP_MARKER_FILENAME).write_text("from the previous run")

        seen = []
        with (
            patch.object(
                DistributedCheckpointer,
                "execute_save",
                side_effect=lambda **kw: seen.append([(d / layout.DCP_MARKER_FILENAME).exists() for d in markers]),
            ),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
            patch("veomni.checkpoint.dcp_checkpointer.OptimizerState"),
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock(), "optimizer": MagicMock()},
                save_async=False,
                global_steps=10,
            )

        assert seen == [[False, False], [False, False]]

    def test_rewriting_a_step_leaves_another_modules_markers_alone(self, tmp_path):
        """Only the module being written loses its markers. A sibling's
        checkpoint is not being overwritten, and deleting its marker would
        strand a module that is still perfectly complete."""
        from veomni.checkpoint import layout
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        step_root = str(final / "global_step_10")
        sibling = Path(layout.weights_dir(step_root, "audio"))
        sibling.mkdir(parents=True)
        (sibling / layout.DCP_MARKER_FILENAME).write_text("a finished module")

        with (
            patch.object(DistributedCheckpointer, "execute_save"),
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock()},
                save_async=False,
                global_steps=10,
                module="vision",
            )

        assert (sibling / layout.DCP_MARKER_FILENAME).exists()

    def test_a_rank_that_cannot_remove_the_markers_fails_the_group(self, tmp_path):
        """One rank owns the marker, so a failure to remove it starts out visible
        to that rank alone. Left there, its peers walk into ``dcp.save`` on a
        collective it never joins."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        with (
            patch("veomni.checkpoint.dcp_checkpointer.dist.is_initialized", return_value=True),
            patch("veomni.checkpoint.dcp_checkpointer.dist.get_rank", return_value=1),
            patch("veomni.utils.dist_utils.any_rank_failed", return_value=True),
            patch.object(DistributedCheckpointer, "execute_save") as execute_save,
        ):
            with pytest.raises(RuntimeError, match="removing the old DCP markers"):
                DistributedCheckpointer.save(
                    path=str(final),
                    state={"model": MagicMock()},
                    save_async=False,
                    global_steps=10,
                )

        # A non-zero rank never touches the file, and still raises — and nothing
        # was written into a step whose markers may still be standing.
        execute_save.assert_not_called()

    def test_save_without_optimizer_writes_only_the_weights(self, tmp_path):
        """A weights-only save leaves no empty optimizer directory behind."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        with (
            patch.object(DistributedCheckpointer, "execute_save") as execute_save,
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
        ):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock(), "optimizer": None},
                save_async=False,
                global_steps=10,
            )

        assert [call.kwargs["slot"] for call in execute_save.call_args_list] == ["ckpt"]

    def test_preparation_reduces_on_the_group_it_is_handed(self, tmp_path):
        """The staging directory's agreement belongs on the stage group like the promotion's."""
        from veomni.checkpoint.dcp_checkpointer import _prepare_stage_dir

        group = object()
        with patch("veomni.checkpoint.dcp_checkpointer.any_rank_failed", return_value=False) as agreed:
            _prepare_stage_dir(str(tmp_path), "/remote/ckpt", group=group)

        assert agreed.call_args.kwargs["group"] is group

    def test_staging_runs_on_the_stage_group(self, tmp_path):
        """On the training backend, a copy outlasting the watchdog aborts the process."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        group = object()
        with (
            patch.object(DistributedCheckpointer, "execute_save") as execute_save,
            patch.object(DistributedCheckpointer, "_create_storage_writer"),
            patch.object(DistributedCheckpointer, "_save_lr_scheduler"),
            patch.object(DistributedCheckpointer, "_get_stage_process_group", return_value=group) as get_group,
            patch("veomni.checkpoint.dcp_checkpointer.ModelState"),
            patch(
                "veomni.checkpoint.dcp_checkpointer._prepare_stage_dir", return_value=str(tmp_path / "s")
            ) as prepare,
            patch("veomni.checkpoint.dcp_checkpointer._promote_staged_checkpoint") as promote,
        ):
            DistributedCheckpointer.save(
                path=str(tmp_path / "ckpt"),
                state={"model": MagicMock()},
                global_steps=10,
                stage_dir=str(tmp_path / "stage"),
                save_timeout_seconds=1800,
            )

        assert get_group.call_args.args == (1800,)
        assert prepare.call_args.kwargs["group"] is group
        assert promote.call_args.kwargs["group"] is group
        assert {call.kwargs["timeout_seconds"] for call in execute_save.call_args_list} == {1800}

    def test_stage_key_does_not_collide_across_similar_paths(self):
        """Separator substitution maps /tmp/a_b/c and /tmp/a/b_c onto one directory."""
        from veomni.checkpoint.dcp_checkpointer import _stage_key

        assert _stage_key("/tmp/a_b/c") != _stage_key("/tmp/a/b_c")
        assert _stage_key("/remote/run") == _stage_key("/remote/run")
        assert _stage_key("/remote/run_a") != _stage_key("/remote/run_b")
        # a relative destination and its absolute spelling stage together
        assert _stage_key("run") == _stage_key(os.path.join(os.getcwd(), "run"))

    def test_stage_dir_with_save_async_is_rejected_before_any_side_effect(self, tmp_path):
        """The staged copy is dropped when save() returns, i.e. before an async write ends."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        final = tmp_path / "ckpt"
        with pytest.raises(ValueError, match="stage_dir cannot be combined with save_async"):
            DistributedCheckpointer.save(
                path=str(final),
                state={"model": MagicMock()},
                save_async=True,
                stage_dir=str(tmp_path / "stage"),
            )
        assert not final.exists()


class TestResumeDiscovery:
    """Which ``global_step_{N}/`` directories ``load_path: auto`` accepts.

    A current-layout step is two independent halves, and both have to be down:
    ``checkpoint_manifest.json`` for the per-rank cursor, and DCP's ``.metadata``
    for each directory the manifest says a module wrote. A pre-split step has
    neither and is recognised by the ``.metadata`` that used to sit at its root.
    """

    @staticmethod
    def _validate(step_dir_path):
        from veomni.utils.checkpoint_utils import _validate_dcp_checkpoint_entry

        return _validate_dcp_checkpoint_entry(str(step_dir_path.parent), step_dir_path.name)

    @staticmethod
    def _write_dcp(step, module="", optimizer=True):
        """Stand in for a finished DCP: the markers are all discovery reads."""
        from veomni.checkpoint import layout

        dirs = [layout.weights_dir(str(step), module)]
        if optimizer:
            dirs.append(layout.optimizer_dir(str(step), module))
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
            Path(directory, layout.DCP_MARKER_FILENAME).write_text("done")

    def test_a_complete_step_is_accepted(self, tmp_path):
        from veomni.checkpoint import layout

        step = tmp_path / "global_step_10"
        self._write_dcp(step)
        layout.write_manifest(str(step), global_step=10, world_size=1)

        assert self._validate(step) == 10

    def test_a_step_without_a_manifest_is_skipped(self, tmp_path):
        """The shards are down but no rank wrote its cursor, so there is nothing
        to resume the dataloader or the step counter from."""
        step = tmp_path / "global_step_10"
        self._write_dcp(step)

        assert self._validate(step) is None

    def test_a_step_whose_shards_are_still_streaming_is_skipped(self, tmp_path):
        """The manifest lands as soon as the cursor does, which under
        ``save_async`` is long before the shards finish. It cannot stand for the
        step on its own -- this is the case that makes the async write worth
        having, and the one that would silently resume half a checkpoint."""
        from veomni.checkpoint import layout

        step = tmp_path / "global_step_10"
        os.makedirs(layout.weights_dir(str(step)))
        layout.write_manifest(str(step), global_step=10, world_size=1)

        assert self._validate(step) is None

    def test_a_step_missing_one_modules_shards_is_skipped(self, tmp_path):
        """Discovery walks ``model/`` for ``ckpt/`` directories — a job is not
        resumable on half its models."""
        from veomni.checkpoint import layout

        step = tmp_path / "global_step_10"
        self._write_dcp(step, module="vision")
        os.makedirs(layout.weights_dir(str(step), "audio"))
        layout.write_manifest(str(step), global_step=10, world_size=1)

        assert self._validate(step) is None

    def test_a_step_saved_without_an_optimizer_is_accepted(self, tmp_path):
        """The export at train end drops the optimizer, so a step can be complete
        without one. Demanding a marker in a directory that does not exist would
        make such a step permanently unresumable."""
        from veomni.checkpoint import layout

        step = tmp_path / "global_step_10"
        self._write_dcp(step, optimizer=False)
        layout.write_manifest(str(step), global_step=10, world_size=1)

        assert self._validate(step) == 10

    def test_a_legacy_step_is_accepted_by_its_own_marker(self, tmp_path):
        from veomni.checkpoint.legacy_v0_1_12 import marker_path

        step = tmp_path / "global_step_10"
        step.mkdir()
        Path(marker_path(str(step))).write_text("legacy")

        assert self._validate(step) == 10

    def test_a_legacy_step_being_rewritten_is_skipped(self, tmp_path):
        """The current layout never writes a ``.metadata`` at the step root, so
        one sitting beside a ``model/`` means a rewrite landed on a pre-split
        step and did not finish. The loader prefers the new tree and would fail
        on whatever part of it never arrived, so the old marker must not keep the
        step discoverable."""
        from veomni.checkpoint import layout
        from veomni.checkpoint.legacy_v0_1_12 import marker_path

        step = tmp_path / "global_step_10"
        (step / layout.MODEL_DIRNAME / layout.WEIGHTS_DIRNAME).mkdir(parents=True)
        Path(marker_path(str(step))).write_text("legacy")

        assert self._validate(step) is None


def _staged_promotion_worker(rank: int, world_size: int, base: str, scenario: str) -> None:
    """One rank of the real promotion check; see ``TestPromotionAcrossRanks``."""
    import shutil
    import time

    from veomni.checkpoint import dcp_checkpointer as module

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = os.environ["_TEST_MASTER_PORT"]
    os.environ["LOCAL_RANK"] = str(rank)
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    stage, final = os.path.join(base, "stage"), os.path.join(base, "final")
    if rank == 0:
        os.makedirs(stage, exist_ok=True)
        for name in ("__0_0.distcp", ".metadata"):
            with open(os.path.join(stage, name), "w") as f:
                f.write(name)
    dist.barrier()

    real_copy = shutil.copyfile

    def instrumented_copy(src, dst):
        """Copy for real, except where the scenario makes rank 0 fail or stall."""
        if scenario == "copy_fails" and rank == 0:
            raise OSError("simulated: destination is full")
        if scenario == "outlives_timeout" and rank == 0:
            time.sleep(5)
        return real_copy(src, dst)

    checkpointer = module.DistributedCheckpointer
    checkpointer._stage_process_group = None
    group = checkpointer._get_stage_process_group(2 if scenario == "outlives_timeout" else 120)
    assert group is not None and group is not dist.group.WORLD, "staging must not run on the training group"

    raised = None
    with patch.object(module.shutil, "copyfile", instrumented_copy):
        try:
            module._promote_staged_checkpoint(stage, final, group=group)
        except BaseException as e:  # noqa: BLE001 - asserted by the test
            raised = f"{type(e).__name__} (runtime_error={isinstance(e, RuntimeError)}): {e}"

    with open(os.path.join(base, f"outcome_{rank}"), "w") as f:
        f.write(raised or "completed")

    dist.barrier()
    dist.destroy_process_group()


class TestPromotionAcrossRanks:
    """The promotion, run for real on four ranks over its own gloo group.

    Mocked reductions pin the bookkeeping. They cannot show that the group is
    separate from the training backend's, or that its timeout is what a rank left
    behind actually hits -- which is what this group exists for.
    """

    @staticmethod
    def _run(tmp_path, scenario):
        """Spawn four ranks for ``scenario`` and return each rank's outcome."""
        from tests.tools.launch_utils import find_free_port

        base = str(tmp_path / scenario)
        os.makedirs(base, exist_ok=True)
        os.environ["_TEST_MASTER_PORT"] = str(find_free_port())
        mp.spawn(_staged_promotion_worker, args=(4, base, scenario), nprocs=4, join=True)
        return base, [open(os.path.join(base, f"outcome_{r}")).read() for r in range(4)]

    def test_promotion_completes_on_every_rank(self, tmp_path):
        """Every rank finishes, the checkpoint lands with its marker, and the staged copy is freed."""
        base, outcomes = self._run(tmp_path, "ok")

        assert outcomes == ["completed"] * 4, outcomes
        assert sorted(os.listdir(os.path.join(base, "final"))) == [".metadata", "__0_0.distcp"]
        assert not os.path.exists(os.path.join(base, "stage")), "the staged copy was not freed"

    def test_one_ranks_failure_stops_every_rank_without_a_marker(self, tmp_path):
        """One rank's failed copy fails every rank and publishes no marker."""
        base, outcomes = self._run(tmp_path, "copy_fails")

        assert all(o != "completed" for o in outcomes), outcomes
        assert "destination is full" in outcomes[0]
        assert not os.path.exists(os.path.join(base, "final", ".metadata")), "published after a failed copy"
        assert not os.path.exists(os.path.join(base, "stage")), "the staged copy was not freed"

    def test_a_rank_that_outlives_the_group_timeout_fails_instead_of_hanging(self, tmp_path):
        """The deadline lives in the group, so every rank gives up on the same collective."""
        base, outcomes = self._run(tmp_path, "outlives_timeout")

        assert all(o != "completed" for o in outcomes), outcomes
        assert not os.path.exists(os.path.join(base, "final", ".metadata")), "published over a timed-out group"
        assert not os.path.exists(os.path.join(base, "stage")), "the staged copy outlived a timed-out group"
        # A collective timeout surfaces as a RuntimeError; its message is the backend's to change.
        assert any("runtime_error=True" in o for o in outcomes), outcomes


class TestSaveTimeoutConfig:
    """``CheckpointConfig`` rejects a timeout that cannot mean what it says."""

    @pytest.mark.parametrize("timeout", [0, -1, True, 1800.5, "1800"])
    def test_non_positive_or_non_integer_timeout_is_rejected(self, timeout):
        """The parser passes YAML scalars through, and ``true`` is an ``int`` to Python."""
        from veomni.arguments.arguments_types import CheckpointConfig

        with pytest.raises(ValueError, match="save_timeout_seconds must be a positive integer"):
            CheckpointConfig(save_timeout_seconds=timeout)

    def test_unset_timeout_defers_to_gloo(self):
        """Unset leaves gloo's own default in place."""
        from datetime import timedelta

        from veomni.arguments.arguments_types import CheckpointConfig
        from veomni.checkpoint.dcp_checkpointer import _gloo_timeout

        assert CheckpointConfig().save_timeout_seconds is None
        # ``new_group(timeout=None)`` is what falls back to the backend default.
        assert _gloo_timeout(None) is None
        assert _gloo_timeout(1800) == timedelta(seconds=1800)

    # An empty ``stage_dir`` is what the checkpointer reads as unset, so it has
    # nothing to bound either.
    @pytest.mark.parametrize("disabled", [{}, {"stage_dir": ""}])
    def test_a_timeout_with_nothing_to_bound_warns(self, disabled):
        """Neither path is on, so no gloo group is created and the value is inert."""
        from veomni.arguments.arguments_types import CheckpointConfig

        with patch("veomni.arguments.arguments_types.logger") as logger:
            CheckpointConfig(save_timeout_seconds=1800, **disabled)

        logger.warning_rank0.assert_called_once()
        assert "no effect" in logger.warning_rank0.call_args[0][0]

    def test_the_forbidden_pair_is_rejected_at_startup(self):
        """``save`` rejects it too, but only when the first checkpoint is due."""
        from veomni.arguments.arguments_types import CheckpointConfig

        with pytest.raises(ValueError, match="stage_dir cannot be combined with save_async"):
            CheckpointConfig(stage_dir="/scratch/ckpt", save_async=True)

    def test_a_numpy_integer_is_an_integer(self):
        """The guard is against ``bool`` and ``float``, not against every integer.

        A numpy scalar is not an ``int`` subclass, so an isinstance check would
        reject 1800 with a message saying it must be a positive integer."""
        from datetime import timedelta

        import numpy as np

        from veomni.arguments.arguments_types import CheckpointConfig
        from veomni.checkpoint.dcp_checkpointer import _gloo_timeout

        config = CheckpointConfig(stage_dir="/scratch/ckpt", save_timeout_seconds=np.int64(1800))
        # Normalized, because ``timedelta(seconds=...)`` rejects a numpy scalar.
        assert type(config.save_timeout_seconds) is int
        assert _gloo_timeout(config.save_timeout_seconds) == timedelta(seconds=1800)

    @pytest.mark.parametrize("enabled", [{"stage_dir": "/scratch/ckpt"}, {"save_async": True}])
    def test_a_timeout_either_path_uses_is_not_warned_about(self, enabled):
        from veomni.arguments.arguments_types import CheckpointConfig

        with patch("veomni.arguments.arguments_types.logger") as logger:
            CheckpointConfig(save_timeout_seconds=1800, **enabled)

        logger.warning_rank0.assert_not_called()


class TestShardingPlanDropHfKeys:
    """``_get_sharding_plan`` plans one shard entry per DCP key.

    A caller that knows two of those keys name the same tensor -- a state dict reports a
    shared tensor under every name it is reachable by -- has no way to say so from the
    checkpoint alone, since the metadata records the two names independently.
    ``drop_hf_keys`` is how it names them.
    """

    @pytest.fixture
    def checkpoint(self, tmp_path):
        """Two equally sized tensors, keyed as a tied checkpoint holds them.

        ``_normalize_key`` maps ``model.model.*`` to ``model.*`` and
        ``model.lm_head.weight`` to ``lm_head.weight``.
        """
        path = tmp_path / "checkpoint"
        dcp.save(
            {
                "model.model.embed_tokens.weight": torch.arange(4, dtype=torch.float32),
                "model.lm_head.weight": torch.arange(4, dtype=torch.float32),
            },
            checkpoint_id=str(path),
        )
        return str(path)

    def test_named_keys_are_left_out_of_the_plan(self, checkpoint):
        from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan

        # ``shard_size=None`` plans a single shard, so the result is one {hf_key: dcp_key}.
        plan, _, _ = _get_sharding_plan(checkpoint, None, "float32", drop_hf_keys={"lm_head.weight"})

        assert set(plan) == {"model.embed_tokens.weight"}

    def test_the_reported_total_follows_the_plan(self, checkpoint):
        """Otherwise the exported index advertises bytes nobody wrote."""
        from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan

        _, unfiltered_size, _ = _get_sharding_plan(checkpoint, None, "float32")
        _, filtered_size, _ = _get_sharding_plan(checkpoint, None, "float32", drop_hf_keys={"lm_head.weight"})

        assert filtered_size == unfiltered_size // 2

    def test_a_key_the_checkpoint_does_not_have_removes_nothing(self, checkpoint):
        """The two key spaces are only conventionally aligned, so a name may not match."""
        from veomni.checkpoint.dcp_checkpointer import _get_sharding_plan

        expected, expected_size, _ = _get_sharding_plan(checkpoint, None, "float32")
        plan, size, _ = _get_sharding_plan(checkpoint, None, "float32", drop_hf_keys={"model.absent.weight"})

        assert set(plan) == set(expected)
        assert size == expected_size

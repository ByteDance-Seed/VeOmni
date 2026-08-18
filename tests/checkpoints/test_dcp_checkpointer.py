"""Unit tests for distributed checkpoint and resume behavior.

Covers: OptimizerState (no placeholder synthesis), key normalization,
extra-state persistence, allow_partial_load planner, skip-HF resume, and
trainer step-counting correctness. Tests marked ``xfail`` document known
in-tree bugs — they become regression guards once the fix lands.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed.checkpoint as dcp
import torch.nn as nn

from veomni.distributed import torch_parallelize
from veomni.distributed.torch_parallelize import build_parallelize_model, parallelize_model_fsdp2
from veomni.models.module_utils import init_empty_weights
from veomni.trainer.callbacks.base import TrainerState
from veomni.utils.checkpoint_utils import should_skip_hf_weight_load


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

        state = {"model": model, "optimizer": optimizer, "extra_state": {}}

        mock_dcp.load = MagicMock()

        with patch.object(DistributedCheckpointer, "_load_extra_state"):
            with patch.object(DistributedCheckpointer, "_create_storage_reader") as mock_reader:
                mock_reader.return_value = MagicMock()
                DistributedCheckpointer.load(path="/fake", state=state)

        mock_dcp.load.assert_called_once()
        planner = mock_dcp.load.call_args.kwargs.get("planner")
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

    @pytest.mark.parametrize("parallelize", [build_parallelize_model, parallelize_model_fsdp2])
    def test_parallelize_apis_reject_renamed_skip_weights_load(self, parallelize):
        with pytest.raises(TypeError, match="'skip_weights_load' was renamed to 'should_skip_hf_weight_load'"):
            parallelize(MagicMock(), skip_weights_load=True)


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

        DistributedCheckpointer.save_future = None

    def test_noop_when_no_pending_save(self):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        DistributedCheckpointer.save_future = None
        # Should be a clean no-op — no exceptions, no barrier
        DistributedCheckpointer.wait_for_pending_save()
        assert DistributedCheckpointer.save_future is None

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_waits_and_clears_future(self, mock_dist):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        future = MagicMock()
        future.result.return_value = None
        DistributedCheckpointer.save_future = future

        DistributedCheckpointer.wait_for_pending_save()

        future.result.assert_called_once()
        assert DistributedCheckpointer.save_future is None
        mock_dist.barrier.assert_called_once()

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_propagates_exception_and_clears_future(self, mock_dist):
        """If the pending save raised, the exception propagates AND the
        future is cleared so retry on the next call is possible."""
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        future = MagicMock()
        future.result.side_effect = RuntimeError("save failed")
        DistributedCheckpointer.save_future = future

        with pytest.raises(RuntimeError, match="save failed"):
            DistributedCheckpointer.wait_for_pending_save()

        # Future must be cleared even on failure — otherwise stuck forever
        assert DistributedCheckpointer.save_future is None

    @patch("veomni.checkpoint.dcp_checkpointer.dist")
    def test_no_barrier_when_dist_not_initialized(self, mock_dist):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        mock_dist.is_initialized.return_value = False

        future = MagicMock()
        DistributedCheckpointer.save_future = future

        DistributedCheckpointer.wait_for_pending_save()

        future.result.assert_called_once()
        mock_dist.barrier.assert_not_called()


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
    @patch("veomni.trainer.callbacks.checkpoint_callback.build_checkpointer")
    @patch("veomni.trainer.callbacks.checkpoint_callback.dist")
    @patch("veomni.trainer.callbacks.checkpoint_callback.helper")
    def test_epoch_end_no_phantom_save_after_stop_iteration(self, mock_helper, mock_dist, mock_build_ckpt):
        from veomni.trainer.callbacks.checkpoint_callback import CheckpointerCallback

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
                ),
                accelerator=SimpleNamespace(fsdp_config=SimpleNamespace(fsdp_mode="fsdp2")),
                global_rank=0,
            ),
        )
        mock_build_ckpt.return_value = trainer.checkpointer
        trainer.checkpointer.save_future = None

        cb = CheckpointerCallback(trainer)
        cb.every_n_epochs = 1

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

        trainer.checkpointer.save.assert_not_called()


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


# ---------------------------------------------------------------------------
# Extra state save/load roundtrip
# ---------------------------------------------------------------------------


@patch("veomni.checkpoint.dcp_checkpointer.dist")
class TestExtraStateSaveLoad:
    def test_roundtrip(self, mock_dist, tmp_path):
        mock_dist.get_rank.return_value = 0
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        original_state = {
            "extra_state": {
                "global_step": 42,
                "lr_scheduler": {"last_epoch": 10, "base_lrs": [1e-4]},
                "torch_rng_state": torch.get_rng_state(),
            }
        }

        DistributedCheckpointer._save_extra_state(str(tmp_path), original_state)

        loaded_state = {"extra_state": {}}
        DistributedCheckpointer._load_extra_state(str(tmp_path), loaded_state)

        assert loaded_state["extra_state"]["global_step"] == 42
        assert loaded_state["extra_state"]["lr_scheduler"]["last_epoch"] == 10
        torch.testing.assert_close(
            loaded_state["extra_state"]["torch_rng_state"],
            original_state["extra_state"]["torch_rng_state"],
        )

    def test_missing_extra_state_key_save(self, mock_dist, tmp_path):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        state = {"model": MagicMock()}
        DistributedCheckpointer._save_extra_state(str(tmp_path), state)

    def test_missing_extra_state_key_load(self, mock_dist, tmp_path):
        from veomni.checkpoint.dcp_checkpointer import DistributedCheckpointer

        state = {"model": MagicMock()}
        DistributedCheckpointer._load_extra_state(str(tmp_path), state)

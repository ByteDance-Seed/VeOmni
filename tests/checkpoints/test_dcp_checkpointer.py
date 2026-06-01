"""Unit tests for DistributedCheckpointer internals.

Covers: OptimizerState (no placeholder synthesis), key normalization,
extra-state persistence, allow_partial_load planner, and trainer
step-counting correctness.  Tests marked ``xfail`` document known
in-tree bugs — they become regression guards once the fix lands.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from veomni.trainer.callbacks.base import TrainerState


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
    """DistributedCheckpointer.load() must pass allow_partial_load=True
    to DCP so checkpoints saved without placeholder state can be loaded."""

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

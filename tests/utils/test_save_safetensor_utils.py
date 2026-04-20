"""
Unit tests for veomni/utils/save_safetensor_utils.py

Covers get_model_save_state and _save_hf_safetensor_distributed,
focusing on the fqn_to_index_mapping filtering fix.

All tests run on CPU without distributed init.

Usage:
    python tests/utils/test_save_safetensor_utils.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Deferred imports inside _save_hf_safetensor_distributed:
#   from torch.distributed.checkpoint import HuggingFaceStorageWriter   (line 90)
#   from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch  (line 95)
#
# These are NOT module-level attributes of save_safetensor_utils, so we must
# patch them at their *source* module rather than the consumer.
#
# Module-level imports (dcp, dist, helper, gc, etc.) can be patched normally
# via "veomni.utils.save_safetensor_utils.<name>".

_MOD = "veomni.utils.save_safetensor_utils"


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 16)
        self.linear = nn.Linear(16, 16)
        self.lm_head = nn.Linear(16, 32, bias=False)


def _call_get_model_save_state(model, fqn_to_index_mapping, fake_state):
    """Invoke get_model_save_state with ModelState mocked at its source module."""
    mock_ms = MagicMock()
    mock_ms.state_dict.return_value = fake_state
    with patch("veomni.checkpoint.dcp_checkpointer.ModelState", return_value=mock_ms):
        from veomni.utils.save_safetensor_utils import get_model_save_state

        return get_model_save_state(model, fqn_to_index_mapping)


def _call_save_distributed(model, save_path, fqn_to_index_mapping, model_assets, fake_save_state):
    """
    Invoke _save_hf_safetensor_distributed with all deps mocked.

    Deferred imports are patched at their source modules;
    module-level imports are patched on save_safetensor_utils directly.

    Returns (mock_get_state, mock_writer_cls, mock_dcp_save, mock_save_assets).
    """
    mock_get_state = MagicMock(return_value=fake_save_state)
    mock_writer_cls = MagicMock()
    mock_dcp_save = MagicMock()
    mock_save_assets = MagicMock()

    with (
        # --- deferred imports: patch at source ---
        patch("torch.distributed.checkpoint.HuggingFaceStorageWriter", mock_writer_cls),
        patch("veomni.checkpoint.dcp_consolidation.apply_dcp_consolidation_patch", MagicMock()),
        # --- module-level imports: patch on consumer ---
        patch(f"{_MOD}.get_model_save_state", mock_get_state),
        patch(f"{_MOD}.dcp", MagicMock(save=mock_dcp_save)),
        patch(f"{_MOD}.dist", MagicMock(is_initialized=MagicMock(return_value=False))),
        patch(f"{_MOD}.save_model_assets", mock_save_assets),
        patch(f"{_MOD}.helper", MagicMock()),
        patch(f"{_MOD}.gc", MagicMock()),
        patch(f"{_MOD}.synchronize", MagicMock()),
    ):
        from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

        _save_hf_safetensor_distributed(
            model=model,
            save_path=save_path,
            fqn_to_index_mapping=fqn_to_index_mapping,
            model_assets=model_assets,
        )

    return mock_get_state, mock_writer_cls, mock_dcp_save, mock_save_assets


# ===========================================================================
# get_model_save_state tests
# ===========================================================================


class TestGetModelSaveState(unittest.TestCase):
    def setUp(self):
        self.model = _TinyModel()

    def test_returns_all_keys_when_mapping_is_none(self):
        fake = {
            "embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "linear.weight": torch.randn(16, 16, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        }
        result = _call_get_model_save_state(self.model, None, fake)
        self.assertEqual(set(result), set(fake))

    def test_fp32_converted_to_bf16(self):
        fake = {
            "fp32": torch.randn(4, 4, dtype=torch.float32),
            "bf16": torch.randn(4, 4, dtype=torch.bfloat16),
            "fp16": torch.randn(4, 4, dtype=torch.float16),
        }
        result = _call_get_model_save_state(self.model, None, fake)
        self.assertEqual(result["fp32"].dtype, torch.bfloat16)
        self.assertEqual(result["bf16"].dtype, torch.bfloat16)
        self.assertEqual(result["fp16"].dtype, torch.float16)

    def test_tied_weight_filtered_out(self):
        fake = {
            "embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "linear.weight": torch.randn(16, 16, dtype=torch.bfloat16),
            "lm_head.weight": torch.randn(32, 16, dtype=torch.bfloat16),
        }
        mapping = {"embed_tokens.weight": 0, "linear.weight": 0}
        result = _call_get_model_save_state(self.model, mapping, fake)
        self.assertNotIn("lm_head.weight", result)
        self.assertEqual(set(result), {"embed_tokens.weight", "linear.weight"})

    def test_excluded_module_keys_filtered(self):
        """Core regression test: MTP keys in mapping but not in save_state."""
        fake = {
            "embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "linear.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        }
        mapping = {
            "embed_tokens.weight": 0,
            "linear.weight": 0,
            "mtp_module.proj.weight": 1,
            "mtp_module.proj.bias": 1,
        }
        result = _call_get_model_save_state(self.model, mapping, fake)
        self.assertEqual(set(result), {"embed_tokens.weight", "linear.weight"})

    def test_empty_mapping_filters_everything(self):
        fake = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        self.assertEqual(_call_get_model_save_state(self.model, {}, fake), {})

    def test_empty_state_returns_empty(self):
        self.assertEqual(_call_get_model_save_state(self.model, {"k": 0}, {}), {})


# ===========================================================================
# _save_hf_safetensor_distributed tests
# ===========================================================================


class TestSaveHfSafetensorDistributed(unittest.TestCase):
    def setUp(self):
        self.model = _TinyModel()

    def test_mapping_filtered_before_writer(self):
        """Core test: extra MTP keys should be stripped before reaching the writer."""
        fake = {
            "embed_tokens.weight": torch.randn(32, 16, dtype=torch.bfloat16),
            "linear.weight": torch.randn(16, 16, dtype=torch.bfloat16),
        }
        mapping = {
            "embed_tokens.weight": 0,
            "linear.weight": 0,
            "mtp_head.proj.weight": 1,
            "mtp_head.proj.bias": 1,
        }
        _, writer_cls, _, _ = _call_save_distributed(
            self.model,
            "/tmp/test",
            mapping.copy(),
            None,
            fake,
        )
        passed = writer_cls.call_args.kwargs["fqn_to_index_mapping"]
        self.assertEqual(set(passed), {"embed_tokens.weight", "linear.weight"})

    def test_none_mapping_passed_through(self):
        fake = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        _, writer_cls, _, _ = _call_save_distributed(
            self.model,
            "/tmp/test",
            None,
            None,
            fake,
        )
        self.assertIsNone(writer_cls.call_args.kwargs["fqn_to_index_mapping"])

    def test_no_filtering_when_all_keys_match(self):
        fake = {"a": torch.randn(2, 2, dtype=torch.bfloat16), "b": torch.randn(2, 2, dtype=torch.bfloat16)}
        mapping = {"a": 0, "b": 0}
        _, writer_cls, _, _ = _call_save_distributed(
            self.model,
            "/tmp/test",
            mapping.copy(),
            None,
            fake,
        )
        self.assertEqual(writer_cls.call_args.kwargs["fqn_to_index_mapping"], mapping)

    def test_save_state_passed_to_dcp_save(self):
        fake = {"w": torch.randn(4, 4, dtype=torch.bfloat16)}
        _, _, dcp_save, _ = _call_save_distributed(
            self.model,
            "/tmp/test",
            None,
            None,
            fake,
        )
        dcp_save.assert_called_once()
        self.assertEqual(set(dcp_save.call_args.kwargs["state_dict"]), {"w"})

    def test_model_assets_saved(self):
        fake = {"w": torch.randn(2, 2, dtype=torch.bfloat16)}
        assets = [MagicMock()]
        _, _, _, save_assets = _call_save_distributed(
            self.model,
            "/tmp/test",
            None,
            assets,
            fake,
        )
        save_assets.assert_called_once_with("/tmp/test", assets)


# ===========================================================================
# Execution ordering: get_model_save_state BEFORE HuggingFaceStorageWriter
# ===========================================================================


class TestExecutionOrdering(unittest.TestCase):
    def test_save_state_computed_before_writer(self):
        call_order = []

        def fake_get_state(model, mapping):
            call_order.append("get_state")
            return {"a": torch.randn(2, 2, dtype=torch.bfloat16)}

        def fake_writer(**kwargs):
            call_order.append("writer_init")
            return MagicMock()

        with (
            patch("torch.distributed.checkpoint.HuggingFaceStorageWriter", MagicMock(side_effect=fake_writer)),
            patch("veomni.checkpoint.dcp_consolidation.apply_dcp_consolidation_patch", MagicMock()),
            patch(f"{_MOD}.get_model_save_state", fake_get_state),
            patch(f"{_MOD}.dcp", MagicMock(save=MagicMock())),
            patch(f"{_MOD}.dist", MagicMock(is_initialized=MagicMock(return_value=False))),
            patch(f"{_MOD}.save_model_assets", MagicMock()),
            patch(f"{_MOD}.helper", MagicMock()),
            patch(f"{_MOD}.gc", MagicMock()),
            patch(f"{_MOD}.synchronize", MagicMock()),
        ):
            from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

            _save_hf_safetensor_distributed(
                model=_TinyModel(),
                save_path="/tmp/test",
                fqn_to_index_mapping={"a": 0, "removed": 1},
                model_assets=None,
            )

        idx_get = call_order.index("get_state")
        idx_writer = call_order.index("writer_init")
        self.assertLess(idx_get, idx_writer, f"Wrong order: {call_order}")


if __name__ == "__main__":
    unittest.main()

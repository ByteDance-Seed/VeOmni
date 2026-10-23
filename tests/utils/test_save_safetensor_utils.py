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
_CONSOLIDATE = "torch.distributed.checkpoint._consolidate_hf_safetensors"


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
        patch(f"{_CONSOLIDATE}.consolidate_safetensors_files", MagicMock()),
        patch(f"{_CONSOLIDATE}.consolidate_safetensors_files_on_every_rank", MagicMock()),
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
            "int64": torch.randint(0, 8, (4, 4), dtype=torch.int64),
        }
        result = _call_get_model_save_state(self.model, None, fake)
        self.assertEqual(result["fp32"].dtype, torch.bfloat16)
        self.assertEqual(result["bf16"].dtype, torch.bfloat16)
        self.assertEqual(result["fp16"].dtype, torch.float16)
        self.assertEqual(result["int64"].dtype, torch.int64)

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

    def test_converter_export_weights_preserves_dtypes_and_filters_mapping(self):
        """Models whose converter defines export_weights bypass ModelState and keep exact dtypes."""
        from veomni.utils.save_safetensor_utils import get_model_save_state

        exported_items = [
            ("embed.weight", torch.ones(4, 4, dtype=torch.bfloat16)),
            ("layers.0.attn.wq_a.weight", torch.zeros(4, 4, dtype=torch.float8_e4m3fn)),
            ("layers.0.attn.wq_a.scale", torch.ones(1, 1, dtype=torch.float32)),
            ("layers.0.ffn.experts.0.w1.weight", torch.ones(4, 2, dtype=torch.int8)),
            ("layers.0.ffn.experts.0.w1.scale", torch.ones(4, 1, dtype=torch.float8_e8m0fnu)),
            ("layers.0.hc_attn_base", torch.ones(4, dtype=torch.float32)),
            ("unmapped.weight", torch.ones(2, dtype=torch.bfloat16)),
        ]

        class _ExportConverter:
            def can_handle(self, name):
                return True

            def convert(self, name, tensor):
                return None

            def export_weights(self, m):
                assert getattr(m, "_veomni_fqn_to_index_mapping", None) == mapping
                yield from exported_items

        model = _TinyModel()
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: _ExportConverter())
        mapping = {k: 1 for k, _ in exported_items if k != "unmapped.weight"}

        with (
            patch("veomni.checkpoint.dcp_checkpointer.ModelState") as mock_ms,
            patch(f"{_MOD}.dist", MagicMock(is_initialized=MagicMock(return_value=False))),
        ):
            result = get_model_save_state(model, mapping)

        mock_ms.assert_not_called()
        self.assertNotIn("unmapped.weight", result)
        self.assertEqual(result["embed.weight"].dtype, torch.bfloat16)
        self.assertEqual(result["layers.0.attn.wq_a.weight"].dtype, torch.float8_e4m3fn)
        self.assertEqual(result["layers.0.attn.wq_a.scale"].dtype, torch.float32)
        self.assertEqual(result["layers.0.ffn.experts.0.w1.weight"].dtype, torch.int8)
        self.assertEqual(result["layers.0.ffn.experts.0.w1.scale"].dtype, torch.float8_e8m0fnu)
        self.assertEqual(result["layers.0.hc_attn_base"].dtype, torch.float32)

    def test_converter_export_weights_partitions_across_ranks_when_distributed(self):
        from veomni.utils.save_safetensor_utils import get_model_save_state

        exported_items = [
            ("k_shard1", torch.ones(2, dtype=torch.bfloat16)),
            ("k_shard2", torch.ones(2, dtype=torch.float32)),
            ("k_shard3", torch.ones(2, dtype=torch.int8)),
        ]
        mapping = {"k_shard1": 1, "k_shard2": 2, "k_shard3": 3}

        class _ExportConverter:
            def can_handle(self, name):
                return True

            def convert(self, name, tensor):
                return None

            def export_weights(self, m):
                yield from exported_items

        model = _TinyModel()
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: _ExportConverter())

        for rank, expected_keys in ((0, {"k_shard2"}), (1, {"k_shard1", "k_shard3"})):
            mock_dist = MagicMock(
                is_initialized=MagicMock(return_value=True),
                get_rank=MagicMock(return_value=rank),
                get_world_size=MagicMock(return_value=2),
            )
            with patch(f"{_MOD}.dist", mock_dist):
                state = get_model_save_state(model, mapping)
            self.assertEqual(set(state.keys()), expected_keys)

    def test_converter_export_weights_passes_should_quantize_and_keeps_scale_on_same_owner(self):
        from veomni.utils.save_safetensor_utils import get_model_save_state

        raw_origins = [
            ("layers.0.attn.wq_a.weight", True),
            ("layers.0.attn.wkv.weight", True),
            ("layers.0.hc_attn_base", False),
        ]
        quant_calls_by_rank: dict[int, list[str]] = {0: [], 1: []}

        class _QuantAwareConverter:
            def can_handle(self, name):
                return True

            def convert(self, name, tensor):
                return None

            def export_weights(self, m, *, should_quantize=None):
                for origin, is_quant in raw_origins:
                    if should_quantize is not None and not should_quantize(origin):
                        continue
                    if is_quant:
                        quant_calls_by_rank[current_rank].append(origin)
                        yield origin, torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
                        yield f"{origin.removesuffix('.weight')}.scale", torch.ones(1, 1, dtype=torch.float32)
                    else:
                        yield origin, torch.ones(2, dtype=torch.float32)

        model = _TinyModel()
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: _QuantAwareConverter())

        states_by_rank = {}
        for current_rank in (0, 1):
            mock_dist = MagicMock(
                is_initialized=MagicMock(return_value=True),
                get_rank=MagicMock(return_value=current_rank),
                get_world_size=MagicMock(return_value=2),
            )
            with patch(f"{_MOD}.dist", mock_dist):
                states_by_rank[current_rank] = get_model_save_state(model, None)

        # Rank 0 owns idx 0 (wq_a + scale) and idx 2 (hc_attn_base); Rank 1 owns idx 1 (wkv + scale)
        self.assertEqual(quant_calls_by_rank[0], ["layers.0.attn.wq_a.weight"])
        self.assertEqual(quant_calls_by_rank[1], ["layers.0.attn.wkv.weight"])
        self.assertEqual(
            set(states_by_rank[0].keys()),
            {"layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale", "layers.0.hc_attn_base"},
        )
        self.assertEqual(
            set(states_by_rank[1].keys()),
            {"layers.0.attn.wkv.weight", "layers.0.attn.wkv.scale"},
        )

    def test_converter_export_weights_passes_should_quantize_with_multi_shard_mapping(self):
        from veomni.utils.save_safetensor_utils import get_model_save_state

        raw_origins = [
            ("layers.0.attn.wq_a.weight", True),
            ("layers.0.attn.wkv.weight", True),
            ("layers.0.hc_attn_base", False),
        ]
        mapping = {
            "layers.0.attn.wq_a.weight": 1,
            "layers.0.attn.wq_a.scale": 1,
            "layers.0.attn.wkv.weight": 2,
            "layers.0.attn.wkv.scale": 2,
            "layers.0.hc_attn_base": 2,
        }
        quant_calls_by_rank: dict[int, list[str]] = {0: [], 1: []}

        class _QuantAwareConverter:
            def can_handle(self, name):
                return True

            def convert(self, name, tensor):
                return None

            def export_weights(self, m, *, should_quantize=None):
                for origin, is_quant in raw_origins:
                    if should_quantize is not None and not should_quantize(origin):
                        continue
                    if is_quant:
                        quant_calls_by_rank[current_rank].append(origin)
                        yield origin, torch.zeros(2, 2, dtype=torch.float8_e4m3fn)
                        yield f"{origin.removesuffix('.weight')}.scale", torch.ones(1, 1, dtype=torch.float32)
                    else:
                        yield origin, torch.ones(2, dtype=torch.float32)

        model = _TinyModel()
        model._create_checkpoint_tensor_converter = staticmethod(lambda m: _QuantAwareConverter())

        states_by_rank = {}
        for current_rank in (0, 1):
            mock_dist = MagicMock(
                is_initialized=MagicMock(return_value=True),
                get_rank=MagicMock(return_value=current_rank),
                get_world_size=MagicMock(return_value=2),
            )
            with patch(f"{_MOD}.dist", mock_dist):
                states_by_rank[current_rank] = get_model_save_state(model, mapping)

        # Rank 0 owns shard 2 (2 % 2 == 0 -> wkv + scale, hc_attn_base); Rank 1 owns shard 1 (1 % 2 == 1 -> wq_a + scale)
        self.assertEqual(quant_calls_by_rank[0], ["layers.0.attn.wkv.weight"])
        self.assertEqual(quant_calls_by_rank[1], ["layers.0.attn.wq_a.weight"])
        self.assertEqual(
            set(states_by_rank[0].keys()),
            {"layers.0.attn.wkv.weight", "layers.0.attn.wkv.scale", "layers.0.hc_attn_base"},
        )
        self.assertEqual(
            set(states_by_rank[1].keys()),
            {"layers.0.attn.wq_a.weight", "layers.0.attn.wq_a.scale"},
        )

    def test_verify_hf_checkpoint_weights_supports_deepseek_v4_release_format(self):
        import os
        import tempfile

        from safetensors.torch import save_file

        from tests.checkpoints.checkpoint_verification_utils import verify_hf_checkpoint_weights
        from tests.models.tiny_configs import tiny_deepseek_v4_config
        from veomni.models.transformers.deepseek_v4.checkpoint_tensor_converter import (
            create_deepseek_v4_checkpoint_tensor_converter,
        )
        from veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_gpu import (
            DeepseekV4ForCausalLM,
        )
        from veomni.utils.save_safetensor_utils import get_model_save_state

        cfg = tiny_deepseek_v4_config()
        model = DeepseekV4ForCausalLM(cfg).float()
        model._create_checkpoint_tensor_converter = staticmethod(create_deepseek_v4_checkpoint_tensor_converter)
        dcp_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        with patch(f"{_MOD}.dist", MagicMock(is_initialized=MagicMock(return_value=False))):
            hf_release_state = get_model_save_state(model, None)

        self.assertIn("embed.weight", hf_release_state)
        self.assertIn("layers.0.ffn.experts.0.w1.weight", hf_release_state)
        self.assertEqual(hf_release_state["layers.0.hc_attn_base"].dtype, torch.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file(hf_release_state, os.path.join(tmp_dir, "model.safetensors"))
            self.assertTrue(verify_hf_checkpoint_weights(tmp_dir, dcp_state, safe_serialization=True))

        from tests.checkpoints.checkpoint_verification_utils import (
            _list_hf_checkpoint_keys,
            verify_dcp_to_hf_conversion,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            save_file(hf_release_state, os.path.join(tmp_dir, "model.safetensors"))
            self.assertEqual(_list_hf_checkpoint_keys(tmp_dir, safe_serialization=True), set(hf_release_state.keys()))
            with patch(
                "tests.checkpoints.checkpoint_verification_utils.load_dcp_checkpoint",
                return_value=dcp_state,
            ):
                self.assertTrue(verify_dcp_to_hf_conversion("/tmp/fake_dcp", tmp_dir, safe_serialization=True))
                self.assertFalse(
                    verify_dcp_to_hf_conversion("/tmp/fake_dcp", "/tmp/nonexistent_hf_dir", safe_serialization=True)
                )


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

    def test_none_mapping_becomes_size_balanced_files(self):
        fake = {"w": torch.randn(4, 4, dtype=torch.bfloat16), "b": torch.randn(4, dtype=torch.bfloat16)}
        with patch(f"{_MOD}._MAX_SHARD_BYTES", 32):
            _, writer_cls, _, _ = _call_save_distributed(
                self.model,
                "/tmp/test",
                None,
                None,
                fake,
            )
        self.assertEqual(writer_cls.call_args.kwargs["fqn_to_index_mapping"], {"w": 1, "b": 2})
        self.assertFalse(writer_cls.call_args.kwargs["enable_consolidation"])

    def test_none_mapping_consolidates_into_several_loadable_files(self):
        import tempfile
        from pathlib import Path

        from safetensors.torch import load_file

        fake = {f"layer{i}.weight": torch.randn(8, 8, dtype=torch.bfloat16) for i in range(5)}
        fake["ids"] = torch.arange(6, dtype=torch.int64)
        with tempfile.TemporaryDirectory() as output_dir, patch(f"{_MOD}._MAX_SHARD_BYTES", 300):
            with (
                patch(f"{_MOD}.get_model_save_state", MagicMock(return_value=dict(fake))),
                patch(f"{_MOD}.helper", MagicMock()),
            ):
                from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

                _save_hf_safetensor_distributed(self.model, output_dir, None, None)

            files = sorted(Path(output_dir).glob("model-*.safetensors"))
            self.assertGreater(len(files), 1)
            self.assertTrue((Path(output_dir) / "model.safetensors.index.json").exists())
            self.assertFalse((Path(output_dir) / "sharded").exists())
            loaded = {}
            for file in files:
                loaded.update(load_file(file))
        self.assertEqual(set(loaded), set(fake))
        for key, value in fake.items():
            torch.testing.assert_close(loaded[key], value)

    def test_consolidation_opens_each_shard_once(self):
        import builtins
        import tempfile
        from pathlib import Path

        import torch.distributed.checkpoint as dcp
        from torch.distributed.checkpoint import HuggingFaceStorageWriter
        from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files

        from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch

        apply_dcp_consolidation_patch()
        fake = {f"t{i}": torch.randn(4, 4, dtype=torch.bfloat16) for i in range(6)}
        with tempfile.TemporaryDirectory() as output_dir:
            sharded = str(Path(output_dir) / "sharded")
            mapping = dict.fromkeys(fake, 1)
            dcp.save(
                fake,
                storage_writer=HuggingFaceStorageWriter(sharded, save_distributed=True, fqn_to_index_mapping=mapping),
            )

            opened = []
            real_open = builtins.open

            def counting_open(file, *args, **kwargs):
                if str(file).startswith(sharded):
                    opened.append(str(file))
                return real_open(file, *args, **kwargs)

            with patch("builtins.open", counting_open):
                consolidate_safetensors_files(sharded, output_dir, mapping)

        shards = set(opened)
        self.assertEqual(len(shards), 1)
        # One read of the header while parsing metadata, one handle for all six tensors.
        self.assertEqual(len(opened), 2)

    def test_consolidation_keeps_e8m0_block_scales(self):
        """Block-scaled exports store their scales as e8m0, which older safetensors cannot name."""
        import tempfile
        from pathlib import Path

        import torch.distributed.checkpoint as dcp
        from safetensors.torch import load_file
        from torch.distributed.checkpoint import HuggingFaceStorageWriter
        from torch.distributed.checkpoint._consolidate_hf_safetensors import consolidate_safetensors_files

        from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch

        apply_dcp_consolidation_patch()
        fake = {
            "w.weight": torch.randn(4, 64).to(torch.float8_e4m3fn),
            "w.scale": torch.randint(120, 135, (4, 2), dtype=torch.uint8).view(torch.float8_e8m0fnu),
        }
        with tempfile.TemporaryDirectory() as output_dir:
            sharded = str(Path(output_dir) / "sharded")
            mapping = dict.fromkeys(fake, 1)
            dcp.save(
                fake,
                storage_writer=HuggingFaceStorageWriter(sharded, save_distributed=True, fqn_to_index_mapping=mapping),
            )
            consolidate_safetensors_files(sharded, output_dir, mapping)
            loaded = {}
            for file in Path(output_dir).glob("*.safetensors"):
                loaded.update(load_file(file))

        for key, value in fake.items():
            self.assertEqual(loaded[key].dtype, value.dtype)
            self.assertTrue(torch.equal(loaded[key].view(torch.uint8), value.view(torch.uint8)), key)

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
        fake = {
            "weight": torch.randn(4, 4, dtype=torch.bfloat16),
            "int64_buffer": torch.randint(0, 8, (4, 4), dtype=torch.int64),
            "bool_buffer": torch.tensor([True, False]),
        }
        _, _, dcp_save, _ = _call_save_distributed(
            self.model,
            "/tmp/test",
            None,
            None,
            fake,
        )
        dcp_save.assert_called_once()
        saved_state = dcp_save.call_args.kwargs["state_dict"]
        self.assertEqual(set(saved_state), {"weight", "int64_buffer", "bool_buffer"})
        self.assertEqual(saved_state["int64_buffer"].dtype, torch.int64)

        import json
        import tempfile
        from pathlib import Path

        import torch.distributed.checkpoint as dcp
        from safetensors.torch import load_file
        from torch.distributed.checkpoint import HuggingFaceStorageWriter
        from torch.distributed.checkpoint._hf_utils import CUSTOM_METADATA_KEY

        from veomni.checkpoint.dcp_consolidation import apply_dcp_consolidation_patch

        apply_dcp_consolidation_patch()
        with tempfile.TemporaryDirectory() as output_dir:
            storage_writer = HuggingFaceStorageWriter(
                path=output_dir,
                save_distributed=True,
                enable_consolidation=True,
                fqn_to_index_mapping=dict.fromkeys(fake, 1),
            )
            dcp.save(fake, storage_writer=storage_writer)

            consolidated_files = list(Path(output_dir).glob("*.safetensors"))
            self.assertEqual(len(consolidated_files), 1)
            loaded_state = load_file(consolidated_files[0])

        torch.testing.assert_close(loaded_state["int64_buffer"], fake["int64_buffer"])
        torch.testing.assert_close(loaded_state["bool_buffer"], fake["bool_buffer"])

        import torch.distributed.checkpoint._consolidate_hf_safetensors as hf_module

        input_files_data = {
            "shard_0": hf_module._InputFileData(
                metadata={
                    "int64_buffer": {hf_module.SHAPE_KEY: [2, 4], hf_module.DTYPE_KEY: "I64"},
                    hf_module.DEFAULT_EXTRA_METADATA_KEY: {
                        CUSTOM_METADATA_KEY: json.dumps({"int64_buffer": {hf_module.SAVED_OFFSETS_KEY: [0, 0]}})
                    },
                }
            ),
            "shard_1": hf_module._InputFileData(
                metadata={
                    "int64_buffer": {hf_module.SHAPE_KEY: [2, 4], hf_module.DTYPE_KEY: "I64"},
                    hf_module.DEFAULT_EXTRA_METADATA_KEY: {
                        CUSTOM_METADATA_KEY: json.dumps({"int64_buffer": {hf_module.SAVED_OFFSETS_KEY: [2, 0]}})
                    },
                }
            ),
        }
        output_files_data = {
            "model.safetensors": hf_module._OutputFileData(fqn_data={"int64_buffer": hf_module._FqnData()})
        }

        hf_module._parse_input_metadata(input_files_data, output_files_data)

        int64_buffer_data = output_files_data["model.safetensors"].fqn_data["int64_buffer"]
        self.assertEqual(int64_buffer_data.shape_in_file, [4, 4])
        self.assertEqual(int64_buffer_data.dtype_size, torch.empty((), dtype=torch.int64).element_size())

        if hasattr(torch, "float8_e8m0fnu"):
            e8m0_input = {
                "shard_0": hf_module._InputFileData(
                    metadata={
                        "scale": {hf_module.SHAPE_KEY: [4, 2], hf_module.DTYPE_KEY: "F8_E8M0"},
                        hf_module.DEFAULT_EXTRA_METADATA_KEY: {
                            CUSTOM_METADATA_KEY: json.dumps({"scale": {hf_module.SAVED_OFFSETS_KEY: [0, 0]}})
                        },
                    }
                ),
            }
            e8m0_output = {"model.safetensors": hf_module._OutputFileData(fqn_data={"scale": hf_module._FqnData()})}
            hf_module._parse_input_metadata(e8m0_input, e8m0_output)
            self.assertEqual(e8m0_output["model.safetensors"].fqn_data["scale"].dtype_size, 1)

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

    def test_mapping_filtered_against_global_keys_when_distributed(self):
        """When save_state keys are partitioned across ranks, every rank must pass the global key union to writer."""
        rank0_state = {"k_shard2": torch.ones(2, dtype=torch.bfloat16)}
        mapping = {"k_shard1": 1, "k_shard2": 2, "mtp.0.weight": 3}

        def fake_all_gather_object(output_list, obj):
            output_list[0] = ["k_shard2"]
            output_list[1] = ["k_shard1"]

        mock_dist = MagicMock(
            is_initialized=MagicMock(return_value=True),
            get_world_size=MagicMock(return_value=2),
            get_rank=MagicMock(return_value=0),
            all_gather_object=MagicMock(side_effect=fake_all_gather_object),
        )
        mock_writer_cls = MagicMock()

        with (
            patch("torch.distributed.checkpoint.HuggingFaceStorageWriter", mock_writer_cls),
            patch("veomni.checkpoint.dcp_consolidation.apply_dcp_consolidation_patch", MagicMock()),
            patch(f"{_CONSOLIDATE}.consolidate_safetensors_files_on_every_rank", MagicMock()),
            patch(f"{_MOD}.get_model_save_state", MagicMock(return_value=rank0_state)),
            patch(f"{_MOD}.dcp", MagicMock(save=MagicMock())),
            patch(f"{_MOD}.dist", mock_dist),
            patch(f"{_MOD}.save_model_assets", MagicMock()),
            patch(f"{_MOD}.helper", MagicMock()),
            patch(f"{_MOD}.gc", MagicMock()),
            patch(f"{_MOD}.synchronize", MagicMock()),
        ):
            from veomni.utils.save_safetensor_utils import _save_hf_safetensor_distributed

            _save_hf_safetensor_distributed(
                model=self.model,
                save_path="/tmp/test",
                fqn_to_index_mapping=mapping.copy(),
                model_assets=None,
            )

        passed = mock_writer_cls.call_args.kwargs["fqn_to_index_mapping"]
        self.assertEqual(passed, {"k_shard1": 1, "k_shard2": 2})


# ===========================================================================
# Execution ordering: get_model_save_state BEFORE HuggingFaceStorageWriter
# ===========================================================================


class TestExecutionOrdering(unittest.TestCase):
    def test_save_state_computed_before_writer(self):
        call_order = []

        def fake_get_state(model, mapping, parallel_state=None):
            call_order.append("get_state")
            return {"a": torch.randn(2, 2, dtype=torch.bfloat16)}

        def fake_writer(**kwargs):
            call_order.append("writer_init")
            return MagicMock()

        with (
            patch("torch.distributed.checkpoint.HuggingFaceStorageWriter", MagicMock(side_effect=fake_writer)),
            patch("veomni.checkpoint.dcp_consolidation.apply_dcp_consolidation_patch", MagicMock()),
            patch(f"{_CONSOLIDATE}.consolidate_safetensors_files", MagicMock()),
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


class TestConsolidationWait(unittest.TestCase):
    def test_ranks_wait_for_consolidation_on_a_long_timeout_cpu_group(self):
        import veomni.utils.save_safetensor_utils as mod

        group = object()
        fake_dist = MagicMock(is_initialized=MagicMock(return_value=True), new_group=MagicMock(return_value=group))
        on_every_rank = MagicMock()
        with (
            patch("torch.distributed.checkpoint.HuggingFaceStorageWriter", MagicMock()),
            patch("veomni.checkpoint.dcp_consolidation.apply_dcp_consolidation_patch", MagicMock()),
            patch(f"{_CONSOLIDATE}.consolidate_safetensors_files_on_every_rank", on_every_rank),
            patch(f"{_MOD}.get_model_save_state", lambda m, f, parallel_state=None: {"a": torch.zeros(2)}),
            patch(f"{_MOD}.dcp", MagicMock(save=MagicMock())),
            patch(f"{_MOD}.dist", fake_dist),
            patch(f"{_MOD}._consolidation_group", None),
            patch(f"{_MOD}.save_model_assets", MagicMock()),
            patch(f"{_MOD}.helper", MagicMock()),
            patch(f"{_MOD}.gc", MagicMock()),
        ):
            mod._save_hf_safetensor_distributed(
                model=_TinyModel(), save_path="/tmp/test", fqn_to_index_mapping=None, model_assets=None
            )

        kwargs = fake_dist.new_group.call_args.kwargs
        self.assertEqual(kwargs["backend"], "gloo")
        self.assertGreaterEqual(kwargs["timeout"].total_seconds(), 3600)
        self.assertIs(on_every_rank.call_args.kwargs["process_group"], group)


class TestLegacySaveModelWeights(unittest.TestCase):
    def test_save_dtype_only_casts_floating_tensors(self):
        import tempfile
        from pathlib import Path

        from safetensors.torch import load_file

        from veomni.models.checkpoint.weights import _get_shard_info, save_model_weights

        state_dict = {
            "weight": torch.ones(4, dtype=torch.float32),
            "expert_ids": torch.arange(4, dtype=torch.int64),
            "enabled": torch.tensor([True, False]),
        }

        _, total_size, _ = _get_shard_info(state_dict, "bfloat16", 5_000_000_000, True)
        expected_size = 4 * 2 + 4 * 8 + 2
        self.assertEqual(total_size, expected_size)

        with tempfile.TemporaryDirectory() as output_dir:
            save_model_weights(output_dir, state_dict, save_dtype="bfloat16")
            loaded = load_file(Path(output_dir) / "model.safetensors")

        self.assertEqual(loaded["weight"].dtype, torch.bfloat16)
        self.assertEqual(loaded["expert_ids"].dtype, torch.int64)
        self.assertEqual(loaded["enabled"].dtype, torch.bool)
        torch.testing.assert_close(loaded["expert_ids"], state_dict["expert_ids"])

    def test_native_dtypes_use_actual_size_for_torch_serialization(self):
        from veomni.models.checkpoint.weights import _get_shard_info

        state_dict = {
            "complex": torch.ones(3, dtype=torch.complex64),
            "quantized": torch.quantize_per_tensor(torch.ones(5), scale=0.1, zero_point=0, dtype=torch.qint8),
        }

        _, total_size, _ = _get_shard_info(state_dict, None, 5_000_000_000, False)
        expected_size = sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
        self.assertEqual(total_size, expected_size)

    def test_safetensors_rejects_unsupported_native_dtype(self):
        from veomni.models.checkpoint.weights import _get_shard_info

        with self.assertRaisesRegex(ValueError, "Unsupported dtype for safetensors serialization: torch.complex64"):
            _get_shard_info({"complex": torch.ones(1, dtype=torch.complex64)}, None, 5_000_000_000, True)

    def test_invalid_save_dtype_is_rejected_for_integer_only_state(self):
        from veomni.models.checkpoint.weights import _get_shard_info

        with self.assertRaisesRegex(ValueError, "Unknown save dtype: not_a_dtype"):
            _get_shard_info({"ids": torch.arange(4)}, "not_a_dtype", 5_000_000_000, False)

    def test_materialize_and_load_weights_caches_fqn_mapping_when_skipping_hf_load(self):
        from veomni.distributed.torch_parallelize import _materialize_and_load_weights
        from veomni.models.checkpoint import init_empty_weights

        with init_empty_weights():
            model = _TinyModel()
        model.init_weights = lambda: None
        mapping = {"embed_tokens.weight": 1, "linear.weight": 2}

        _materialize_and_load_weights(
            model,
            weights_path="/tmp/nonexistent",
            materialize_device="cpu",
            should_skip_hf_weight_load=True,
            is_peft_model=False,
            adapter_path=None,
            broadcast_from_rank0=False,
            fqn_to_index_mapping=mapping,
        )

        self.assertEqual(getattr(model, "_veomni_fqn_to_index_mapping", None), mapping)
        self.assertEqual(getattr(model, "_veomni_prepared_fqn_to_index_mapping", None), mapping)


if __name__ == "__main__":
    unittest.main()

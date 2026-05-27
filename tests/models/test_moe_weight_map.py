"""
Unit tests for MoE fqn_to_index_mapping conversion (model-class registration).

Usage:
    pytest tests/models/test_moe_weight_map.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from veomni.models.checkpoint_tensor_loading import (
    get_fqn_to_index_mapping_converter,
    maybe_convert_fqn_to_index_mapping,
    parse_fqn_to_index_mapping_from_json,
    resolve_fqn_to_index_mapping_for_save,
    shard_index_from_filename,
)
from veomni.models.transformers.qwen3_moe.checkpoint_tensor_converter import (
    convert_qwen3_moe_fqn_to_index_mapping,
)


class _ModelWithQwen3MoeFqnConverter:
    _convert_fqn_to_index_mapping = staticmethod(convert_qwen3_moe_fqn_to_index_mapping)


class TestMoeFqnMapping(unittest.TestCase):
    def test_shard_index_from_filename(self):
        self.assertEqual(shard_index_from_filename("model-00003-of-00014.safetensors"), 3)

    def test_converter_from_model_class(self):
        model = _ModelWithQwen3MoeFqnConverter()
        converter = get_fqn_to_index_mapping_converter(model)
        self.assertIs(converter, convert_qwen3_moe_fqn_to_index_mapping)

    def test_converter_missing_on_plain_module(self):
        self.assertIsNone(get_fqn_to_index_mapping_converter(torch.nn.Linear(1, 1)))

    def test_convert_per_expert_to_fused(self):
        mapping = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": 1,
            "model.layers.0.mlp.experts.0.up_proj.weight": 1,
            "model.layers.0.mlp.experts.0.down_proj.weight": 2,
            "model.layers.0.mlp.experts.1.gate_proj.weight": 3,
            "model.layers.0.mlp.experts.1.up_proj.weight": 3,
            "model.layers.0.mlp.experts.1.down_proj.weight": 4,
            "model.embed_tokens.weight": 1,
            "lm_head.weight": 1,
        }
        model = _ModelWithQwen3MoeFqnConverter()
        converted = maybe_convert_fqn_to_index_mapping(mapping, model)

        self.assertEqual(converted["model.embed_tokens.weight"], 1)
        self.assertEqual(converted["lm_head.weight"], 1)
        self.assertEqual(converted["model.layers.0.mlp.experts.gate_up_proj"], 1)
        self.assertEqual(converted["model.layers.0.mlp.experts.down_proj"], 2)
        self.assertNotIn("model.layers.0.mlp.experts.0.gate_proj.weight", converted)

    def test_maybe_convert_passes_through_without_handler(self):
        fused = {
            "model.layers.0.mlp.experts.gate_up_proj": 1,
            "model.layers.0.mlp.experts.down_proj": 2,
            "model.embed_tokens.weight": 1,
        }
        result = maybe_convert_fqn_to_index_mapping(fused, torch.nn.Linear(1, 1))
        self.assertIs(result, fused)

    def test_maybe_convert_none(self):
        self.assertIsNone(maybe_convert_fqn_to_index_mapping(None, MagicMock()))

    def test_resolve_uses_cache_from_load(self):
        model = MagicMock()
        cached = {"model.layers.0.mlp.experts.gate_up_proj": 1}
        model._veomni_prepared_fqn_to_index_mapping = cached
        raw = {"model.layers.0.mlp.experts.0.gate_proj.weight": 1}
        self.assertIs(resolve_fqn_to_index_mapping_for_save(model, raw), cached)


class TestSaveStateKeepsExpertsWithConvertedMapping(unittest.TestCase):
    def test_expert_keys_not_filtered_after_mapping_convert(self):
        per_expert_mapping = {
            "model.layers.0.mlp.experts.0.gate_proj.weight": 1,
            "model.layers.0.mlp.experts.0.up_proj.weight": 1,
            "model.layers.0.mlp.experts.0.down_proj.weight": 1,
            "model.embed_tokens.weight": 1,
        }
        converted_mapping = convert_qwen3_moe_fqn_to_index_mapping(per_expert_mapping)

        fake_state = {
            "model.embed_tokens.weight": torch.randn(8, 4, dtype=torch.bfloat16),
            "model.layers.0.mlp.experts.gate_up_proj": torch.randn(2, 8, 4, dtype=torch.bfloat16),
            "model.layers.0.mlp.experts.down_proj": torch.randn(2, 4, 8, dtype=torch.bfloat16),
        }

        mock_ms = MagicMock()
        mock_ms.state_dict.return_value = fake_state
        with patch("veomni.checkpoint.dcp_checkpointer.ModelState", return_value=mock_ms):
            from veomni.utils.save_safetensor_utils import get_model_save_state

            result = get_model_save_state(MagicMock(), converted_mapping)

        self.assertIn("model.layers.0.mlp.experts.gate_up_proj", result)
        self.assertIn("model.layers.0.mlp.experts.down_proj", result)
        self.assertIn("model.embed_tokens.weight", result)


if __name__ == "__main__":
    unittest.main()

import importlib

import pytest
import torch
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig


_MODEL_CASES = (
    (
        "veomni.models.transformers.qwen3_5.generated.patched_modeling_qwen3_5_gpu",
        "Qwen3_5TextModel",
    ),
    (
        "veomni.models.transformers.qwen3_5_moe.generated.patched_modeling_qwen3_5_moe_gpu",
        "Qwen3_5MoeTextModel",
    ),
)


@pytest.mark.parametrize(("module_name", "class_name"), _MODEL_CASES)
def test_linear_attention_mask_uses_cache_metadata(module_name, class_name):
    model_class = getattr(importlib.import_module(module_name), class_name)
    attention_mask = torch.ones(1, 3, dtype=torch.long)
    config = Qwen3_5TextConfig(num_hidden_layers=1, layer_types=["linear_attention"])
    cache = DynamicCache(config=config)

    assert model_class._update_linear_attn_mask(None, attention_mask, cache) is attention_mask

    cache.update_conv_state(torch.zeros(1, 1, config.linear_conv_kernel_dim), layer_idx=0)
    assert model_class._update_linear_attn_mask(None, attention_mask, cache) is None

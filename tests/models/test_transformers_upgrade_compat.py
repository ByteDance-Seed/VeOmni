"""Checkpoint and generation contracts that must survive a Transformers upgrade."""

import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import AutoConfig

from veomni.utils.device import IS_NPU_AVAILABLE


_TOY_CONFIGS = Path(__file__).parents[1] / "toy_config"


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_deepseek_v4_indexer_preserves_model_and_optimizer_keys(backend):
    modeling = importlib.import_module(
        f"veomni.models.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_{backend}"
    )
    config = AutoConfig.from_pretrained(_TOY_CONFIGS / "deepseek_v4_toy")
    indexer = modeling.DeepseekV4Indexer(config)
    expected = {
        "position_bias",
        "kv_proj.weight",
        "gate_proj.weight",
        "kv_norm.weight",
        "q_b_proj.weight",
        "weights_proj.weight",
    }
    assert set(indexer.state_dict()) == expected
    optimizer = torch.optim.AdamW(indexer.parameters())
    state = get_optimizer_state_dict(indexer, optimizer)
    assert set(state["param_groups"][0]["params"]) == expected
    model = modeling.DeepseekV4PreTrainedModel(config)
    assert "self_attn.compressor.indexer.weights_proj" in model._keep_in_fp32_modules
    assert all(".indexer.scorer." not in name for name in model._keep_in_fp32_modules)


@pytest.mark.parametrize(
    "family,prefix",
    [("qwen2_5_omni", "Qwen2_5Omni"), ("qwen3_omni_moe", "Qwen3OmniMoe")],
)
def test_omni_generation_prepares_text_positions(family, prefix):
    # Qwen2.5-Omni uses one shared generated module on both accelerators.
    backend = "npu" if IS_NPU_AVAILABLE and family == "qwen3_omni_moe" else "gpu"
    modeling = importlib.import_module(
        f"veomni.models.transformers.{family}.generated.patched_modeling_{family}_{backend}"
    )
    cls = getattr(modeling, f"{prefix}ThinkerForConditionalGeneration")
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    toy_name = "qwen25omni_toy" if family == "qwen2_5_omni" else "qwen3omni_toy"
    config = json.loads((_TOY_CONFIGS / toy_name / "config.json").read_text())
    model.config = getattr(modeling, f"{prefix}ThinkerConfig")(**config["thinker_config"])
    model.spatial_merge_size = model.config.vision_config.spatial_merge_size
    ids = torch.ones((1, 3), dtype=torch.long)
    positions = model._prepare_position_ids_for_generation(ids, {"attention_mask": torch.ones_like(ids)})
    torch.testing.assert_close(positions, torch.arange(3, dtype=positions.dtype).view(1, 1, 3).expand(4, 1, 3))

    # HF generation supplies a global False and no dummy audio lengths for a
    # silent video; training supplies a zero-length per-video placeholder.
    video_ids = torch.tensor(
        [[model.config.vision_start_token_id, model.config.video_token_id, model.config.vision_end_token_id]]
    )
    video_kwargs = {
        "video_grid_thw": torch.tensor([[1, model.spatial_merge_size, model.spatial_merge_size]]),
        "attention_mask": torch.ones_like(video_ids),
        "second_per_grids": torch.tensor([1.0]),
    }
    generation_positions = model.get_rope_index(video_ids, use_audio_in_video=False, **video_kwargs)
    training_positions = model.get_rope_index(video_ids, audio_seqlens=torch.tensor([0]), **video_kwargs)
    for generated, trained in zip(generation_positions, training_positions):
        torch.testing.assert_close(generated, trained)

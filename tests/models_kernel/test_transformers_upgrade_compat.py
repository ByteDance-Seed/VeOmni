"""Checkpoint and generation contracts that must survive a Transformers upgrade."""

import importlib
import json
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutputWithPast

from veomni.utils.device import IS_NPU_AVAILABLE


_TOY_CONFIGS = Path(__file__).parents[1] / "toy_config"


@pytest.mark.parametrize("backend", ["gpu", "npu"])
def test_deepseek_v4_indexer_matches_hf_model_and_optimizer_keys(backend):
    from transformers.models.deepseek_v4.modeling_deepseek_v4 import DeepseekV4Indexer

    modeling = importlib.import_module(
        f"veomni.models_kernel.transformers.deepseek_v4.generated.patched_modeling_deepseek_v4_{backend}"
    )
    config = AutoConfig.from_pretrained(_TOY_CONFIGS / "deepseek_v4_toy")
    indexer = modeling.DeepseekV4Indexer(config)
    expected = {
        "position_bias",
        "kv_proj.weight",
        "gate_proj.weight",
        "kv_norm.weight",
        "q_b_proj.weight",
        "scorer.weights_proj.weight",
    }
    assert set(indexer.state_dict()) == expected
    reference = DeepseekV4Indexer(config)
    indexer.load_state_dict(reference.state_dict(), strict=True)
    optimizer = torch.optim.AdamW(indexer.parameters())
    state = get_optimizer_state_dict(indexer, optimizer)
    assert set(state["param_groups"][0]["params"]) == expected
    reference_state = get_optimizer_state_dict(reference, torch.optim.AdamW(reference.parameters()))
    assert state["param_groups"] == reference_state["param_groups"]
    model = modeling.DeepseekV4PreTrainedModel(config)
    assert "self_attn.compressor.indexer.scorer.weights_proj" in model._keep_in_fp32_modules
    assert "self_attn.compressor.indexer.weights_proj" not in model._keep_in_fp32_modules


def _make_omni_thinker(family, prefix):
    # Qwen2.5-Omni uses one shared generated module on both accelerators.
    backend = "npu" if IS_NPU_AVAILABLE and family == "qwen3_omni_moe" else "gpu"
    modeling = importlib.import_module(
        f"veomni.models_kernel.transformers.{family}.generated.patched_modeling_{family}_{backend}"
    )
    cls = getattr(modeling, f"{prefix}ThinkerForConditionalGeneration")
    model = object.__new__(cls)
    torch.nn.Module.__init__(model)
    toy_name = "qwen25omni_toy" if family == "qwen2_5_omni" else "qwen3omni_toy"
    config = json.loads((_TOY_CONFIGS / toy_name / "config.json").read_text())
    model.config = getattr(modeling, f"{prefix}ThinkerConfig")(**config["thinker_config"])
    model.spatial_merge_size = model.config.vision_config.spatial_merge_size
    return model


@pytest.mark.parametrize(
    "family,prefix",
    [("qwen2_5_omni", "Qwen2_5Omni"), ("qwen3_omni_moe", "Qwen3OmniMoe")],
)
def test_omni_generation_prepares_text_positions(family, prefix):
    model = _make_omni_thinker(family, prefix)
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


class _PositionRecordingDecoder(torch.nn.Module):
    def forward(self, inputs_embeds, position_ids, **kwargs):
        self.position_ids = position_ids
        return BaseModelOutputWithPast(last_hidden_state=inputs_embeds)


@pytest.mark.parametrize(
    "family,prefix,use_audio_in_video",
    [
        ("qwen2_5_omni", "Qwen2_5Omni", None),
        ("qwen3_omni_moe", "Qwen3OmniMoe", None),
        ("qwen3_omni_moe", "Qwen3OmniMoe", False),
    ],
)
@pytest.mark.parametrize("num_videos", [1, 2])
def test_omni_thinker_forward_computes_video_positions(family, prefix, use_audio_in_video, num_videos):
    """Exercise the real forward-to-RoPE call when positions are not precomputed."""
    model = _make_omni_thinker(family, prefix)
    model.model = _PositionRecordingDecoder()
    model.lm_head = torch.nn.Identity()
    model.rope_deltas = None
    model.eval()
    config = model.config
    video_tokens = [config.vision_start_token_id, config.video_token_id, config.vision_end_token_id]
    ids = torch.tensor([[0, *video_tokens * num_videos]])
    mask = torch.ones_like(ids)
    mask[:, 0] = 0
    grid = torch.tensor([[1, model.spatial_merge_size, model.spatial_merge_size]] * num_videos)
    audio_lengths = torch.zeros(num_videos, dtype=torch.long) if use_audio_in_video is None else None
    seconds = torch.arange(1, num_videos + 1, dtype=torch.float)
    expected_positions, expected_deltas = model.get_rope_index(
        input_ids=ids,
        video_grid_thw=grid,
        attention_mask=mask,
        use_audio_in_video=use_audio_in_video,
        audio_seqlens=audio_lengths,
        second_per_grids=seconds,
    )
    # Supply already embedded inputs to isolate position computation from the
    # modality towers and decoder math; forward and get_rope_index remain real.
    embeds = torch.zeros(1, ids.shape[1], 4)
    output = model(
        input_ids=ids,
        inputs_embeds=embeds,
        attention_mask=mask,
        video_grid_thw=grid,
        use_audio_in_video=use_audio_in_video,
        audio_feature_lengths=audio_lengths,
        video_second_per_grid=seconds,
        image_mask=ids == config.image_token_id,
        video_mask=ids == config.video_token_id,
        audio_mask=ids == config.audio_token_id,
    )
    torch.testing.assert_close(model.model.position_ids, expected_positions, rtol=0, atol=0)
    torch.testing.assert_close(output.rope_deltas, expected_deltas - 1, rtol=0, atol=0)
    torch.testing.assert_close(output.logits, embeds, rtol=0, atol=0)

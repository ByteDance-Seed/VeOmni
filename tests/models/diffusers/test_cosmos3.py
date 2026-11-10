# SPDX-License-Identifier: Apache-2.0
"""Cosmos3-Nano tests: checkpoint names, the freeze switch, the loss reduction,
packed-vs-per-sample equivalence, the mRoPE time base, packing cost and the JSONL preprocessor."""

from __future__ import annotations

import pytest
import torch

from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_cosmos3_config
from veomni.models.diffusers.cosmos3 import cosmos3_condition, cosmos3_transformer  # noqa: F401
from veomni.models.diffusers.cosmos3.cosmos3_condition.configuration_cosmos3_condition import (
    Cosmos3ConditionModelConfig,
)
from veomni.models.diffusers.cosmos3.cosmos3_core import flow_matching_loss, latent_temporal_positions
from veomni.models.diffusers.cosmos3.cosmos3_transformer.configuration_cosmos3_transformer import (
    Cosmos3MoTTransformerConfig,
)
from veomni.models.diffusers.cosmos3.cosmos3_transformer.modeling_cosmos3_transformer import (
    TRAINABLE_NAME_KEYS,
    Cosmos3MoTTransformer,
)
from veomni.models.registry import MODEL_CONFIG_REGISTRY


# The released checkpoint's per-layer and network-level keys.
LAYER_KEY_SUFFIXES = frozenset(
    {
        "self_attn.to_q.weight",
        "self_attn.to_k.weight",
        "self_attn.to_v.weight",
        "self_attn.to_out.weight",
        "self_attn.norm_q.weight",
        "self_attn.norm_k.weight",
        "self_attn.add_q_proj.weight",
        "self_attn.add_k_proj.weight",
        "self_attn.add_v_proj.weight",
        "self_attn.to_add_out.weight",
        "self_attn.norm_added_q.weight",
        "self_attn.norm_added_k.weight",
        "mlp.gate_proj.weight",
        "mlp.up_proj.weight",
        "mlp.down_proj.weight",
        "mlp_moe_gen.gate_proj.weight",
        "mlp_moe_gen.up_proj.weight",
        "mlp_moe_gen.down_proj.weight",
        "input_layernorm.weight",
        "input_layernorm_moe_gen.weight",
        "post_attention_layernorm.weight",
        "post_attention_layernorm_moe_gen.weight",
    }
)
TOP_LEVEL_KEYS = frozenset(
    {
        "embed_tokens.weight",
        "lm_head.weight",
        "norm.weight",
        "norm_moe_gen.weight",
        "proj_in.weight",
        "proj_in.bias",
        "proj_out.weight",
        "proj_out.bias",
        "time_embedder.linear_1.weight",
        "time_embedder.linear_1.bias",
        "time_embedder.linear_2.weight",
        "time_embedder.linear_2.bias",
    }
)


def _tiny_model(attn_implementation: str = "eager", **overrides) -> Cosmos3MoTTransformer:
    torch.manual_seed(0)
    config = tiny_cosmos3_config(**overrides)
    ops = eager_ops_config()
    ops.attn_implementation = attn_implementation
    config._set_ops_implementation(ops)
    model = Cosmos3MoTTransformer(config)
    model.eval()
    return model


def _columns() -> dict[str, list]:
    """Two samples of different prompt length and latent geometry, as ``DiTDataCollator`` emits them."""
    generator = torch.Generator().manual_seed(1234)
    shapes = [(2, 3, 4, 6), (2, 2, 2, 4)]
    return {
        "input_ids": [torch.randint(0, 128, (length,), generator=generator) for length in (7, 3)],
        "hidden_states": [torch.randn(shape, generator=generator) for shape in shapes],
        "timestep": [torch.tensor([500.0]), torch.tensor([125.0])],
        "training_target": [torch.randn(shape, generator=generator) for shape in shapes],
        "condition_mask": [torch.tensor([1.0, 0.0, 0.0]).view(3, 1, 1), torch.zeros(2, 1, 1)],
        "fps": [5.0, 24.0],
    }


def test_the_transformer_points_at_its_condition_model():
    """``DiTTrainer`` builds the frozen side from this key; registry membership is covered by
    ``tests/models/base/test_auto_registry.py`` through ``_MODEL_CASES``."""
    condition_key = Cosmos3MoTTransformerConfig.condition_model_type
    assert Cosmos3ConditionModelConfig.model_type == condition_key
    assert MODEL_CONFIG_REGISTRY[condition_key]() is Cosmos3ConditionModelConfig


def test_weight_names_match_the_released_checkpoint():
    """Set equality with the checkpoint keys: a renamed module would otherwise load partially."""
    model = _tiny_model()
    names = {name for name, _ in model.named_parameters()}
    expected = set(TOP_LEVEL_KEYS)
    for index in range(model.config.num_hidden_layers):
        expected |= {f"layers.{index}.{suffix}" for suffix in LAYER_KEY_SUFFIXES}
    assert names == expected, sorted(names ^ expected)


def test_freezing_the_reasoner_keeps_only_the_generation_tower_trainable():
    model = _tiny_model(train_joint_reasoner=False)
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == any(key in name for key in TRAINABLE_NAME_KEYS), name
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    # The generation attention is exported as ``add_*``; the reasoner's ``to_q`` / ``norm_q`` stay frozen.
    assert "layers.0.self_attn.add_q_proj.weight" in trainable
    assert "layers.0.self_attn.norm_added_q.weight" in trainable
    assert "proj_in.weight" in trainable and "norm_moe_gen.weight" in trainable
    assert "layers.0.self_attn.to_q.weight" not in trainable
    assert "layers.0.self_attn.norm_q.weight" not in trainable
    assert "embed_tokens.weight" not in trainable


@pytest.mark.parametrize("train_joint", [False, True])
def test_lm_head_is_always_frozen(train_joint):
    """No AR loss, so ``lm_head`` never gets a gradient; joint training releases everything else."""
    model = _tiny_model(train_joint_reasoner=train_joint)
    model.freeze_model()
    assert model.lm_head.weight.requires_grad is False
    if train_joint:
        assert all(p.requires_grad for name, p in model.named_parameters() if not name.startswith("lm_head."))


def test_joint_training_is_rejected_under_ddp(monkeypatch):
    """The final norm and the last block's reasoner side never reach the loss, which DDP cannot tolerate."""
    from types import SimpleNamespace

    from veomni.models.diffusers.cosmos3.cosmos3_transformer import modeling_cosmos3_transformer as modeling

    monkeypatch.setattr(modeling, "get_parallel_state", lambda: SimpleNamespace(dp_mode="ddp"))
    with pytest.raises(NotImplementedError, match="fsdp2"):
        _tiny_model(train_joint_reasoner=True)
    _tiny_model(train_joint_reasoner=False)


def test_loss_is_a_scaled_per_sample_mean():
    """``DiTTrainer.postforward`` divides by the micro-batch count, so the model must not sum."""
    model = _tiny_model()
    columns = _columns()
    single = {key: values[:1] for key, values in columns.items()}
    duplicated = {key: values[:1] * 2 for key, values in columns.items()}
    loss_single = model(**single).loss["dm_loss"]
    torch.testing.assert_close(model(**duplicated).loss["dm_loss"], loss_single)

    prediction = model(**single).predictions[0]
    direct = flow_matching_loss(
        prediction.unsqueeze(0),
        columns["training_target"][0].unsqueeze(0),
        condition_mask=[columns["condition_mask"][0].unsqueeze(0)],
        loss_scale=model.config.loss_scale,
    )
    torch.testing.assert_close(loss_single, direct)


@pytest.mark.parametrize("attn_implementation", ["eager", "sdpa"])
def test_packed_forward_matches_per_sample_forward(attn_implementation):
    """Packing must isolate samples: each keeps its own positions, timestep, fps and mask."""
    model = _tiny_model(attn_implementation)
    columns = _columns()
    with torch.no_grad():
        packed = model(**columns).predictions
        for index, prediction in enumerate(packed):
            alone = model(**{key: values[index : index + 1] for key, values in columns.items()}).predictions[0]
            torch.testing.assert_close(prediction, alone, rtol=1e-5, atol=1e-5)


def test_gradient_checkpointing_keeps_the_gradients():
    columns = _columns()
    grads = []
    for checkpointing in (False, True):
        model = _tiny_model(train_joint_reasoner=True)
        model.train()
        if checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model(**columns).loss["dm_loss"].backward()
        grads.append({name: p.grad for name, p in model.named_parameters() if p.grad is not None})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[1][name], grads[0][name], msg=name)


def test_latent_temporal_step_is_base_fps_over_fps():
    """A 5 fps clip at base_fps=24 advances 4.8 per latent frame, not 1.0."""
    positions = latent_temporal_positions(6, fps=5.0)
    assert (positions[1:] - positions[:-1]).tolist() == pytest.approx([4.8] * 5, abs=1e-6)
    assert positions[0].item() == 0.0


def test_packed_length_counts_text_plus_latent_patches():
    """Frames and their offline latents price the same: (prompt + eos) + patches + 2 markers."""
    from veomni.models.diffusers.cosmos3.cosmos3_condition.packing import cosmos3_packed_length

    geometry = dict(spatial_compression_factor=16, temporal_compression_factor=4, latent_patch_size=2)
    ids = torch.ones(7, dtype=torch.long)
    # 17 frames -> 5 latent frames; 256x320 -> 16x20 latents -> 8x10 patches.
    from_frames = cosmos3_packed_length({"input_ids": ids, "videos": torch.zeros(17, 3, 256, 320)}, **geometry)
    from_latents = cosmos3_packed_length({"input_ids": ids, "latents": torch.zeros(48, 5, 16, 20)}, **geometry)
    assert from_frames == from_latents == (7 + 1) + 8 * 10 * 5 + 2

"""Tests for preparing the MiniMax-H3-Image release for offline T2I SFT."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME

from scripts.ci.accelerator.launch_plan import LaunchSpec
from scripts.prepare_minimax_h3_image_t2i import prepare_checkpoint
from tests.models.tiny_configs import tiny_minimax_h3_config
from tests.tools.training_utils import make_eager_ops_config
from veomni.models import build_config, build_foundation_model, load_model_weights
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import MiniMaxH3DiTModel
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.configuration_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModel,
)
from veomni.models.diffusers.minimax_h3_image.pdd import DEFAULT_SIGMAS


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def test_h3_image_dit_config_validates_effective_condition_type(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    saved = tiny_minimax_h3_config()
    saved.save_pretrained(tmp_path)
    ops = make_eager_ops_config()
    legacy = build_config(str(tmp_path), ops_implementation=ops)
    assert legacy.condition_model_type == "MiniMaxH3ConditionModel"
    assert legacy.audio_enabled and legacy.num_pdd_heads == 0

    overrides = {"condition_model_type": "MiniMaxH3ImageConditionModel", "audio_enabled": False, "num_pdd_heads": 2}
    for invalid_override in ({"audio_enabled": True}, {"num_pdd_heads": 0}):
        with pytest.raises(ValueError, match="audio-free DiT with PDD heads"):
            build_config(str(tmp_path), ops_implementation=ops, **(overrides | invalid_override))

    image = build_config(str(tmp_path), ops_implementation=ops, **overrides)
    assert image.condition_model_type == "MiniMaxH3ImageConditionModel"
    assert not image.audio_enabled and image.num_pdd_heads == 2


@pytest.fixture
def native_release(tmp_path):
    model_dir = tmp_path / "MiniMax-H3-Image"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = tiny_minimax_h3_config(audio_enabled=False, patch_size=(1, 2, 2))
    config._set_ops_implementation(make_eager_ops_config())
    model = MiniMaxH3DiTModel(config)
    state = {name: tensor.to(torch.bfloat16).contiguous() for name, tensor in model.dit.state_dict().items()}
    state["rope.inv_freq"] = model.dit.rope.inv_freq
    state["final_layer.video_out.weight"] = model.dit.final_layer.video_out.weight.detach()
    state["final_layer.video_out.bias"] = model.dit.final_layer.video_out.bias.detach()
    keys = sorted(state)
    weight_map = {}
    for shard_number, shard_keys in enumerate((keys[::2], keys[1::2]), start=1):
        filename = f"model-{shard_number:05d}-of-00002.safetensors"
        save_file({name: state[name] for name in shard_keys}, transformer_dir / filename)
        weight_map.update(dict.fromkeys(shard_keys, filename))
    (transformer_dir / SAFE_WEIGHTS_INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {"total_size": sum(tensor.nbytes for tensor in state.values())}, "weight_map": weight_map}
        )
    )
    config_dict = config.to_dict()
    for name in ("model_type", "architectures", "ops_implementation", "num_pdd_heads", "condition_model_type"):
        config_dict.pop(name, None)
    config_dict["_class_name"] = "MiniMaxH3DiTModel"
    (transformer_dir / "config.json").write_text(json.dumps(config_dict))

    pdd = {}
    sigmas = [1.0, 0.93, 0.45, 0.0]
    for suffix in ("weight", "bias"):
        shape = state[f"final_layer.video_out.{suffix}"].shape
        pdd[f"video_out.{suffix}"] = torch.randn(len(sigmas) - 1, *shape)
    save_file(pdd, model_dir / "pdd_fused_heads.safetensors")
    (model_dir / "pdd_config.json").write_text(json.dumps({"sigmas": sigmas}))
    return model_dir, state, pdd


def test_prepared_h3_image_checkpoint_loads_through_foundation_model(tmp_path, monkeypatch, native_release):
    monkeypatch.setenv("MODELING_BACKEND", "veomni")
    model_dir, state, pdd = native_release
    transformer_dir = model_dir / "transformer"
    before = {file: file.read_bytes() for file in model_dir.rglob("*") if file.is_file()}
    prepared = tmp_path / "prepared"
    prepared.mkdir()

    prepare_checkpoint(model_dir, prepared)

    assert before == {file: file.read_bytes() for file in model_dir.rglob("*") if file.is_file()}
    source_index = json.loads(before[transformer_dir / SAFE_WEIGHTS_INDEX_NAME])
    index = json.loads((prepared / SAFE_WEIGHTS_INDEX_NAME).read_text())
    for name, filename in source_index["weight_map"].items():
        assert index["weight_map"][name] == filename
        assert (prepared / filename).is_symlink()
        assert (prepared / filename).resolve() == transformer_dir / filename
    (supplemental,) = set(index["weight_map"].values()) - set(source_index["weight_map"].values())
    assert not (prepared / supplemental).is_symlink()
    assert index["metadata"]["total_size"] == source_index["metadata"]["total_size"] + sum(
        tensor.nbytes for tensor in pdd.values()
    )
    saved_heads = load_file(prepared / supplemental)
    expected_heads = {
        f"final_layer.pdd_out.{step}.{suffix}": tensor
        for suffix in ("weight", "bias")
        for step, tensor in enumerate(pdd[f"video_out.{suffix}"])
    }
    assert (
        saved_heads.keys() == expected_heads.keys() == index["weight_map"].keys() - source_index["weight_map"].keys()
    )
    for name, expected in expected_heads.items():
        torch.testing.assert_close(saved_heads[name], expected, rtol=0, atol=0)

    model = build_foundation_model(
        str(prepared),
        weights_path=str(prepared),
        torch_dtype="float32",
        init_device="meta",
        ops_implementation=make_eager_ops_config(),
    )
    load_model_weights(model, str(prepared), init_device="cpu")
    assert type(model) is MiniMaxH3DiTModel
    assert model.config.condition_model_type == "MiniMaxH3ImageConditionModel"
    assert model.config.audio_enabled is False
    assert "image_training" not in model.config.to_dict()
    condition_config = MiniMaxH3ImageConditionModelConfig.from_pretrained(
        str(prepared / "condition.json"), skip_encoder_load=True, ops_implementation=make_eager_ops_config()
    )
    sigmas = json.loads(before[model_dir / "pdd_config.json"])["sigmas"]
    assert condition_config.pdd_sigmas == tuple(sigmas)
    assert condition_config.num_pdd_heads == model.config.num_pdd_heads == len(model.dit.final_layer.pdd_out) == 3
    loaded = model.dit.state_dict() | {"rope.inv_freq": model.dit.rope.inv_freq}
    for name, expected in (state | expected_heads).items():
        torch.testing.assert_close(loaded[name], expected.to(loaded[name].dtype), rtol=0, atol=0)

    cached_batch = {
        "prompt_embeds": [torch.zeros(1, 3, model.config.text_dim)],
        "text_token_tags": [torch.ones(3, dtype=torch.long)],
        "clean_latents": [torch.zeros(1, model.config.latents_dim, 2, 4)],
    }
    condition_model = MiniMaxH3ImageConditionModel._from_config(condition_config, meta_init=True)
    assert model(**condition_model.process_condition(**cached_batch)).loss["mse_image"].isfinite()

    wrong_config = MiniMaxH3ImageConditionModelConfig(skip_encoder_load=True, pdd_sigmas=(1.0, 0.5, 0.0))
    wrong_config._set_ops_implementation(make_eager_ops_config())
    wrong_condition = MiniMaxH3ImageConditionModel._from_config(wrong_config, meta_init=True)
    with pytest.raises(ValueError, match="same number of PDD heads"):
        model(**wrong_condition.process_condition(**cached_batch))


def test_prepare_h3_image_checkpoint_rejects_pdd_schedule_mismatch(tmp_path, native_release):
    model_dir, _, pdd = native_release
    save_file({name: tensor[:-1] for name, tensor in pdd.items()}, model_dir / "pdd_fused_heads.safetensors")
    prepared = tmp_path / "prepared"
    with pytest.raises(ValueError, match="3 transitions but the checkpoint has 2 PDD heads"):
        prepare_checkpoint(model_dir, prepared)
    assert not prepared.exists()


@pytest.mark.parametrize(
    ("missing_file", "error_type"),
    [(True, FileNotFoundError), (False, ValueError)],
    ids=["missing-file", "missing-key"],
)
def test_prepare_h3_image_checkpoint_requires_explicit_pdd_sigmas(tmp_path, native_release, missing_file, error_type):
    model_dir, _, pdd = native_release
    num_heads = len(DEFAULT_SIGMAS) - 1
    save_file(
        {name: tensor[:1].expand(num_heads, *tensor.shape[1:]).contiguous() for name, tensor in pdd.items()},
        model_dir / "pdd_fused_heads.safetensors",
    )
    pdd_config = model_dir / "pdd_config.json"
    if missing_file:
        pdd_config.unlink()
    else:
        pdd_config.write_text("{}")
    prepared = tmp_path / "prepared"
    with pytest.raises(error_type, match="pdd_config.json"):
        prepare_checkpoint(model_dir, prepared)
    assert not prepared.exists()


def test_prepare_h3_image_checkpoint_rejects_index_shard_mismatch(tmp_path, native_release):
    model_dir, _, _ = native_release
    index_path = model_dir / "transformer" / SAFE_WEIGHTS_INDEX_NAME
    index = json.loads(index_path.read_text())
    weight_map = index["weight_map"]
    names = list(weight_map)
    weight_map[names[0]], weight_map[names[-1]] = weight_map[names[-1]], weight_map[names[0]]
    index_path.write_text(json.dumps(index))
    prepared = tmp_path / "prepared"
    with pytest.raises(ValueError, match="keys do not match its index entries"):
        prepare_checkpoint(model_dir, prepared)
    assert not prepared.exists()


def test_prepare_h3_image_checkpoint_preserves_existing_output(tmp_path, native_release):
    model_dir, _, _ = native_release
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    keep = prepared / "keep"
    keep.write_text("existing data")
    with pytest.raises(FileExistsError, match="Output must be a new or empty directory"):
        prepare_checkpoint(model_dir, prepared)
    assert list(prepared.iterdir()) == [keep]
    assert keep.read_text() == "existing data"

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
from veomni.models import build_foundation_model, load_model_weights
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import MiniMaxH3DiTModel


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


@pytest.fixture
def native_release(tmp_path):
    model_dir = tmp_path / "MiniMax-H3-Image"
    transformer_dir = model_dir / "transformer"
    transformer_dir.mkdir(parents=True)
    config = tiny_minimax_h3_config(audio_enabled=False)
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
    for name in ("model_type", "architectures", "ops_implementation", "num_pdd_heads", "image_training"):
        config_dict.pop(name, None)
    config_dict["_class_name"] = "MiniMaxH3DiTModel"
    (transformer_dir / "config.json").write_text(json.dumps(config_dict))

    pdd = {}
    for suffix in ("weight", "bias"):
        shape = state[f"final_layer.video_out.{suffix}"].shape
        pdd[f"video_out.{suffix}"] = torch.randn(6, *shape)
    save_file(pdd, model_dir / "pdd_fused_heads.safetensors")
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
    assert model.config.image_training is True
    assert len(model.dit.final_layer.pdd_out) == 6
    loaded = model.dit.state_dict() | {"rope.inv_freq": model.dit.rope.inv_freq}
    for name, expected in (state | expected_heads).items():
        torch.testing.assert_close(loaded[name], expected.to(loaded[name].dtype), rtol=0, atol=0)


def test_prepare_h3_image_checkpoint_rejects_incomplete_pdd_stack(tmp_path, native_release):
    model_dir, _, pdd = native_release
    save_file({name: tensor[:-1] for name, tensor in pdd.items()}, model_dir / "pdd_fused_heads.safetensors")
    prepared = tmp_path / "prepared"
    with pytest.raises(ValueError, match=r"PDD video_out.weight must be floating point with shape \(6,"):
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

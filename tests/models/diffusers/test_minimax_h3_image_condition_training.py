"""CPU coverage for MiniMax-H3-Image text-to-image and editing conditions."""

import json
from types import SimpleNamespace

import pytest
import torch
from datasets import load_dataset
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_minimax_h3_config
from veomni.data import build_data_transform
from veomni.models.diffusers.minimax_h3.minimax_h3_core.flow_match_scheduler import FlowMatchScheduler
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from veomni.models.diffusers.minimax_h3_image.layout import unpatchify_image_rows
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition import modeling_minimax_h3_image_condition
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.configuration_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModel,
)
from veomni.trainer.dit_trainer import DiTDataCollator, DiTTrainer, OfflineEmbeddingSaver


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _condition_model(seed=42, **kwargs):
    config = MiniMaxH3ImageConditionModelConfig(skip_encoder_load=True, height=32, width=64, seed=seed)
    config._set_ops_implementation(eager_ops_config())
    return MiniMaxH3ImageConditionModel._from_config(config, **kwargs)


def _cached_batch():
    return DiTDataCollator()(
        [
            {
                "clean_latents": torch.arange(16, dtype=torch.float32).view(1, 2, 2, 4) / 8,
                "prompt_embeds": torch.zeros(1, 3, 8, dtype=torch.bfloat16),
                "text_token_tags": torch.ones(3, dtype=torch.long),
            }
        ]
    )


def test_t2i_raw_condition_round_trips_through_dit_cache(tmp_path, monkeypatch):
    image = Image.new("L", (70, 20), 192)
    image.paste(33, (0, 0, 12, image.height))
    image.paste(230, (58, 0, 70, image.height))
    image.save(tmp_path / "target.png")
    transform = build_data_transform("minimax_h3_image_t2i", data_dir=str(tmp_path))
    rows = transform({"prompt": "A rectangle", "image": "target.png"}, source_name="example")
    assert rows[0]["images"][0].size == (70, 20)

    posterior_mode = torch.tensor([1.5, 3.0]).view(1, 2, 1, 1).expand(1, 2, 2, 4)

    def encode(pixels, return_dict):
        assert not return_dict
        assert not torch.is_grad_enabled()
        expected = (192.0 / 255 - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor([0.229, 0.224, 0.225])
        torch.testing.assert_close(pixels, expected.view(1, 3, 1, 1).expand(1, 3, 32, 64))
        return (SimpleNamespace(mode=lambda: posterior_mode),)

    def encode_text(prompt):
        assert prompt == "A rectangle"
        assert not torch.is_grad_enabled()
        return torch.ones(1, 3, 8, dtype=torch.bfloat16), torch.ones(3, dtype=torch.long)

    model = _condition_model()
    model._vae = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(latents_mean=(0.5, 1.0), latents_std=(2.0, 0.5)),
        encode=encode,
    )
    monkeypatch.setattr(model, "encode_text", encode_text)
    state = model.rng_state_dict()["generator"]
    raw_batch = DiTDataCollator()(rows)
    first_cache = model.get_condition(**raw_batch)
    second_cache = model.get_condition(**raw_batch)
    torch.testing.assert_close(first_cache, second_cache, rtol=0, atol=0)
    torch.testing.assert_close(model.rng_state_dict()["generator"], state)
    expected_latents = torch.tensor([0.5, 4.0]).view(1, 2, 1, 1).expand(1, 2, 2, 4)
    torch.testing.assert_close(first_cache["clean_latents"][0], expected_latents)

    import veomni.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))
    saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=1)
    saver.save(DiTTrainer._unpack_dict_of_list(first_cache)[0])
    dataset = load_dataset("parquet", data_files=str(tmp_path / "rank_0_shard_0.parquet"), split="train")
    cached_batch = DiTDataCollator()(build_data_transform("dit_offline")(dataset[0]))
    torch.testing.assert_close(dict(cached_batch), first_cache, rtol=0, atol=0)

    offline_model = _condition_model(meta_init=True)
    assert offline_model.text_encoder is None and offline_model.vae is None
    assert offline_model.process_condition(**cached_batch)["image_inputs"]["x"].shape == (2, 8)


def test_i2i_raw_condition_round_trips_through_dit_cache(tmp_path, monkeypatch):
    for name in ("target.png", "first.png", "second.png"):
        Image.new("RGB", (64, 32), 192).save(tmp_path / name)
    transform = build_data_transform("minimax_h3_image_i2i", data_dir=str(tmp_path))
    rows = transform(
        {"image": "target.png", "prompt": "Make it blue", "reference_images": json.dumps(["first.png", "second.png"])},
        source_name="example",
    )
    assert len(rows[0]["reference_images"]) == 2
    assert rows[0]["images"][0].size == (64, 32)

    model = _condition_model()
    target_mode = torch.ones(1, 2, 2, 4)
    model._vae = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(latents_mean=(0, 0), latents_std=(1, 1)),
        encode=lambda pixels, return_dict: (SimpleNamespace(mode=lambda: target_mode),),
    )

    def encode_text(prompt, reference_images):
        assert prompt == "Make it blue"
        assert [image.size for image in reference_images] == [(2048, 1024)] * 2
        return torch.ones(1, 3, 8), torch.tensor([1, 0, 1])

    def encode_references(images, sample_posterior):
        assert not sample_posterior
        assert [image.size for image in images] == [(2048, 1024)] * 2
        return [torch.ones(1, 2, 2, 4), torch.ones(1, 2, 4, 2)]

    monkeypatch.setattr(model, "encode_text", encode_text)
    monkeypatch.setattr(model, "encode_reference_images", encode_references)
    cached = model.get_condition(**DiTDataCollator()(rows))
    assert [ref.shape for ref in cached["reference_latents"][0]] == [(1, 2, 2, 4), (1, 2, 4, 2)]
    state = model.rng_state_dict()
    online = model.process_condition(**cached)
    assert online["image_num_reference_rows"] == 4
    assert online["image_inputs"]["x"].shape == (6, 8)
    assert online["image_training_target"].shape == (2, 8)
    torch.testing.assert_close(
        online["image_inputs"]["timestep_indices"][online["image_inputs"]["image_indices"]],
        torch.tensor([0, 0, 0, 0, 1, 1]),
    )
    torch.testing.assert_close(online["image_inputs"]["unique_timesteps"][0], torch.tensor(0.999))

    import veomni.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))
    saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=1)
    saver.save(DiTTrainer._unpack_dict_of_list(cached)[0])
    dataset = load_dataset("parquet", data_files=str(tmp_path / "rank_0_shard_0.parquet"), split="train")
    offline_batch = DiTDataCollator()(build_data_transform("dit_offline")(dataset[0]))
    offline_model = _condition_model(meta_init=True)
    offline_model.load_rng_state_dict(state)
    torch.testing.assert_close(offline_model.process_condition(**offline_batch), online, rtol=0, atol=0)


def test_mixed_raw_microbatch_matches_online_and_offline_conditions(tmp_path, monkeypatch):
    model = _condition_model()
    model._vae = SimpleNamespace(
        device=torch.device("cpu"),
        config=SimpleNamespace(latents_mean=(0, 0), latents_std=(1, 1)),
        encode=lambda pixels, return_dict: (SimpleNamespace(mode=lambda: pixels[:, :2, ::16, ::16]),),
    )

    def encode_text(prompt, reference_images=None):
        length = len(prompt.split()) + len(reference_images or [])
        embeds = torch.arange(length * 8, dtype=torch.float32).reshape(1, length, 8)
        return embeds, torch.ones(length, dtype=torch.long)

    def encode_references(images, sample_posterior):
        assert not sample_posterior
        return [torch.full((1, 2, 2, 4), index + 0.5) for index in range(len(images))]

    monkeypatch.setattr(model, "encode_text", encode_text)
    monkeypatch.setattr(model, "encode_reference_images", encode_references)
    raw = DiTDataCollator()(
        [
            {"inputs": "A rectangle", "images": [Image.new("RGB", (64, 32), 128)]},
            {
                "inputs": "Make the rectangle blue",
                "images": [Image.new("RGB", (32, 64), 192)],
                "reference_images": [Image.new("RGB", (64, 32)), Image.new("RGB", (32, 64))],
            },
        ]
    )
    state = model.rng_state_dict()
    cached = model.get_condition(**raw)
    assert cached["reference_latents"][0] == []
    assert len(cached["reference_latents"][1]) == 2
    assert [embeds.shape[1] for embeds in cached["prompt_embeds"]] == [2, 6]
    for index in range(2):
        expected = model.get_condition(**{key: values[index : index + 1] for key, values in raw.items()})
        torch.testing.assert_close({key: values[index : index + 1] for key, values in cached.items()}, expected)
    torch.testing.assert_close(model.rng_state_dict(), state, rtol=0, atol=0)
    online = model.process_condition(**cached)

    import veomni.distributed.parallel_state as parallel_state

    monkeypatch.setattr(parallel_state, "get_parallel_state", lambda: SimpleNamespace(dp_rank=0, dp_size=1))
    saver = OfflineEmbeddingSaver(str(tmp_path), dataset_length=2)
    for item in DiTTrainer._unpack_dict_of_list(cached):
        saver.save(item)
    dataset = load_dataset("parquet", data_files=str(tmp_path / "rank_0_shard_0.parquet"), split="train")
    transform = build_data_transform("dit_offline")
    offline = DiTDataCollator()([transform(item)[0] for item in dataset])
    torch.testing.assert_close(dict(offline), cached, rtol=0, atol=0)
    restored = _condition_model(meta_init=True)
    restored.load_rng_state_dict(state)
    torch.testing.assert_close(restored.process_condition(**offline), online, rtol=0, atol=0)
    torch.testing.assert_close(restored.rng_state_dict(), model.rng_state_dict(), rtol=0, atol=0)


def test_mixed_cached_microbatch_preserves_sample_geometry_and_rng_across_grouping():
    samples = [
        {key: values[0] for key, values in _cached_batch().items()},
        {
            "clean_latents": torch.ones(1, 2, 4, 2),
            "prompt_embeds": torch.ones(1, 5, 8, dtype=torch.bfloat16),
            "text_token_tags": torch.tensor([1, 0, 0, 1, 1]),
            "reference_latents": [torch.zeros(1, 2, 2, 2)],
        },
        {
            "clean_latents": torch.ones(1, 2, 4, 4),
            "prompt_embeds": torch.ones(1, 2, 8, dtype=torch.bfloat16),
            "text_token_tags": torch.ones(2, dtype=torch.long),
            "reference_latents": [torch.zeros(1, 2, 2, 4), torch.ones(1, 2, 4, 2)],
        },
    ]
    model, sequential = _condition_model(), _condition_model()
    batch = DiTDataCollator()(samples)
    packed = model.process_condition(**batch)
    expected = [sequential.process_condition(**DiTDataCollator()([sample])) for sample in samples]
    torch.testing.assert_close(DiTTrainer._unpack_dict_of_list(packed), expected, rtol=0, atol=0)
    torch.testing.assert_close(model.rng_state_dict(), sequential.rng_state_dict(), rtol=0, atol=0)
    assert [target.shape for target in packed["image_training_target"]] == [(2, 8), (2, 8), (4, 8)]
    assert packed["image_num_reference_rows"] == [0, 1, 4]
    assert [item["x"].shape[0] for item in packed["image_inputs"]] == [2, 3, 8]
    assert [item["prompt_embeds"].shape[1] for item in packed["image_inputs"]] == [3, 5, 2]

    state = model.rng_state_dict()
    next_batch = model.process_condition(**batch)
    restored = _condition_model(seed=0, meta_init=True)
    restored.load_rng_state_dict(state)
    replayed = [restored.process_condition(**DiTDataCollator()([sample])) for sample in samples]
    torch.testing.assert_close(DiTTrainer._unpack_dict_of_list(next_batch), replayed, rtol=0, atol=0)
    torch.testing.assert_close(model.rng_state_dict(), restored.rng_state_dict(), rtol=0, atol=0)


def test_i2i_requires_references_and_uses_deterministic_posterior(tmp_path):
    Image.new("RGB", (32, 32)).save(tmp_path / "target.png")
    transform = build_data_transform("minimax_h3_image_i2i", data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="reference_images"):
        transform({"image": "target.png", "prompt": "Edit it", "reference_images": "[]"}, source_name="example")

    model = _condition_model()
    model._vae = SimpleNamespace(
        device=torch.device("cpu"),
        spatial_compression_ratio=16,
        config=SimpleNamespace(latents_mean=(0, 0), latents_std=(1, 1)),
        encode=lambda pixels, return_dict: (SimpleNamespace(mode=lambda: torch.ones(1, 2, 2, 2)),),
    )
    state = model.rng_state_dict()
    reference = model.encode_reference_images([Image.new("RGB", (32, 32))], sample_posterior=False)
    torch.testing.assert_close(reference[0], torch.ones(1, 2, 2, 2))
    torch.testing.assert_close(model.rng_state_dict(), state, rtol=0, atol=0)


def test_i2i_condition_trains_tiny_transformer():
    condition = _condition_model(meta_init=True)
    cached = _cached_batch()
    cached["prompt_embeds"] = [torch.zeros(1, 3, 8)]
    cached["reference_latents"] = [[torch.ones(1, 2, 2, 4)]]
    batch = condition.process_condition(**cached)
    config = tiny_minimax_h3_config(
        audio_enabled=False, num_pdd_heads=6, image_training=True, patch_size=(1, 2, 2), text_dim=8
    )
    config._set_ops_implementation(eager_ops_config())
    model = MiniMaxH3DiTModel(config)
    output = model(**batch)
    assert output.predictions[0].shape == batch["image_training_target"].shape
    assert torch.isfinite(output.loss["mse_image"])
    output.loss["mse_image"].backward()
    assert torch.count_nonzero(model.dit.video_patch_proj.weight.grad) > 0


@pytest.mark.parametrize(
    ("seed", "step", "sigma", "model_time", "weight", "pdd_index"),
    [(42, 542, 0.6522877812, 0.3477122188, 1.4969605207, 3), (52, 989, 0.0240966268, 0.9759033918, 0.1255879402, 5)],
)
def test_t2i_schedule_matches_public_fp32_objective(seed, step, sigma, model_time, weight, pdd_index):
    model = _condition_model(seed=seed, meta_init=True)
    batch = _cached_batch()
    result = model.process_condition(**batch)
    image_inputs = result["image_inputs"]

    assert image_inputs["pdd_step_index"] == pdd_index
    torch.testing.assert_close(image_inputs["unique_timesteps"], torch.tensor([model_time]))
    torch.testing.assert_close(result["image_loss_weight"], torch.tensor(weight))
    torch.testing.assert_close(image_inputs["timestep_indices"], torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(image_inputs["text_indices"], torch.arange(3))
    torch.testing.assert_close(image_inputs["image_indices"], torch.arange(3, 5))
    torch.testing.assert_close(image_inputs["token_tags"], torch.tensor([1, 1, 1, 0, 0]))
    torch.testing.assert_close(image_inputs["prompt_embeds"], batch["prompt_embeds"][0])
    assert image_inputs["position_ids"].shape == (5, 3)
    assert image_inputs["position_ids"].dtype == torch.float32

    generator = torch.Generator(device="cpu").manual_seed(seed)
    assert torch.randint(1000, (1,), generator=generator).item() == step
    clean_latents = batch["clean_latents"][0]
    noise = torch.randn(clean_latents.shape, generator=generator)
    noised_latents = unpatchify_image_rows(image_inputs["x"], 2, 4, 2, (1, 2, 2))
    target = unpatchify_image_rows(result["image_training_target"], 2, 4, 2, (1, 2, 2))
    torch.testing.assert_close(noised_latents, (1 - sigma) * clean_latents + sigma * noise)
    torch.testing.assert_close(target, noise - clean_latents)


def test_t2i_schedule_matches_h3_scheduler_at_all_training_steps():
    model = _condition_model(meta_init=True)
    reference = FlowMatchScheduler(shift=2.22)
    reference.set_timesteps(1000, training=True)
    torch.testing.assert_close(model._scheduler.sigmas, reference.sigmas, rtol=0, atol=0)
    torch.testing.assert_close(
        model._scheduler.linear_timesteps_weights, reference.linear_timesteps_weights, rtol=0, atol=0
    )
    torch.testing.assert_close(model._training_model_times, 1 - reference.timesteps / 1000, rtol=0, atol=0)


def test_t2i_rng_checkpoint_replays_next_timestep_and_noise():
    model = _condition_model()
    batch = _cached_batch()
    first = model.process_condition(**batch)
    state = model.rng_state_dict()
    second = model.process_condition(**batch)
    restored_model = _condition_model(seed=0, meta_init=True)
    restored_model.load_rng_state_dict(state)
    repeated = restored_model.process_condition(**batch)

    assert not torch.equal(first["image_training_target"], second["image_training_target"])
    assert not torch.equal(first["image_inputs"]["unique_timesteps"], second["image_inputs"]["unique_timesteps"])
    torch.testing.assert_close(second, repeated, rtol=0, atol=0)


def test_t2i_rng_uses_independent_data_parallel_streams(monkeypatch):
    monkeypatch.setattr(modeling_minimax_h3_image_condition, "is_parallel_state_initialized", lambda: True)
    mesh_state = SimpleNamespace(dp_rank=0)
    monkeypatch.setattr(modeling_minimax_h3_image_condition, "get_parallel_state", lambda: mesh_state)
    first_rank = _condition_model()
    mesh_state.dp_rank = 1
    second_rank = _condition_model()

    batch = _cached_batch()
    assert not torch.equal(
        first_rank.process_condition(**batch)["image_training_target"],
        second_rank.process_condition(**batch)["image_training_target"],
    )
    state = second_rank.rng_state_dict()
    expected = second_rank.process_condition(**batch)
    resumed = _condition_model()
    resumed.load_rng_state_dict(state)
    torch.testing.assert_close(resumed.process_condition(**batch), expected, rtol=0, atol=0)


def test_t2i_rejects_invalid_training_geometry_and_microbatch():
    with pytest.raises(ValueError, match="positive multiples of 32"):
        MiniMaxH3ImageConditionModelConfig(width=1000)

    model = _condition_model()
    batch = _cached_batch()
    batch["text_token_tags"] *= 2
    with pytest.raises(ValueError, match="aligned condition columns"):
        model.process_condition(**batch)
    with pytest.raises(ValueError, match="nonempty aligned condition columns"):
        model.process_condition(prompt_embeds=[], text_token_tags=[], clean_latents=[])
    with pytest.raises(ValueError, match="aligned input and image columns"):
        model.get_condition(["A rectangle", "Another rectangle"], [[]])
    with pytest.raises(ValueError, match="exactly one target image"):
        model.get_condition(["A rectangle"], [[]])

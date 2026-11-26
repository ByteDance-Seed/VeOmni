"""CPU coverage for MiniMax-H3-Image offline text-to-image conditions."""

from types import SimpleNamespace

import pytest
import torch
from datasets import load_dataset
from PIL import Image

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import eager_ops_config
from veomni.data import build_data_transform
from veomni.models.diffusers.minimax_h3.minimax_h3_core.flow_match_scheduler import FlowMatchScheduler
from veomni.models.diffusers.minimax_h3_image.layout import unpatchify_image_rows
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition import modeling_minimax_h3_image_condition
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.configuration_minimax_h3_image_condition import (
    MiniMaxH3ImageConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
    REFERENCE_MODEL_TIME,
    MiniMaxH3ImageConditionModel,
)
from veomni.models.diffusers.minimax_h3_image.pdd import DEFAULT_SIGMAS
from veomni.trainer.dit_trainer import DiTDataCollator, DiTTrainer, OfflineEmbeddingSaver


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _condition_model(seed=42, pdd_sigmas=DEFAULT_SIGMAS, **kwargs):
    config = MiniMaxH3ImageConditionModelConfig(
        skip_encoder_load=True, height=32, width=64, seed=seed, pdd_sigmas=pdd_sigmas
    )
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
    assert offline_model.process_condition(**cached_batch)["x"].shape == (2, 8)


@pytest.mark.parametrize(
    ("seed", "step", "sigma", "model_time", "weight", "pdd_index"),
    [(42, 542, 0.6522877812, 0.3477122188, 1.4969605207, 3), (52, 989, 0.0240966268, 0.9759033918, 0.1255879402, 5)],
)
def test_t2i_schedule_matches_public_fp32_objective(seed, step, sigma, model_time, weight, pdd_index):
    model = _condition_model(seed=seed, meta_init=True)
    batch = _cached_batch()
    result = model.process_condition(**batch)

    assert result["pdd_step_index"] == pdd_index
    torch.testing.assert_close(result["unique_timesteps"], torch.tensor([model_time]))
    torch.testing.assert_close(result["image_loss_weight"], torch.tensor(weight))
    torch.testing.assert_close(result["inverse_indices"], torch.zeros(5, dtype=torch.long))
    torch.testing.assert_close(result["text_pos_info"]["position_ids"], torch.arange(3))
    torch.testing.assert_close(result["img_pos_info"]["position_ids"], torch.arange(3, 5))
    torch.testing.assert_close(result["token_tags"], torch.tensor([1, 1, 1, 0, 0]))
    torch.testing.assert_close(result["prompt_embeds"], batch["prompt_embeds"][0][0])
    assert result["img_position_ids"].shape == (1, 5, 3)
    assert result["img_position_ids"].dtype == torch.float32

    generator = torch.Generator(device="cpu").manual_seed(seed)
    assert torch.randint(1000, (1,), generator=generator).item() == step
    clean_latents = batch["clean_latents"][0]
    noise = torch.randn(clean_latents.shape, generator=generator)
    noised_latents = unpatchify_image_rows(result["x"], 2, 4, 2, (1, 2, 2))
    target = unpatchify_image_rows(result["training_target"], 2, 4, 2, (1, 2, 2))
    torch.testing.assert_close(noised_latents, (1 - sigma) * clean_latents + sigma * noise)
    torch.testing.assert_close(target, noise - clean_latents)


def test_t2i_schedule_uses_configured_release_heads():
    model = _condition_model(seed=42, pdd_sigmas=(1.0, 0.65, 0.2, 0.0), meta_init=True)
    result = model.process_condition(**_cached_batch())
    assert result["pdd_step_index"] == 1
    assert result["num_pdd_heads"] == 3


def test_t2i_condition_accepts_optional_cached_reference_rows():
    model = _condition_model()
    batch = _cached_batch()
    state = model.rng_state_dict()
    t2i = model.process_condition(**batch)
    model.load_rng_state_dict(state)
    references = [torch.ones(1, 2, 2, 2), -torch.ones(1, 2, 2, 4)]
    edit = model.process_condition(**batch, reference_latents=[references])

    assert edit["cond_rows"] == 3
    assert edit["x"].shape == (5, 8)
    torch.testing.assert_close(edit["x"][3:], t2i["x"], rtol=0, atol=0)
    torch.testing.assert_close(edit["training_target"], t2i["training_target"], rtol=0, atol=0)
    torch.testing.assert_close(edit["inverse_indices"], torch.tensor([1, 1, 1, 0, 0, 0, 1, 1]))
    torch.testing.assert_close(
        edit["unique_timesteps"], torch.cat([torch.tensor([REFERENCE_MODEL_TIME]), t2i["unique_timesteps"]])
    )
    torch.testing.assert_close(edit["img_pos_for_infer_output_info"]["position_ids"], torch.arange(3, 8))


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

    assert not torch.equal(first["training_target"], second["training_target"])
    assert not torch.equal(first["unique_timesteps"], second["unique_timesteps"])
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
        first_rank.process_condition(**batch)["training_target"],
        second_rank.process_condition(**batch)["training_target"],
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
    with pytest.raises(ValueError, match="micro_batch_size=1"):
        model.process_condition(**{key: values * 2 for key, values in _cached_batch().items()})
    with pytest.raises(ValueError, match="exactly one target image"):
        model.get_condition(["A rectangle"], [[]])


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"height": 0}, "positive multiples of 32"),
        ({"width": 48}, "positive multiples of 32"),
        ({"pdd_sigmas": [1.0, 0.2, 0.5, 0.0]}, "must strictly decrease"),
        ({"pdd_sigmas": [1.0, 0.5, 0.0], "num_pdd_heads": 6}, "2 transitions.*6 PDD heads"),
        ({"num_pdd_heads": 2}, "6 transitions.*2 PDD heads"),
    ],
)
def test_t2i_condition_config_rejects_invalid_load_overrides(tmp_path, overrides, error):
    path = tmp_path / "condition.json"
    _condition_model().config.to_json_file(path, use_diff=False)
    with pytest.raises(ValueError, match=error):
        MiniMaxH3ImageConditionModelConfig.from_pretrained(path, **overrides)


def test_t2i_condition_config_uses_effective_load_overrides(tmp_path):
    path = tmp_path / "condition.json"
    saved = _condition_model().config
    saved.to_json_file(path, use_diff=False)
    restored = MiniMaxH3ImageConditionModelConfig.from_pretrained(path)
    assert restored.pdd_sigmas == DEFAULT_SIGMAS and restored.num_pdd_heads == 6

    overridden = MiniMaxH3ImageConditionModelConfig.from_pretrained(
        path, height=512, width=256, pdd_sigmas=[1.0, 0.5, 0.0]
    )
    assert (overridden.height, overridden.width) == (512, 256)
    assert overridden.pdd_sigmas == (1.0, 0.5, 0.0)
    assert overridden.num_pdd_heads == 2


@pytest.mark.parametrize("sigmas", [(1.0, 0.4, 0.6, 0.0), (1.0, 0.9, 0.1), (0.9, 0.4, 0.0)])
def test_t2i_rejects_invalid_pdd_schedule(sigmas):
    with pytest.raises(ValueError, match="PDD sigmas must strictly decrease from 1 to 0"):
        MiniMaxH3ImageConditionModelConfig(pdd_sigmas=sigmas)

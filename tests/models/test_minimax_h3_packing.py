"""Native H3 model-owned packing, without pretrained weights or encoders."""

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from veomni.models.diffusers.minimax_h3.minimax_h3_condition.configuration_minimax_h3_condition import (
    MiniMaxH3ConditionModelConfig,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_condition.modeling_minimax_h3_condition import (
    MiniMaxH3ConditionModel,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_core import core, minimax_h3_dit, packed_sequence
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
    MiniMaxH3DiTModelConfig,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
    MiniMaxH3DiTOutput,
)
from veomni.trainer.dit_trainer import DiTDataCollator
from veomni.utils.import_utils import is_torch_npu_available


def tiny_model():
    return MiniMaxH3DiTModel(
        MiniMaxH3DiTModelConfig(
            hidden_size=32,
            num_layers=2,
            token_refiner_num_layers=1,
            num_attention_heads=2,
            attention_head_dim=16,
            ffn_hidden_size=64,
            text_dim=32,
            timestep_input_dim=16,
            time_embed_hidden_size=32,
            time_embed_dim=16,
            adaln_out_features=576,
            final_adaln_out_features=64,
            rope_inv_freq_len=2,
        )
    )


def condition_model():
    return MiniMaxH3ConditionModel(MiniMaxH3ConditionModelConfig(skip_encoder_load=True, num_train_timesteps=16))


@pytest.mark.parametrize("power,expected_loss,expected_grad", [(2.0, 4.0, 1.0), (1.0, 16.0, 4.0), (0.0, 64.0, 16.0)])
def test_cfg_objective_curvature_and_detached_unconditional(power, expected_loss, expected_grad):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.training_objectives import cfg_calibrated_mse

    pred = torch.tensor([6.0], requires_grad=True)
    uncond = torch.tensor([2.0], requires_grad=True)
    loss = cfg_calibrated_mse(pred, torch.ones(1), unconditional=uncond, scale=4.0, curvature_power=power)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    torch.testing.assert_close(pred.grad, torch.tensor([expected_grad]))
    assert uncond.grad is None


def test_cfg_scales_use_each_modality_sigma():
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.training_objectives import resolve_cfg_scales

    assert resolve_cfg_scales(4, "constant", t_video=0.25, t_audio=0.8) == (4, 4)
    assert resolve_cfg_scales(4, "sigma", t_video=0.25, t_audio=0.8) == pytest.approx((3.25, 1.6))


def test_cfg_hessian_and_shape_contract():
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.training_objectives import cfg_calibrated_mse

    for power, expected in ((2.0, 2 / 16), (1.0, 2 / 4), (0.0, 2.0), (0.5, 1.0)):
        p = torch.tensor([6.0], requires_grad=True)
        loss = cfg_calibrated_mse(p, torch.ones(1), unconditional=torch.ones(1), scale=4, curvature_power=power)
        gradient = torch.autograd.grad(loss, p, create_graph=True)[0]
        hessian = torch.autograd.grad(gradient, p)[0]
        torch.testing.assert_close(hessian, torch.tensor([expected]))
    with pytest.raises(ValueError, match="shapes"):
        cfg_calibrated_mse(torch.ones(2), torch.ones(1))
    with pytest.raises(ValueError, match="unconditional"):
        cfg_calibrated_mse(torch.ones(2), torch.ones(2), scale=4)


@pytest.mark.parametrize(
    "options",
    [
        {"training_cfg_scale": 0.5},
        {"training_cfg_scale": float("nan")},
        {"training_cfg_schedule": "unknown"},
        {"training_cfg_curvature_power": -1},
        {"training_cfg_curvature_power": 3},
        {"training_cfg_curvature_power": float("nan")},
        {"training_cfg_curvature_power": float("inf")},
    ],
)
def test_cfg_configuration_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        MiniMaxH3ConditionModelConfig(**options)


def test_default_noise_sampling_preserves_rng():
    row = raw_sample()
    cond = condition_model()
    torch.manual_seed(17)
    index = torch.randint(0, cond.config.num_train_timesteps, (1,)).item()
    video_noise = torch.randn_like(row["input_latents"])
    audio_noise = torch.randn_like(row["audio_input_latents"])
    state = torch.get_rng_state()
    torch.manual_seed(17)
    sample = prepare(cond, [row])[0]
    assert torch.equal(torch.get_rng_state(), state)
    torch.testing.assert_close(sample["training_target"], video_noise - row["input_latents"])
    torch.testing.assert_close(sample["training_target_audio"], audio_noise - row["audio_input_latents"])
    for modality in ("video", "audio"):
        scheduler = getattr(cond, f"_scheduler_{modality}")
        timestep = scheduler.timesteps[index].to(row["input_latents"].dtype)
        assert sample[f"t_{modality}"] == 1 - timestep.float().item() / scheduler.num_train_timesteps


def raw_sample(text_len=3, task="fl2va", refs=None, latent_t=2, latent_h=4, latent_w=6, audio_t=3, keyframes=(0,)):
    geometry = dict(
        text_len=text_len, latent_t=latent_t, latent_h=latent_h, latent_w=latent_w, audio_t=audio_t, audio_channel=2
    )
    if task == "ref2va":
        refs = refs or [{"kind": "image", "latent_t": 1, "latent_h": 4, "latent_w": 4}]
        pk = packed_sequence.build_packed_ref2va(**geometry, ref_blocks=refs)
        anchor_key = "ref_visual_anchor"
    else:
        pk = packed_sequence.build_packed_fl2va(**geometry, keyframe_indices=list(keyframes))
        anchor_key = "keyframe_cond_anchor"
    row = dict(
        input_latents=torch.randn(1, 24, latent_t, latent_h, latent_w),
        audio_input_latents=torch.randn(2, 32, audio_t),
        prompt_embeds=torch.randn(text_len, 32),
        packed=pk,
        use_gradient_checkpointing=False,
    )
    if pk["cond_rows"]:
        row[anchor_key] = torch.randn(pk["cond_rows"], 96)
    return row


def legacy_tail(sample):
    """Append the pre-#1204 64-row sample tail to a prepared single sample."""
    legacy = copy.deepcopy(sample)
    length = sample["x"].shape[1]
    pad = (-length) % 64
    for key in ("x", "audio_x", "img_position_ids"):
        legacy[key] = F.pad(legacy[key], (0, 0, 0, pad))
    legacy["token_tags"] = F.pad(legacy["token_tags"], (0, pad), value=-1)
    legacy["inverse_indices"] = F.pad(legacy["inverse_indices"], (0, pad))
    legacy["packed_seq_params"]["cu_seqlens_q"] = torch.tensor([0, length, length + pad], dtype=torch.int32)
    legacy["packed_seq_params"]["cu_seqlens_host"] = (0, length, length + pad)
    return legacy


def prepare(condition, raws):
    columns = condition.process_condition(**DiTDataCollator()(raws))
    if len(raws) == 1:
        return [columns]
    assert all(isinstance(value, list) and len(value) == len(raws) for value in columns.values())
    return [{key: value[i] for key, value in columns.items()} for i in range(len(raws))]


def batch(samples):
    return {key: [sample[key] for sample in samples] for key in samples[0]}


def cfg_condition(**options):
    return MiniMaxH3ConditionModel(
        MiniMaxH3ConditionModelConfig(skip_encoder_load=True, num_train_timesteps=16, training_cfg_scale=4, **options)
    )


def cfg_raw(text_len=3, task="ref2va"):
    row = raw_sample(text_len, task)
    negative = raw_sample(2, task)
    row["unconditional_prompt_embeds"] = negative["prompt_embeds"]
    row["unconditional_packed"] = negative["packed"]
    return row


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_cfg_branches_share_noise_references_and_rng(task):
    row = cfg_raw(task=task)
    torch.manual_seed(12)
    ordinary = prepare(condition_model(), [row])[0]
    state = torch.get_rng_state()
    torch.manual_seed(12)
    sample = prepare(cfg_condition(), [row])[0]
    assert torch.equal(torch.get_rng_state(), state)
    negative = sample["unconditional_inputs"]
    for key, positions in (("x", "img_pos_info"), ("audio_x", "audio_pos_info")):
        torch.testing.assert_close(sample[key], ordinary[key])
        torch.testing.assert_close(
            sample[key][0, sample[positions]["position_ids"]],
            negative[key][0, negative[positions]["position_ids"]],
        )
    assert "training_target" not in negative
    assert not negative["use_gradient_checkpointing"]
    torch.testing.assert_close(negative["unique_timesteps"], sample["unique_timesteps"])
    for name in ("img_pos_info", "audio_pos_info"):
        torch.testing.assert_close(
            negative["inverse_indices"][negative[name]["position_ids"]],
            sample["inverse_indices"][sample[name]["position_ids"]],
        )


def test_cfg_training_does_not_repeat_cache_value_validation(monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import cfg_conditioning

    row = cfg_raw()
    cfg_conditioning.validate_unconditional(
        row["packed"], row["unconditional_packed"], row["unconditional_prompt_embeds"], row["prompt_embeds"]
    )

    def unexpected_validation(*args):
        raise AssertionError("Cache values must not be revalidated during training.")

    monkeypatch.setattr(cfg_conditioning, "_validate_layout", unexpected_validation)
    sample = prepare(cfg_condition(), [row])[0]
    assert "unconditional_inputs" in sample


@pytest.mark.parametrize(
    "malformation", [None, "width", "geometry", "image_rows", "audio_rows", "coordinates", "tags"]
)
def test_cfg_metadata_validation_needs_no_tensor_values(malformation):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.cfg_conditioning import validate_unconditional_metadata

    row = cfg_raw()
    positive, negative = (
        {key: value.to("meta") if isinstance(value, torch.Tensor) else value for key, value in row[name].items()}
        for name in ("packed", "unconditional_packed")
    )
    prompt = row["unconditional_prompt_embeds"].to("meta")
    if malformation == "width":
        prompt = prompt[:, :16]
    elif malformation == "geometry":
        negative["latent_t"] += 1
    elif malformation in ("image_rows", "audio_rows"):
        key = "img_pos" if malformation == "image_rows" else "audio_pos"
        negative[key] = negative[key][:-1]
    elif malformation == "coordinates":
        negative["img_position_ids"] = negative["img_position_ids"][:, :-1]
    elif malformation == "tags":
        negative["token_tags"] = negative["token_tags"][:-1]
    if malformation:
        with pytest.raises(ValueError):
            validate_unconditional_metadata(positive, negative, prompt, row["prompt_embeds"].to("meta"))
    else:
        validate_unconditional_metadata(positive, negative, prompt, row["prompt_embeds"].to("meta"))


def test_cfg_remapping_does_not_recompute_unique(monkeypatch):
    original = torch.unique
    calls = []

    def unique(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "unique", unique)
    prepare(cfg_condition(), [cfg_raw()])
    assert len(calls) == 1


@pytest.mark.parametrize("mode", ["fl2va", "ref2va"])
@pytest.mark.parametrize("finite", [True, False])
@pytest.mark.parametrize("drop_visual", [False, True])
def test_unconditional_cache_producer_matches_native_presentation(monkeypatch, mode, finite, drop_visual):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_text_encoder as text

    class Tokenizer:
        def __call__(self, value, **kwargs):
            return {"input_ids": [ord(char) for char in value]}

        def convert_tokens_to_ids(self, token):
            return {text.VISION_START: 1000, text.VISION_END: 1001, text.IMAGE_PAD: 1002, text.VIDEO_PAD: 1003}[token]

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.calls = []

        def forward(self, **kwargs):
            assert not torch.is_grad_enabled()
            self.calls.append(kwargs)
            return kwargs["input_ids"][0, :, None].float().expand(-1, 32) * (self.weight if finite else float("nan"))

    monkeypatch.setattr(
        text, "image_token_counts", lambda processor, images: (torch.ones(1, 3), torch.ones(1, 3), [2])
    )
    cond = condition_model()
    cond._text_encoder = Encoder()
    cond._tokenizer = Tokenizer()
    image = object()
    row = raw_sample(task=mode)
    kwargs = (
        {"keyframe_images": [image]}
        if mode == "fl2va"
        else {"ref_blocks": [{"kind": "image", "prepared_image": image}]}
    )
    if drop_visual:
        kwargs["drop_visual"] = True
    if not finite:
        with pytest.raises(ValueError, match="finite"):
            cond.add_unconditional_cache(row, **kwargs)
        return
    paired = cond.add_unconditional_cache(row, **kwargs)
    encoded = {
        "prompt_embeds": paired["unconditional_prompt_embeds"],
        "text_token_tags": paired["unconditional_packed"]["token_tags"][: paired["unconditional_packed"]["text_len"]],
    }
    if drop_visual:
        expected_ids, expected_tags = text.presentation_t2va(cond._tokenizer, " ")
        assert "pixel_values" not in cond._text_encoder.calls[0]
    elif mode == "fl2va":
        expected_ids, expected_tags = text.presentation_fl2va(cond._tokenizer, " ", [2])
    else:
        expected_ids, expected_tags = text.presentation_ref2va(cond._tokenizer, " ", [("image", 1)], [2], [], [])
    assert paired["input_latents"] is row["input_latents"]
    anchor = "keyframe_cond_anchor" if mode == "fl2va" else "ref_visual_anchor"
    assert paired[anchor] is row[anchor]
    prepared = prepare(cfg_condition(), [paired])[0]
    negative = prepared["unconditional_inputs"]
    torch.testing.assert_close(
        negative["x"][0, negative["img_pos_info"]["position_ids"]],
        prepared["x"][0, prepared["img_pos_info"]["position_ids"]],
    )
    torch.testing.assert_close(cond._text_encoder.calls[0]["input_ids"][0], expected_ids)
    torch.testing.assert_close(encoded["text_token_tags"], expected_tags)
    assert encoded["prompt_embeds"].shape == (len(expected_ids), 32)
    assert not encoded["prompt_embeds"].requires_grad


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_text_only_negative_does_not_require_visual_sources(monkeypatch, task):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_text_encoder as text

    def encode(encoder, processor, tokenizer, prompt, **kwargs):
        assert prompt == " "
        assert kwargs["keyframes"] is None
        assert kwargs["ref_blocks"] is None
        return {"prompt_embeds": torch.ones(1, 32), "text_token_tags": torch.ones(1, dtype=torch.long)}

    monkeypatch.setattr(text, "encode_prompt", encode)
    cond = condition_model()
    cond._text_encoder = torch.nn.Linear(1, 1)
    cond._tokenizer = object()
    row = raw_sample(task=task)
    paired = cond.add_unconditional_cache(row, drop_visual=True)
    assert "unconditional_prompt_embeds" not in row
    assert paired["unconditional_packed"]["cond_rows"] == row["packed"]["cond_rows"]
    output = tiny_model()(**prepare(cfg_condition(), [paired])[0])
    assert all(torch.isfinite(loss) for loss in output.loss.values())
    sum(output.loss.values()).backward()


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_paired_cache_rejects_missing_visual_sources(task):
    with pytest.raises(ValueError, match="original"):
        condition_model().add_unconditional_cache(raw_sample(task=task))


def test_ref_prompt_preserves_reference_order_and_video_timestamps(monkeypatch):
    import numpy as np

    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_text_encoder as text

    observed = {}

    def presentation(tokenizer, prompt, labels, images, videos, timestamps):
        observed.update(prompt=prompt, labels=labels, images=images, videos=videos, timestamps=timestamps)
        return torch.tensor([1]), torch.tensor([1])

    def process_videos(processor, videos, timestamps):
        assert videos[0].shape[0] == 2  # native 24 FPS -> Qwen 2 FPS sampling
        assert timestamps == [[0.25]]
        return torch.ones(1, 3), torch.ones(1, 3), [[2]], timestamps

    monkeypatch.setattr(text, "presentation_ref2va", presentation)
    monkeypatch.setattr(
        text, "image_token_counts", lambda processor, images: (torch.ones(1, 3), torch.ones(1, 3), [3])
    )
    monkeypatch.setattr(text, "video_token_counts", process_videos)
    refs = [
        {"kind": "video", "prepared_frames": [np.zeros((2, 2, 3)) for _ in range(24)], "ref_audio_t": 0},
        {"kind": "image", "prepared_image": object()},
    ]
    text.prepare_ref_prompt(None, None, " ", refs)
    assert observed == {
        "prompt": " ",
        "labels": [("video", 1), ("image", 1)],
        "images": [3],
        "videos": [[2]],
        "timestamps": [[0.25]],
    }


def test_cfg_missing_or_incompatible_negative_fails():
    with pytest.raises(ValueError, match="unconditional"):
        prepare(cfg_condition(), [raw_sample()])
    row = cfg_raw()
    row["unconditional_packed"]["latent_t"] += 1
    with pytest.raises(ValueError, match="geometry"):
        prepare(cfg_condition(), [row])


@pytest.mark.parametrize("power", [0.0, 0.5, 1.0, 2.0])
def test_cfg_model_loss_raw_predictions_and_gradient_boundary(power):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.training_objectives import cfg_calibrated_mse

    model = tiny_model()
    row = cfg_raw()
    row["prompt_embeds"].requires_grad_()
    row["unconditional_prompt_embeds"].requires_grad_()
    sample = prepare(cfg_condition(training_cfg_curvature_power=power, training_cfg_schedule="sigma"), [row])[0]
    branch_modes = []
    hook = model.dit.register_forward_pre_hook(lambda *args: branch_modes.append(torch.is_grad_enabled()))
    out = model(**sample)
    hook.remove()
    assert branch_modes == [False, True]
    ordinary = {k: v for k, v in sample.items() if k != "unconditional_inputs"}
    ordinary["training_cfg_scales"] = (1, 1)
    with torch.no_grad():
        negative = model(**sample["unconditional_inputs"])
        raw = model(**ordinary)
    for i, modality in enumerate(("video", "audio")):
        torch.testing.assert_close(out.predictions[i], raw.predictions[i])
        target = sample["training_target" if i == 0 else "training_target_audio"]
        expected = cfg_calibrated_mse(
            out.predictions[i],
            target,
            unconditional=negative.predictions[i],
            scale=sample["training_cfg_scales"][i],
            curvature_power=power,
        )
        scheduler = sample[f"scheduler_{modality}"]
        weight = scheduler.training_weight(torch.tensor((1 - sample[f"t_{modality}"]) * scheduler.num_train_timesteps))
        torch.testing.assert_close(out.loss[f"mse_{modality}"], expected * weight)
    sum(out.loss.values()).backward()
    assert row["prompt_embeds"].grad is not None
    assert row["unconditional_prompt_embeds"].grad is None


def test_cfg_packed_matches_serial_loss_and_gradients():
    torch.manual_seed(11)
    samples = prepare(cfg_condition(), [cfg_raw(3, "fl2va"), cfg_raw(7, "ref2va")])
    one = tiny_model()
    many = copy.deepcopy(one)
    outputs = serial(one, samples)
    expected = sum(sum(out.loss.values()) for out in outputs) / 2
    actual = many(**batch(samples))
    torch.testing.assert_close(sum(actual.loss.values()), expected, rtol=2e-5, atol=2e-5)
    expected.backward()
    sum(actual.loss.values()).backward()
    for a, b in zip(one.parameters(), many.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=3e-4, atol=3e-5)


def test_scale_one_requires_no_negative_and_preserves_output():
    row = raw_sample()
    torch.manual_seed(8)
    expected = prepare(condition_model(), [row])[0]
    state = torch.get_rng_state()
    torch.manual_seed(8)
    cond = MiniMaxH3ConditionModel(
        MiniMaxH3ConditionModelConfig(
            skip_encoder_load=True,
            num_train_timesteps=16,
            training_cfg_scale=1,
        )
    )
    actual = prepare(cond, [row])[0]
    assert torch.equal(torch.get_rng_state(), state)
    assert expected.keys() == actual.keys()
    model = tiny_model()
    modes = []
    hook = model.dit.register_forward_pre_hook(lambda *args: modes.append(torch.is_grad_enabled()))
    a = model(**actual)
    hook.remove()
    assert modes == [True]
    b = model(**expected)
    torch.testing.assert_close(a.loss, b.loss, rtol=0, atol=0)


@pytest.mark.parametrize("malformation", ["width", "nan", "reference", "tags", "layout"])
def test_cfg_offline_negative_validation(malformation):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.cfg_conditioning import validate_unconditional

    row = cfg_raw()
    if malformation == "width":
        row["unconditional_prompt_embeds"] = torch.randn(2, 16)
    elif malformation == "nan":
        row["unconditional_prompt_embeds"][0, 0] = float("nan")
    elif malformation == "reference":
        pk = row["unconditional_packed"]
        pk["img_position_ids"][0, pk["img_pos"][0], 1] += 1
    elif malformation == "tags":
        row["unconditional_packed"]["token_tags"].fill_(2)
    else:
        row["unconditional_packed"]["text_pos"] += 1
    with pytest.raises(ValueError):
        validate_unconditional(
            row["packed"], row["unconditional_packed"], row["unconditional_prompt_embeds"], row["prompt_embeds"]
        )


def test_cfg_checkpointing_and_silent_audio_match():
    torch.manual_seed(16)
    row = cfg_raw()
    row["has_audio"] = False
    sample = prepare(cfg_condition(), [row])[0]
    one = tiny_model()
    checkpointed = copy.deepcopy(one)
    out = one(**sample)
    sample["use_gradient_checkpointing"] = True
    other = checkpointed(**sample)
    assert out.loss["mse_audio"].item() == 0
    torch.testing.assert_close(out.loss, other.loss)
    sum(out.loss.values()).backward()
    sum(other.loss.values()).backward()
    for a, b in zip(one.parameters(), checkpointed.parameters()):
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad)


def test_cfg_offline_cache_roundtrip():
    import pickle

    from veomni.data.multimodal.dit.data_transform import process_dit_offline_example

    row = cfg_raw()
    cached = {key: pickle.dumps(value) for key, value in row.items()}
    restored = process_dit_offline_example(cached)[0]
    out = tiny_model()(**prepare(cfg_condition(), [restored])[0])
    assert all(torch.isfinite(loss) for loss in out.loss.values())


def test_cfg_recipe_native_lora_and_condition_config_roundtrip():
    from pathlib import Path

    import yaml

    from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel

    path = Path(__file__).resolve().parents[2] / "configs/dit/minimax_h3_ref2va_cfg_offline.yaml"
    recipe = yaml.safe_load(path.read_text())
    config = MiniMaxH3ConditionModelConfig(**recipe["model"]["condition_model_cfg"])
    restored = MiniMaxH3ConditionModelConfig.from_dict(config.to_dict())
    assert restored.to_dict() == config.to_dict()
    model = VeOmniLoraModel(tiny_model(), VeOmniLoraConfig.from_yaml(recipe["model"]["lora_config"]))
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    assert len(trainable) == 2 * 4 * 2  # two blocks, four linears, A/B
    assert all("dit.blocks." in name and ".lora_" in name for name, _ in trainable)
    sample = prepare(MiniMaxH3ConditionModel(config), [cfg_raw()])[0]
    out = model(**sample)
    sum(out.loss.values()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for _, p in trainable)
    assert any(p.grad.abs().sum() > 0 for _, p in trainable)


def serial(model, samples):
    return [model(**sample) for sample in samples]


@pytest.fixture(autouse=True)
def cpu_attention(monkeypatch):
    monkeypatch.setattr(minimax_h3_dit, "IS_NPU_AVAILABLE", False)
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: None)


@pytest.mark.parametrize("backend", ["eager", "sdpa"])
def test_native_foundation_loader_uses_ordinary_batch_contract(backend):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model

    ops = OpsImplementationConfig(
        attn_implementation=backend,
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        tiny_model().config, init_device="cpu", torch_dtype="float32", ops_implementation=ops
    )
    assert isinstance(model, MiniMaxH3DiTModel)
    keys = set(model.state_dict())
    columns = condition_model().process_condition(**DiTDataCollator()([raw_sample(), raw_sample(7)]))
    out = model(**columns)
    assert isinstance(out, MiniMaxH3DiTOutput)
    assert all(value.ndim == 0 for value in out.loss.values())
    assert [p.shape for p in out.predictions[0]] == [(1, 24, 2, 4, 6)] * 2
    assert keys == set(model.state_dict())


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_preparation_preserves_single_sample_rng_and_weights(task):
    cond = condition_model()
    raws = [raw_sample(3, task), raw_sample(7, task)]
    torch.manual_seed(12)
    expected = [cond.process_condition(**DiTDataCollator()([r])) for r in raws]
    expected_rng = torch.get_rng_state()
    torch.manual_seed(12)
    actual = prepare(cond, raws)
    assert torch.equal(torch.get_rng_state(), expected_rng)
    for sample, ref in zip(actual, expected):
        for key, val in ref.items():
            if torch.is_tensor(val):
                torch.testing.assert_close(sample[key], val, rtol=0, atol=0)
        assert sample["t_video"] == ref["t_video"]
        assert sample["t_audio"] == ref["t_audio"]


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_outputs_losses_and_gradients_match_serial(task, checkpointing):
    torch.manual_seed(7)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3, task), raw_sample(7, task)]
    for row in raws:
        row["use_gradient_checkpointing"] = checkpointing
    samples = prepare(condition_model(), raws)
    expected = serial(base, samples)
    entries = {id(block): 0 for block in packed.dit.blocks}

    def record(module, inputs):
        entries[id(module)] += 1

    handles = [block.register_forward_pre_hook(record) for block in packed.dit.blocks]
    calls = []
    handle = packed.dit.register_forward_hook(lambda *args: calls.append(1))
    actual = packed(**batch(samples))
    handle.remove()
    assert len(calls) == 1
    assert set(entries.values()) == {1}
    for i, ref in enumerate(expected):
        torch.testing.assert_close(actual.predictions[0][i], ref.predictions[0], rtol=2e-5, atol=2e-5)
        torch.testing.assert_close(actual.predictions[1][i], ref.predictions[1], rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    assert set(entries.values()) == {2 if checkpointing else 1}
    for handle in handles:
        handle.remove()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        assert p.grad is not None, name
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_mixed_target_geometry_matches_serial(task):
    torch.manual_seed(11)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [
        raw_sample(3, task),
        raw_sample(7, task, latent_t=3, latent_h=6, latent_w=4, audio_t=5),
    ]
    samples = prepare(condition_model(), raws)
    expected = serial(base, samples)
    actual = packed(**batch(samples))
    assert [p.shape for p in actual.predictions[0]] == [(1, 24, 2, 4, 6), (1, 24, 3, 6, 4)]
    assert [p.shape for p in actual.predictions[1]] == [(2, 32, 3), (2, 32, 5)]
    for i, ref in enumerate(expected):
        for a, b in zip((actual.predictions[0][i], actual.predictions[1][i]), ref.predictions):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_mixed_tasks_and_keyframes_match_serial():
    torch.manual_seed(13)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3), raw_sample(5, "ref2va"), raw_sample(4, keyframes=())]
    collated = DiTDataCollator()(raws)
    assert [anchor is None for anchor in collated["keyframe_cond_anchor"]] == [False, True, True]
    assert [anchor is None for anchor in collated["ref_visual_anchor"]] == [True, False, True]
    samples = prepare(condition_model(), raws)
    assert [sample["cond_rows"] for sample in samples][2] == 0
    expected = serial(base, samples)
    actual = packed(**batch(samples))
    for i, ref in enumerate(expected):
        for a, b in zip((actual.predictions[0][i], actual.predictions[1][i]), ref.predictions):
            torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], torch.stack([out.loss[key] for out in expected]).mean())
    sum(sum(out.loss.values()) for out in expected).div(len(samples)).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)


def test_get_condition_flags_silent_audio_placeholder(monkeypatch):
    cond = MiniMaxH3ConditionModel(
        MiniMaxH3ConditionModelConfig(skip_encoder_load=True, num_train_timesteps=16, use_keyframe_condition=False)
    )
    cond._video_vae = torch.nn.Linear(1, 1)
    monkeypatch.setattr(cond, "_encode_text", lambda prompt, images, device: (torch.randn(3, 32), None))
    monkeypatch.setattr(cond, "_encode_video", lambda video, device: torch.randn(1, 24, 2, 4, 6))
    monkeypatch.setattr(cond, "_encode_audio", lambda audio, frames, device: torch.randn(2, 32, 3))
    monkeypatch.setattr(cond, "_make_silent_audio_latent", lambda frames, device: torch.zeros(2, 32, 3))
    frames = [torch.zeros(3, 8, 8)] * 22
    out = cond.get_condition(inputs=["a", "b"], videos=[frames, frames], audios=[(torch.zeros(1, 8), 16000), None])
    assert out["has_audio"] == [True, False]


def test_audio_loss_skips_samples_without_audio():
    torch.manual_seed(17)
    base = tiny_model()
    packed = copy.deepcopy(base)
    raws = [raw_sample(3), raw_sample(5), raw_sample(4, "ref2va")]
    raws[1]["has_audio"] = False
    samples = prepare(condition_model(), raws)
    assert [sample["has_audio"] for sample in samples] == [True, False, True]
    expected = serial(base, samples)
    assert expected[1].loss["mse_audio"] == 0
    actual = packed(**batch(samples))
    torch.testing.assert_close(actual.loss["mse_video"], torch.stack([o.loss["mse_video"] for o in expected]).mean())
    torch.testing.assert_close(actual.loss["mse_audio"], torch.stack([o.loss["mse_audio"] for o in expected]).mean())
    (sum(o.loss["mse_video"] + o.loss["mse_audio"] for o in expected) / 3).backward()
    sum(actual.loss.values()).backward()
    for (name, p), (_, q) in zip(base.named_parameters(), packed.named_parameters()):
        torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-5, msg=name)

    silent = [raw_sample(3), raw_sample(5)]
    for row in silent:
        row["has_audio"] = False
    silent_samples = prepare(condition_model(), silent)
    for out in (base(**silent_samples[0]), base(**batch(silent_samples))):
        assert out.loss["mse_audio"] == 0 and out.loss["mse_audio"].requires_grad


def test_accumulated_microbatches_weight_each_sample_like_single_sample():
    """With the trainer's /K, packed microbatches weight every sample 1/G whatever their audio mix."""
    torch.manual_seed(19)
    model = tiny_model()
    raws = [raw_sample(3), raw_sample(5), raw_sample(4, "ref2va"), raw_sample(6)]
    raws[1]["has_audio"] = False  # microbatches [audio, no-audio] and [audio, audio]
    samples = prepare(condition_model(), raws)
    reference = sum(sum(out.loss.values()) for out in serial(model, samples)) / len(samples)
    accumulated = sum(sum(model(**batch(part)).loss.values()) / 2 for part in (samples[:2], samples[2:]))
    torch.testing.assert_close(accumulated, reference)


def test_unsupported_backend_fails_only_on_packed_forward():
    model = tiny_model()
    model._configure_packed_attention("veomni_flash_attention_4_with_sp")  # what __init__ runs; must not raise
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(5)])
    serial(model, samples)
    with pytest.raises(ValueError, match="Unsupported H3 packing backend"):
        model(**batch(samples))


def test_sample_isolation_boundaries_and_zero_valid_rows():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(9)])
    expected = model(**batch(samples))
    altered = copy.deepcopy(samples)
    altered[1]["prompt_embeds"].add_(100)
    captured = []
    hook = model.dit.register_forward_pre_hook(lambda module, args, kwargs: captured.append(kwargs), with_kwargs=True)
    actual = model(**batch(altered))
    hook.remove()
    for a, b in zip(expected.predictions, actual.predictions):
        torch.testing.assert_close(a[0], b[0])
    inp = captured[0]
    lengths = [sample["x"].shape[1] for sample in samples]
    assert inp["x"].shape[1] == sum(lengths)
    assert inp["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, lengths[0], sum(lengths)]
    assert inp["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [0, 3, 12]
    for sample in samples:
        sample["x"].zero_()
        sample["audio_x"].zero_()
    assert [p.shape for p in model(**batch(samples)).predictions[0]] == [(1, 24, 2, 4, 6)] * 2


def test_ref2va_variable_reference_layouts_and_target_only_loss():
    refs = [
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]
    samples = prepare(condition_model(), [raw_sample(2, "ref2va"), raw_sample(11, "ref2va", refs)])
    model = tiny_model()
    expected = serial(model, samples)
    actual = model(**batch(samples))
    for i in range(2):
        torch.testing.assert_close(actual.predictions[0][i], expected[i].predictions[0], rtol=2e-5, atol=2e-5)
    assert [p.shape for p in actual.predictions[1]] == [(2, 32, 3)] * 2


def test_audio_refs_fail_closed():
    cond = condition_model()
    row = raw_sample(task="ref2va")
    row["ref_audio_anchor"] = torch.ones(2, 32)
    with pytest.raises(NotImplementedError, match="audio"):
        prepare(cond, [row])
    with pytest.raises(NotImplementedError, match="audio"):
        raw_sample(task="ref2va", refs=[{"kind": "audio", "ref_audio_t": 2}])


@pytest.mark.parametrize("audio_channel", [1, 2])
def test_visual_ref_layout_matches_native_inference(audio_channel):
    from veomni.models.diffusers.minimax_h3.inference import MiniMaxH3Unit_PackedSequenceBuilder

    refs = [
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
        {"kind": "video", "latent_t": 3, "latent_h": 6, "latent_w": 4},
    ]
    kwargs = dict(
        text_len=7, latent_t=2, latent_h=4, latent_w=6, audio_t=3, ref_blocks=refs, audio_channel=audio_channel
    )
    expected = MiniMaxH3Unit_PackedSequenceBuilder()._build_packed_ref2va(**kwargs)
    actual = packed_sequence.build_packed_ref2va(**kwargs)
    for key, value in expected.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(actual[key], value, rtol=0, atol=1e-12)
        else:
            assert actual[key] == value
    assert actual["cu_seqlens"].tolist() == [0, actual["seq_len"]]


def test_host_bounds_match_packed_layouts():
    refs = [
        {"kind": "video", "latent_t": 2, "latent_h": 4, "latent_w": 6},
        {"kind": "image", "latent_t": 1, "latent_h": 2, "latent_w": 4},
    ]
    rows = [
        raw_sample(3),
        raw_sample(5, keyframes=(0, -1)),
        raw_sample(4, keyframes=()),
        raw_sample(6, "ref2va", refs),
    ]
    for row in rows:
        pk = row["packed"]
        assert packed_sequence.host_cu_seqlens(pk) == tuple(pk["cu_seqlens"].tolist())
        sample = prepare(condition_model(), [row])[0]
        assert sample["packed_seq_params"]["cu_seqlens_host"] == tuple(pk["cu_seqlens"].tolist())
        assert sample["refiner_packed_seq_params"]["cu_seqlens_host"] == (0, pk["text_len"])
        assert sample["refiner_packed_seq_params"]["cu_seqlens_q"].tolist() == [0, pk["text_len"]]
    legacy = dict(rows[0]["packed"])
    used = legacy["seq_len"]
    legacy["seq_len"] = used + (-used) % 64
    assert packed_sequence.host_cu_seqlens(legacy) == (0, used, legacy["seq_len"])


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_single_sample_valid_outputs_match_legacy_64_tail(task):
    model = tiny_model()
    sample = prepare(condition_model(), [raw_sample(3, task)])[0]
    length = sample["x"].shape[1]
    assert length % 64 != 0
    assert sample["packed_seq_params"]["cu_seqlens_q"].tolist() == [0, length]
    actual, expected = model(**sample), model(**legacy_tail(sample))
    for a, b in zip(actual.predictions, expected.predictions):
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-5)
    for key in actual.loss:
        torch.testing.assert_close(actual.loss[key], expected.loss[key])


def test_uncovered_sp_attention_tail_cannot_poison_gradients(monkeypatch):
    attention = tiny_model().dit.blocks[0].attn
    x = torch.randn(8, 32, requires_grad=True)
    monkeypatch.setattr(torch, "empty_like", lambda value: torch.full_like(value, float("nan")))
    output = attention(x, rope_cos=None, rope_sin=None, cu_seqlens=(0, 7), max_seqlen=7)
    output[:7].square().mean().backward()
    assert all(torch.isfinite(param.grad).all() for param in attention.parameters())
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(output).all()


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_sequence_parallel_padding_remains_forward_local(monkeypatch, task):
    model = tiny_model()
    sample = prepare(condition_model(), [raw_sample(3, task)])[0]
    length = sample["x"].shape[1]
    assert length % 2 == 1
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: object())
    monkeypatch.setattr(minimax_h3_dit.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(minimax_h3_dit.dist, "get_rank", lambda group: 0)
    seen = []

    def block(hidden, **kwargs):
        seen.append((hidden.shape[0], kwargs["rope_cos"].shape[0], kwargs["cu_seqlens"], kwargs["use_ulysses"]))
        return hidden

    for module in model.dit.blocks:
        monkeypatch.setattr(module, "forward", block)
    monkeypatch.setattr(minimax_h3_dit._Gather, "apply", lambda group, x, *args: x.repeat(2, 1))
    out = model(**sample)
    assert seen == [((length + 1) // 2, length + 1, (0, length), True)] * 2
    assert sample["x"].shape[1] == sample["img_position_ids"].shape[1] == length
    assert out.predictions[0].shape == (1, 24, 2, 4, 6)
    with pytest.raises(ValueError, match="sequence parallelism"):
        model(**batch([sample, sample]))


_NPU_REJECTS_HUB = pytest.mark.skipif(is_torch_npu_available(), reason="Hub attention is rejected on Ascend NPU.")


@pytest.mark.parametrize(
    "backend",
    [
        "flash_attention_2",
        pytest.param("flash_attention_2_hub", marks=_NPU_REJECTS_HUB),
        "flash_attention_3",
        pytest.param("flash_attention_3_hub", marks=_NPU_REJECTS_HUB),
    ],
)
def test_fused_dispatch_keeps_refiners_sample_local_and_single_sample_legacy(monkeypatch, backend):
    from veomni.arguments import OpsImplementationConfig
    from veomni.models.auto import build_foundation_model
    from veomni.ops.kernels.attention import flash

    calls, loads = [], []

    config = tiny_model().config

    def kernel(q, k, v, *, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, softmax_scale, causal):
        assert q.ndim == 3 and q.shape[1:] == (config.num_attention_heads, config.attention_head_dim)
        assert q.is_contiguous() and k.shape == v.shape == q.shape
        assert cu_seqlens_q.dtype == torch.int32 and cu_seqlens_k is cu_seqlens_q
        assert cu_seqlens_q[-1] == q.shape[0] and max_seqlen_q == max_seqlen_k
        assert not causal
        calls.append(cu_seqlens_q.tolist())
        return minimax_h3_dit._sdpa_varlen_attention(q, k, v, tuple(cu_seqlens_q.tolist()), softmax_scale, True)

    def loader(name):
        loads.append(name)
        return SimpleNamespace(flash_attn_varlen_func=kernel)

    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", loader)
    ops = OpsImplementationConfig(
        attn_implementation=backend,
        rms_norm_implementation="eager",
        rotary_pos_emb_implementation="eager",
        swiglu_mlp_implementation="eager",
        cross_entropy_loss_implementation="eager",
        moe_implementation="eager",
        load_balancing_loss_implementation="eager",
    )
    model = build_foundation_model(
        tiny_model().config, init_device="cpu", torch_dtype="float32", ops_implementation=ops
    ).bfloat16()
    raws = [raw_sample(3), raw_sample(7)]
    for row in raws:
        for key, value in row.items():
            if torch.is_tensor(value) and value.is_floating_point():
                row[key] = value.bfloat16()
    samples = prepare(condition_model(), raws)
    serial(model, samples)
    assert calls == []
    loads.clear()  # Transformers may already have resolved the backend at construction.
    out = model(**batch(samples))
    sum(out.loss.values()).backward()
    assert calls[:2] == [[0, 3], [0, 7]]
    assert len(calls) == 4 and calls[2] == calls[3] and len(calls[2]) == 3
    assert loads == [f"veomni_{backend}_with_sp"]


def test_flash_backend_defers_packed_kernel_until_multisample_forward(monkeypatch):
    import sys

    from transformers import modeling_flash_attention_utils as hf_flash

    from veomni.ops.kernels.attention import flash

    loads = []

    def unavailable(name):
        loads.append(name)
        raise ImportError("flash_attn unavailable")

    def npu_attention(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, max_seqlen_k=None):
        raise AssertionError("construction must not run NPU attention")

    # Mirror Ascend: Transformers resolves its native NPU FA before VeOmni's loader.
    monkeypatch.setattr(flash, "_load_veomni_flash_kernel", unavailable)
    monkeypatch.setattr(hf_flash, "is_flash_attn_2_available", lambda: False)
    monkeypatch.setattr(hf_flash, "is_torch_npu_available", lambda: True)
    for name in (
        "_loaded_implementation",
        "_flash_fn",
        "_flash_varlen_fn",
        "_flash_with_kvcache_fn",
        "_pad_fn",
        "_unpad_fn",
        "_process_flash_kwargs_fn",
    ):
        monkeypatch.setattr(hf_flash, name, None if name == "_loaded_implementation" else getattr(hf_flash, name))
    monkeypatch.setitem(
        sys.modules,
        "transformers.integrations.npu_flash_attention",
        SimpleNamespace(
            npu_flash_attn_func=npu_attention,
            npu_flash_attn_varlen_func=npu_attention,
            npu_flash_attn_with_kvcache=npu_attention,
        ),
    )
    config = tiny_model().config
    config._attn_implementation = "veomni_flash_attention_2_with_sp"
    model = MiniMaxH3DiTModel(config)
    samples = prepare(condition_model(), [raw_sample(3), raw_sample(7)])

    serial(model, samples[:1])
    assert loads == []
    with pytest.raises(ImportError, match="flash_attn unavailable"):
        model(**batch(samples))
    assert loads == ["veomni_flash_attention_2_with_sp"]


@pytest.mark.parametrize("checkpointing", [False, True])
def test_packed_sdpa_slices_with_host_bounds(monkeypatch, checkpointing):
    sdpa = minimax_h3_dit._sdpa_varlen_attention
    bounds = []

    def record(q, k, v, cu_seqlens, softmax_scale, compatibility_mode=False):
        bounds.append(cu_seqlens)
        return sdpa(q, k, v, cu_seqlens, softmax_scale, compatibility_mode)

    monkeypatch.setattr(minimax_h3_dit, "_sdpa_varlen_attention", record)
    raws = [raw_sample(3), raw_sample(7)]
    for row in raws:
        row["use_gradient_checkpointing"] = checkpointing
    samples = prepare(condition_model(), raws)
    model = tiny_model()
    reads = []
    tolist = torch.Tensor.tolist
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self: reads.append(self.shape) or tolist(self))
    out = model(**batch(samples))
    sum(out.loss.values()).backward()
    assert reads == []  # packing and the DiT use host bounds; no device read-back

    assert len(bounds) == 4 + 2 * checkpointing
    assert all(type(bound) is tuple and all(type(value) is int for value in bound) for bound in bounds)


@pytest.mark.parametrize("task", ["fl2va", "ref2va"])
def test_refiner_preserves_linear_row_counts_with_main_dit_packed(task):
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(3, task), raw_sample(9, task)])
    rows = {"out_proj": [], "fc2": [], "main": []}
    refiner = model.dit.token_refiner.blocks[0]
    modules = {"out_proj": refiner.attn.out_proj, "fc2": refiner.mlp.fc2, "main": model.dit.blocks[0]}
    handles = [
        module.register_forward_pre_hook(lambda mod, args, name=name: rows[name].append(args[0].shape[0]))
        for name, module in modules.items()
    ]
    try:
        sum(model(**batch(samples)).loss.values()).backward()
    finally:
        for handle in handles:
            handle.remove()
    assert rows["out_proj"] == rows["fc2"] == [3, 9]
    assert rows["main"] == [sum(sample["x"].shape[1] for sample in samples)]


@pytest.mark.parametrize("offload", [True, [False, True]])
def test_multisample_wrapper_rejects_checkpoint_offload(offload):
    inputs = batch(prepare(condition_model(), [raw_sample(), raw_sample()]))
    inputs["use_gradient_checkpointing_offload"] = offload
    with pytest.raises(ValueError, match="checkpoint offload"):
        tiny_model()(**inputs)


@pytest.mark.parametrize("offload", [False, True])
def test_inference_forwards_checkpoint_offload_to_core(offload):
    from veomni.models.diffusers.minimax_h3.inference import model_fn_minimax_h3

    row = raw_sample()
    model = tiny_model()

    def capture(module, args, kwargs):
        assert kwargs["use_gradient_checkpointing_offload"] is offload
        assert kwargs["packed_seq_params"]["cu_seqlens_host"] == tuple(row["packed"]["cu_seqlens"].tolist())
        assert kwargs["refiner_packed_seq_params"]["cu_seqlens_host"] == (0, 3)
        raise RuntimeError("inference offload forwarded")

    handle = model.dit.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="inference offload forwarded"):
            model_fn_minimax_h3(
                model,
                row["input_latents"],
                row["audio_input_latents"],
                row["packed"],
                row["prompt_embeds"],
                t_video=0.5,
                t_audio=0.5,
                keyframe_cond_anchor=row["keyframe_cond_anchor"],
                use_gradient_checkpointing_offload=offload,
            )
    finally:
        handle.remove()


def test_condition_preserves_checkpoint_offload_for_model_validation():
    raws = [raw_sample(), raw_sample()]
    for row in raws:
        row["use_gradient_checkpointing_offload"] = True
    samples = prepare(condition_model(), raws)
    assert all(sample["use_gradient_checkpointing_offload"] is True for sample in samples)
    model = tiny_model()
    with pytest.raises(ValueError, match="checkpoint offload"):
        model(**batch(samples))

    def capture(module, args, kwargs):
        assert kwargs["use_gradient_checkpointing_offload"] is True
        raise RuntimeError("single-sample offload forwarded")

    handle = model.dit.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with pytest.raises(RuntimeError, match="single-sample offload forwarded"):
            model(**samples[0])
    finally:
        handle.remove()


def test_invalid_precision_and_legacy_tail_fail_closed():
    model = tiny_model()
    samples = prepare(condition_model(), [raw_sample(), raw_sample()])
    changed = copy.deepcopy(samples)
    changed[0]["unique_timesteps"] = changed[0]["unique_timesteps"].bfloat16()
    with pytest.raises(ValueError, match="cast_forward_inputs"):
        model(**batch(changed))
    with pytest.raises(ValueError, match="tail padding"):
        model(**batch([legacy_tail(samples[0]), samples[1]]))
    row = raw_sample()
    row["keyframe_cond_anchor"] = None
    with pytest.raises(ValueError, match="anchor rows"):
        prepare(condition_model(), [row])
    with pytest.raises(ValueError, match="nonempty"):
        model(x=[])

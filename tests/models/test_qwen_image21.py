# ruff: noqa: E402

import copy
import os
from pathlib import Path

import diffusers
import pytest
import torch
import torch.nn.functional as F
import yaml
from PIL import Image


_REQUIRED_COMPONENTS = ("AutoencoderKLQwenImage21", "QwenImage21Pipeline", "QwenImage21Transformer2DModel")
_MISSING_COMPONENTS = [component for component in _REQUIRED_COMPONENTS if not hasattr(diffusers, component)]
if _MISSING_COMPONENTS and os.environ.get("VEOMNI_REQUIRE_QWEN_IMAGE21") == "1":
    raise ImportError(
        "The Qwen-Image-2.1 test job requires a Diffusers build with these components: "
        + ", ".join(_MISSING_COMPONENTS)
    )
if _MISSING_COMPONENTS:
    pytest.skip("Qwen-Image-2.1 requires a Diffusers build with QwenImage21 support.", allow_module_level=True)

from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers import QwenImage21Transformer2DModel as DiffusersQwenImage21Transformer2DModel

from veomni.lora import VeOmniLoraConfig, VeOmniLoraModel
from veomni.lora.state_dict import get_lora_state_dict
from veomni.lora.weight_loading import load_lora_weights
from veomni.models.diffusers.qwen_image21.qwen_image21_condition.configuration_qwen_image21_condition import (
    QwenImage21ConditionModelConfig,
)
from veomni.models.diffusers.qwen_image21.qwen_image21_condition.modeling_qwen_image21_condition import (
    QwenImage21ConditionModel,
)
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.configuration_qwen_image21_transformer import (
    QwenImage21Transformer2DModelConfig,
)
from veomni.models.diffusers.qwen_image21.qwen_image21_transformer.modeling_qwen_image21_transformer import (
    QwenImage21Transformer2DModel,
)


def _toy_model():
    config = QwenImage21Transformer2DModelConfig(
        patch_size=1,
        in_channels=4,
        out_channels=4,
        num_layers=1,
        attention_head_dim=8,
        num_attention_heads=1,
        context_in_dim=16,
        mlp_ratio=2,
        axes_dims_rope=(2, 2, 4),
        causal_condition=True,
    )
    return QwenImage21Transformer2DModel(config).eval()


def test_qwen_image21_supervised_forward_matches_diffusers_tail():
    torch.manual_seed(7)
    model = _toy_model()
    hidden_states = torch.randn(1, 4, 4)
    encoder_hidden_states = torch.randn(1, 3, 16)
    timestep = torch.tensor([0.25])
    img_shapes = [[(1, 2, 2)]]
    img_mask = torch.tensor([[False, False, False, True]])
    target = torch.randn_like(hidden_states)

    with torch.no_grad():
        upstream = DiffusersQwenImage21Transformer2DModel.forward(
            model,
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            img_mask=img_mask,
            return_dict=False,
        )[0][:, -hidden_states.shape[1] :]
        output = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_shapes=img_shapes,
            img_mask=img_mask,
            training_target=target,
            latents=target,
        )

    torch.testing.assert_close(output.predictions[0], upstream)
    torch.testing.assert_close(output.loss["mse_loss"], F.mse_loss(upstream.float(), target.float()))


def test_qwen_image21_transformer_save_reload(tmp_path):
    torch.manual_seed(13)
    model = _toy_model()
    expected = {key: value.detach().clone() for key, value in model.state_dict().items()}

    model.save_pretrained(str(tmp_path))
    reloaded = QwenImage21Transformer2DModel.from_pretrained(str(tmp_path))

    assert isinstance(reloaded.config, QwenImage21Transformer2DModelConfig)
    actual = reloaded.state_dict()
    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])


class _FakePosterior:
    def __init__(self, parameters):
        self.parameters = parameters


class _FakeVAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("anchor", torch.zeros(1))
        self.config = type(
            "VAEConfig",
            (),
            {"z_dim": 64, "latents_mean": [0.0] * 64, "latents_std": [1.0] * 64},
        )()

    @property
    def dtype(self):
        return self.anchor.dtype

    @property
    def device(self):
        return self.anchor.device

    def encode(self, image):
        assert image.shape[:3] == (1, 4, 1)
        assert image.shape[-2] == image.shape[-1]
        torch.testing.assert_close(image[:, 3], torch.ones_like(image[:, 3]))
        latent_size = image.shape[-1] // 16
        parameters = image.new_zeros((1, 128, 1, latent_size, latent_size))
        return type("EncoderOutput", (), {"latent_dist": _FakePosterior(parameters)})()


class _FakeScheduler(FlowMatchEulerDiscreteScheduler):
    def __init__(self):
        super().__init__(
            num_train_timesteps=1000,
            use_dynamic_shifting=True,
            base_image_seq_len=256,
            max_image_seq_len=8192,
            base_shift=0.5,
            max_shift=0.9,
            shift_terminal=0.02,
            time_shift_type="exponential",
        )
        self.last_mu = None
        self.scale_noise_called = False

    def set_timesteps(self, *args, mu=None, **kwargs):
        self.last_mu = mu
        return super().set_timesteps(*args, mu=mu, **kwargs)

    def scale_noise(self, *args, **kwargs):
        self.scale_noise_called = True
        return super().scale_noise(*args, **kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Qwen-Image-2.1 condition preprocessing is GPU-only.")
@pytest.mark.parametrize(("resolution", "latent_size"), [(512, 32), (1024, 64)])
def test_qwen_image21_condition_uses_resolution_shifted_schedule(monkeypatch, resolution, latent_size):
    device = torch.device("cuda")

    def fake_load_components(self):
        self.vae = _FakeVAE().to(device)
        self.scheduler = _FakeScheduler()

    monkeypatch.setattr(QwenImage21ConditionModel, "_load_components", fake_load_components)
    model = QwenImage21ConditionModel(QwenImage21ConditionModelConfig(height=resolution, width=resolution, seed=11))
    prompt_embeds = torch.randn(1, 6, 16, device=device)
    monkeypatch.setattr(
        model,
        "encode_prompt",
        lambda prompt, device=None: (
            prompt_embeds,
            torch.ones(1, 6, dtype=torch.long, device=prompt_embeds.device),
        ),
    )

    condition = model.get_condition(["a red seed"], [[Image.new("RGB", (resolution, resolution), "red")]])
    assert condition["latents"][0].shape == (1, 128, 1, latent_size, latent_size)
    assert condition["img_shapes"] == [[(1, latent_size, latent_size)]]

    ready = model.process_condition(**condition)
    latent_tokens = latent_size**2
    expected_mu = (latent_tokens * (0.9 - 0.5) / (8192 - 256)) + 0.5 - ((0.9 - 0.5) * 256 / (8192 - 256))
    assert model.scheduler.last_mu == pytest.approx(expected_mu)
    assert model.scheduler.scale_noise_called
    assert ready["hidden_states"][0].shape == (1, latent_tokens, 64)
    assert ready["training_target"][0].shape == (1, latent_tokens, 64)
    assert ready["img_mask"][0].shape == (1, 6 + latent_tokens // 4)
    assert ready["img_mask"][0][:, :6].sum() == 0
    assert ready["img_mask"][0][:, 6:].all()
    assert torch.isfinite(ready["hidden_states"][0]).all()
    assert torch.isfinite(ready["training_target"][0]).all()


def test_qwen_image21_lora_config_round_trips_adapter(tmp_path):
    config_path = Path("configs/dit/qwen_image21_lora.yaml")
    model_yaml = yaml.safe_load(config_path.read_text())["model"]
    assert model_yaml["accelerator"]["init_device"] == "cuda"
    assert model_yaml["accelerator"]["fsdp_config"]["fsdp_mode"] == "ddp"
    lora_yaml = model_yaml["lora_config"]
    assert lora_yaml["lora_modules"] == ["to_q", "to_k", "to_v", "to_out.0"]

    torch.manual_seed(19)
    base = _toy_model()
    wrapped = VeOmniLoraModel(copy.deepcopy(base), VeOmniLoraConfig.from_yaml(lora_yaml))
    trainable = [name for name, parameter in wrapped.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(".lora_A." in name or ".lora_B." in name for name in trainable)
    with torch.no_grad():
        for name, parameter in wrapped.named_parameters():
            if ".lora_B." in name:
                parameter.normal_()

    expected = {
        key: value.detach().clone()
        for key, value in get_lora_state_dict(wrapped, config=wrapped.get_lora_config()).items()
    }
    wrapped.save_pretrained(str(tmp_path))
    reloaded = VeOmniLoraModel.from_pretrained(copy.deepcopy(base), str(tmp_path))
    load_lora_weights(reloaded, str(tmp_path), init_device="cpu")
    actual = get_lora_state_dict(reloaded, config=reloaded.get_lora_config())

    assert set(actual) == set(expected)
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key])

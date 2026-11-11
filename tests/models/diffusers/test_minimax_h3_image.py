# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax-H3-Image transformer, VAE, layout, and registry tests."""

from __future__ import annotations

import json

import torch
from PIL import Image
from safetensors.torch import save_file

from tests.models.compare import eager_ops_config
from tests.models.tiny_configs import tiny_minimax_h3_image_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_minimax_h3_image_config as _tiny_config
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from veomni.models.diffusers.minimax_h3_image.inference import MiniMaxH3ImagePipeline
from veomni.models.diffusers.minimax_h3_image.layout import (
    build_image_packed_sequence,
    get_image_canvas_size,
    patchify_image_latents,
    unpatchify_image_rows,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_vae.configuration_minimax_h3_image_vae import (
    MiniMaxH3ImageVAEConfig,
)
from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_vae.modeling_minimax_h3_image_vae import (
    MiniMaxH3ImageVAE,
)


def _build_model() -> MiniMaxH3DiTModel:
    config = _tiny_config()
    config._set_ops_implementation(eager_ops_config())
    return MiniMaxH3DiTModel(config)


def _image_inputs():
    sequence_length = 7
    position_ids = torch.zeros(sequence_length, 3)
    position_ids[:3, 0] = torch.arange(3)
    position_ids[3:, 1] = torch.arange(4)
    return {
        "x": torch.randn(4, 2, requires_grad=True),
        "prompt_embeds": torch.randn(3, 16, requires_grad=True),
        "unique_timesteps": torch.zeros(1),
        "timestep_indices": torch.zeros(sequence_length, dtype=torch.long),
        "token_tags": torch.tensor([1, 1, 1, 0, 0, 0, 0]),
        "position_ids": position_ids,
        "image_indices": torch.arange(3, 7),
        "text_indices": torch.arange(3),
    }


def test_minimax_h3_image_condition_builds_from_registered_class_without_assets():
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition import (
        modeling_minimax_h3_image_condition,
    )

    config = _tiny_condition_config()
    config._set_ops_implementation(eager_ops_config())
    model_class = MODELING_REGISTRY["MiniMaxH3ImageConditionModel"]()
    model = model_class._from_config(config)

    assert type(model) is modeling_minimax_h3_image_condition.MiniMaxH3ImageConditionModel
    assert model.config.model_type == "MiniMaxH3ImageConditionModel"


def test_minimax_h3_image_forward_uses_pdd_heads_and_has_no_audio_modules():
    torch.manual_seed(0)
    model = _build_model().eval()
    assert model.dit.audio_enabled is False
    assert model.dit.audio_patch_proj is None
    assert model.dit.final_layer.audio_out is None
    assert len(model.dit.final_layer.pdd_out) == 2

    inputs = _image_inputs()
    with torch.no_grad():
        first = model.forward_image(**inputs, pdd_step_index=0)
        second = model.forward_image(**inputs, pdd_step_index=1)

    assert first.shape == (4, 2)
    assert second.shape == (4, 2)
    assert not torch.allclose(first, second)


def test_minimax_h3_image_forward_backpropagates_to_image_and_text_rows():
    model = _build_model()
    inputs = _image_inputs()
    output = model.forward_image(**inputs, pdd_step_index=0)
    output.sum().backward()

    assert inputs["x"].grad is not None
    assert inputs["prompt_embeds"].grad is not None


def test_minimax_h3_image_forward_keeps_fp32_projection_modules():
    model = _build_model().to(torch.bfloat16)
    model.dit.video_patch_proj.to(torch.float32)
    model.dit.time_embedder.to(torch.float32)
    model.dit.final_layer.video_out.to(torch.float32)
    for head in model.dit.final_layer.pdd_out:
        head.to(torch.float32)

    inputs = _image_inputs()
    inputs["x"] = inputs["x"].to(torch.bfloat16)
    inputs["prompt_embeds"] = inputs["prompt_embeds"].to(torch.bfloat16)
    with torch.no_grad():
        output = model.forward_image(**inputs, pdd_step_index=0)

    assert output.shape == (4, 2)
    assert output.dtype == torch.float32


def test_minimax_h3_image_pdd_heads_follow_checkpoint_layout():
    model = _build_model()
    pipe = MiniMaxH3ImagePipeline()
    pipe.dit = model
    weight = torch.randn(2, 2, 24)
    bias = torch.randn(2, 2)
    pipe._load_pdd_heads({"video_out.weight": weight, "video_out.bias": bias})

    for index, head in enumerate(model.dit.final_layer.pdd_out):
        assert torch.equal(head.weight, weight[index])
        assert torch.equal(head.bias, bias[index])


def _tiny_vae() -> MiniMaxH3ImageVAE:
    config = MiniMaxH3ImageVAEConfig(
        block_out_channels=(8, 8, 8, 8, 8, 8),
        layers_per_block=1,
        norm_num_groups=4,
        decoder_num_layers=1,
        decoder_num_attention_heads=2,
        decoder_attention_head_dim=8,
        decoder_num_register_tokens=2,
        decoder_ffn_mult=2,
    )
    return MiniMaxH3ImageVAE(config)


def test_minimax_h3_image_vae_roundtrip():
    vae = _tiny_vae().eval()
    vae.disable_tiling()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        posterior = vae.encode(image).latent_dist
        latent = posterior.mode()
        decoded = vae.decode(latent).sample

    assert latent.shape == (1, 24, 4, 4)
    assert decoded.shape == image.shape


def test_minimax_h3_image_vae_uses_diffusers_checkpoint_names():
    keys = set(_tiny_vae().state_dict())
    expected = {
        "encoder.conv_in.weight",
        "encoder.down_blocks.0.resnets.0.conv1.weight",
        "encoder.down_blocks.0.downsamplers.0.conv.weight",
        "quant_conv.weight",
        "post_quant_conv.weight",
        "decoder.proj_in.weight",
        "decoder.register_tokens",
        "decoder.transformer_blocks.0.attn.to_q.weight",
        "decoder.transformer_blocks.0.attn.to_out.0.weight",
        "decoder.transformer_blocks.0.ff.net.0.proj.weight",
        "decoder.transformer_blocks.0.ff.net.2.weight",
        "decoder.transformer_blocks.0.scale1",
    }
    assert expected <= keys
    assert not any("to_qkv" in key or "mask_token" in key for key in keys)


def test_minimax_h3_image_layout_places_text_references_and_target():
    text_token_tags = torch.tensor([1, 1])
    position_ids, token_tags, image_indices, text_indices, num_reference_rows = build_image_packed_sequence(
        text_token_tags,
        [(4, 4)],
        8,
        8,
        (1, 2, 2),
    )

    assert position_ids.shape == (22, 3)
    assert token_tags.tolist()[:2] == [1, 1]
    assert token_tags.tolist()[2:] == [0] * 20
    assert text_indices.tolist() == [0, 1]
    assert image_indices.tolist() == list(range(2, 22))
    assert num_reference_rows == 4
    assert position_ids[0, 0] == 0
    assert position_ids[1, 0] == 1
    assert position_ids[2, 0] == 2
    assert position_ids[6, 0] == 3


def test_minimax_h3_image_patchify_matches_unpatchify():
    latents = torch.randn(1, 24, 1, 8, 8)
    rows = patchify_image_latents(latents, (1, 2, 2))
    restored = unpatchify_image_rows(rows, 8, 8, 24, (1, 2, 2))
    assert rows.shape == (16, 96)
    assert torch.equal(restored, latents[:, :, 0])


def test_minimax_h3_image_canvas_size_uses_minimum_area():
    assert get_image_canvas_size(1024, 1024, 2048 * 2048, 32) == (2048, 2048)
    assert get_image_canvas_size(2048, 2048, 2048 * 2048, 32) == (2048, 2048)


def test_minimax_h3_image_edit_presentation_tags_picture_blocks():
    from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
        MiniMaxH3ImageConditionModel,
    )

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1] * len(text)}

        def convert_tokens_to_ids(self, token):
            return 99

    class FakeImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors="pt"):
            return {
                "pixel_values": torch.zeros(len(images), 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
            }

    class FakeProcessor:
        image_processor = FakeImageProcessor()

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return torch.randn(input_ids.shape[1], 16, dtype=torch.bfloat16)

    config = _tiny_condition_config()
    config._set_ops_implementation(eager_ops_config())
    model = MiniMaxH3ImageConditionModel(config)
    model._tokenizer = FakeTokenizer()
    model._processor = FakeProcessor()
    model._text_encoder = FakeTextEncoder()

    images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
    prompt_embeds, token_tags = model.encode_text("edit", reference_images=images)

    # Two "Picture i: " labels (11 characters each) and the 4-character prompt
    # are text; each 1x4x4 grid contributes 16 / merge_size**2 = 4 image pads
    # plus the vision start/end tokens.
    assert prompt_embeds.shape == (1, 38, 16)
    assert int((token_tags == 1).sum()) == 26
    assert int((token_tags == 0).sum()) == 12


def test_minimax_h3_image_pipeline_loads_tiny_checkpoint(tmp_path, monkeypatch):
    transformer_dir = tmp_path / "transformer"
    transformer_dir.mkdir()
    vae_dir = tmp_path / "vae_diffusers"
    vae_dir.mkdir()

    config = _tiny_config(latents_dim=24, patch_size=(1, 2, 2))
    config._set_ops_implementation(eager_ops_config())
    model = MiniMaxH3DiTModel(config)
    state = {
        key[len("dit.") :]: value
        for key, value in model.state_dict().items()
        if not key.startswith("dit.final_layer.pdd_out.")
    }
    save_file({key: value.contiguous() for key, value in state.items()}, transformer_dir / "model.safetensors")
    config.save_pretrained(transformer_dir)

    vae = _tiny_vae()
    save_file(
        {key: value.contiguous() for key, value in vae.state_dict().items()},
        vae_dir / "diffusion_pytorch_model.safetensors",
    )
    vae.config.save_pretrained(vae_dir)
    save_file(
        {"video_out.weight": torch.randn(2, 96, 24), "video_out.bias": torch.randn(2, 96)},
        tmp_path / "pdd_fused_heads.safetensors",
    )
    (tmp_path / "pdd_config.json").write_text(json.dumps({"sigmas": [1.0, 0.5, 0.0]}))

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [1] * min(4, max(1, len(text.split())))}

        def convert_tokens_to_ids(self, token):
            return 99

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = torch.nn.Parameter(torch.zeros(1))

        def forward(self, input_ids, attention_mask=None, **kwargs):
            return torch.randn(input_ids.shape[1], 16, dtype=torch.bfloat16)

    def fake_init(self, condition_config):
        self._text_encoder = FakeTextEncoder()
        self._vae = vae
        self._tokenizer = FakeTokenizer()
        self._processor = None

    from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
        MiniMaxH3ImageConditionModel,
    )

    monkeypatch.setattr(MiniMaxH3ImageConditionModel, "_init_encoders", fake_init)
    pipe = MiniMaxH3ImagePipeline.from_pretrained(str(tmp_path), torch_dtype=torch.bfloat16)
    image = pipe("a cat", height=32, width=32, min_generation_area=1024, seed=0)

    assert image.size == (32, 32)

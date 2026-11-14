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
from contextlib import nullcontext

import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from scripts.ci.accelerator.launch_plan import LaunchSpec
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_forward_backpropagates_to_image_and_text_rows():
    model = _build_model()
    inputs = _image_inputs()
    output = model.forward_image(**inputs, pdd_step_index=0)
    output.sum().backward()

    assert inputs["x"].grad is not None
    assert inputs["prompt_embeds"].grad is not None


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_forward_rejects_sequence_parallel(monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_dit

    model = _build_model()
    monkeypatch.setattr(minimax_h3_dit, "get_ulysses_sequence_parallel_group", lambda: object())
    with pytest.raises(NotImplementedError, match="sequence parallelism"):
        model.forward_image(**_image_inputs(), pdd_step_index=0)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_pdd_heads_follow_checkpoint_layout():
    model = _build_model()
    pipe = MiniMaxH3ImagePipeline(device="cpu")
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_tiles_and_stitches():
    vae = _tiny_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=16,
        tile_sample_min_overlap_width=16,
    )
    image = torch.randn(1, 3, 96, 128)
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent).sample

    assert latent.shape == (1, 24, 6, 8)
    assert decoded.shape == image.shape


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize(
    ("tile_height", "min_overlap", "image_height"),
    [(65, 16, 96), (64, 17, 112)],
)
def test_minimax_h3_image_vae_tiles_on_latent_grid(tile_height, min_overlap, image_height):
    vae = _tiny_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=tile_height,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=min_overlap,
        tile_sample_min_overlap_width=16,
    )
    indices, lengths, overlaps = vae._split_tiles(image_height, tile_height, min_overlap)
    ratio = vae.spatial_compression_ratio
    assert all(index % ratio == 0 for index in indices)
    assert all(length % ratio == 0 for length in lengths)
    assert all(overlap % ratio == 0 for overlap in overlaps)
    assert sum(lengths) - sum(overlaps) == image_height

    image = torch.randn(1, 3, image_height, 64)
    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent).sample

    assert latent.shape == (1, 24, image_height // ratio, 4)
    assert decoded.shape == image.shape


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_tiled_encoding_matches_untiled_shape_for_partial_edge():
    vae = _tiny_vae().eval()
    image = torch.randn(1, 3, 97, 64)

    with torch.no_grad():
        vae.disable_tiling()
        untiled_shape = vae.encode(image).latent_dist.mode().shape
        vae.enable_tiling(
            tile_sample_min_height=65,
            tile_sample_min_width=64,
            tile_sample_min_overlap_height=16,
            tile_sample_min_overlap_width=16,
        )
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent).sample

    assert latent.shape == untiled_shape == (1, 24, 6, 4)
    assert decoded.shape == (1, 3, 96, 64)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_preserves_blended_neighbors_at_tile_junction():
    vae = _tiny_vae()
    tiles = [[torch.full((1, 1, 4, 4), value) for value in values] for values in ((0.0, 4.0), (8.0, 12.0))]

    stitched = vae._stitch_tiles(tiles, [2], [2])

    expected = torch.tensor(
        [
            [0, 0, 0, 2, 4, 4],
            [0, 0, 0, 2, 4, 4],
            [0, 0, 0, 1, 4, 4],
            [4, 4, 4, 5.5, 8, 8],
            [8, 8, 8, 10, 12, 12],
            [8, 8, 8, 10, 12, 12],
        ]
    )
    torch.testing.assert_close(stitched[0, 0], expected)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("image_shape", [(112, 64), (64, 112)])
def test_minimax_h3_image_vae_rejects_zero_latent_tile_overlap(image_shape):
    vae = _tiny_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=1,
        tile_sample_min_overlap_width=1,
    )

    with torch.no_grad(), pytest.raises(ValueError, match="at least one latent"):
        vae.encode(torch.zeros(1, 3, *image_shape))


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_allows_small_overlap_for_single_tile():
    vae = _tiny_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=1,
        tile_sample_min_overlap_width=1,
    )
    image = torch.zeros(1, 3, 64, 64)

    with torch.no_grad():
        latent = vae.encode(image).latent_dist.mode()
        decoded = vae.decode(latent).sample

    assert latent.shape == (1, 24, 4, 4)
    assert decoded.shape == image.shape


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_allows_small_overlap_for_tiled_decode():
    vae = _tiny_vae().eval()
    vae.enable_tiling(
        tile_sample_min_height=64,
        tile_sample_min_width=64,
        tile_sample_min_overlap_height=1,
        tile_sample_min_overlap_width=1,
    )
    with torch.no_grad():
        decoded = vae.decode(torch.zeros(1, 24, 7, 4)).sample

    assert decoded.shape == (1, 3, 112, 64)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_vae_rejects_non_advancing_tiles():
    vae = _tiny_vae()
    with pytest.raises(ValueError, match="positive"):
        vae.enable_tiling(tile_sample_min_height=0)
    with pytest.raises(ValueError, match="positive"):
        vae.enable_tiling(tile_sample_min_overlap_height=0)
    with pytest.raises(ValueError, match="greater than"):
        vae.enable_tiling(
            tile_sample_min_height=64,
            tile_sample_min_width=64,
            tile_sample_min_overlap_height=64,
            tile_sample_min_overlap_width=64,
        )
    with pytest.raises(ValueError, match="greater than the overlap"):
        vae._split_tiles(128, 64, 64)
    with pytest.raises(ValueError, match="overlap must be positive"):
        vae._split_tiles(128, 64, 0)
    with pytest.raises(ValueError, match="latent grid"):
        vae._split_tiles(64, 17, 16)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_patchify_matches_unpatchify():
    latents = torch.randn(1, 24, 1, 8, 8)
    rows = patchify_image_latents(latents, (1, 2, 2))
    restored = unpatchify_image_rows(rows, 8, 8, 24, (1, 2, 2))
    assert rows.shape == (16, 96)
    assert torch.equal(restored, latents[:, :, 0])


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_canvas_size_uses_minimum_area():
    assert get_image_canvas_size(1024, 1024, 2048 * 2048, 32) == (2048, 2048)
    assert get_image_canvas_size(2048, 2048, 2048 * 2048, 32) == (2048, 2048)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_text_encoder_inherits_condition_ops(monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_text_encoder as text_encoder_module

    captured = {}

    class FakeLanguageModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = torch.nn.Identity()

    class FakeQwen3VLModel(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.language_model = FakeLanguageModel()
            captured["config"] = config

    monkeypatch.setattr(text_encoder_module, "Qwen3VLModel", FakeQwen3VLModel)
    config = _tiny_condition_config()
    config._set_ops_implementation(eager_ops_config())
    text_encoder_module.MiniMaxH3TextEncoder(num_retained_layers=1, config=config)

    assert captured["config"].ops_config.attn_implementation == "eager"


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
def test_minimax_h3_image_condition_forwards_config_to_text_encoder(tmp_path, monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_text_encoder as text_encoder_module
    from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition import (
        modeling_minimax_h3_image_condition as condition_module,
    )

    base = tmp_path
    (base / "processor").mkdir()
    (base / "text_encoder").mkdir()
    vae_dir = base / "vae_diffusers"
    vae_dir.mkdir()
    (vae_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"")

    captured = {}

    class FakeTextEncoder(torch.nn.Module):
        def __init__(self, num_retained_layers, config=None):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            captured["num_retained_layers"] = num_retained_layers
            captured["config"] = config

    class FakeVAEConfig:
        @classmethod
        def from_pretrained(cls, path):
            return cls()

    class FakeVAE(torch.nn.Module):
        def __init__(self, config):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))
            self.config = config

        def load_state_dict(self, *args, **kwargs):
            return None

    class FakeProcessor:
        tokenizer = object()

        @classmethod
        def from_pretrained(cls, path):
            return cls()

    monkeypatch.setattr(condition_module, "AutoProcessor", FakeProcessor)
    monkeypatch.setattr(condition_module, "MiniMaxH3ImageVAEConfig", FakeVAEConfig)
    monkeypatch.setattr(condition_module, "MiniMaxH3ImageVAE", FakeVAE)
    monkeypatch.setattr(condition_module, "load_file", lambda path: {})
    monkeypatch.setattr(condition_module, "_init_empty_weights", lambda: nullcontext())
    monkeypatch.setattr(
        condition_module.MiniMaxH3ImageConditionModel,
        "_stream_load_weights",
        staticmethod(lambda *args, **kwargs: 1),
    )
    monkeypatch.setattr(text_encoder_module, "MiniMaxH3TextEncoder", FakeTextEncoder)

    config = _tiny_condition_config(
        base_model_path=str(base),
        skip_encoder_load=False,
    )
    config._set_ops_implementation(eager_ops_config())
    condition_module.MiniMaxH3ImageConditionModel(config)

    assert captured["num_retained_layers"] == 50
    assert captured["config"] is config


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
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

    class FakeImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors="pt"):
            return {
                "pixel_values": torch.zeros(len(images), 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 4, 4]] * len(images)),
            }

    class FakeProcessor:
        image_processor = FakeImageProcessor()

    def fake_init(self, condition_config):
        self._text_encoder = FakeTextEncoder()
        self._vae = vae
        self._tokenizer = FakeTokenizer()
        self._processor = FakeProcessor()

    from veomni.models.diffusers.minimax_h3_image.minimax_h3_image_condition.modeling_minimax_h3_image_condition import (
        MiniMaxH3ImageConditionModel,
    )

    monkeypatch.setattr(MiniMaxH3ImageConditionModel, "_init_encoders", fake_init)
    pipe = MiniMaxH3ImagePipeline.from_pretrained(str(tmp_path), torch_dtype=torch.bfloat16)
    assert pipe.dit.dit.video_patch_proj.weight.dtype == torch.float32
    assert pipe.dit.dit.time_embedder.proj_in.weight.dtype == torch.float32
    assert pipe.dit.dit.final_layer.video_out.weight.dtype == torch.float32
    assert pipe.dit.dit.final_layer.pdd_out[0].weight.dtype == torch.float32

    torch.manual_seed(0)
    image = pipe("a cat", height=32, width=32, min_generation_area=1024, seed=0)
    assert image.size == (32, 32)
    torch.manual_seed(0)
    array = pipe("a cat", height=32, width=32, min_generation_area=1024, seed=0, output_type="np")
    assert array.dtype.name == "uint8"
    assert array.tobytes() == image.tobytes()

    from veomni.models.diffusers.minimax_h3_image import inference as inference_module

    monkeypatch.setattr(
        inference_module,
        "_resize_reference",
        lambda image, min_short_edge, max_short_edge: image.convert("RGB"),
    )
    edit = pipe(
        "edit <Picture 1>",
        height=64,
        width=64,
        min_generation_area=1024,
        references=[Image.new("RGB", (64, 64), "red")],
        seed=0,
    )
    assert edit.size == (64, 64)


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("sigmas", [[1.0, 0.0], [1.0, 0.75, 0.5, 0.0]])
def test_minimax_h3_image_pipeline_rejects_mismatched_pdd_schedule_before_loading(tmp_path, sigmas):
    save_file(
        {"video_out.weight": torch.zeros(2, 1, 1), "video_out.bias": torch.zeros(2, 1)},
        tmp_path / "pdd_fused_heads.safetensors",
    )
    (tmp_path / "pdd_config.json").write_text(json.dumps({"sigmas": sigmas}))

    with pytest.raises(ValueError, match=rf"{len(sigmas) - 1}.*2 PDD heads"):
        MiniMaxH3ImagePipeline.from_pretrained(str(tmp_path), device="cpu")


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("failure_phase", ["text_stage", "text_encode", "vae_stage", "reference_encode"])
def test_minimax_h3_image_pipeline_offloads_conditioning_on_failure(monkeypatch, failure_phase):
    from veomni.models.diffusers.minimax_h3_image import inference as inference_module

    failure = RuntimeError("conditioning failed")
    calls = []

    class FakeConditionModel:
        def encode_text(self, prompt, reference_images=None, device=None, dtype=None):
            if failure_phase == "text_encode":
                raise failure
            return torch.zeros(1, 1, 16), torch.tensor([1])

        def encode_reference_images(self, images, generator=None, device=None):
            raise failure

    pipe = MiniMaxH3ImagePipeline(device="cpu")
    pipe.condition_model = FakeConditionModel()
    pipe.text_encoder = torch.nn.Linear(1, 1)
    pipe.vae = torch.nn.Linear(1, 1)
    pipe.dit = torch.nn.Linear(1, 1)

    def stage(names):
        calls.append(("stage", tuple(names)))
        if (failure_phase == "text_stage" and names == ["text_encoder"]) or (
            failure_phase == "vae_stage" and names == ["vae"]
        ):
            raise failure

    monkeypatch.setattr(pipe, "_stage", stage)
    monkeypatch.setattr(pipe, "_offload", lambda names: calls.append(("offload", tuple(names))))
    monkeypatch.setattr(inference_module, "_resize_reference", lambda image, *_args: image)
    references = [Image.new("RGB", (32, 32))] if failure_phase in ("vae_stage", "reference_encode") else None

    with pytest.raises(RuntimeError, match="conditioning failed") as error:
        pipe("a cat", height=32, width=32, min_generation_area=1024, references=references)

    assert error.value is failure
    assert ("offload", ("text_encoder", "vae")) in calls


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("failure_phase", ["stage", "forward"])
def test_minimax_h3_image_pipeline_offloads_dit_on_failure(monkeypatch, failure_phase):
    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.param = torch.nn.Parameter(torch.zeros(1))

    class FakeConditionModel:
        def encode_text(self, prompt, reference_images=None, device=None, dtype=None):
            return torch.zeros(1, 1, 16), torch.tensor([1])

    failure = RuntimeError("boom")

    class FailingDiT(FakeModule):
        def forward_image(self, **kwargs):
            raise failure

    pipe = MiniMaxH3ImagePipeline(device="cpu")
    pipe.condition_model = FakeConditionModel()
    pipe.text_encoder = FakeModule()
    pipe.vae = FakeModule()
    pipe.dit = FailingDiT()
    calls = []

    def stage(names):
        calls.append(("stage", tuple(names)))
        if names == ["dit"] and failure_phase == "stage":
            raise failure

    monkeypatch.setattr(pipe, "_stage", stage)
    monkeypatch.setattr(pipe, "_offload", lambda names: calls.append(("offload", tuple(names))))

    with pytest.raises(RuntimeError, match="boom") as error:
        pipe("a cat", height=32, width=32, min_generation_area=1024)

    assert error.value is failure
    assert ("offload", ("dit",)) in calls


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("staged_tensor", ["parameter", "buffer", "buffer_only"])
def test_minimax_h3_image_pipeline_offloads_partially_staged_module(monkeypatch, staged_tensor):
    module = torch.nn.Module()
    if staged_tensor != "buffer_only":
        module.register_parameter("first", torch.nn.Parameter(torch.zeros(1)))
    tensor = torch.empty(1, device="meta")
    if staged_tensor == "parameter":
        module.register_parameter("staged", torch.nn.Parameter(tensor))
    else:
        module.register_buffer("staged", tensor)
    destinations = []
    monkeypatch.setattr(module, "to", lambda device: destinations.append(device))
    pipe = MiniMaxH3ImagePipeline(device="cpu")
    pipe.dit = module

    pipe._offload(["dit"])

    assert destinations == ["cpu"]


@pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))
@pytest.mark.parametrize("misplaced_tensor", ["parameter", "buffer", "buffer_only"])
def test_minimax_h3_image_pipeline_stages_all_misplaced_tensors(monkeypatch, misplaced_tensor):
    module = torch.nn.Module()
    if misplaced_tensor != "buffer_only":
        module.register_parameter("first", torch.nn.Parameter(torch.empty(1, device="meta")))
    if misplaced_tensor == "parameter":
        module.register_parameter("misplaced", torch.nn.Parameter(torch.ones(1)))
    else:
        module.register_buffer("misplaced", torch.ones(1))
    destinations = []
    monkeypatch.setattr(module, "to", lambda device: destinations.append(device))
    pipe = MiniMaxH3ImagePipeline(device="cpu")
    pipe.device = torch.device("meta")
    pipe.dit = module

    pipe._stage(["dit"])

    assert destinations == [torch.device("meta")]


@pytest.mark.parametrize("failure_phase", ["stage", "forward", "text_stage"])
def test_minimax_h3_image_pipeline_accelerator_offload_and_retry(failure_phase):
    from types import SimpleNamespace

    from veomni.utils.device import get_device_type, get_torch_device

    if get_device_type() == "cpu" or not get_torch_device().is_available():
        pytest.skip("DiT staging and offload require an accelerator.")

    pipe = MiniMaxH3ImagePipeline()
    failure = RuntimeError("injected DiT failure")

    class ConditionModel:
        def encode_text(self, prompt, reference_images=None, device=None, dtype=None):
            assert all(tensor.device == pipe.device for tensor in pipe.text_encoder.parameters())
            return torch.zeros(1, 1, 16, device=device, dtype=dtype), torch.tensor([1])

    class TextEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.projection = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.fail = failure_phase == "text_stage"

        def to(self, device):
            if torch.device(device).type != "cpu" and self.fail:
                self.projection.to(device)
                assert self.first.device.type == "cpu"
                assert self.projection.weight.device == pipe.device
                raise failure
            return super().to(device)

    class VAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(1))
            self.config = SimpleNamespace(latents_mean=[0] * 24, latents_std=[1] * 24)

        def decode(self, latents, return_dict=False):
            assert self.weight.device == latents.device == pipe.device
            return (latents[:, :3].repeat_interleave(16, dim=-2).repeat_interleave(16, dim=-1),)

    class DiT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Parameter(torch.ones(1, dtype=torch.float32))
            self.projection = torch.nn.Linear(1, 1, bias=False, dtype=torch.bfloat16)
            self.register_buffer("offset", torch.ones(1, dtype=torch.float32))
            self.fail = True
            self.forward_calls = 0

        def to(self, device):
            if torch.device(device).type != "cpu" and self.fail and failure_phase == "stage":
                self.projection.to(device)
                self.offset = self.offset.to(device)
                assert self.first.device.type == "cpu"
                assert self.projection.weight.device == self.offset.device == pipe.device
                raise failure
            return super().to(device)

        def forward_image(self, x, **kwargs):
            assert all(tensor.device == pipe.device for tensor in [*self.parameters(), *self.buffers()])
            self.forward_calls += 1
            if self.fail:
                raise failure
            return torch.zeros_like(x)

    pipe.condition_model = ConditionModel()
    pipe.text_encoder = TextEncoder()
    pipe.vae = VAE()
    pipe.dit = DiT()
    initial_text_state = {name: tensor.clone() for name, tensor in pipe.text_encoder.state_dict().items()}
    initial_state = {name: tensor.clone() for name, tensor in pipe.dit.state_dict().items()}

    with pytest.raises(RuntimeError, match="injected DiT failure") as error:
        pipe("a cat", height=32, width=32, min_generation_area=1024)

    assert error.value is failure
    assert all(tensor.device.type == "cpu" for tensor in pipe.text_encoder.parameters())
    assert all(tensor.device.type == "cpu" for tensor in [*pipe.dit.parameters(), *pipe.dit.buffers()])
    assert pipe.dit.forward_calls == (1 if failure_phase == "forward" else 0)
    for name, tensor in pipe.text_encoder.state_dict().items():
        torch.testing.assert_close(tensor, initial_text_state[name], rtol=0, atol=0)
    for name, tensor in pipe.dit.state_dict().items():
        torch.testing.assert_close(tensor, initial_state[name], rtol=0, atol=0)

    pipe.text_encoder.fail = False
    pipe.dit.fail = False
    image = pipe("a cat", height=32, width=32, min_generation_area=1024)

    assert image.size == (32, 32)
    assert pipe.dit.forward_calls == len(pipe.scheduler.sigmas) - 1 + (1 if failure_phase == "forward" else 0)
    assert all(tensor.device.type == "cpu" for tensor in pipe.text_encoder.parameters())
    assert all(tensor.device.type == "cpu" for tensor in [*pipe.dit.parameters(), *pipe.dit.buffers()])
    for name, tensor in pipe.text_encoder.state_dict().items():
        torch.testing.assert_close(tensor, initial_text_state[name], rtol=0, atol=0)
    for name, tensor in pipe.dit.state_dict().items():
        torch.testing.assert_close(tensor, initial_state[name], rtol=0, atol=0)
    pipe._offload(["text_encoder", "vae", "dit"])

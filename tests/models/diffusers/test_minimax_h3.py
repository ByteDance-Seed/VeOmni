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
# See the License for the specific language governing limitations
# under the License.

"""MiniMax H3 registry, public prediction/loss, and RMSNorm numerical tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config, ops_config_scope
from tests.models.tiny_configs import tiny_minimax_h3_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_minimax_h3_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import (
    MiniMaxH3Attention,
    VeomniRMSNorm,
    _PackedBounds,
)
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from veomni.ops.config import get_ops_config, set_ops_config


_VEOMNI_RMS_NORM_FORWARD = VeomniRMSNorm.forward


def _torch_rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _sdpa_ops_config() -> SimpleNamespace:
    """Portable MiniMax H3 attention path. This family has no local eager forward."""
    ops = eager_ops_config()
    ops.attn_implementation = "sdpa"
    return ops


def _build_norm(size: int = 16):
    with ops_config_scope(eager_ops_config()):
        return VeomniRMSNorm(size, eps=1e-6)


def _build_model():
    with ops_config_scope(_sdpa_ops_config()):
        # Keep the production latent channels/patch size: the public wrapper
        # unpacks 24-channel video and 32-dimensional audio tokens.
        return MiniMaxH3DiTModel(_tiny_config(latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2)))


def _minimax_h3_inputs(cond_rows: int) -> dict:
    video_rows = cond_rows + 4  # Two frames, each containing a 1x2 patch grid.
    audio_rows = 4  # Two channels, each containing two timesteps.
    seq_len = video_rows + audio_rows + 1
    return {
        "x": torch.randn(1, seq_len, 96, requires_grad=True),
        "audio_x": torch.randn(1, seq_len, 32, requires_grad=True),
        "img_position_ids": torch.arange(seq_len * 3).reshape(1, seq_len, 3),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(seq_len, dtype=torch.long),
        "update_mask": torch.arange(video_rows).remainder(2).float(),
        "token_tags": torch.tensor([0] * video_rows + [2] * audio_rows + [1]),
        "prompt_embeds": torch.randn(1, 16, requires_grad=True),
        "img_pos_info": {"position_ids": torch.arange(video_rows)},
        "audio_pos_info": {"position_ids": torch.arange(video_rows, video_rows + audio_rows)},
        "text_pos_info": {"position_ids": torch.tensor([seq_len - 1])},
        "img_pos_for_infer_output_info": {"position_ids": torch.arange(video_rows)},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, seq_len]), "max_seqlen_q": seq_len},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 1]), "max_seqlen_q": 1},
    }


def test_minimax_h3_condition_builds_from_registered_class_without_assets():
    from veomni.models import MODELING_REGISTRY
    from veomni.models.diffusers.minimax_h3.minimax_h3_condition import modeling_minimax_h3_condition

    model_class = MODELING_REGISTRY["MiniMaxH3ConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_minimax_h3_condition.MiniMaxH3ConditionModel
    assert model.config.model_type == "MiniMaxH3ConditionModel"


@pytest.mark.parametrize(
    ("cond_rows", "skip_mask", "training", "weighted"),
    [(0, False, True, False), (1, True, True, True), (1, True, False, False)],
    ids=["masked-training", "conditioned-weighted-training", "conditioned-inference"],
)
def test_minimax_h3_public_forward_matches_token_reference(monkeypatch, cond_rows, skip_mask, training, weighted):
    """Check the wrapper against raw DiT tokens plus independent latent/loss math.

    The raw DiT is shared; this is a wrapper and norm integration reference,
    not an independent implementation of the entire MiniMax backbone.
    """
    torch.manual_seed(1)
    reference = _build_model()
    ours = _build_model()
    assert ours.dit.blocks[0].attn.veomni_attn.impl == "sdpa"
    ours.load_state_dict(reference.state_dict())
    inputs = _minimax_h3_inputs(cond_rows)
    ours_inputs = {
        key: value.detach().clone().requires_grad_(value.requires_grad) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }
    video_target = torch.randn(1, 24, 2, 2, 4) + 1
    audio_target = torch.randn(2, 32, 2) - 2
    public_kwargs = {
        "cond_rows": cond_rows,
        "skip_mask_out_condition": skip_mask,
        "video_latent_shape": (2, 1, 2),
        "audio_latent_shape": (2, 2),
    }
    if training:
        public_kwargs.update(training_target=video_target, training_target_audio=audio_target)
    if weighted:
        public_kwargs.update(
            scheduler_video=SimpleNamespace(num_train_timesteps=1000, training_weight=lambda ts: 1 + ts / 1000),
            scheduler_audio=SimpleNamespace(num_train_timesteps=100, training_weight=lambda ts: 2 + ts / 100),
            t_video=0.25,
            t_audio=0.6,
        )
    expected_predictions = []

    previous = get_ops_config()
    set_ops_config(_sdpa_ops_config())
    monkeypatch.setattr(VeomniRMSNorm, "forward", VeomniRMSNorm.forward)
    try:

        def call(model):
            VeomniRMSNorm.forward = _torch_rms_norm_forward if model is reference else _VEOMNI_RMS_NORM_FORWARD
            if model is reference:
                video_tokens, audio_tokens = model.dit(
                    **dict(inputs, update_mask=None if skip_mask else inputs["update_mask"]),
                    skip_mask_out_condition=skip_mask,
                )
                # Each frame has two flattened Cx2x2 patches. Fold reconstructs
                # the spatial grid independently of production unpatchify_video.
                patches = video_tokens[cond_rows:].reshape(2, 2, 96).transpose(1, 2)
                video = -F.fold(patches, output_size=(2, 4), kernel_size=2, stride=2).transpose(0, 1).unsqueeze(0)
                audio = -torch.stack([audio_tokens[:2].T, audio_tokens[2:].T])
                expected_predictions.extend([video, audio])
                if not training:
                    return video, audio
                return (
                    F.mse_loss(video, video_target) * (1.75 if weighted else 1),
                    F.mse_loss(audio, audio_target) * (2.4 if weighted else 1),
                )
            output = model(**ours_inputs, **public_kwargs)
            assert len(output.predictions) == 2
            for actual, expected in zip(output.predictions, expected_predictions, strict=True):
                torch.testing.assert_close(actual, expected, atol=EAGER_ATOL, rtol=EAGER_RTOL)
            if not training:
                assert output.loss is None
                return tuple(output.predictions)
            assert set(output.loss) == {"mse_video", "mse_audio"}
            return output.loss["mse_video"], output.loss["mse_audio"]

        assert_outputs_and_grads_match(reference, ours, call)
        for key in ("x", "audio_x", "prompt_embeds"):
            assert inputs[key].grad is not None
            torch.testing.assert_close(
                ours_inputs[key].grad, inputs[key].grad, atol=EAGER_GRAD_ATOL, rtol=EAGER_GRAD_RTOL
            )
    finally:
        set_ops_config(previous)


def _tiny_attention():
    with ops_config_scope(_sdpa_ops_config()):
        return MiniMaxH3Attention(hidden_size=16, num_attention_heads=2, attention_head_dim=8, qk_norm_eps=1e-5)


def test_minimax_h3_packed_sdpa_matches_independent_segments():
    torch.manual_seed(0)
    attn = _tiny_attention()
    packed = torch.randn(6, 16)
    cu_seqlens = torch.tensor([0, 2, 6], dtype=torch.int32)
    out_packed = attn(packed, rope_cos=None, rope_sin=None, cu_seqlens=cu_seqlens, max_seqlen=4, valid_seqlen=6)
    out_a = attn(
        packed[:2],
        rope_cos=None,
        rope_sin=None,
        cu_seqlens=torch.tensor([0, 2], dtype=torch.int32),
        max_seqlen=2,
        valid_seqlen=2,
    )
    out_b = attn(
        packed[2:],
        rope_cos=None,
        rope_sin=None,
        cu_seqlens=torch.tensor([0, 4], dtype=torch.int32),
        max_seqlen=4,
        valid_seqlen=4,
    )
    torch.testing.assert_close(out_packed, torch.cat((out_a, out_b), dim=0), atol=EAGER_ATOL, rtol=EAGER_RTOL)


def test_minimax_h3_sdpa_packed_uses_host_slices_not_varlen_kwargs(monkeypatch):
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import minimax_h3_dit

    attn = _tiny_attention()
    captured: dict = {}
    orig = minimax_h3_dit._sdpa_varlen_attention

    def record(q, k, v, cu_seqlens, softmax_scale, compatibility_mode=False):
        captured["cu_seqlens"] = cu_seqlens
        captured["compatibility_mode"] = compatibility_mode
        return orig(q, k, v, cu_seqlens, softmax_scale, compatibility_mode)

    monkeypatch.setattr(minimax_h3_dit, "_sdpa_varlen_attention", record)

    def boom(*args, **kwargs):
        raise AssertionError("SDPA packed must not dispatch through veomni_attn")

    attn.veomni_attn = boom
    hidden = torch.randn(6, 16)
    attn(
        hidden,
        rope_cos=None,
        rope_sin=None,
        cu_seqlens=torch.tensor([0, 2, 6], dtype=torch.int32),
        max_seqlen=4,
        valid_seqlen=6,
    )
    assert captured["cu_seqlens"].tolist() == [0, 2, 6]
    assert captured["compatibility_mode"] is False


def test_minimax_h3_flash2_bind_defers_until_packed_bounds():
    ops = eager_ops_config()
    ops.attn_implementation = "flash_attention_2"
    with ops_config_scope(ops):
        attn = MiniMaxH3Attention(hidden_size=16, num_attention_heads=2, attention_head_dim=8, qk_norm_eps=1e-5)
    assert attn.veomni_attn.impl == "sdpa"
    assert attn.veomni_rope.op == "rope"
    assert attn.veomni_rope.variant == "partial"

    captured: dict = {}

    def record(_module, query, _key, _value, attention_mask=None, **kwargs):
        captured["attention_mask"] = attention_mask
        captured.update(kwargs)
        return query.transpose(1, 2), None

    attn.veomni_attn = record
    attn.config._attn_implementation = "veomni_flash_attention_2"
    hidden = torch.randn(6, 16)
    attn(
        hidden,
        rope_cos=None,
        rope_sin=None,
        cu_seqlens=_PackedBounds(torch.tensor([0, 2, 6], dtype=torch.int32), (0, 2, 6)),
        max_seqlen=4,
        valid_seqlen=6,
    )
    assert captured["attention_mask"] is None
    assert captured["max_length_q"] == 4
    assert captured["max_length_k"] == 4
    assert captured["cu_seq_lens_q"].tolist() == [0, 2, 6]
    assert captured["cu_seq_lens_k"].tolist() == [0, 2, 6]


def test_minimax_h3_video_vae_attention_uses_partial_rope():
    from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_video_vae import Attention

    with ops_config_scope(_sdpa_ops_config()):
        attn = Attention(heads=2, dim_head=8, qk_norm_type=None)
    assert attn.veomni_rope.op == "rope"
    assert attn.veomni_rope.variant == "partial"

    torch.manual_seed(0)
    hidden = torch.randn(2, 4, 16)
    cos = torch.randn(2, 4, 6)
    sin = torch.randn(2, 4, 6)
    captured: dict = {}
    orig = attn.veomni_rope

    def record(query, key, rope_cos, rope_sin, unsqueeze_dim=1):
        captured["unsqueeze_dim"] = unsqueeze_dim
        captured["q_shape"] = tuple(query.shape)
        captured["cos_shape"] = tuple(rope_cos.shape)
        return orig(query, key, rope_cos, rope_sin, unsqueeze_dim=unsqueeze_dim)

    attn.veomni_rope = record
    out = attn(hidden, rotary_pos_emb=(cos, sin))
    assert out.shape == hidden.shape
    assert captured["unsqueeze_dim"] == 2
    assert captured["q_shape"] == (2, 4, 2, 8)
    assert captured["cos_shape"] == (2, 4, 6)


def test_minimax_h3_pipeline_constructs_without_weights():
    from veomni.models.diffusers.minimax_h3.inference import MiniMaxH3Pipeline

    pipe = MiniMaxH3Pipeline(device="cpu")
    assert pipe.dit is None
    assert pipe.units
    assert callable(pipe.model_fn)


def test_minimax_h3_rms_norm_matches_official():
    torch.manual_seed(0)
    official = nn.RMSNorm(16, eps=1e-6)
    ours = _build_norm()
    ours.load_state_dict(official.state_dict())
    hidden = torch.randn(2, 8, 16)

    def call(model):
        return model(hidden)

    assert_outputs_and_grads_match(official, ours, call)

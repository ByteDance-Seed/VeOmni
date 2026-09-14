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

from tests.models.compare import assert_outputs_and_grads_match, eager_ops_config
from tests.models.tiny_configs import tiny_minimax_h3_condition_config as _tiny_condition_config
from tests.models.tiny_configs import tiny_minimax_h3_config as _tiny_config
from tests.ops.tol import EAGER_ATOL, EAGER_GRAD_ATOL, EAGER_GRAD_RTOL, EAGER_RTOL
from veomni.models.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import VeomniRMSNorm
from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
    MiniMaxH3DiTModel,
)
from veomni.ops.config import get_ops_config, set_ops_config


_VEOMNI_RMS_NORM_FORWARD = VeomniRMSNorm.forward


def _torch_rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)


def _build_norm(size: int = 16):
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        return VeomniRMSNorm(size, eps=1e-6)
    finally:
        set_ops_config(previous)


def _build_model():
    previous = get_ops_config()
    set_ops_config(eager_ops_config())
    try:
        # Keep the production latent channels/patch size: the public wrapper
        # unpacks 24-channel video and 32-dimensional audio tokens.
        return MiniMaxH3DiTModel(_tiny_config(latents_dim=24, audio_latents_dim=32, patch_size=(1, 2, 2)))
    finally:
        set_ops_config(previous)


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


def test_minimax_h3_configs_roundtrip_through_registry(tmp_path):
    from veomni.models import build_config, get_model_class
    from veomni.models.diffusers.minimax_h3.minimax_h3_condition.configuration_minimax_h3_condition import (
        MiniMaxH3ConditionModelConfig,
    )
    from veomni.models.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
        MiniMaxH3DiTModelConfig,
    )

    transformer_path = tmp_path / "transformer"
    condition_path = tmp_path / "condition"
    _tiny_config().save_pretrained(transformer_path)
    _tiny_condition_config().save_pretrained(condition_path)

    transformer = build_config(str(transformer_path))
    condition = build_config(str(condition_path))

    assert type(transformer) is MiniMaxH3DiTModelConfig
    assert type(condition) is MiniMaxH3ConditionModelConfig
    assert get_model_class(transformer).__name__ == "MiniMaxH3DiTModel"
    assert get_model_class(condition).__name__ == "MiniMaxH3ConditionModel"


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
    from veomni.models.diffusers.minimax_h3.minimax_h3_core import core

    torch.manual_seed(1)
    reference = _build_model()
    ours = _build_model()
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
    set_ops_config(eager_ops_config())
    monkeypatch.setattr(core, "ATTENTION_IMPLEMENTATION", "torch")
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


def test_minimax_h3_pipeline_constructs_without_weights():
    from veomni.models.diffusers.minimax_h3.inference import MiniMaxH3Pipeline

    pipe = MiniMaxH3Pipeline(device="cpu")
    assert pipe.dit is None
    assert len(pipe.units) == 8
    assert pipe.model_fn is not None


def test_minimax_h3_rms_norm_matches_official():
    torch.manual_seed(0)
    official = nn.RMSNorm(16, eps=1e-6)
    ours = _build_norm()
    ours.load_state_dict(official.state_dict())
    hidden = torch.randn(2, 8, 16)

    def call(model):
        return model(hidden)

    assert_outputs_and_grads_match(official, ours, call)

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

"""MiniMax H3 models_kernel registry and eager parity tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tests.models_kernel.compare import assert_outputs_and_grads_match, eager_ops_config
from tests.models_kernel.tiny_configs import tiny_minimax_h3_condition_config as _tiny_condition_config
from tests.models_kernel.tiny_configs import tiny_minimax_h3_config as _tiny_config
from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import VeomniRMSNorm
from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_transformer.modeling_minimax_h3_transformer import (
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
        return MiniMaxH3DiTModel(_tiny_config())
    finally:
        set_ops_config(previous)


def _minimax_h3_inputs() -> dict:
    return {
        "x": torch.randn(1, 4, 2),
        "audio_x": torch.randn(1, 4, 4),
        "img_position_ids": torch.tensor([[[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]]),
        "unique_timesteps": torch.tensor([0.5]),
        "inverse_indices": torch.zeros(4, dtype=torch.long),
        "update_mask": torch.ones(2),
        "token_tags": torch.tensor([0, 0, 1, 2]),
        "prompt_embeds": torch.randn(1, 16),
        "img_pos_info": {"position_ids": torch.tensor([0, 1])},
        "audio_pos_info": {"position_ids": torch.tensor([3])},
        "text_pos_info": {"position_ids": torch.tensor([2])},
        "img_pos_for_infer_output_info": {"position_ids": torch.tensor([0, 1])},
        "packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 4]), "max_seqlen_q": 4},
        "refiner_packed_seq_params": {"cu_seqlens_q": torch.tensor([0, 1]), "max_seqlen_q": 1},
    }


def test_minimax_h3_configs_roundtrip_through_registry(tmp_path):
    from veomni.models_kernel import build_config, get_model_class
    from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_condition.configuration_minimax_h3_condition import (
        MiniMaxH3ConditionModelConfig,
    )
    from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_transformer.configuration_minimax_h3_transformer import (
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
    from veomni.models_kernel import MODELING_REGISTRY
    from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_condition import modeling_minimax_h3_condition

    model_class = MODELING_REGISTRY["MiniMaxH3ConditionModel"]()
    model = model_class._from_config(_tiny_condition_config())

    assert type(model) is modeling_minimax_h3_condition.MiniMaxH3ConditionModel
    assert model.config.model_type == "MiniMaxH3ConditionModel"


def test_minimax_h3_eager_forward_and_backward_match_torch_reference():
    from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_core import core

    torch.manual_seed(1)
    reference = _build_model()
    ours = _build_model()
    ours.load_state_dict(reference.state_dict())
    inputs = _minimax_h3_inputs()

    previous = get_ops_config()
    previous_attention_impl = core.ATTENTION_IMPLEMENTATION
    previous_norm_forward = VeomniRMSNorm.forward
    set_ops_config(eager_ops_config())
    core.ATTENTION_IMPLEMENTATION = "torch"
    try:

        def call(model):
            VeomniRMSNorm.forward = _torch_rms_norm_forward if model is reference else _VEOMNI_RMS_NORM_FORWARD
            return model.dit(**inputs)

        assert_outputs_and_grads_match(reference, ours, call)
    finally:
        VeomniRMSNorm.forward = previous_norm_forward
        core.ATTENTION_IMPLEMENTATION = previous_attention_impl
        set_ops_config(previous)


def test_minimax_h3_pipeline_constructs_without_weights():
    from veomni.models_kernel.diffusers.minimax_h3.inference import MiniMaxH3Pipeline

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

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

"""MoVQGAN models_kernel consume tests.

Direct-import the staged class. Do not register or use
``build_foundation_model``. There is no RMS / RoPE / CE kernel site.
Compare a tiny encode/decode against ``tests/models_kernel/refs/movqgan/``.
"""

from __future__ import annotations

import torch

from tests.models_kernel.compare import assert_no_ops_or_old_models_import, assert_outputs_and_grads_match
from tests.models_kernel.refs.movqgan import MoVQGAN as RefMoVQGAN
from tests.models_kernel.refs.movqgan import MoVQGANConfig as RefMoVQGANConfig
from veomni.models_kernel.transformers.movqgan.configuration_movqgan import MoVQGANConfig
from veomni.models_kernel.transformers.movqgan.modeling_movqgan import MoVQGAN


def _tiny_kwargs() -> dict:
    return {
        "embed_dim": 4,
        "n_embed": 32,
        "double_z": False,
        "z_channels": 4,
        "resolution": 32,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 32,
        "ch_mult": (1, 2),
        "num_res_blocks": 1,
        "attn_resolutions": (16,),
        "dropout": 0.0,
    }


def test_movqgan_modeling_has_no_opslot_or_ops_import():
    from veomni.models_kernel.transformers.movqgan import modeling_movqgan

    assert_no_ops_or_old_models_import(modeling_movqgan, require_loss_utils=False)


def test_movqgan_eager_matches_official():
    torch.manual_seed(0)
    official = RefMoVQGAN(RefMoVQGANConfig(**_tiny_kwargs()))
    ours = MoVQGAN(MoVQGANConfig(**_tiny_kwargs()))
    ours.load_state_dict(official.state_dict())
    features = torch.randn(2, 3, 32, 32)

    def call(model):
        return model(features)

    assert_outputs_and_grads_match(official, ours, call)

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

"""MiniMax H3 models_kernel consume tests.

Direct-import staged classes. Compare ``VeomniRMSNorm`` against official
``torch.nn.RMSNorm``.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tests.models_kernel.compare import (
    assert_outputs_and_grads_match,
    eager_kernels_config,
)
from veomni.kernels import VeomniKernel
from veomni.kernels.config import get_kernels_config, set_kernels_config


def _build_ours(size: int = 16, kernels: SimpleNamespace | None = None):
    from veomni.models_kernel.diffusers.minimax_h3.minimax_h3_core.minimax_h3_dit import VeomniRMSNorm

    previous = get_kernels_config()
    set_kernels_config(kernels if kernels is not None else eager_kernels_config())
    try:
        return VeomniRMSNorm(size, eps=1e-6)
    finally:
        set_kernels_config(previous)


def test_minimax_h3_constructs_local_kernels():
    norm = _build_ours()
    assert isinstance(norm.veomni_rms_norm, VeomniKernel)
    assert norm.veomni_rms_norm.kernel == "rms_norm"
    assert norm.veomni_rms_norm.variant == "standard"
    assert norm.veomni_rms_norm.impl == "eager"


def test_minimax_h3_instances_keep_distinct_impls():
    eager = _build_ours(kernels=eager_kernels_config())
    other_cfg = eager_kernels_config()
    other_cfg.rms_norm_implementation = "liger_kernel"
    other = _build_ours(kernels=other_cfg)

    assert eager.veomni_rms_norm.impl == "eager"
    assert other.veomni_rms_norm.impl == "liger_kernel"

    set_kernels_config(other_cfg)
    assert eager.veomni_rms_norm.impl == "eager"


def test_minimax_h3_pipeline_constructs_without_weights():
    from veomni.models_kernel.diffusers.minimax_h3.inference import MiniMaxH3Pipeline

    pipe = MiniMaxH3Pipeline(device="cpu")
    assert pipe.dit is None
    assert len(pipe.units) == 8
    assert pipe.model_fn is not None


def test_minimax_h3_rms_norm_matches_official():
    torch.manual_seed(0)
    official = nn.RMSNorm(16, eps=1e-6)
    ours = _build_ours()
    ours.load_state_dict(official.state_dict())
    hidden = torch.randn(2, 8, 16)

    def call(model):
        return model(hidden)

    assert_outputs_and_grads_match(official, ours, call)

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

"""SeedOss models consume tests.

Direct-import the generated class. Compare a toy CausalLM against HuggingFace.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch
from transformers.models.seed_oss.configuration_seed_oss import SeedOssConfig
from transformers.models.seed_oss.modeling_seed_oss import SeedOssForCausalLM as HFSeedOssForCausalLM

from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    ops_config_scope,
)
from tests.models.tiny_configs import tiny_seed_oss_config as _tiny_config
from veomni.ops import VeomniOp


def _seed_oss_cls():
    from veomni.utils.device import IS_NPU_AVAILABLE

    if IS_NPU_AVAILABLE:
        from veomni.models.transformers.seed_oss.generated.patched_modeling_seed_oss_npu import (
            SeedOssForCausalLM,
        )
    else:
        from veomni.models.transformers.seed_oss.generated.patched_modeling_seed_oss_gpu import (
            SeedOssForCausalLM,
        )
    return SeedOssForCausalLM


def _build_ours(config: SeedOssConfig, ops: SimpleNamespace | None = None):
    with ops_config_scope(ops if ops is not None else eager_ops_config()):
        return _seed_oss_cls()(config)


def test_seed_oss_constructs_local_kernels():
    model = _build_ours(_tiny_config())
    assert isinstance(model.veomni_ce, VeomniOp)
    assert model.veomni_ce.impl == "eager"
    layer = model.model.layers[0]
    assert layer.input_layernorm.veomni_rms_norm.impl == "eager"
    assert layer.mlp.veomni_swiglu_mlp.impl == "eager"


def test_seed_oss_eager_matches_hf():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFSeedOssForCausalLM(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, config.vocab_size, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids)

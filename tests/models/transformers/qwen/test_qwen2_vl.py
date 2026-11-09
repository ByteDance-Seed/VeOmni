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

"""Qwen2-VL text and multimodal Hugging Face parity tests.

Direct-import the generated class. Compare a toy model against Hugging Face on
both the text-only and image+text paths.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
from transformers.models.qwen2_vl.configuration_qwen2_vl import (
    Qwen2VLConfig,
)
from transformers.models.qwen2_vl.modeling_qwen2_vl import (
    Qwen2VLForConditionalGeneration as HFQwen2VLForConditionalGeneration,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    qwen_image_inputs,
)
from tests.models.tiny_configs import tiny_qwen2_vl_config as _tiny_config
from tests.ops.tol import EAGER_GRAD_ATOL, EAGER_GRAD_RTOL
from tests.tools.launch_utils import torchrun
from veomni.utils.device import IS_CUDA_AVAILABLE, get_device_type


_CI_ACCELERATOR_2 = LaunchSpec(nproc=2, device="accelerator")


IMAGE_TOKEN_ID = 120
VIDEO_TOKEN_ID = 121


def _qwen2_vl_vision_fsdp_checkpoint_worker():
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.models.transformers.qwen2_vl.generated.patched_modeling_qwen2_vl_gpu import (
        Qwen2VisionTransformerPretrainedModel,
    )

    world_size = torch.distributed.get_world_size()
    _init_parallel_state(dp_size=world_size, dp_shard_size=world_size)
    torch.manual_seed(43)
    config = _tiny_config().vision_config
    ops = eager_ops_config()
    ops.rotary_pos_emb_vision_implementation = None
    config._set_ops_implementation(ops)
    reference = Qwen2VisionTransformerPretrainedModel(config).to(get_device_type()).train()
    checkpointed = copy.deepcopy(reference)
    checkpointed.gradient_checkpointing_enable({"use_reentrant": False})
    assert all(block.attn.veomni_rope.impl == "liger_kernel" for block in checkpointed.blocks)
    mesh = init_device_mesh(get_device_type(), (world_size,))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, cast_forward_inputs=True)
    features = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixels = torch.randn(16, features, device=get_device_type())
    grid = torch.tensor([[1, 4, 4]], device=get_device_type())
    results = []
    for model in (reference, checkpointed):
        for block in model.blocks:
            fully_shard(block, mesh=mesh, mp_policy=policy)
        fully_shard(model, mesh=mesh, mp_policy=policy)
        output = model(pixels, grid).pooler_output
        output.float().square().mean().backward()
        grads = {
            name: parameter.grad.to_local().detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        assert grads and all(torch.isfinite(grad).all() for grad in grads.values())
        results.append((output.detach().clone(), grads))
    expected, actual = results
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    assert actual[1].keys() == expected[1].keys()
    for name in expected[1]:
        # FSDP recomputes with FP32 tables instead of the first pass's BF16
        # tables, so gradients can differ by small low-precision roundoff.
        torch.testing.assert_close(
            actual[1][name],
            expected[1][name],
            rtol=EAGER_GRAD_RTOL,
            atol=EAGER_GRAD_ATOL,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )


@pytest.mark.launch_plan(provider=_CI_ACCELERATOR_2)
@pytest.mark.skipif(not IS_CUDA_AVAILABLE, reason="Liger vision RoPE with FSDP2 needs GPU")
def test_qwen2_vl_default_vision_rope_with_fsdp_checkpoint():
    pytest.importorskip("liger_kernel.ops.rope")
    torchrun(_qwen2_vl_vision_fsdp_checkpoint_worker, world_size=_CI_ACCELERATOR_2.nproc)


def _build_ours(config: Qwen2VLConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_vl.generated.patched_modeling_qwen2_vl_gpu import (
        Qwen2VLForConditionalGeneration,
    )

    config._set_ops_implementation(ops if ops is not None else eager_ops_config())
    return Qwen2VLForConditionalGeneration(config)


def test_qwen2_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(hf, ours, input_ids=input_ids, logits_equal=True)


def test_qwen2_vl_eager_matches_hf_image_and_text():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    image = qwen_image_inputs(config, input_ids)
    ids = image.pop("input_ids")
    labels = image.pop("labels")
    assert_eager_matches_hf(hf, ours, input_ids=ids, labels=labels, fwd_kwargs=image, logits_equal=True)

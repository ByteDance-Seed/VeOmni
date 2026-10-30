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

"""Qwen2.5-VL text, image, and video Hugging Face parity tests.

Direct-import the generated class. Compare a toy model against Hugging Face on
the text-only, image+text, and video+text paths.

This file owns Qwen2.5 regular vision forward, non-default dummy geometry, and
CPU/Gloo SP execution. The shared vision suite in
tests/distributed/test_dummy_forward.py owns standard geometry and accelerator
rejection coverage.
"""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import (
    Qwen2_5_VLConfig,
)
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VLForConditionalGeneration as HFQwen2_5_VLForConditionalGeneration,
)

from scripts.ci.accelerator.launch_plan import LaunchSpec
from tests.models.compare import (
    assert_eager_matches_hf,
    eager_ops_config,
    qwen_image_inputs,
    qwen_video_inputs,
)
from tests.models.tiny_configs import tiny_qwen2_5_vl_config as _tiny_config


pytestmark = pytest.mark.launch_plan(provider=LaunchSpec(device="cpu"))


def _build_ours(config: Qwen2_5_VLConfig, ops: SimpleNamespace | None = None):
    from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
        Qwen2_5_VLForConditionalGeneration,
    )

    config._set_ops_implementation(ops if ops is not None else eager_ops_config())
    return Qwen2_5_VLForConditionalGeneration(config)


def _mask_kwargs(input_ids: torch.Tensor) -> dict:
    zeros = torch.zeros_like(input_ids, dtype=torch.bool)
    return {
        "image_mask": zeros,
        "video_mask": zeros,
    }


@pytest.mark.parametrize(
    "channels,temporal,patch,merge",
    [(3, 2, 8, 2), (1, 1, 8, 2), (3, 2, 8, 1), (3, 2, 8, 4)],
)
def test_qwen2_5_vl_dummy_forward_uses_vision_geometry(channels, temporal, patch, merge):
    from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    config = _tiny_config().vision_config
    config.in_channels = channels
    config.temporal_patch_size = temporal
    config.patch_size = patch
    config.spatial_merge_size = merge
    config.window_size = 4 * merge * patch
    config._set_ops_implementation(eager_ops_config())
    model = Qwen2_5_VisionTransformerPretrainedModel(config).train()
    output = model.dummy_forward().pooler_output
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in model.parameters())


@pytest.mark.parametrize("attention", ["eager", "veomni_sdpa"])
def test_qwen2_5_vl_regular_vision_forward(attention):
    from veomni.models.transformers.qwen2_5vl.generated import patched_modeling_qwen2_5_vl_gpu as module

    config = _tiny_config().vision_config
    ops = eager_ops_config()
    ops.attn_implementation = attention
    config._set_ops_implementation(ops)
    torch.manual_seed(177)
    model = module.Qwen2_5_VisionTransformerPretrainedModel(config).train()
    merge = config.spatial_merge_size
    patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
    pixels = torch.randn(merge * merge, patch_dim, requires_grad=True)
    grid_thw = torch.tensor([[1, merge, merge]], dtype=torch.long)
    metadata = module._qwen2_5_vit_metadata(grid_thw.tolist(), 0, model.window_size, merge, model.patch_size)
    output = model(pixels, grid_thw, return_dict=True, vit_metadata=metadata).pooler_output
    assert torch.isfinite(output).all()
    output.float().square().mean().backward()
    assert pixels.grad is not None and torch.isfinite(pixels.grad).all()
    assert torch.count_nonzero(pixels.grad)
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
    assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in model.parameters())


def _qwen2_5_vl_dummy_sp_worker(rank, merge, endpoint):
    from veomni.distributed.parallel_state import _init_parallel_state
    from veomni.models.transformers.qwen2_5vl.generated.patched_modeling_qwen2_5_vl_gpu import (
        Qwen2_5_VisionTransformerPretrainedModel,
    )

    os.environ.update(
        MASTER_ADDR=endpoint.host,
        MASTER_PORT=str(endpoint.port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE="2",
    )
    dist.init_process_group("gloo", store=endpoint.connect(), rank=rank, world_size=2)
    try:
        _init_parallel_state(ulysses_size=2, device_type="cpu")
        torch.manual_seed(177)
        config = _tiny_config().vision_config
        config.patch_size = 8
        config.spatial_merge_size = merge
        config.window_size = 4 * merge * config.patch_size
        ops = eager_ops_config()
        ops.attn_implementation = "veomni_sdpa"
        config._set_ops_implementation(ops)
        model = Qwen2_5_VisionTransformerPretrainedModel(config).train()
        output = model.dummy_forward().pooler_output
        assert torch.isfinite(output).all()
        output.float().square().mean().backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, name
                assert torch.isfinite(param.grad).all(), name
        assert any(param.grad is not None and torch.count_nonzero(param.grad) for param in model.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.launch_plan(provider=LaunchSpec(nproc=2, device="cpu"))
@pytest.mark.parametrize("merge", [2, 4])
def test_qwen2_5_vl_dummy_forward_with_sequence_parallel(merge):
    from tests.tools.launch_utils import spawn_with_rendezvous

    spawn_with_rendezvous(_qwen2_5_vl_dummy_sp_worker, args=(merge,), nprocs=2)


def test_qwen2_5_vl_eager_matches_hf_text_only():
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2_5_VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 8))
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=input_ids,
        ours_fwd_kwargs=_mask_kwargs(input_ids),
        logits_equal=True,
    )


@pytest.mark.parametrize("modality", ["image", "video"])
def test_qwen2_5_vl_eager_matches_hf_vision_and_text(modality):
    torch.manual_seed(0)
    config = _tiny_config()
    hf = HFQwen2_5_VLForConditionalGeneration(config)
    ours = _build_ours(config)
    ours.load_state_dict(hf.state_dict())

    input_ids = torch.randint(3, 100, (2, 20))
    vision_inputs = (
        qwen_image_inputs(config, input_ids)
        if modality == "image"
        else qwen_video_inputs(config, input_ids, split_frames=False)
    )
    ids = vision_inputs.pop("input_ids")
    labels = vision_inputs.pop("labels")
    ours_masks = {
        "image_mask": vision_inputs.pop("image_mask"),
        "video_mask": vision_inputs.pop("video_mask"),
    }
    assert_eager_matches_hf(
        hf,
        ours,
        input_ids=ids,
        labels=labels,
        fwd_kwargs=vision_inputs,
        ours_fwd_kwargs=ours_masks,
        logits_equal=True,
    )

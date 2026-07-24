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

"""CPU-always reference-path invariants for HunyuanImage 3.

The GPU E2E covers full training numerics; this file locks the pure-CPU
invariants that the GPU path relies on and that would otherwise only surface
as a loss/attention regression far downstream:

  - segment compiler freezes mixed-parity image coordinates + dense attention
    mask (payload-only full attention);
  - segment compiler rejects unsupported segment patterns AND equal-length
    batches with different image grids (both are silent-corruption vectors);
  - 2D RoPE preserves the official frequency interleave (the exact per-head
    ordering the fused-QKV forward assumes);
  - flow-matching RNG is stable across calls and micro-step-scoped (per-
    micro-step identity is what makes the training bit-reproducible across
    resumes).
"""

from copy import deepcopy

import pytest
import torch

from veomni.models.transformers.hunyuan_image_3.reference_rope import build_reference_2d_rope
from veomni.models.transformers.hunyuan_image_3.sequence_compiler import (
    UnsupportedSegmentPattern,
    build_single_gen_t2i_plan,
    compile_single_gen_t2i_plans,
)
from veomni.schedulers.flow_matching import derive_seed_v1, prepare_reference_flow_batch


@pytest.mark.parametrize(
    ("grid_hw", "expected_y", "expected_x"),
    [
        ((2, 3), [8, 8, 8, 9, 9, 9], [7, 8, 9, 7, 8, 9]),
        ((3, 2), [7, 7, 8, 8, 9, 9], [8, 9, 8, 9, 8, 9]),
    ],
)
def test_segment_compiler_freezes_mixed_parity_coordinates_and_dense_gca(grid_hw, expected_y, expected_x):
    """Mixed-parity grids centre on half-integer offsets that must truncate the
    same way across every consumer (compiler + RoPE + upstream). The dense mask
    also asserts image-payload tokens attend everywhere while text is causal.
    """
    plan = build_single_gen_t2i_plan(sample_id="sample", text_token_count=2, grid_hw=grid_hw)
    metadata = compile_single_gen_t2i_plans([plan])

    payload_start = 6
    payload_stop = 12
    assert metadata["position_ids"][0, 0, payload_start:payload_stop].tolist() == expected_y
    assert metadata["position_ids"][0, 1, payload_start:payload_stop].tolist() == expected_x
    assert metadata["position_ids"][0, :, -1].tolist() == [12, 12]
    assert metadata["timestep_sample_index"][0, 5].item() == 0
    assert metadata["image_output_mask"][0].nonzero().flatten().tolist() == list(range(payload_start, payload_stop))

    attention_mask = metadata["dense_attention_mask"][0]
    assert not attention_mask[5, 6]  # text is causal (cannot see image)
    assert attention_mask[6].all()  # image payload is full-attention
    assert attention_mask[-1].all()


def test_segment_compiler_rejects_patterns_outside_single_gen_t2i_v1():
    """Any change to the payload-region contract must fail loud, not silently
    train on a mis-typed loss role."""
    plan = build_single_gen_t2i_plan(sample_id="sample", text_token_count=2, grid_hw=(2, 2))
    invalid_plan = deepcopy(plan)
    invalid_plan["segments"][1]["regions"][2]["loss_role"] = "none"

    with pytest.raises(UnsupportedSegmentPattern, match="payload region contract"):
        compile_single_gen_t2i_plans([invalid_plan])


def test_segment_compiler_rejects_equal_length_batch_with_different_image_grids():
    """Equal seq_len with mismatched grid_hw silently corrupts RoPE broadcasts."""
    plans = [
        build_single_gen_t2i_plan(sample_id="wide", text_token_count=2, grid_hw=(2, 3)),
        build_single_gen_t2i_plan(sample_id="tall", text_token_count=2, grid_hw=(3, 2)),
    ]

    with pytest.raises(UnsupportedSegmentPattern, match="shared image grid"):
        compile_single_gen_t2i_plans(plans)


def test_reference_2d_rope_preserves_official_frequency_interleave():
    """Lock the per-head frequency interleave the fused-QKV forward assumes."""
    position_ids = torch.tensor([[[0, 2], [0, 3]]], dtype=torch.long)
    cos, sin = build_reference_2d_rope(position_ids, head_dim=8)
    expected_angles = torch.tensor([2.0, 0.3, 0.02, 0.003] * 2)

    torch.testing.assert_close(cos[0, 1], expected_angles.cos())
    torch.testing.assert_close(sin[0, 1], expected_angles.sin())


def test_reference_flow_rng_is_stable_and_stream_scoped():
    """Per-micro-step RNG identity: same context -> same noise; changing the
    micro_step must produce different noise (otherwise resume can silently
    reuse a stale flow sample)."""
    assert derive_seed_v1(1234, 2, 17, 3, "posterior") == 7220414926050156979
    posterior_mean = torch.zeros(2, 4, 2, 2)
    posterior_logvar = torch.zeros_like(posterior_mean)
    vae_config = {"scaling_factor": 0.5, "shift_factor": None}
    step_context = {"train_seed": 1234, "data_replica_rank": 2, "optimizer_step": 17, "micro_step": 3}

    first = prepare_reference_flow_batch(
        posterior_mean, posterior_logvar, vae_config=vae_config, flow_config=None, flow_step_context=step_context
    )
    second = prepare_reference_flow_batch(
        posterior_mean, posterior_logvar, vae_config=vae_config, flow_config=None, flow_step_context=step_context
    )
    changed = prepare_reference_flow_batch(
        posterior_mean,
        posterior_logvar,
        vae_config=vae_config,
        flow_config=None,
        flow_step_context=dict(step_context, micro_step=4),
    )

    for name in first:
        torch.testing.assert_close(first[name], second[name])
    assert not torch.equal(first["noised_latents"], changed["noised_latents"])

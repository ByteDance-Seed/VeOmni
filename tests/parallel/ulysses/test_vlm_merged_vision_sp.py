# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
"""SP=2 equivalence for the collator-merged (exactly-once) VLM vision forward.

Each worker first computes a non-SP reference forward (default single-process
parallel state, in-forward image+video merge), then initializes
``init_parallel_state(ulysses_size=2)`` and runs the full VeOmni SP pipeline:
``MainCollator`` with the model's ``get_pre_sp_collate_func`` /
``get_metadata_collate_func`` hooks -> ``pixel_values_merged`` -> patched
``Qwen3VLModel.forward``. Asserts:

1. the vision tower executes exactly ONCE per rank (forward-hook counter);
2. the gathered SP logits match the non-SP reference;
3. under SP, raw ``pixel_values`` (bypassing the merge hook) raises.

Runs on a mixed image+video packed batch so the merged path is exercised
end-to-end, deepstack included (toy config remaps deepstack indexes).
"""

import os
import tempfile

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import AutoConfig


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TOY_CONFIG = os.path.join(REPO_ROOT, "tests", "toy_config", "qwen3vl_toy")

IMAGE_TOKEN_ID = 2030
VIDEO_TOKEN_ID = 2031


def _build_config():
    config = AutoConfig.from_pretrained(TOY_CONFIG)
    config.image_token_id = IMAGE_TOKEN_ID
    config.video_token_id = VIDEO_TOKEN_ID
    text_overrides = {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        # Keep the toy head_dim (128): rope_parameters.mrope_section sums to
        # head_dim/2 and must stay consistent with it.
        "vocab_size": 2048,
    }
    vision_overrides = {
        "depth": 2,
        "hidden_size": 128,
        "num_heads": 4,
        "intermediate_size": 256,
        "out_hidden_size": 128,
        # Toy deepstack indexes sit above the toy depth; remap so the
        # deepstack path actually runs.
        "deepstack_visual_indexes": [0, 1],
    }
    for key, value in text_overrides.items():
        setattr(config.text_config, key, value)
    for key, value in vision_overrides.items():
        setattr(config.vision_config, key, value)
    # SP awareness lives in VeOmni's registered attention implementation
    # (`veomni_flash_attention_2_with_sp` in veomni/ops/kernels/attention):
    # plain `flash_attention_2` (and the sdpa/eager fallbacks) are not
    # SP-aware. Non-SP it degrades to plain FA2, so the reference uses the
    # same kernel. bf16 end to end (FA requirement).
    attn_impl = "veomni_flash_attention_2_with_sp"
    config._attn_implementation = attn_impl
    config.text_config._attn_implementation = attn_impl
    config.vision_config._attn_implementation = attn_impl
    return config


def _make_features(config):
    """Two packed samples: one mixed image+video, one text-only."""
    vc = config.vision_config
    feat_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size
    gen = torch.Generator().manual_seed(1)

    seq1 = 10
    input_ids1 = torch.randint(10, 1000, (seq1,), generator=gen)
    input_ids1[1] = IMAGE_TOKEN_ID  # 1x2x2 image -> 1 merged token
    input_ids1[3] = VIDEO_TOKEN_ID  # 2x2x2 video -> 2 merged tokens
    input_ids1[4] = VIDEO_TOKEN_ID
    f1 = {
        "input_ids": input_ids1,
        "attention_mask": torch.ones(seq1, dtype=torch.long),
        "labels": input_ids1.clone(),
        "position_ids": torch.arange(seq1, dtype=torch.int64).unsqueeze(0).expand(3, -1).contiguous(),
        "image_mask": input_ids1 == IMAGE_TOKEN_ID,
        "video_mask": input_ids1 == VIDEO_TOKEN_ID,
        "pixel_values": torch.randn(4, feat_dim, generator=gen),
        "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
        "pixel_values_videos": torch.randn(8, feat_dim, generator=gen),
        "video_grid_thw": torch.tensor([[2, 2, 2]], dtype=torch.long),
    }
    seq2 = 6
    input_ids2 = torch.randint(10, 1000, (seq2,), generator=gen)
    f2 = {
        "input_ids": input_ids2,
        "attention_mask": torch.ones(seq2, dtype=torch.long),
        "labels": input_ids2.clone(),
        "position_ids": torch.arange(seq2, dtype=torch.int64).unsqueeze(0).expand(3, -1).contiguous(),
        "image_mask": torch.zeros(seq2, dtype=torch.bool),
        "video_mask": torch.zeros(seq2, dtype=torch.bool),
    }
    return [f1, f2]


def _forward_logits(model, batch, device, visual_calls=None):
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    batch.pop("labels", None)  # logits-only comparison; labels trigger the loss path
    if visual_calls is not None:
        handle = model.model.visual.register_forward_hook(lambda *args: visual_calls.append(1))
    with torch.no_grad():
        logits = model(**batch, use_cache=False).logits
    if visual_calls is not None:
        handle.remove()
    return logits


def _run_worker(rank, world_size, init_file):
    import importlib

    from veomni.arguments.arguments_types import OpsImplementationConfig
    from veomni.data.data_collator import MainCollator
    from veomni.distributed.parallel_state import init_parallel_state
    from veomni.models.auto import _bind_veomni_ops
    from veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu import (
        Qwen3VLForConditionalGeneration,
    )

    # mp.spawn children are fresh interpreters: the generated module's OpSlots
    # (rotary, rms_norm, attention glue) are unbound there. Bind them exactly
    # like build_foundation_model does so the SP async-ulysses path sees the
    # production op implementations (mirrors the gated-deltanet SP test).
    _bind_veomni_ops(
        importlib.import_module("veomni.models.transformers.qwen3_vl.generated.patched_modeling_qwen3_vl_gpu"),
        OpsImplementationConfig(),
    )

    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # ── Non-SP reference BEFORE dist init ───────────────────────────────────
    # With no process group the default parallel state is single-process
    # (sp off); once dist is initialized the default state would fail its
    # world-size check, so the reference must be computed first.
    config = _build_config()
    torch.manual_seed(0)  # identical weights on every rank
    model = Qwen3VLForConditionalGeneration(config).bfloat16().eval().to(device)
    features = _make_features(config)

    ref_collator = MainCollator(
        metadata_collate_func=model.get_metadata_collate_func(),
    )
    ref_batch = ref_collator([{k: v.clone() for k, v in f.items()} for f in features])
    ref_len = ref_batch["input_ids"].shape[-1]
    ref_calls = []
    ref_logits = _forward_logits(model, ref_batch, device, ref_calls)
    assert len(ref_calls) == 1, f"non-SP vision tower ran {len(ref_calls)}x, expected exactly once"

    dist.init_process_group(
        backend="nccl",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        # ── SP pipeline: merged stream through the collator hooks ───────────
        init_parallel_state(dp_size=1, ulysses_size=world_size, device_type="cuda")
        sp_collator = MainCollator(
            metadata_collate_func=model.get_metadata_collate_func(),
            pre_sp_collate_func=model.get_pre_sp_collate_func(),
        )
        sp_batch = sp_collator([{k: v.clone() for k, v in f.items()} for f in features])
        assert "pixel_values_merged" in sp_batch and "pixel_values" not in sp_batch
        sp_calls = []
        sp_logits = _forward_logits(model, sp_batch, device, sp_calls)
        assert len(sp_calls) == 1, f"SP vision tower ran {len(sp_calls)}x, expected exactly once"

        gathered = [torch.empty_like(sp_logits) for _ in range(world_size)]
        dist.all_gather(gathered, sp_logits.contiguous())
        full_logits = torch.cat(gathered, dim=1)[:, :ref_len]
        # Observed bitwise-equal on H100 (the Ulysses exchange feeds the same
        # FA kernel the same full-sequence data); keep a small margin for
        # kernel-version variance.
        torch.testing.assert_close(full_logits, ref_logits, rtol=2e-3, atol=2e-3)

        # ── Raw per-modality streams under SP must be rejected ──────────────
        raw_batch = {
            k: (v.clone() if torch.is_tensor(v) else v) for k, v in sp_batch.items() if k != "pixel_values_merged"
        }
        raw_batch["pixel_values"] = features[0]["pixel_values"]
        with pytest.raises(ValueError, match="pixel_values_merged"):
            _forward_logits(model, raw_batch, device)

        if rank == 0:
            print(f"[vlm_merged_vision_sp] sp={world_size} OK")
    finally:
        dist.barrier()
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2])
def test_vlm_merged_vision_forward_sp_equivalence(world_size):
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA devices")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        init_file = tmp.name
    mp.spawn(_run_worker, args=(world_size, init_file), nprocs=world_size, join=True)
